#!/usr/bin/env python3
"""Pre-serve normalisation to a plain text-only causal LM (same step the preview runs before serving a model).

Why: many decision-model checkpoints are saved as multimodal *ForConditionalGeneration / *Model (headless) with
legacy key layouts. The HF DI stock engine loads AutoModelForCausalLM and scores option-key log-probs with the LM head.
We therefore serve exactly the text LM + LM head:
  * config := text_config (if present) with architectures = [<Family>ForCausalLM]
  * weights := language-model tensors renamed to model.<...>; lm_head kept if present
  * no own lm_head and tie_word_embeddings false  -> HEADLESS: abort (needs the author's native engine; a random head
    would be garbage). Exit code 3 + receipt.
Tensor values are copied unchanged (safetensors, same dtype). Writes ../normalize-receipt.json. Needs torch + safetensors (present in a vLLM environment).
"""
import json, sys, os
from pathlib import Path

CAUSAL = {'qwen3_5': 'Qwen3_5ForCausalLM', 'qwen3_5_text': 'Qwen3_5ForCausalLM', 'qwen3_5_moe': 'Qwen3_5MoeForCausalLM',
          'qwen3_5_moe_text': 'Qwen3_5MoeForCausalLM', 'gemma4': 'Gemma4ForCausalLM', 'gemma4_text': 'Gemma4ForCausalLM'}
LM_PREFIXES = ['model.language_model.', 'language_model.model.', 'language_model.', 'model.text_model.', 'thinker.model.']
DROP = ('visual.', 'model.visual.', 'vision_tower.', 'model.vision_tower.', 'embed_vision.', 'model.embed_vision.',
        'audio_tower.', 'model.audio_tower.', 'embed_audio.', 'model.embed_audio.', 'mtp.', 'model.vision_embedder.',
        'multi_modal_projector.', 'model.multi_modal_projector.', 'readout.', 'talker.', 'code2wav.')


def keymap(k):
    if k.startswith('lm_head.') or k.startswith('language_model.lm_head.'):
        return 'lm_head.' + k.split('lm_head.', 1)[1]
    if k.startswith(DROP):
        return None
    for p in LM_PREFIXES:
        if k.startswith(p):
            return 'model.' + k[len(p):]
    if k.startswith('model.'):
        return k
    if k.split('.')[0] in ('embed_tokens', 'layers', 'norm'):
        return 'model.' + k
    return None


def main(model_dir):
    d = Path(model_dir); rec = {'actions': []}
    out = d.parent / 'normalize-receipt.json'
    cfg = json.loads((d / 'config.json').read_text())
    mt = cfg.get('model_type', ''); arch = ' '.join(cfg.get('architectures') or [])
    tc = cfg.get('text_config') or {}
    tie = cfg.get('tie_word_embeddings', tc.get('tie_word_embeddings'))
    from safetensors import safe_open
    shards = sorted(d.glob('*.safetensors'))
    keys = {}
    for s in shards:
        with safe_open(str(s), 'pt') as f:
            for k in f.keys(): keys[k] = s
    has_head = any('lm_head.' in k for k in keys)
    rec.update(model_type=mt, architectures=arch, tie=tie, has_lm_head=has_head, n_keys=len(keys))
    if not has_head and not tie:
        rec['actions'].append('HEADLESS: no lm_head and untied embeddings; stock LM readout impossible')
        out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec)); sys.exit(3)
    if mt not in CAUSAL:
        rec['actions'].append('model_type not in conversion table: served as-is')
        out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec)); return
    target = CAUSAL[mt]
    mapped = {k: keymap(k) for k in keys}
    already = arch == target and all(v == k for k, v in mapped.items() if v)
    if already and all(mapped.values()):
        rec['actions'].append('already plain causal LM: no-op')
        out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec)); return
    import torch
    from safetensors.torch import save_file
    newcfg = dict(tc) if tc else dict(cfg)
    for k in ('torch_dtype', 'dtype', 'transformers_version'):
        if k in cfg and k not in newcfg: newcfg[k] = cfg[k]
    newcfg['architectures'] = [target]
    newcfg['tie_word_embeddings'] = bool(tie) and not has_head
    if tc and 'model_type' in tc: newcfg['model_type'] = tc['model_type']
    kept = dropped = 0; index = {}
    for i, s in enumerate(shards):
        tensors = {}
        with safe_open(str(s), 'pt') as f:
            for k in f.keys():
                nk = mapped[k]
                if nk is None: dropped += 1; continue
                tensors[nk] = f.get_tensor(k); kept += 1
        name = f'model-norm-{i:05d}.safetensors'
        if tensors:
            save_file(tensors, str(d / name), metadata={'format': 'pt'})
            for nk in tensors: index[nk] = name
        os.remove(s)
    (d / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': index}, indent=1))
    (d / 'config.json.orig').write_text(json.dumps(cfg, indent=1))
    (d / 'config.json').write_text(json.dumps(newcfg, indent=1))
    for f in ('preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'):
        if (d / f).exists(): (d / f).rename(d / (f + '.disabled'))
    rec['actions'].append(f'converted to {target}: kept {kept} tensors, dropped {dropped} (vision/audio/mtp/readout), tie={newcfg["tie_word_embeddings"]}')
    out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))


if __name__ == '__main__':
    main(sys.argv[1])
