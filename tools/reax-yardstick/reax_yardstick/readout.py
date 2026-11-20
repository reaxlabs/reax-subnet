"""Prompts and the fast next-token label readout: the same code path the REAX preview uses.

For every question the model sees one chat (system + user message), rendered by the served model's own chat
template (vLLM `/tokenize` with `add_generation_prompt`). Every option key ("label") is scored by its log-likelihood
as the assistant's reply: one-token labels in one request, longer labels teacher-forced one token at a time through a
label-prefix trie. Probabilities are a softmax over the label scores; the chosen answer is the highest-scoring label
(first one on a tie). No sampling, logit bias, truncation, label filtering or prompt changes.

Two prompt styles, exactly as in the preview:
  * "hf" (PV-HF): HF Decision Index style renderer (`render_prompt`), score rubrics turned into choices first
  * "aa" (PV-HLE, PV-W2C, PV-ONET): the AA-style renderer (`render_prompt_aa`)
The server must be vLLM's OpenAI-compatible server (the preview pins vLLM 0.33.0): `/tokenize` and `/v1/completions`
with `logprob_token_ids`.
"""
import json
import math
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------------------------------------- prompts
SYSTEM_PROMPT = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."
LABEL_SYSTEM = 'You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain.'


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_prompt(state, question):
    """HF-style user message (PV-HF)."""
    lines = ["State:", text(state) if state not in ("", None, {}, []) else "(empty)", "", "Question:", text(question.get("instructions", "")), "", "Options:"]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    for key, description in criteria.items():
        lines.append(f"{key}: {key if description is None else text(description)}")
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def render_prompt_aa(state, q):
    """AA-style user message (PV-HLE, PV-W2C, PV-ONET)."""
    opts = q.get('criteria', {'false': 'False', 'true': 'True'})
    return 'State:\n' + (state if isinstance(state, str) else json.dumps(state)) + '\n\nQuestion:\n' + q['instructions'] + '\n\nOptions:\n' + '\n'.join(f'{k}: {v}' for k, v in opts.items()) + '\n\nReply with exactly one option key.'


def normalized(logprobs):
    if not logprobs or any(not math.isfinite(v) for v in logprobs):
        raise ValueError('finite candidate scores required')
    top = max(logprobs)
    values = [math.exp(x - top) for x in logprobs]
    return [x / sum(values) for x in values]


# ------------------------------------------------------------------------------------- HF score rubrics and checks
def hf_wire(row):
    """HF score rubrics are ordinal categories: score every supplied level as a choice, then restore the score type."""
    questions = {}
    for key, q in row['questions'].items():
        q = dict(q)
        if q['type'] == 'score':
            q['type'] = 'choice'
            if isinstance(q['criteria'], list):
                q['criteria'] = {str(i): v for i, v in enumerate(q['criteria'])}
        questions[key] = q
    return {**row, 'questions': questions}


def restore_scores(row, response):
    for key, q in row['questions'].items():
        if q['type'] == 'score':
            a = response['answers'][key]
            a['type'] = 'score'
            a['score'] = a.pop('choice')
    return response


def validate(questions, response):
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError('response must answer exactly all question keys')
    for k, q in questions.items():
        a = answers[k]
        if a.get('type') != q['type']:
            raise ValueError('answer type mismatch')
        if q['type'] == 'choice':
            probs = a.get('probabilities', {})
            if set(probs) != set(q['criteria']) or a.get('choice') not in probs:
                raise ValueError('invalid choice or missing probabilities')
            ps = list(probs.values())
            if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in ps):
                raise ValueError('nonfinite or out-of-range probability')
            if abs(sum(ps) - 1) > 0.01:
                raise ValueError('probabilities must sum to one')
        elif q['type'] == 'noul':
            p = a.get('noul')
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('invalid noul probability')
        else:
            raise ValueError('unsupported question type')
    return response


# ---------------------------------------------------------------------------------------------------------- client
class Client:
    """Minimal JSON-over-HTTP client for one item (the label tokenisation cache lives per item, as in the preview)."""

    def __init__(self, base_url, model, timeout=120, api_key=''):
        self.base = base_url.rstrip('/')
        if self.base.endswith('/v1'):
            self.base = self.base[:-3]
        self.model, self.timeout, self.api_key = model, timeout, api_key
        self.calls = 0
        self.labels_cache = {}

    def post(self, path, payload):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), headers)
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as f:
                return json.load(f)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            raise RuntimeError(f'HTTP {exc.code}: {detail[:500]}') from exc


# ------------------------------------------------------------------------------------------------ fast readout
def fast_score(client, row, style):
    """Answers for one request row. `row` must already be hf_wire()d for style "hf"."""
    if style == 'hf':
        system, render = SYSTEM_PROMPT, render_prompt
    else:
        system, render = LABEL_SYSTEM, render_prompt_aa
    post = client.post
    answers = {}
    cache = client.labels_cache
    for qid, q in row['questions'].items():
        labels = ['false', 'true'] if q['type'] == 'noul' else list(q['criteria'])
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': render(row['state'], q)}]
        prompt = post('/tokenize', {'model': client.model, 'messages': messages, 'add_generation_prompt': True})['tokens']
        for label in labels:
            if label not in cache:
                cache[label] = post('/tokenize', {'model': client.model, 'prompt': label, 'add_special_tokens': False})['tokens']
            if not cache[label]:
                raise ValueError('empty label tokenization')
        nodes = {}
        for label in labels:
            suffix = cache[label]
            for j, token in enumerate(suffix):
                nodes.setdefault(tuple(suffix[:j]), set()).add(token)
        values = {}
        for prefix, tokens in nodes.items():
            payload = {'model': client.model, 'prompt': prompt + list(prefix), 'max_tokens': 1, 'temperature': 0, 'logprobs': 0,
                       'logprob_token_ids': sorted(tokens), 'return_tokens_as_token_ids': True, 'ignore_eos': True,
                       'skip_special_tokens': False, 'add_special_tokens': False}
            result = post('/v1/completions', payload)
            top = result['choices'][0]['logprobs']['top_logprobs'][0]
            for token in tokens:
                key = 'token_id:' + str(token)
                v = top.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError('missing explicit next-token label logprob ' + key)
                values[prefix, token] = v
        scores = [sum(values[tuple(cache[label][:j]), t] for j, t in enumerate(cache[label])) for label in labels]
        probs = normalized(scores)
        if q['type'] == 'noul':
            answers[qid] = {'type': 'noul', 'noul': probs[1]}
        else:
            answers[qid] = {'type': q['type'], 'choice': labels[max(range(len(labels)), key=lambda j: scores[j])],
                            'probabilities': dict(zip(labels, probs))}
    return {'answers': answers}


def answer_row(client, row, style):
    """One item through the readout, with the same post-processing as the preview's pool runner."""
    if style == 'hf':
        wired = hf_wire(row)
        return restore_scores(row, validate(wired['questions'], fast_score(client, wired, 'hf')))
    return fast_score(client, row, 'aa')
