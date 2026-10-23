# Reference scorer run

The eval v5.1 scorer was run from the directory containing these release files, using one uniform-probability response per question as both candidate and base. Exit status: 0.

Run from this directory (`eval-v5/devset-v1` in a reax-subnet checkout):

```bash
PYTHONPATH=../../src python3 ../../tools/score_v5.py \
  --items dev.jsonl --responses example_responses.jsonl \
  --base-responses example_responses.jsonl --unscaled \
  --min-type-items 0 --seed 0 --resamples 200
```

Exact scorer output:

```json
{
  "A": null,
  "G": null,
  "N": 0.0,
  "N_eq": 0.0,
  "Pub": null,
  "SE_G": null,
  "S_v5.1": 0.5385416666666667,
  "T": 0.8975694444444445,
  "T_eq": 0.8975694444444445,
  "detail": {
    "N_base": 0.0,
    "N_eq_base": 0.0,
    "Pub_base": null,
    "S_v5.1_base": 0.5385416666666667,
    "T_base": 0.8975694444444445,
    "T_eq_base": 0.8975694444444445,
    "coverage": {
      "base_responses": {
        "answered": 300,
        "expected": 300,
        "extra_ids": [],
        "fraction": 1.0
      },
      "responses": {
        "answered": 300,
        "expected": 300,
        "extra_ids": [],
        "fraction": 1.0
      }
    },
    "gap_check": "report-only (no score effect)",
    "gap_scaled": true,
    "ignored_items": 0,
    "internal": {
      "areas_T": {
        "arts": 0.0,
        "knowledge": 3.4722222222222223,
        "language": 0.0,
        "retrieval": 0.0,
        "tools": 0.0
      },
      "areas_T_base": {
        "arts": 0.0,
        "knowledge": 3.4722222222222223,
        "language": 0.0,
        "retrieval": 0.0,
        "tools": 0.0
      },
      "areas_pub": {},
      "overlap_excluded_types": [],
      "overlap_unmatched_types": [],
      "tiers": {},
      "type_items": {
        "arith_word_10": 6,
        "arith_word_4": 6,
        "arxiv_category": 12,
        "best_of_32": 12,
        "causal_yesno": 6,
        "chess_best_move": 3,
        "chord_from_notes": 8,
        "code_output": 9,
        "expense_policy": 21,
        "intent_utterance": 12,
        "logic_multistep": 12,
        "next_api": 8,
        "nhtsa_recall_component": 34,
        "nli": 20,
        "omniscience": 5,
        "poem_pair": 9,
        "python_pep_status": 21,
        "rfc_status": 22,
        "shipping_rate": 22,
        "stance": 9,
        "story_continuation": 10,
        "tool_call_decision": 9,
        "tool_retrieval": 8,
        "when2call": 8,
        "wildguard": 8
      },
      "type_meta": {
        "arith_word_10": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "arith_word_4": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "arxiv_category": {
          "area": "retrieval",
          "gold": true,
          "part": "T"
        },
        "best_of_32": {
          "area": "retrieval",
          "gold": true,
          "part": "T"
        },
        "causal_yesno": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "chess_best_move": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "chord_from_notes": {
          "area": "arts",
          "gold": false,
          "part": "T"
        },
        "code_output": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "expense_policy": {
          "area": "knowledge",
          "gold": false,
          "part": "N"
        },
        "intent_utterance": {
          "area": "retrieval",
          "gold": true,
          "part": "T"
        },
        "logic_multistep": {
          "area": "knowledge",
          "gold": true,
          "part": "T"
        },
        "next_api": {
          "area": "tools",
          "gold": true,
          "part": "T"
        },
        "nhtsa_recall_component": {
          "area": "retrieval",
          "gold": false,
          "part": "N"
        },
        "nli": {
          "area": "language",
          "gold": true,
          "part": "T"
        },
        "omniscience": {
          "area": "knowledge",
          "gold": false,
          "part": "T"
        },
        "poem_pair": {
          "area": "arts",
          "gold": false,
          "part": "T"
        },
        "python_pep_status": {
          "area": "language",
          "gold": false,
          "part": "N"
        },
        "rfc_status": {
          "area": "language",
          "gold": false,
          "part": "N"
        },
        "shipping_rate": {
          "area": "knowledge",
          "gold": false,
          "part": "N"
        },
        "stance": {
          "area": "language",
          "gold": false,
          "part": "T"
        },
        "story_continuation": {
          "area": "language",
          "gold": true,
          "part": "T"
        },
        "tool_call_decision": {
          "area": "tools",
          "gold": true,
          "part": "T"
        },
        "tool_retrieval": {
          "area": "tools",
          "gold": false,
          "part": "T"
        },
        "when2call": {
          "area": "tools",
          "gold": false,
          "part": "T"
        },
        "wildguard": {
          "area": "language",
          "gold": false,
          "part": "T"
        }
      },
      "types_N": {
        "expense_policy": 0.0,
        "nhtsa_recall_component": 0.0,
        "python_pep_status": 0.0,
        "rfc_status": 0.0,
        "shipping_rate": 0.0
      },
      "types_N_base": {
        "expense_policy": 0.0,
        "nhtsa_recall_component": 0.0,
        "python_pep_status": 0.0,
        "rfc_status": 0.0,
        "shipping_rate": 0.0
      },
      "types_T": {
        "arith_word_10": 0.0,
        "arith_word_4": 0.0,
        "arxiv_category": 0.0,
        "best_of_32": 0.0,
        "causal_yesno": 0.0,
        "chess_best_move": 0.0,
        "chord_from_notes": 0.0,
        "code_output": 0.0,
        "intent_utterance": 0.0,
        "logic_multistep": 0.0,
        "next_api": 0.0,
        "nli": 0.0,
        "omniscience": 25.0,
        "poem_pair": 0.0,
        "stance": 0.0,
        "story_continuation": 0.0,
        "tool_call_decision": 0.0,
        "tool_retrieval": 0.0,
        "when2call": 0.0,
        "wildguard": 0.0
      },
      "types_T_base": {
        "arith_word_10": 0.0,
        "arith_word_4": 0.0,
        "arxiv_category": 0.0,
        "best_of_32": 0.0,
        "causal_yesno": 0.0,
        "chess_best_move": 0.0,
        "chord_from_notes": 0.0,
        "code_output": 0.0,
        "intent_utterance": 0.0,
        "logic_multistep": 0.0,
        "next_api": 0.0,
        "nli": 0.0,
        "omniscience": 25.0,
        "poem_pair": 0.0,
        "stance": 0.0,
        "story_continuation": 0.0,
        "tool_call_decision": 0.0,
        "tool_retrieval": 0.0,
        "when2call": 0.0,
        "wildguard": 0.0
      },
      "types_pub": {},
      "wildguard_extra": {
        "wildguard": {
          "f1_const": 43.82284382284382,
          "raw_f1": 32.804232804232804
        }
      }
    },
    "n_N": 120,
    "n_T": 180,
    "n_old": 0,
    "n_pub": 0,
    "params": {
      "a_floor": 1.5,
      "a_se_mult": 2.0,
      "const_baseline": true,
      "gold_weight": 1.2,
      "m_b": 23.4973,
      "omni_acc_weight": 0.5,
      "omniscience_mode": "aa_index",
      "panel": null,
      "resamples": 200,
      "s_b": 16.539081,
      "seed": 0,
      "w_n": 0.4,
      "wildguard_mode": "const_corrected"
    },
    "pool_version": null,
    "pub_scaled": false,
    "scaled": false,
    "w_N": 0.4
  },
  "gap_flag": null,
  "overlap_excluded": false
}
```
