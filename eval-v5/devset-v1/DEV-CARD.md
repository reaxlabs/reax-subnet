# Dev set DEV-CARD — eval v5.1 public dev set (300 items)

`dev.jsonl` sha256 `64ef7f8620961c8e4b9d43e275d4261544a20c801371f68889bbed60cf5bb404`; generated 2026-10-22 (collected date), seed 20261023 (procedural) / 20261023+offsets (real-data selection).

**What it is.** 300 public items in the exact v5.1 mirror format (`tier: mirror`, `part` T / N, `public: true`): 180 T items across the five areas at the DI weights and 120 N (new-domain) items. Weight 0 in the score; disjoint from the private test pools; the private pools use different sources. No item text was written by a language model; gold labels come from real records (human or agency labels) or from computed constructions (executed code, solver, engine, arithmetic). Row format, prompts, readout and scoring: `FORMAT.md`.

## Counts per part and area

| part | area | items | questions |
|---|---|---|---|
| N | knowledge | 43 | 127 |
| N | language | 43 | 43 |
| N | retrieval | 34 | 34 |
| T | knowledge | 47 | 47 |
| T | language | 47 | 63 |
| T | retrieval | 36 | 36 |
| T | tools | 33 | 65 |
| T | arts | 17 | 17 |
| T total | | 180 | |
| N total | | 120 | |

T target at DI weights (knowledge 47, language 47, retrieval 36, tools 33, arts 17) is met exactly; N covers 5 task types / formats: many-option choice (26 options), 7- and 5-option choice, multi-question yes/no (5 questions per item) and two decision-under-rules tasks.

## Counts per task type, sources and licences

| task_type | part | area | gold | items | questions | options | fresh items | source id(s) | licence / terms |
|---|---|---|---|---|---|---|---|---|---|
| expense_policy | N | knowledge |  | 21 | 105 | 2–2 | 0 | procedural_expense_policy | none (own generator) |
| shipping_rate | N | knowledge |  | 22 | 22 | 4–5 | 0 | procedural_shipping_tariff | none (own generator) |
| python_pep_status | N | language |  | 21 | 21 | 5–5 | 0 | python_peps_repository_public_domain | Python PEPs (placed in the public domain / CC0) |
| rfc_status | N | language |  | 22 | 22 | 7–7 | 0 | rfc_editor_index_public | RFC Editor index and RFC abstracts (IETF Trust legal provisions allow reproduction; short abstracts) |
| nhtsa_recall_component | N | retrieval |  | 34 | 34 | 26–26 | 0 | nhtsa_recalls_api_public_domain | NHTSA recalls API (US government, public domain) |
| arith_word_10 | T | knowledge |  | 6 | 6 | 10–10 | 6 | publicgen_arithmetic | none (own typed generator) |
| arith_word_4 | T | knowledge |  | 6 | 6 | 4–4 | 6 | publicgen_arithmetic | none (own typed generator) |
| causal_yesno | T | knowledge |  | 6 | 6 | 2–2 | 6 | publicgen_causal | none (own generator) |
| chess_best_move | T | knowledge |  | 3 | 3 | 28–41 | 3 | procedural_random_playout_stockfish_gold | none: positions from engine-vs-random playouts; gold = Stockfish (GPLv3 engine, tool only), depth 16 with a >= 1.5 pawn margin, re-checked at depth 22 |
| code_output | T | knowledge |  | 9 | 9 | 2–10 | 9 | publicgen_code | none (own generator) |
| logic_multistep | T | knowledge | yes | 12 | 12 | 2–7 | 12 | publicgen_logic | none (own generator) |
| omniscience | T | knowledge |  | 5 | 5 | 5–5 | 0 | nobelprize_org_api_cc0 | Nobel Prize API, CC0 |
| nli | T | language | yes | 20 | 20 | 3–3 | 0 | snli_test_cc_by_sa_4 | SNLI (Bowman et al. 2015), CC BY-SA 4.0 — attribution + share-alike apply to these rows |
| stance | T | language |  | 9 | 9 | 2–2 | 0 | ibm_argument_quality_30k_cc_by_sa_3 | IBM Argument Quality Ranking 30k (Gretz et al. 2020), CC BY-SA 3.0 — attribution + share-alike apply to these rows |
| story_continuation | T | language | yes | 10 | 10 | 4–4 | 0 | standard_ebooks_cc0_austen_pride_and_prejudice, standard_ebooks_cc0_doyle_a_study_in_scarlet, standard_ebooks_cc0_doyle_adventures_of_sherlock_holmes, standard_ebooks_cc0_shelley_frankenstein, standard_ebooks_cc0_stoker_dracula | Standard Ebooks editions (CC0; texts public domain in the US): Doyle (Adventures of Sherlock Holmes, A Study in Scarlet), Austen, Stoker, Shelley |
| wildguard | T | language |  | 8 | 24 | 2–2 | 0 | aegis2_prompts_cc_by_4_plus_fixed_templates | NVIDIA Aegis AI Content Safety 2.0 prompts and human-labelled safe responses, CC BY 4.0 — attribution; refusal texts are fixed templates written for this set |
| arxiv_category | T | retrieval | yes | 12 | 12 | 41–41 | 12 | arxiv_api_metadata_cc0 | arXiv API metadata (titles, abstracts), CC0 |
| best_of_32 | T | retrieval | yes | 12 | 12 | 32–32 | 12 | arxiv_api_metadata_cc0 | arXiv API metadata (titles, abstracts), CC0 |
| intent_utterance | T | retrieval | yes | 12 | 12 | 59–59 | 0 | amazon_massive_1_1_en_us_cc_by_4 | MASSIVE 1.1 en-US (Amazon), CC BY 4.0 — attribution |
| next_api | T | tools | yes | 8 | 8 | 25–56 | 8 | publicgen_fictional_api | procedurally generated fictional APIs (own word lists and templates, CC0 1.0); no third-party API text; requests are fixed templates with synthetic values |
| tool_call_decision | T | tools | yes | 9 | 41 | 2–2 | 9 | publicgen_fictional_api | procedurally generated fictional APIs (own word lists and templates, CC0 1.0); no third-party API text; requests are fixed templates with synthetic values |
| tool_retrieval | T | tools |  | 8 | 8 | 32–32 | 8 | publicgen_fictional_api | procedurally generated fictional APIs (own word lists and templates, CC0 1.0); no third-party API text; requests are fixed templates with synthetic values |
| when2call | T | tools |  | 8 | 8 | 4–4 | 8 | publicgen_fictional_api | procedurally generated fictional APIs (own word lists and templates, CC0 1.0); no third-party API text; requests are fixed templates with synthetic values |
| chord_from_notes | T | arts |  | 8 | 8 | 108–108 | 0 | procedural_chord_construction | none (computed construction: root = lowest note, tones from the quality) |
| poem_pair | T | arts |  | 9 | 9 | 2–2 | 0 | poetrydb_public_domain_poems | PoetryDB (public-domain poems), the original four lines are re-located in the poem text; the other version differs only in the last word of line 4 (deterministic edit) |

`gold` = counts x1.2 inside its area (DI gold benchmark formats). Types in half `aa`: `omniscience` (IDK offered), `when2call`, `wildguard`.

## Required item kinds present

* IDK fact items (omniscience): 5 (`dev-omniscience-1..5`).
* WildGuard 3-question rows: 8.
* Items with more than 26 options: 62 items (up to 108 options; live v0 serves such questions capped at 26: chess, MASSIVE utterance intent, arXiv category, chord, best-of-32, tool retrieval, next API, NHTSA 26).
* Multi-question items: 38 (tool call decision, logic yes/no families, expense policy, WildGuard).

## Label balance and trivial baselines (dev sample, small n)

| task_type | chance | first option | longest option | most frequent gold key | yes share (yes/no) |
|---|---|---|---|---|---|
| expense_policy | 0.5 | 0.467 | 0.467 | 0.533 | 0.467 |
| shipping_rate | 0.214 | 0.273 | 0.318 | 0.273 |  |
| python_pep_status | 0.2 | 0.238 | 0.238 | 0.238 |  |
| rfc_status | 0.143 | 0.182 | 0.182 | 0.182 |  |
| nhtsa_recall_component | 0.038 | 0.029 | 0.029 | 0.147 |  |
| arith_word_10 | 0.1 | 0.167 | 0.333 | 0.167 |  |
| arith_word_4 | 0.25 | 0.333 | 0.333 | 0.333 |  |
| causal_yesno | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| chess_best_move | 0.029 | 0.0 | 0.333 | 0.333 |  |
| code_output | 0.212 | 0.333 | 0.222 | 0.333 |  |
| logic_multistep | 0.345 | 0.333 | 0.333 | 0.167 | 0.5 |
| omniscience | 0.167 | 0.2 | 0.2 | 0.2 |  |
| nli | 0.333 | 0.35 | 0.35 | 0.35 |  |
| stance | 0.5 | 0.556 | 0.556 | 0.556 |  |
| story_continuation | 0.25 | 0.3 | 0.2 | 0.3 |  |
| wildguard | 0.5 | 0.333 | 0.333 | 0.667 | 0.333 |
| arxiv_category | 0.024 | 0.0 | 0.083 | 0.167 |  |
| best_of_32 | 0.031 | 0.083 | 0.0 | 0.167 |  |
| intent_utterance | 0.017 | 0.0 | 0.083 | 0.167 |  |
| next_api | 0.025 | 0.125 | 0.0 | 0.25 |  |
| tool_call_decision | 0.5 | 0.512 | 0.512 | 0.512 | 0.512 |
| tool_retrieval | 0.031 | 0.125 | 0.0 | 0.125 |  |
| when2call | 0.25 | 0.25 | 0.25 | 0.25 |  |
| chord_from_notes | 0.009 | 0.0 | 0.0 | 0.125 |  |
| poem_pair | 0.5 | 0.556 | 0.222 | 0.556 |  |

Gold positions of single-question multiple-choice types are spread evenly over the option positions by construction; yes/no types are ~50/50 except where the natural base rate matters (WildGuard: harmful-response label is always `no`).

## Quality checks

* 300 / 300 rows pass `validate_mirror_row`; ids unique (`dev-<task_type>-<n>`); 0 duplicate instructions.
* Gold check on 100 % of items with an independent re-derivation from the raw source or a second implementation: 0 mismatches in every task type (`work/verify_real.json`, procedural types `scripts/proc/verify_*.py`).
* Aggregate leakage check against the final pools and sibling outputs: 0 equal source ids, 0 exact duplicate instructions; maximum 13-gram overlap fraction 0.4. Detailed source names remain internal.

## Known limits (honest list)

* Dev samples per task type are small (3–34 items): per-type scores are very noisy.
* WildGuard dev rows have no harmful responses (all 'response harmful?' answers are `no`) because no human-labelled harmful responses with a permissive licence were found; that label scores 0 in the dev set.
* Tool-API items use fictional, procedurally generated APIs (invented names, operations and parameters) with fixed request templates and synthetic parameter values; they are deliberately simpler and more regular than real API catalogues, and parameter values can look odd.

## Licences and attribution

| Source | Licence | Licence page |
|---|---|---|
| `aegis2_prompts_cc_by_4_plus_fixed_templates` | CC BY 4.0 for Aegis 2.0 source prompts/responses; added fixed templates are original | https://creativecommons.org/licenses/by/4.0/ |
| `amazon_massive_1_1_en_us_cc_by_4` | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| `arxiv_api_metadata_cc0` | CC0 1.0 for descriptive metadata | https://info.arxiv.org/help/api/tou.html |
| `ibm_argument_quality_30k_cc_by_sa_3` | CC BY-SA 3.0; attribution and share-alike apply to these rows | https://creativecommons.org/licenses/by-sa/3.0/ |
| `nhtsa_recalls_api_public_domain` | U.S. federal government work; public domain under 17 U.S.C. § 105 | https://www.copyright.gov/title17/92chap1.html#105 |
| `nobelprize_org_api_cc0` | CC0 1.0; cite the Nobel Prize API as the source | https://www.nobelprize.org/about/terms-of-use-for-api-nobelprize-org-and-data-nobelprize-org/#licence |
| `poetrydb_public_domain_poems` | CC BY 4.0 for PoetryDB data; the poem texts are public domain in the United States | https://github.com/thundercomb/poetrydb/blob/master/DATA_LICENSE.md |
| `procedural_chord_construction` | Original computed items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `procedural_expense_policy` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `procedural_random_playout_stockfish_gold` | Original generated items; CC0 1.0; Stockfish GPL-3.0 used only as a labeling tool and is not included | https://creativecommons.org/publicdomain/zero/1.0/ ; https://www.gnu.org/licenses/gpl-3.0.html |
| `procedural_shipping_tariff` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `publicgen_arithmetic` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `publicgen_causal` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `publicgen_code` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `publicgen_logic` | Original generated items; CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `publicgen_fictional_api` | Original procedurally generated fictional APIs (own word lists and templates); CC0 1.0 | https://creativecommons.org/publicdomain/zero/1.0/ |
| `python_peps_repository_public_domain` | Public domain / CC0 1.0 | https://peps.python.org/pep-0001/#copyright |
| `rfc_editor_index_public` | RFC Editor reproduction under the IETF Trust Legal Provisions | https://www.rfc-editor.org/series/rfc-use/ |
| `snli_test_cc_by_sa_4` | CC BY-SA 4.0; attribution and share-alike apply to these rows | https://creativecommons.org/licenses/by-sa/4.0/ |
| `standard_ebooks_cc0_austen_pride_and_prejudice` | CC0 1.0 edition; source text is public domain in the United States | https://standardebooks.org/about |
| `standard_ebooks_cc0_doyle_a_study_in_scarlet` | CC0 1.0 edition; source text is public domain in the United States | https://standardebooks.org/about |
| `standard_ebooks_cc0_doyle_adventures_of_sherlock_holmes` | CC0 1.0 edition; source text is public domain in the United States | https://standardebooks.org/about |
| `standard_ebooks_cc0_shelley_frankenstein` | CC0 1.0 edition; source text is public domain in the United States | https://standardebooks.org/about |
| `standard_ebooks_cc0_stoker_dracula` | CC0 1.0 edition; source text is public domain in the United States | https://standardebooks.org/about |

### Attribution lines

* **SNLI:** Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. “A large annotated corpus for learning natural language inference.” Proceedings of EMNLP 2015. Stanford Natural Language Inference Corpus v1.0. [Dataset](https://nlp.stanford.edu/projects/snli/). CC BY-SA 4.0. Attribution is required; share-alike applies to the `nli` rows and adaptations of them.
* **Aegis 2.0:** Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese, and Christopher Parisien. 2025. “AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails.” NAACL 2025. [NVIDIA dataset](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0). CC BY 4.0.
* **MASSIVE:** Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gökhan Tür, and Prem Natarajan. 2022. “MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages.” [Paper](https://arxiv.org/abs/2204.08582); [dataset](https://huggingface.co/datasets/AmazonScience/massive). CC BY 4.0.
* **IBM Argument Quality Ranking 30k:** Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit Aharonov, and Noam Slonim. 2020. “A Large-Scale Dataset for Argument Quality Ranking: Construction and Analysis.” AAAI 34(5), DOI [10.1609/aaai.v34i05.6285](https://doi.org/10.1609/aaai.v34i05.6285). CC BY-SA 3.0. Attribution is required; share-alike applies to the `stance` rows and adaptations of them.
* **PoetryDB:** “Poetry data from PoetryDB” ([poetrydb.org](https://poetrydb.org)), CC BY 4.0; poem texts are public domain in the United States.

The attribution and share-alike notices above apply to the indicated rows. The dataset includes transformed prompts or options only where noted; it does not change those source terms.
