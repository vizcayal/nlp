# Assignment 4: Fact-checking Claims with AVeriTeC

Implement three methods in `factcheck.py`: lexical verification, neural NLI
verification, and verification-question selection. Submit this file only.
The handout gives the complete assignment and course policy; the targets below
remain provisional until course staff confirms them.

## Setup and local commands

Use Python 3.10 and the versions in `requirements.txt`. All grading runs on CPU.
The Torch 2.5.0 pin has no native Intel macOS wheel. Intel Mac users should use
a Linux environment with the same pins. Apple Silicon users should use native
ARM Python rather than Python running through Rosetta.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python factchecking_main.py --mode always_refuted
python factchecking_main.py --mode random
```

After implementing each part, run:

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
python question_decomposition_main.py
```

The three unfinished methods raise `NotImplementedError` until you implement
them. The first entailment run downloads the pinned DeBERTa checkpoint unless
cached. Grading uses a preinstalled model and runs offline. You may use
`--max_examples 10` for a quick verification run, then check the full set.

## Files

- `factcheck.py`: starter classes, NLI wrapper, and the three methods to complete.
- `factchecking_main.py`: supplied driver for Parts 1 and 2.
- `question_decomposition_main.py`: supplied driver for Part 3.
- `data/dev.jsonl`: 260 labeled claims, 130 Supported and 130 Refuted.
- `data/questions_dev.jsonl`: 64 labeled question-selection tasks.
- `data/AVERITEC-LICENSE.txt`: dataset attribution and license.
- `requirements.txt`: pinned runtime dependencies.

Keep the supplied drivers, method signatures, and NLI wrapper unchanged.

## Data and interfaces

The data are JSONL: one JSON object per line. The verification file contains
`claim`, `label`, and `passages`. Each passage contains `question`, `answer`,
and `answer_type`, plus `boolean_explanation` when supplied. The explanation
is evidence text. An Unanswerable item alone does not imply Refuted.

The driver derives `title` from `question` and `text` from `answer`. Source IDs,
URLs, annotation indices, and duplicate stored fields are omitted. Gold labels
are used for local scoring and are not passed to a checker.

| Part | Method | Inputs | Required output |
| --- | --- | --- | --- |
| 1 | `WordOverlapFactChecker.predict(fact, passages)` | Claim and answer-side evidence; no question text. | `Supported` or `Refuted` |
| 2 | `EntailmentFactChecker.predict(fact, passages)` | Claim and supplied question-answer evidence. | `Supported` or `Refuted` |
| 3 | `QuestionDecompositionFactChecker.select_questions(claim, candidates)` | Claim and six candidates containing `id` and `question`. | Two distinct candidate IDs, e.g. `['Q2', 'Q5']` |

For Part 1, use non-neural lexical features. You may tune or fit a small feature
combiner on the visible labels; inference must remain self-contained in
`factcheck.py`. Question wording is excluded to prevent inflated lexical overlap.

For Part 2, pair each question with its answer when forming an NLI premise.
You may join Q+A items into one premise, score items separately, or combine
these approaches. Individual answers can depend on context in other items.
`check_entailments(premises, fact)` returns entailment, contradiction, and neutral
probabilities. `check_entailment(premise, fact)` handles one pair. The wrapper
permits seven pairs per claim and truncates each pair to 256 tokens: up to six
individual evidence items and one optional joined bundle.
**The limit is seven premise–claim pairs per claim, summed across all NLI calls.
Requesting an eighth pair on any claim causes the entire Part 2 run to receive
zero points. Using fewer than seven is allowed.** Use the supplied wrapper for
all NLI inference; do not reset its budget inside `predict`, change its accounting,
or call the underlying model directly.

The Part 3 file contains `claim`, `candidates`, and `gold_ids`. The selector
receives only the claim and candidates. Select specific questions that jointly
cover the claim's central facts, rather than a question asking whether the whole
claim is true. For a comparison between 2021 and 2022, questions asking for the
value in each year form a useful plan. Candidate questions and keys are authored
by course staff from AVeriTeC claims; the original verification annotations are
preserved in Parts 1 and 2. Candidate order varies across menus. Some different claims share the same
candidate menu and require different selections; use the claim to determine
which facts matter. For a claim containing a value judgment, select questions
about its empirical premise. Historical comparisons refer to the period named
in the questions. Answers may be negative or unavailable; the selection is a
plan for checking the facts, not an assumption that the claim is true.

## Scoring and submission

| Part | Points | Provisional full-credit target | Staff run wall limit |
| --- | --- | --- | --- |
| 1 | 40 | 0.63 label macro-F1 | 60 seconds |
| 2 | 40 | 0.82 label macro-F1 | 300 seconds |
| 3 | 20 | 0.70 selection F1 | 60 seconds |

Staff evaluates 200 separate balanced verification claims and 64 separate
question tasks. Parts 1 and 2 scale linearly from zero performance points at
0.50 macro-F1 to 40 at their targets; values outside that interval are capped
at zero or 40. Part 3 scales linearly from one-third to 0.70 selection F1.
Because each task has two gold questions and requires two selections, selection
F1 equals the fraction of gold questions selected. Full credit requires at least
90 of 128 gold questions. Invalid selections earn zero on the affected task.
Each part runs independently. An uncaught exception, invalid verdict label,
or timeout fails that part without changing the other two parts' scores.

Keep Part 2 below approximately 6 GB peak memory. Do not retrieve, train a neural
model, download resources, or call an external model during grading. Do not
hard-code claims, access answer keys inside a checker, or use candidate positions
as the answer. Follow the course collaboration policy and submit your own work.

The NLI model is
[MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli),
revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
