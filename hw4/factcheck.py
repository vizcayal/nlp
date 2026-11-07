"""Assignment 4 starter. Complete the three TODO methods; submit this file only."""

import math
import re
from typing import Dict, List, Sequence

import numpy as np
import torch


SUPPORTED = "Supported"
REFUTED = "Refuted"
LABELS = (SUPPORTED, REFUTED)


class FactExample:
    def __init__(self, fact: str, passages: List[dict], label: str, example_id: str = ""):
        self.fact = fact
        self.passages = passages
        self.label = label
        self.example_id = example_id


def answer_text(passage: Dict) -> str:
    """Answer text with its Boolean explanation, excluding the question."""
    answer = str(passage.get("answer") or passage.get("text") or "").strip()
    explanation = str(passage.get("boolean_explanation") or "").strip()
    if explanation and explanation.lower() not in answer.lower():
        return f"{answer} {explanation}".strip()
    return answer


def qa_text(passage: Dict) -> str:
    """Self-contained Q+A premise for Part 2's NLI model."""
    question = str(passage.get("question") or passage.get("title") or "").strip()
    answer = answer_text(passage)
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}"
    return answer or question


def is_unanswerable(passage: Dict) -> bool:
    """The source dataset uses this answer type for unanswered questions."""
    kind = str(passage.get("answer_type") or "").strip().lower()
    text = answer_text(passage).strip().lower().rstrip(" .!?:;")
    return kind == "unanswerable" or not text or text == "no answer could be found"

def normalize_text(text: str) ->str:
    """lowercase + remove punctuation, thousands separators (keep dots)"""
    text  = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    return text

def tokenize(text:str)->List[str]:
    text = normalize_text(text)
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", text)

def stem(token:str) -> str:
    """stemmer"""

    if token.endswith("ies") and len(token) >4:
        token= token[:len(token)-3] + "y"

    if token.isdigit() or len(token)<= 3:
        return token

    for suffix in ("ing","ed","es","s","ly"):
        if token.endswith(suffix) and len(token) - len(suffix) >=3:
            token = token[:len(token)-len(suffix)]
            break
    return token

def has_negation(text:str) ->bool:
    tokens =tokenize(text)
    return any(t in NEGATIONS  for t in tokens )

def extract_numbers(text:str) -> List[float]:
    tokens = tokenize(text)
    numbers= []
    for t in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?",t):
            numbers.append(float(t))
    return numbers

def overlap_calc(claim:str, answer:str) -> float:
    claim_tokens = content_tokens(claim)
    answer_tokens = content_tokens(answer)
    set_claim = set(claim_tokens)
    set_answer= set(answer_tokens )
    if not set_claim:
        overlap = 0
        return overlap

    intersection = set_claim &  set_answer
    overlap = len(intersection) / len(set_claim)
    return overlap


STOPWORDS = frozenset("""
a an the and or but as at by for from in into of on to with
about above after against between during through
is am are was were be been being
do does did have has had
it its this that these those
i me my we our you your he him his she her they them their
who what when where which how
can could may might must shall should will would
there here then than too very also s
""".split())

NEGATIONS = frozenset("""
not no never none nobody nothing neither nor without
""".split())

def content_tokens(text:str) -> List[str]:
    tokens = tokenize(text)
    content= [stem(t) for t in tokens if t not in STOPWORDS and t not in NEGATIONS]
    return content


class EntailmentModel:
    """Provided batching and probability wrapper around the pretrained NLI model."""

    def __init__(self, model, tokenizer, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()
        names = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        indices = {}
        for target in ("contradiction", "entailment", "neutral"):
            matches = [idx for idx, name in names.items() if target in name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {target} label in model.config.id2label")
            indices[target] = matches[0]
        if len(set(indices.values())) != 3:
            raise ValueError("NLI labels must use distinct logit indices")
        self.indices = indices
        self._budget_exceeded = False
        self.reset_budget()

    def reset_budget(self):
        """Called by the driver before each claim; do not change this helper."""
        self.pairs_used = 0

    def assert_budget(self):
        if self._budget_exceeded or self.pairs_used > 7:
            raise ValueError("Part 2 exceeded seven NLI pairs on a claim; the entire part receives zero")

    def check_entailments(self, premises: Sequence[str], hypothesis: str, batch_size: int = 8):
        """Return one probability dictionary per premise, in input order."""
        if isinstance(premises, str) or not isinstance(hypothesis, str):
            raise ValueError("Pass a sequence of premises and a string hypothesis")
        premises = list(premises)
        if not all(isinstance(p, str) for p in premises) or batch_size <= 0:
            raise ValueError("Premises must be strings and batch_size positive")
        self.assert_budget()
        self.pairs_used += len(premises)
        if self.pairs_used > 7:
            self._budget_exceeded = True
            raise ValueError("At most seven NLI pairs per claim: six items and one optional bundle")
        results = []
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(premises), batch_size):
                batch = list(premises[start:start + batch_size])
                inputs = self.tokenizer(
                    batch,
                    [hypothesis] * len(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()
                for row in probabilities:
                    results.append({name: float(row[idx]) for name, idx in self.indices.items()})
        return results

    def check_entailment(self, premise: str, hypothesis: str):
        return self.check_entailments([premise], hypothesis, batch_size=1)[0]


class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise NotImplementedError


class RandomGuessFactChecker(FactChecker):
    def __init__(self, seed: int = 13):
        self.rng = np.random.default_rng(seed)

    def predict(self, fact: str, passages: List[dict]) -> str:
        return str(self.rng.choice(LABELS))


class AlwaysRefutedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return REFUTED



class WordOverlapFactChecker(FactChecker):
    """Part 1: a non-neural answer-side word-overlap classifier."""

    def predict(self, fact: str, passages: List[dict]) -> str:
        # TODO: Use answer-side lexical features; return Supported or Refuted.
        raise NotImplementedError("Complete Part 1")


class EntailmentFactChecker(FactChecker):
    """Part 2: use fixed gold Q+A evidence with the supplied NLI wrapper."""

    def __init__(self, ent_model: EntailmentModel, max_nli_evidence: int = 6):
        if max_nli_evidence <= 0:
            raise ValueError("max_nli_evidence must be positive")
        self.ent_model = ent_model
        self.max_nli_evidence = max_nli_evidence

    def predict(self, fact: str, passages: List[dict]) -> str:
        # TODO: Score Q+A premises with self.ent_model; return Supported or Refuted.
        raise NotImplementedError("Complete Part 2")


class QuestionDecompositionFactChecker:
    """Part 3: select two distinct IDs from candidates containing id/question."""

    def select_questions(self, claim: str, candidates: List[dict]) -> List[str]:
        # TODO: Return the IDs of two complementary verification questions.
        raise NotImplementedError("Complete Part 3")
