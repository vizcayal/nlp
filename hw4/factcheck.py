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

def is_year(n:float) -> bool:
    year_is = n.is_integer() and 1800<= int(n) <=2040
    return year_is


def numbers_match(c: float, a: float, rel_tol: float = 0.05) -> bool:
    if is_year(c) or is_year(a):
        return c == a
    if abs(c - a) < 1e-6:
        return True
    if round(a) == c and abs(c - a) <= 0.55:
        return True
    denom = max(abs(c), 1e-6)
    return abs(c - a) / denom <= rel_tol

def number_match_features(claim: str, answer: str, rel_tol: float = 0.05) -> Dict[str, float]:
    claim_nums = extract_numbers(claim)
    answer_nums = extract_numbers(answer)
    quant_claim = [n for n in claim_nums if not is_year(n)]
    year_claim = [n for n in claim_nums if is_year(n)]
    #1.¿El claim tiene cifra cuantitativa?
    has_quant = 1.0 if quant_claim else 0.0

    #2.Proporción de cifras cuantitativas del claim respaldadas en el answer
    if quant_claim:
        matched_q = sum(1 for c in quant_claim if any(numbers_match(c, a, rel_tol) for a in answer_nums))
        quant_match_ratio = matched_q / len(quant_claim)
    else:
        quant_match_ratio = 0.0

    # 3. Bandera de discrepancia: el claim pide cifras, el answer tiene cifras, pero no coinciden
    quant_mismatch = 1.0 if (quant_claim and answer_nums and quant_match_ratio < 0.5) else 0.0

    # 4. Proporción de años del claim presentes en el answer
    if year_claim:
        matched_y = sum(1 for y in year_claim if any(y == a for a in answer_nums))
        year_match_ratio = matched_y / len(year_claim)
    else:
        year_match_ratio = 0.0

    return {
        "has_quant": has_quant,
        "quant_match_ratio": quant_match_ratio,
        "quant_mismatch": quant_mismatch,
        "year_match_ratio": year_match_ratio,
    }

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

REFUTATION_WORDS = frozenset("""
fake false hoax satire satirical parody spoof scam fabricated
forged fiction humorous unscientific unfounded baseless
incorrectly inaccurate misattributed untrue denied
""".split())

REFUTATION_PHRASES = (
    "no evidence",
    "no record",
    "no such",
    "not true",
    "never said",
    "no cure",
    "cannot be cured",
    "can t be cured",
    "nothing found",
    "no coherent data",
    "no link",
    "no impact",
    "no effect",
    "didn t work",
    "did not work",
    "remains a concept",
)

def content_tokens(text:str) -> List[str]:
    tokens = tokenize(text)
    content= [stem(t) for t in tokens if t not in STOPWORDS and t not in NEGATIONS]
    return content

def passage_signal_features(claim: str, passages: List[dict]) -> Dict[str, float]:
    if not passages:
        return {
            "unanswerable_ratio": 1.0,
            "bool_yes_ratio": 0.0,
            "bool_no_ratio": 0.0,
            "refute_cues": 0.0,
            "neg_mismatch": 0.0,
            "claim_neg": 1.0 if has_negation(claim) else 0.0,
        }

    claim_tok_list = tokenize(claim)
    claim_tok_set = set(claim_tok_list)
    norm_claim = " ".join(claim_tok_list)
    claim_neg = 1.0 if has_negation(claim) else 0.0

    unans_count = 0
    valid_count = 0
    yes_count = 0
    no_count = 0
    refute_cues = 0
    ans_neg_count = 0

    for p in passages:
        if is_unanswerable(p):
            unans_count += 1
            continue

        valid_count += 1
        raw_ans = str(p.get("answer") or p.get("text") or "").strip()
        raw_tokens = tokenize(raw_ans)
        first_tok = raw_tokens[0] if raw_tokens else ""

        # 1. Respuestas que inician con afirmación o negación directa
        if first_tok in ("yes", "correct", "true"):
            yes_count += 1
        elif first_tok in ("no", "false", "incorrect"):
            no_count += 1

        # 2. Marcadores de refutación en el texto completo (answer + boolean_explanation)
        full_ans = answer_text(p)
        ans_tok_list = tokenize(full_ans)
        ans_tok_set = set(ans_tok_list)
        norm_ans = " ".join(ans_tok_list)

        for w in REFUTATION_WORDS:
            if w in ans_tok_set and w not in claim_tok_set:
                refute_cues += 1

        for phrase in REFUTATION_PHRASES:
            if phrase in norm_ans and phrase not in norm_claim:
                refute_cues += 1

        # 3. Negación en el pasaje
        if has_negation(full_ans):
            ans_neg_count += 1

    denom = max(valid_count, 1)
    ans_neg_ratio = ans_neg_count / denom
    neg_mismatch = ans_neg_ratio if claim_neg == 0.0 else 0.0

    return {
        "unanswerable_ratio": unans_count / len(passages),
        "bool_yes_ratio": yes_count / denom,
        "bool_no_ratio": no_count / denom,
        "refute_cues": float(refute_cues),
        "neg_mismatch": neg_mismatch,
        "claim_neg": claim_neg,
    }

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
        valid_answers = [answer_text(p) for p in passages if not is_unanswerable(p)]
        if not valid_answers:
            return REFUTED
        joined_answer = " ".join(valid_answers)
        # 1. Bloque de Solapamiento Léxico
        overlap_all = overlap_calc(fact, joined_answer)
        overlap_max = max(overlap_calc(fact, a) for a in valid_answers)
        # 2. Bloque de Coincidencia Numérica
        num_feats = number_match_features(fact, joined_answer)
        # 3. Bloque de Señales del Pasaje (Yes/No, Refutación, Negación)
        sig_feats = passage_signal_features(fact, passages)
        # 4. Puntaje combinado:
        #    (+) suman a favor de Supported
        #    (-) restan a favor de Refuted
        score = (
            1.2 * overlap_all
            + 0.8 * overlap_max
            + 0.9 * num_feats["quant_match_ratio"]
            + 0.3 * num_feats["year_match_ratio"]
            + 1.1 * sig_feats["bool_yes_ratio"]
            - 1.4 * num_feats["quant_mismatch"]
            - 1.3 * sig_feats["bool_no_ratio"]
            - 0.9 * sig_feats["refute_cues"]
            - 0.6 * sig_feats["neg_mismatch"]
            - 0.5 * sig_feats["unanswerable_ratio"]
            - 0.45  # umbral base (bias)
        )
        return SUPPORTED if score >= 0.0 else REFUTED


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
