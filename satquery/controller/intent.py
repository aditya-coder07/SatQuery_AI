"""Tier-1 intent classifier (plan task 1.3/1.4).

TF-IDF character and word n-grams into multinomial logistic regression. This
is deliberately not a neural model: intent classification over nine closed
classes is a solved problem at this scale, the training bank is synthetic, and
a linear model trains in under a second on CPU, which means it can be fitted
at process start rather than shipped as a pickled artifact. That removes a
deserialisation trust issue and guarantees the model always matches the
template bank it was trained from.

The classifier only ever *proposes*. The router constrains the proposal to
tasks that are legal for the actual input configuration, so a misclassification
can never produce an illegal plan.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from satquery.contracts.plan import TaskID
from satquery.controller.understanding import CUE_NAMES, cue_vector
from satquery.synth.query_bank import QueryExample, generate

# v2 (2026-09-20): the input configuration is a token the classifier sees,
# and the lexical cues of `understanding.py` are a feature block beside the
# n-grams. Measured on evaluation/nl/queries.jsonl (held out, hand-written):
# v1 routed 71.7% (bitemporal pairs 58.3%); see the NL benchmark report for
# v2. The name is in every trace, so a golden regenerated under v2 says so.
CLASSIFIER_NAME = "tfidf_cues_logreg_v2"

# The input configuration, as a token in front of the text. Change tasks are
# only legal on a bitemporal pair, fusion only on a cross-modal pair, and the
# single-image tasks on all three; with the token present the model learns
# that "are there any new structures?" on a pair is a change question, while
# the same words on one image are a presence question. Every example is
# also trained WITHOUT a token, so the unconstrained prediction the router
# uses to detect "asked for something these inputs cannot do" still works.
CONFIG_TOKENS: dict[str, str] = {
    "SINGLE": "[single]",
    "CROSSMODAL_PAIR": "[crossmodal]",
    "BITEMPORAL_PAIR": "[bitemporal]",
}
CONFIG_TO_LEGAL_TASKS: dict[str, list[str]] = {
    "SINGLE": [
        "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER",
        "CLARIFY_OR_ABSTAIN",
    ],
    "CROSSMODAL_PAIR": [
        "XMODAL_JOINT_EXTRACT", "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND",
        "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN",
    ],
    "BITEMPORAL_PAIR": [
        "TEMPORAL_CHANGE_DESC", "TEMPORAL_CHANGE_VQA", "TEMPORAL_CHANGE_MAP",
        "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER",
        "CLARIFY_OR_ABSTAIN",
    ],
}
LEGAL_CONFIGS_BY_TASK: dict[str, list[str]] = {
    task: [cfg for cfg, tasks in CONFIG_TO_LEGAL_TASKS.items() if task in tasks]
    for tasks in CONFIG_TO_LEGAL_TASKS.values() for task in tasks
}


def with_config(text: str, config: str | None) -> str:
    token = CONFIG_TOKENS.get(config or "")
    return f"{token} {text}" if token else text


def _strip_token(text: str) -> str:
    for token in CONFIG_TOKENS.values():
        if text.startswith(token + " "):
            return text[len(token) + 1:]
    return text


def _cue_matrix(texts) -> sparse.csr_matrix:
    # The token is not a cue; strip it so the cue block sees the sentence.
    rows = [cue_vector(_strip_token(t)) for t in texts]
    return sparse.csr_matrix(np.asarray(rows, dtype=float).reshape(len(rows), len(CUE_NAMES)))

# Below this top-1 probability the router treats the prediction as unreliable
# and falls back to the configuration default rather than trusting it.
LOW_CONFIDENCE_TOP1 = 0.35

# Below this margin the top two classes are effectively tied.
LOW_MARGIN = 0.10


@dataclass(frozen=True)
class IntentPrediction:
    task: TaskID
    top1: float
    margin: float
    ranked: list[tuple[str, float]]

    @property
    def is_confident(self) -> bool:
        return self.top1 >= LOW_CONFIDENCE_TOP1 and self.margin >= LOW_MARGIN


def build_pipeline() -> Pipeline:
    """Word n-grams catch phrasing; char n-grams give robustness to typos;
    the cue block gives a few high-level features (change / locate / count
    ...) a weight of their own, so a paraphrase built from words the bank
    never used can still reach the right class through its cues."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "cues",
                            FunctionTransformer(_cue_matrix, validate=False),
                        ),
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                min_df=2,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=4.0,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


class IntentClassifier:
    """Fit-on-construction intent classifier over the nine task IDs."""

    def __init__(self, examples: list[QueryExample] | None = None):
        self.examples = examples if examples is not None else generate()
        self.pipeline = build_pipeline()
        texts: list[str] = []
        labels: list[str] = []
        for i, e in enumerate(self.examples):
            texts.append(e.text)
            labels.append(e.task)
            configs = LEGAL_CONFIGS_BY_TASK.get(e.task, [])
            if configs:
                # One tokened copy per example, cycling through the task's
                # legal configurations so each token sees every task it can
                # carry, and the bank size stays 2x rather than 4x.
                texts.append(with_config(e.text, configs[i % len(configs)]))
                labels.append(e.task)
        self.pipeline.fit(texts, labels)
        self.classes_: list[str] = list(self.pipeline.named_steps["clf"].classes_)

    def predict(
        self, query: str, candidates: list[str] | None = None, config: str | None = None
    ) -> IntentPrediction:
        """Classify a query, optionally restricted to a set of legal tasks.

        Restricting to `candidates` renormalises the probabilities over just
        those classes, so top1 and margin remain interpretable rather than
        being diluted by tasks that were never possible for this input.
        `config` prefixes the input-configuration token (see CONFIG_TOKENS);
        None classifies the bare sentence.
        """
        probs = self.pipeline.predict_proba([with_config(query, config)])[0]
        pairs = list(zip(self.classes_, probs, strict=True))

        if candidates:
            allowed = set(candidates)
            filtered = [(c, p) for c, p in pairs if c in allowed]
            if filtered:
                total = sum(p for _, p in filtered)
                if total > 0:
                    pairs = [(c, p / total) for c, p in filtered]
                else:
                    pairs = [(c, 1.0 / len(filtered)) for c, _ in filtered]

        pairs.sort(key=lambda kv: kv[1], reverse=True)
        top1 = float(pairs[0][1])
        second = float(pairs[1][1]) if len(pairs) > 1 else 0.0

        return IntentPrediction(
            task=pairs[0][0],  # type: ignore[arg-type]
            top1=round(top1, 6),
            margin=round(top1 - second, 6),
            ranked=[(c, round(float(p), 6)) for c, p in pairs],
        )

    def evaluate(self, examples: list[QueryExample]) -> dict:
        """Accuracy, per-class report and confusion matrix on held-out data."""
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
        )

        texts = [e.text for e in examples]
        truth = [e.task for e in examples]
        predicted = list(self.pipeline.predict(texts))

        labels = sorted(set(truth) | set(predicted))
        matrix = confusion_matrix(truth, predicted, labels=labels)
        report = classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        )
        accuracy = float(np.mean([t == p for t, p in zip(truth, predicted, strict=True)]))

        return {
            "n": len(examples),
            "accuracy": round(accuracy, 6),
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
            "per_class": {
                label: {
                    "precision": round(report[label]["precision"], 6),
                    "recall": round(report[label]["recall"], 6),
                    "f1": round(report[label]["f1-score"], 6),
                    "support": int(report[label]["support"]),
                }
                for label in labels
                if label in report
            },
        }


_default: IntentClassifier | None = None
_lock = threading.Lock()


def default_classifier() -> IntentClassifier:
    """Process-wide classifier, fitted once on first use."""
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = IntentClassifier()
    return _default
