"""Corpus captioning metrics: sanity properties, not reference-implementation parity."""

import pytest

from evaluation.metrics.caption_corpus import cider_d, corpus_bleu, rouge_l, score_corpus

REFS = [["a large airport with many planes", "many planes parked at a large airport"],
        ["some buildings and green trees are in a residential area"],
        ["a river runs through the farmland"]]


def test_perfect_hypotheses_score_one():
    hyps = [r[0] for r in REFS]
    b = corpus_bleu(hyps, REFS)
    assert abs(b["bleu4"] - 1.0) < 1e-9 and abs(b["bleu1"] - 1.0) < 1e-9
    assert abs(rouge_l(hyps, REFS) - 1.0) < 1e-9


def test_unrelated_hypotheses_score_low():
    hyps = ["zebra", "elephant xylophone", "quantum"]
    assert corpus_bleu(hyps, REFS)["bleu4"] == 0.0
    assert rouge_l(hyps, REFS) == 0.0
    assert cider_d(hyps, REFS) == 0.0


def test_partial_overlap_is_between():
    hyps = ["a large airport", "some buildings in a residential area", "a river"]
    s = score_corpus(hyps, REFS)
    assert 0.0 < s["bleu1"] < 1.0
    assert s["brevity_penalty"] < 1.0  # hypotheses are shorter than references
    assert 0.0 < s["rouge_l"] < 1.0
    assert s["cider_d"] > 0.0
    assert s["n"] == 3


def test_cider_rewards_rare_ngrams_more_than_common():
    # "a" appears in every reference (df high); "farmland" in one (df low).
    common = ["a", "a", "a"]
    rare = ["farmland", "farmland", "farmland"]
    assert cider_d(rare, REFS) > cider_d(common, REFS)


def test_meteor_exact_bounds_and_order():
    from evaluation.metrics.caption_corpus import meteor_exact

    # an identical sentence is one chunk, so METEOR is 1 - 0.5 * (1/m)^3, not exactly 1
    assert meteor_exact(["a road next to a river"], [["a road next to a river"]]) > 0.99
    assert meteor_exact(["nothing"], [["a road next to a river"]]) == 0.0
    ordered = meteor_exact(["a road next to a river"], [["a road next to a river and trees"]])
    scrambled = meteor_exact(["river a to next road a"], [["a road next to a river and trees"]])
    assert 0 < scrambled < ordered < 1
