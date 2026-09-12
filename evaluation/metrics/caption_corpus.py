"""Corpus-level captioning metrics: BLEU-1..4, ROUGE-L, CIDEr-D.

Why a second module
-------------------
`all_tasks.bleu` is sentence-level with add-one smoothing, chosen so a
single short caption never scores exactly zero. Papers report **corpus**
BLEU (Papineni 2002: n-gram counts summed over the corpus before the
geometric mean, one brevity penalty for the whole set) and CIDEr-D
(Vedantam 2015). The two BLEUs differ by several points on RSICD, which is
why the Phase 5 comparison to published numbers was Category B. This
module implements the corpus conventions so the comparison can be A.

Conventions match pycocoevalcap's Python paths (BLEU with no smoothing,
CIDEr-D with sigma 6 and n = 1..4, ROUGE-L with beta 1.2) on the same
lower-cased alphanumeric tokenisation used everywhere in this repository.
METEOR and SPICE are not implemented: both need external Java or WordNet
resources and are reported by fewer remote-sensing papers.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from evaluation.metrics.all_tasks import _ngrams, tokenize


def corpus_bleu(hyps: list[str], refs: list[list[str]], max_n: int = 4) -> dict[str, float]:
    """BLEU-1..4 over the corpus. No smoothing: a zero n-gram match anywhere
    in the corpus is essentially impossible at corpus scale, so smoothing
    only matters per-sentence."""
    match = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len = ref_len = 0
    for hyp, rs in zip(hyps, refs):
        h = tokenize(hyp)
        r_tok = [tokenize(r) for r in rs if str(r).strip()]
        if not r_tok:
            continue
        hyp_len += len(h)
        # Closest reference length (ties -> shorter), as in the original BLEU.
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in r_tok)[1]
        for n in range(1, max_n + 1):
            hg = _ngrams(h, n)
            if not hg:
                continue
            best = Counter()
            for r in r_tok:
                for g, c in _ngrams(r, n).items():
                    best[g] = max(best[g], c)
            match[n] += sum(min(c, best[g]) for g, c in hg.items())
            total[n] += sum(hg.values())
    bp = 1.0 if hyp_len > ref_len else (math.exp(1 - ref_len / hyp_len) if hyp_len else 0.0)
    out = {}
    for k in range(1, max_n + 1):
        logs = []
        for n in range(1, k + 1):
            if match[n] == 0 or total[n] == 0:
                logs = None
                break
            logs.append(math.log(match[n] / total[n]))
        out[f"bleu{k}"] = bp * math.exp(sum(logs) / k) if logs else 0.0
    out["brevity_penalty"] = bp
    return out


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyps: list[str], refs: list[list[str]], beta: float = 1.2) -> float:
    scores = []
    for hyp, rs in zip(hyps, refs):
        h = tokenize(hyp)
        best = 0.0
        for r in rs:
            rt = tokenize(r)
            l = _lcs(h, rt)
            if l == 0:
                continue
            p, rc = l / len(h), l / len(rt)
            best = max(best, (1 + beta**2) * p * rc / (rc + beta**2 * p))
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def cider_d(hyps: list[str], refs: list[list[str]], max_n: int = 4, sigma: float = 6.0) -> float:
    """CIDEr-D. Document frequencies are computed over the reference set of
    the corpus being scored, as in pycocoevalcap."""
    ref_tok = [[tokenize(r) for r in rs] for rs in refs]
    hyp_tok = [tokenize(h) for h in hyps]
    df: dict[tuple, float] = defaultdict(float)
    for rs in ref_tok:
        seen = set()
        for r in rs:
            for n in range(1, max_n + 1):
                seen.update(_ngrams(r, n).keys())
        for g in seen:
            df[g] += 1
    n_docs = len(ref_tok)
    log_n = math.log(max(1, n_docs))

    def tfidf(tokens):
        vecs, norms, length = [], [], len(tokens)
        for n in range(1, max_n + 1):
            v = {}
            for g, c in _ngrams(tokens, n).items():
                v[g] = c * (log_n - math.log(max(1.0, df[g])))
            vecs.append(v)
            norms.append(math.sqrt(sum(x * x for x in v.values())))
        return vecs, norms, length

    scores = []
    for h, rs in zip(hyp_tok, ref_tok):
        hv, hn, hl = tfidf(h)
        per_n = [0.0] * max_n
        for r in rs:
            rv, rn, rl = tfidf(r)
            for n in range(max_n):
                # CIDEr-D clips the hypothesis count at the reference count.
                dot = sum(min(hv[n][g], rv[n].get(g, 0.0)) * rv[n].get(g, 0.0) for g in hv[n])
                val = dot / (hn[n] * rn[n]) if hn[n] and rn[n] else 0.0
                val *= math.exp(-((hl - rl) ** 2) / (2 * sigma**2))
                per_n[n] += val
        k = max(1, len(rs))
        scores.append(10.0 * sum(v / k for v in per_n) / max_n)
    return sum(scores) / len(scores) if scores else 0.0


def meteor_exact(hyps: list[str], refs: list[list[str]], alpha: float = 0.9, beta: float = 3.0,
                 gamma: float = 0.5) -> float:
    """METEOR with the exact-match module only, averaged over sentences.

    Banerjee & Lavie 2005 / Denkowski & Lavie 2014 parameters (alpha 0.9,
    beta 3, gamma 0.5) but no stemming, synonym or paraphrase matching, so
    this is a LOWER BOUND on the METEOR 1.5 number papers quote (which
    needs Java and the paraphrase tables). Multi-reference: the best
    reference per sentence, as METEOR does. Reported as `meteor_exact`,
    never as "METEOR".
    """
    def align(h: list[str], r: list[str]) -> tuple[int, int]:
        # Greedy left-to-right exact alignment (each ref token used once);
        # chunks = maximal runs of adjacent, monotone matches.
        used = [False] * len(r)
        pairs = []
        for i, tok in enumerate(h):
            for j, rt in enumerate(r):
                if not used[j] and rt == tok:
                    used[j] = True
                    pairs.append((i, j))
                    break
        chunks = 0
        for k, (i, j) in enumerate(pairs):
            if k == 0 or pairs[k - 1][0] != i - 1 or pairs[k - 1][1] != j - 1:
                chunks += 1
        return len(pairs), chunks

    total = 0.0
    for hyp, rs in zip(hyps, refs):
        h = tokenize(hyp)
        best = 0.0
        for ref in rs:
            r = tokenize(ref)
            m, chunks = align(h, r)
            if m == 0 or not h or not r:
                continue
            prec, rec = m / len(h), m / len(r)
            fmean = prec * rec / (alpha * prec + (1 - alpha) * rec)
            penalty = gamma * (chunks / m) ** beta
            best = max(best, fmean * (1 - penalty))
        total += best
    return total / max(1, len(hyps))


def score_corpus(hyps: list[str], refs: list[list[str]]) -> dict[str, float]:
    out = corpus_bleu(hyps, refs)
    out["rouge_l"] = rouge_l(hyps, refs)
    out["cider_d"] = cider_d(hyps, refs)
    out["meteor_exact"] = meteor_exact(hyps, refs)
    out["n"] = len(hyps)
    return out
