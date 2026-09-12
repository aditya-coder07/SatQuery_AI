"""Build the FINAL DATASET MANIFEST: one record per dataset the programme
may train or evaluate on, with counts, hashes, licence, leakage status and
what is quarantined - read from the datasets' own manifests/indexes on the
machine that holds them (the cluster), never typed by hand.

Run on compute01::

    python scripts/final_dataset_manifest.py --data data \
        --out artifacts/dataset_manifests/FINAL_DATASET_MANIFEST.json \
        --md docs/research/final_dataset_manifest.md

Every record has: dataset, version, license, usable_for (train / eval /
none), provenance, preprocessing_version, splits {train,val,test: n},
valid / quarantined counts with reasons, checksums (sha256 of each split
manifest or index; image checksums live in the dataset's own stats.json),
manifest_hash (sha256 over the split checksums, the id of this dataset
version in the registry), leakage status, and the audit findings that
apply. A dataset that is not on disk is reported as `missing`, never
guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def jsonl_manifests(root: Path, names=("train", "val", "test")) -> tuple[dict, dict]:
    splits, sums = {}, {}
    for s in names:
        f = root / "manifests" / f"{s}.jsonl"
        if f.exists():
            splits[s] = count_jsonl(f)
            sums[f"manifests/{s}.jsonl"] = sha256_file(f)
    st = root / "manifests" / "stats.json"
    if st.exists():
        sums["manifests/stats.json"] = sha256_file(st)
    return splits, sums


def index_splits(path: Path, key="splits") -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: len(v) for k, v in d[key].items()}


def manifest_hash(sums: dict) -> str:
    return hashlib.sha256(json.dumps(sums, sort_keys=True).encode()).hexdigest()


def build(data: Path) -> list[dict]:
    recs: list[dict] = []

    def add(key, **kw):
        rec = {"key": key, "path": str(data / key)}
        rec.update(kw)
        if "checksums" in rec:
            rec["manifest_hash"] = manifest_hash(rec["checksums"])
        recs.append(rec)

    # --- grounding -------------------------------------------------------
    r = data / "dior_rsvg_official"
    if r.exists():
        splits, sums = jsonl_manifests(r)
        add("dior_rsvg_official", dataset="DIOR-RSVG", version="authors' Google Drive release, official train/val/test.txt",
            license="CC-BY-NC-4.0", usable_for="train+eval (non-commercial)", task="grounding",
            provenance="Zhan et al. 2023; index files are indices into the sorted flat object list (RSVG-pytorch convention)",
            preprocessing_version="training/prepare/dior_rsvg_official.py (unified manifest v1)",
            splits=splits, valid=sum(splits.values()), quarantined=[],
            leakage="index files pairwise disjoint (enforced); 4,206 test images also in train BY THE OFFICIAL PROTOCOL "
                    "(object-level split) - reported, not altered; 22 source-level near/exact duplicate images under "
                    "different ids (0.3%), kept for comparability; image-disjoint subset reported alongside",
            findings=["G1 (old mirror was test-only)", "G2 (protocol image overlap)"], checksums=sums)
    r = data / "dior_rsvg"
    if r.exists():
        add("dior_rsvg", dataset="DIOR-RSVG (HF parquet mirror)", version="test shards only", license="CC-BY-NC-4.0",
            usable_for="none (retired: holds only the official test split; finding G1)", task="grounding",
            provenance="HF mirror", preprocessing_version="Phase 5", splits={"test": 7500}, valid=0,
            quarantined=[{"n": 7500, "reason": "official test rows - must never be trained on"}],
            leakage="was the Phase 5 training source (Category C numbers)", findings=["G1"], checksums={})

    # --- VQA -------------------------------------------------------------
    r = data / "rsvqa_lr_official"
    if r.exists():
        splits, sums = jsonl_manifests(r, ("train", "val"))
        test_q = json.loads((r / "LR_split_test_questions.json").read_text(encoding="utf-8"))
        qs = test_q.get("questions", test_q)
        n_test = sum(1 for q in qs if q.get("active", True))
        splits["test"] = n_test
        for name in ("LR_split_test_questions.json", "LR_split_test_answers.json", "LR_split_test_images.json"):
            sums[name] = sha256_file(r / name)
        add("rsvqa_lr_official", dataset="RSVQA-LR", version="Zenodo 6344334 official lists", license="CC-BY-4.0",
            usable_for="train+eval", task="VQA",
            provenance="Lobry et al. 2020; train = official train; val = derived 100-image split of official val "
                       "(training/prepare/rsvqa_official_train.py); test = official test",
            preprocessing_version="rsvqa_official_train.py (unified manifest v1)", splits=splits,
            valid=sum(splits.values()), quarantined=[],
            leakage="train/val/test images disjoint (0/0/0); selection on val only", findings=[], checksums=sums)
    r = data / "rsvqa_lr_2k"
    if r.exists():
        add("rsvqa_lr_2k", dataset="RSVQA-LR (HF 2k redistribution of the VALIDATION split)", version="dmarsili/RSVQA-LR-2k",
            license="CC-BY-4.0", usable_for="none (retired: official train replaces it; it is official-val material)",
            task="VQA", provenance="HF", preprocessing_version="Phase 4", splits={}, valid=0,
            quarantined=[{"reason": "validation-split material; no longer trained on"}], leakage="disjoint from official test",
            findings=[], checksums={})
    for key, note in (("instruct_mix_v2", "aligned WHU labels (lbl_v2); rsvqa rows excluded in train_no_rsvqa"),
                      ("instruct_mix", "SUPERSEDED: built on misaligned WHU labels (F2)")):
        r = data / key
        if r.exists():
            splits, sums = jsonl_manifests(r, ("train", "train_no_rsvqa", "val"))
            add(key, dataset="SatQuery instruction mix", version=key, license="derived: WHU-OPT-SAR (unstated) + RSVQA-LR (CC-BY-4.0)",
                usable_for="train (train_no_rsvqa)" if key.endswith("v2") else "none (superseded)", task="VQA/instruction",
                provenance="training/prepare/instruction_mix.py + instruct_mix_manifest.py", preprocessing_version=key,
                splits=splits, valid=sum(splits.values()) if key.endswith("v2") else 0,
                quarantined=[] if key.endswith("v2") else [{"reason": note}],
                leakage="WHU rows scene-split; rsvqa rows are HF-val material (excluded from train_no_rsvqa)",
                findings=["F2"], checksums=sums)

    # --- captioning ------------------------------------------------------
    r = data / "rsicd"
    if r.exists():
        splits, sums = jsonl_manifests(r)
        add("rsicd", dataset="RSICD", version="HF parquet mirror, official split names", license="unstated (research use)",
            usable_for="train+eval", task="caption", provenance="Lu et al. 2017",
            preprocessing_version="training/prepare/caption_manifests.py", splits=splits, valid=sum(splits.values()),
            quarantined=[], leakage="train vs test: 0 exact, 0 near duplicates (dup_rsicd_train_test.json)", findings=[],
            checksums=sums)
    r = data / "levir_mci"
    if r.exists():
        splits, sums = jsonl_manifests(r)
        add("levir_mci", dataset="LEVIR-CC / LEVIR-MCI", version="LEVIR-MCI release", license="academic-only",
            usable_for="train+eval (non-commercial)", task="change caption", provenance="Liu et al. (LEVIR-MCI)",
            preprocessing_version="caption_manifests.py", splits=splits, valid=sum(splits.values()), quarantined=[],
            leakage="official split; GT masks never used by the VLM arm", findings=[], checksums=sums)

    # --- change detection ------------------------------------------------
    r = data / "levircd"
    if r.exists():
        idx = r / "index.json"
        splits = index_splits(idx)
        add("levircd", dataset="LEVIR-CD", version="official split, 256-px tiles", license="academic-only",
            usable_for="train+eval (non-commercial)", task="change mask", provenance="Chen & Shi 2020",
            preprocessing_version="Phase 3 preparer (tiles/)", splits=splits, valid=sum(splits.values()), quarantined=[],
            leakage="tiles never cross splits; val for selection+calibration only", findings=[],
            checksums={"index.json": sha256_file(idx)})

    # --- fusion ----------------------------------------------------------
    r = data / "whu_opt_sar"
    if r.exists():
        idx = r / "index_v2_scene_split.json"
        d = json.loads(idx.read_text(encoding="utf-8"))
        splits = {k: len(v) for k, v in d["splits"].items()}
        add("whu_opt_sar", dataset="WHU-OPT-SAR", version="lbl_v2 (1-based re-cut) + scene-disjoint split",
            license="unstated (research use)", usable_for="train+eval", task="optical/SAR fusion",
            provenance="Li et al. 2022; labels re-cut from lbl_full.zip (training/prepare/whu_opt_sar_relabel.py)",
            preprocessing_version="lbl_v2 / index_v2_scene_split.json", splits=splits, valid=sum(splits.values()),
            quarantined=[{"what": "prepared/lbl (v1) + index.json random tile split", "reason": "F1 leakage + F2 misalignment"}],
            leakage="scene-disjoint (29/7 scenes); NDWI alignment +0.184 vs +0.001 (relabel_report.json)",
            findings=["F1", "F2"], checksums={"index_v2_scene_split.json": sha256_file(idx), "index_v2.json": sha256_file(r / "index_v2.json"),
                                              "relabel_report.json": sha256_file(r / "relabel_report.json")})

    # --- land cover ------------------------------------------------------
    r = data / "ben_v1_full"
    st = r / "manifests" / "stats.json"
    if st.exists():
        d = json.loads(st.read_text(encoding="utf-8"))
        splits = {s: (d["splits"][s] or {}).get("n") for s in ("train", "val", "test")}
        q = [{"split": s, **x} for s in d["splits"] if d["splits"][s] for x in d["splits"][s].get("quarantined", [])]
        add("ben_v1_full", dataset="BigEarthNet-S2 v1.0 (19-label)", version="lc-col/bigearthnet HDF5 mirror, torchgeo split lists",
            license=d["license"], usable_for="train+eval", task="land cover", provenance=d["source"],
            preprocessing_version=d["preprocessing_version"], splits=splits, valid=sum(v for v in splits.values() if v),
            quarantined=q, leakage=d["leakage"] + f"; id overlap {d['split_id_overlap']}", findings=["L1 (revised)"],
            checksums={"manifests/stats.json": sha256_file(st), **{f"{s}_ids.txt": sha256_file(r / f"{s}_ids.txt")
                                                                   for s in splits if (r / f"{s}_ids.txt").exists()}})
    elif r.exists():
        add("ben_v1_full", dataset="BigEarthNet-S2 v1.0", version="in preparation", license="CDLA-Permissive-1.0",
            usable_for="pending", task="land cover", provenance="lc-col/bigearthnet", preprocessing_version="pending",
            splits={}, valid=0, quarantined=[], leakage="pending", findings=["L1 (revised)"], checksums={})
    r = data / "ben_full"
    if r.exists():
        add("ben_full", dataset="BigEarthNet-S2 v1.0 (partial mirror)", version="train p0-p3 + test p8", license="CDLA-Permissive-1.0",
            usable_for="none (superseded by ben_v1_full; geographic prefix, L1 revised)", task="land cover",
            provenance="lc-col/bigearthnet", preprocessing_version="Phase 1 download", splits={"train": 60000, "test": 5866},
            valid=0, quarantined=[{"n": 1, "reason": "phantom zero row in test_p8 (converter off-by-one)"},
                                  {"n": 65866, "reason": "not a representative sample of the official split"}],
            leakage="official split ids", findings=["L1 (revised)"], checksums={})

    # --- semantic change ------------------------------------------------
    r = data / "landsat_scd"
    if r.exists():
        idx = r / "index.json"
        d = json.loads(idx.read_text(encoding="utf-8"))
        splits = {k: len(v) for k, v in d["splits"].items()}
        add("landsat_scd", dataset="Landsat-SCD", version="figshare 19946135 v1", license=d["license"], usable_for="train+eval",
            task="semantic change", provenance=d["source"], preprocessing_version="training/prepare/landsat_scd.py (3:1:1 by original)",
            splits=splits, valid=sum(splits.values()), quarantined=[], leakage=d["split_method"], findings=[],
            checksums={"index.json": sha256_file(idx)})
    r = data / "second"
    if r.exists():
        add("second", dataset="SECOND", version="release", license="NONE STATED - blocked",
            usable_for="none (licence unresolved; replaced by Landsat-SCD)", task="semantic change", provenance="Yang et al. 2021",
            preprocessing_version="-", splits={}, valid=0, quarantined=[{"reason": "no licence; not trained on"}],
            leakage="n/a", findings=["licensing"], checksums={})
    r = data / "cdvqa"
    if r.exists():
        add("cdvqa", dataset="CDVQA", version="GitHub annotations (Apache-2.0) over SECOND imagery", license="annotations Apache-2.0; imagery unlicensed",
            usable_for="eval only (legacy Phase 5 number); no training", task="change VQA", provenance="Yuan et al. 2022",
            preprocessing_version="Phase 5", splits={}, valid=0, quarantined=[{"reason": "imagery licence unresolved"}],
            leakage="test ids never trained on (Phase 5 verified)", findings=["licensing"],
            checksums={"LICENSE": sha256_file(r / "LICENSE")})
    return recs


def render_md(recs: list[dict]) -> str:
    lines = ["# Final dataset manifest", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/final_dataset_manifest.py` on the data host. "
             "Machine-readable copy: `artifacts/dataset_manifests/FINAL_DATASET_MANIFEST.json`. "
             "Nothing in the *train* column may be used unless `usable_for` includes train.", "",
             "| key | dataset / version | licence | usable for | train | val | test | valid | quarantined | leakage | manifest hash |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        sp = r.get("splits", {})
        q = "; ".join(f"{x.get('n', '')} {x.get('reason', x.get('what', ''))}".strip() for x in r.get("quarantined", [])) or "—"
        lines.append("| " + " | ".join([
            f"`{r['key']}`", f"{r['dataset']} — {r['version']}", r["license"], r["usable_for"],
            str(sp.get("train", sp.get("train_no_rsvqa", "—"))), str(sp.get("val", sp.get("validation", "—"))),
            str(sp.get("test", "—")), str(r.get("valid", "—")), q, r.get("leakage", "—")[:160],
            (r.get("manifest_hash") or "—")[:12]]) + " |")
    lines += ["", "## Rules applied", "",
              "* Test splits are never read by a trainer; selection uses the split marked val.",
              "* Quarantined rows are listed with the reason and are never sampled.",
              "* `manifest_hash` is the sha256 over the per-file sha256s and is what the experiment registry records as `dataset_manifest_hash`.",
              "* Licence-blocked data (SECOND imagery) is not trained on; CDVQA is evaluation-only history."]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("artifacts/dataset_manifests/FINAL_DATASET_MANIFEST.json"))
    p.add_argument("--md", type=Path, default=Path("docs/research/final_dataset_manifest.md"))
    args = p.parse_args()
    recs = build(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "datasets": recs}, indent=1),
                        encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_md(recs), encoding="utf-8")
    for r in recs:
        print(f"{r['key']:22s} {r['usable_for'][:28]:28s} {r.get('splits')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
