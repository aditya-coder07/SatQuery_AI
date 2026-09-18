# Licensing register — datasets and pretrained weights

**Written 2026-09-12.** One row per source this project trains on, evaluates
on, or considers. "Weights" means our adapters/checkpoints trained on the
source. Where the licence is not stated by the publisher, the row says so:
"no licence" is a finding, not a gap to be filled by assumption.

Rule applied throughout (from `docs/verification.md` §"SECOND"): a public
**benchmark claim** may be made on any dataset we can lawfully evaluate on;
**redistributing weights** requires every training source to permit it.

## Datasets

| Source | Where obtained | Licence | Allowed | Redistribution / commercial | Citation | Our use | Status |
|---|---|---|---|---|---|---|---|
| RSVQA-LR (official test) | Zenodo 10.5281/zenodo.6344334 | **CC-BY-4.0** | any, with attribution | yes / yes | Lobry et al. 2020 | Track B eval | OK |
| RSVQA-LR-2k (val subset) | HF `dmarsili/RSVQA-LR-2k` | inherits CC-BY-4.0 | any | yes / yes | same | Track B train | OK |
| BigEarthNet-S2 **v1.0** (the mirror we hold is v1, not reBEN) | HF `lc-col/bigearthnet` (HDF5 export of torchgeo's dataset; converter Apache-2.0) | **CDLA-Permissive-1.0** | any | yes / yes | Sumbul et al. 2019/2021 | landcover train/eval (complete split from 2026-09-12) | OK |
| BigEarthNet.txt | HF `BIFOLD-BigEarthNetTextual` | CDLA-Permissive-1.0 | any | yes / yes | 2603.29630 | planned | OK |
| **DIOR-RSVG** (official, added 2026-09-12) | authors' Google Drive (ZhanYang-nwpu/RSVG-pytorch) | **CC-BY-NC-4.0** | research, non-commercial | yes with attribution / **no** | Zhan et al. 2023 | grounding train/val/test | **OK for SIH (non-commercial). Weights trained on it are NC.** |
| DIOR (underlying imagery) | via DIOR-RSVG zip | authors: research use; HF `torchgeo/dior` card says CC-BY-SA-4.0 | research | share-alike if the HF card is authoritative | Li et al. 2020 | same | note SA |
| `data/dior_rsvg` parquet mirror (test only) | HF mirror, unattributed | inherits DIOR-RSVG NC | research | — | — | **superseded** by official | retire |
| RSICD | GitHub `201528014227051/RSICD_optimal` | **none stated** ("for research") | research, by convention | unclear | Lu et al. 2017 | caption train/eval | benchmark OK; weights **unresolved** |
| LEVIR-CD | justchenhao.github.io/LEVIR | **academic only, commercial prohibited** (stated on site) | research | no commercial | Chen & Shi 2020 | change_mask train/eval | benchmark OK; weights NC |
| LEVIR-CC / LEVIR-MCI | Chg2Cap / RSICC repos | built on LEVIR-CD — inherits **academic only** | research | no commercial | Liu et al. 2022, 2024 | change_caption train/eval | benchmark OK; weights NC |
| CDVQA annotations | GitHub `YZHJessica/CDVQA` | **Apache-2.0** | any | yes / yes | Yuan et al. 2022 | change_vqa eval | OK |
| SECOND (CDVQA imagery) | captain-whu / HF `ljx620/CDVQA` | **NO licence stated at all** | undefined | undefined | Yang et al. 2021 | change_vqa train | **BLOCKED for weight release; benchmark use documented as risk** |
| WHU-OPT-SAR | GitHub `AmberHen/WHU-OPT-SAR-dataset` | **none stated** in repo; paper says open | research, by convention | unclear | Li et al. 2022 | optsar_fusion, Track B mix | benchmark OK; weights unresolved |
| **VRSBench** (fetched 2026-09-12 21:20) | HF `xiang709/VRSBench` | **CC-BY-4.0** (dataset card, verified via the Hub API) | any, with attribution | yes / yes | Li et al. 2024 (NeurIPS D&B) | caption + VQA + grounding train; official val for eval | OK — the most permissive VLM-training set in the programme; DIOR-derived half overlaps DIOR-RSVG (see `dataset_audit.md`, V1) |
| RSVQA-HR | rsvqa.sylvainlobry.com | USGS imagery public domain; annotations CC-BY-4.0 | any | yes | Lobry 2020 | planned | verify on fetch |

## Pretrained weights

| Model | Source | Licence | `trust_remote_code` needed? | Decision |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | HF `Qwen/Qwen2.5-VL-3B-Instruct` | **Qwen Research Licence** (3B: research; commercial needs licence) | no (native in transformers) | in use; commercial deployment would need the 7B (Apache-2.0) or a licence |
| Qwen2.5-VL-7B-Instruct | HF | **Apache-2.0** | no | candidate |
| torchvision ResNet-50 (ImageNet) | torchvision | BSD-3 | no | in use (grounding_pre, caption_pre) |
| SSL4EO-S12 (ResNet/ViT, S1+S2) | zhu-xlab/SSL4EO-S12 | code Apache-2.0; **weights CC-BY-4.0** | no (plain timm/torchvision state dicts) | **candidate for landcover + fusion** |
| SpectralGPT | Zenodo 8412455 | CC-BY (per Zenodo) | no (MAE-ViT code in repo, vendorable) | candidate |
| Prithvi-EO-2.0-300M/600M | HF `ibm-nasa-geospatial` | **Apache-2.0** | loads via `terratorch` package, not remote code | candidate (HLS bands; S2 subset mapping needed) |
| Florence-2 | Microsoft | MIT | **yes** | excluded (policy) |
| InternVL3 | OpenGVLab | MIT-ish | **yes** | excluded (policy) |

## Consequences for this programme

1. **Public benchmark claims** are permitted on every dataset above; SECOND
   claims are made with the "no licence" caveat stated next to the number.
2. **Adapter/weight release** is currently blocked for anything trained on
   DIOR-RSVG (NC), LEVIR-* (academic only), RSICD / WHU-OPT-SAR (unstated),
   or SECOND (none). The only weights releasable without caveat today would
   be Track A (BigEarthNet, CDLA) and Track B trained on RSVQA-LR only.
   This does not affect SIH, which is non-commercial, but it must be said in
   the final report.
3. A **licensed semantic-change replacement for SECOND** is still needed
   (prompt §27); candidates and their licences go in
   `docs/research/dataset_audit.md` when investigated.
4. The Qwen2.5-VL-**3B** research licence is fine for SIH; a commercial
   deployment must move to the Apache-2.0 7B or obtain a licence.
