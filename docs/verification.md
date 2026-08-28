# Week-0 Verification Gate

Tracks resolution of the 12-item verification gate defined in
`docs/03-Models-and-Datasets.md` §6. Plan item 0.1 requires every row below to
carry a written answer before GPU time is spent against an unverified
assumption.

| # | Claim | Status | Answer / Evidence | Owner | Resolved |
|---|---|---|---|---|---|
| 1 | BigEarthNet.txt dataset contents + licence | **Resolved** | Paper `2603.29630v2` read: 464,044 pairs, 9.6M annotations, captions + VQA + referring. HF card confirms licence **CDLA-Permissive-1.0** (permissive). Format: 467 MB Parquet, 9,553,962 rows (1 per annotation), text only — S1/S2 imagery is a separate reBEN download. | ML lead | 2026-08-27 |
| 2 | BigEarthNet.txt headline figures | **Resolved** | `2603.29630v2` verified — 464,044 pairs / 9.6M annotations / 1,082-pair benchmark all match. | ML lead | 2026-08-27 |
| 3 | CROMA / DOFA checkpoints downloadable, permissive licence | **Open** | Not yet checked. Fallback if false: torchgeo SSL weights (guaranteed). | ML #2 | Pending — Phase 1 |
| 4 | Change-Agent / LEVIR-MCI weights available | **Open** | Not yet checked. Fallback: TinyCD + separate caption head. | ML #2 | Pending — Phase 1 |
| 5 | Which RISAT sensor/mode ISRO/SAC will use (C-band vs X-band, look count) | **Largely de-risked** | Which platform ISRO picks is still their call. But (2026-08-29): **EOS-04 / RISAT-1A is C-band at 5.35 GHz**, with modes HRS, FRS-1, FRS-2 (12 m, 30 km swath, quad-pol), MRS (25 m, 120 km), CRS (50 m, 240 km); selectable resolution 1–50 m, swath 10–223 km, co- and cross-pol. **5.35 GHz is within ~1% of Sentinel-1's C-band 5.405 GHz**, so backscatter physics transfers almost directly from the S1 training data — a major de-risking of the SAR track. RISAT-2B/2BR1 remain X-band if ISRO chooses those instead. **RISAT-1 MRS/CRS are open data for all**; FRS (Stripmap) is priced. Design mitigation unchanged: σ⁰ thresholds stay adaptive. | Geo lead | Frequency confirmed 2026-08-29; platform choice pending |
| 6 | Cartosat-2S MX band composition — confirm 4-band VNIR, no SWIR | **Resolvable now** | A **Cartosat-2S MX sample product (`5132611.zip`) is directly downloadable** from the Bhoonidhi sample-products page — no order, no payment, no wait. Download it and run `scripts/inspect_product.py` to close this item. If SWIR turns out to be present, MNDWI/NDBI paths can be enabled (pure upside, no rework). | Geo lead | Pending — unblocked, hours of work |
| 7 | Newer ≤4B VLM beats Qwen2.5-VL-3B on the §2.1 criteria | **Open** | InternVL3-1B (used by the BigEarthNet.txt authors) flagged as the candidate to evaluate first. Not yet benchmarked. | ML lead | Pending — Phase 1 |
| 8 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | **Open** | Not yet checked. Fallback: run Stage A3 optical-only and document the limitation. | Geo lead | Pending — Phase 1 |
| 9 | Prescribed benchmark test splits downloadable (VRSBench, RSVQA, CDVQA) | **Open** | Not yet checked. Fallback: use published splits from the papers' own repos. | Eval lead | Pending — Phase 1 |
| 10 | SIH 2026 timeline: internal deadline, grand finale dates, submission format | **Open** | Not yet confirmed with team/organizers. Fallback: compress the phase plan in doc `04` proportionally once the real deadline is known. | Team lead | Pending |
| 11 | Bhoonidhi registration approved; Cartosat-2S + RISAT products downloaded | **Partially resolved** | **Registration completed 2026-08-29.** Product download still outstanding — items 5 and 6 stay blocked until a real Cartosat-2S MX and a real RISAT product are on disk and their metadata has been read. Fallback if downloads prove unavailable: use any open Indian-context high-res imagery for qualitative work. | Geo lead | Registration 2026-08-29; download pending |
| 12 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | **Open** | Not yet checked. Fallback: baseline against the un-finetuned base VLM only. | ML #2 | Pending — Phase 1 |

## Summary

- **Resolved: 2/12** (items 1, 2 — both dataset-paper claims, closed 2026-08-27, before this repo's code existed).
- **Open: 10/12**, but items 5, 6 and 11 are substantially de-risked as of 2026-08-29 (see below). All ten open items have a documented, costed fallback per doc `03` §6, so **none blocks starting Phase 1**.

## NEW RISK (discovered 2026-08-29, not in the original 12 items)

**Cartosat-2S full-scene data is priced, not open, for our likely user class.**
Under the Indian Space Policy 2023 implementation at Bhoonidhi, data finer than
5 m resolution is **open only to Indian Government Entities** and is **priced for
Non-Government Entities** (Cartosat-2S ordering is handled by NSIL). A student
competition team is most likely an NGE, so routine bulk Cartosat-2S acquisition
may cost money or require institutional sponsorship.

Why this is not currently blocking:

- The **Cartosat-2S MX sample product is a free direct download**, which is
  sufficient to answer item 6 (band composition) — the only thing we actually
  need Cartosat data for at this stage.
- Doc `03` §5 already specifies that Bhoonidhi products are a **small curated
  qualitative/OOD set that is never trained on**. A handful of scenes is enough;
  we were never going to need bulk Cartosat data.
- SAR is better off: **RISAT-1 MRS/CRS are open to all**, and **EOS-04
  (RISAT-1A, C-band)** samples are freely downloadable.

Action: if bulk high-res optical is ever needed beyond the samples, route the
request through the team's academic institution (which may qualify differently)
or budget for NSIL pricing. Flag to the team lead alongside item 10.

## Tooling

`scripts/inspect_product.py` reads a downloaded product and prints exactly what
items 5 and 6 need — band count, band descriptions, CRS, ground sample distance,
subdatasets, and any sensor/frequency/polarization metadata tags, plus a list of
sidecar XML/metadata files (where RISAT frequency and look count usually live,
rather than in the raster header):

```
python scripts/inspect_product.py <path-to-product-or-directory>
```

## Recommended next actions

1. **Download the free Cartosat-2S MX sample product** (`5132611.zip`) from the
   Bhoonidhi sample-products page and run the inspection script on it. This
   closes item 6 in an afternoon, with no order and no payment. Grab an
   **EOS-04 (RISAT-1A) sample** at the same time for the SAR side of item 5.
   Sample products page:
   `https://bhoonidhi.nrsc.gov.in/bhoonidhi/help/sampleProducts.html`
2. Confirm the SIH 2026 submission deadline and format (item 10) with the team/organizers — this affects how the W0–W14 phase plan in `docs/04-Implementation-Plan.md` should be paced.
3. Items 3, 4, 7, 8, 9, 12 are model/dataset-availability checks with zero-cost fallbacks already specified — verify opportunistically during Phase 1 rather than blocking on them.
