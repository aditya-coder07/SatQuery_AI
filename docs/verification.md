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
| 5 | Which RISAT sensor/mode ISRO/SAC will use (C-band vs X-band, look count) | **RESOLVED for EOS-04** | Read from a real Bhoonidhi product's `product.xml` (2026-08-29): **`radarCenterFrequency = 5.40e09 Hz` = 5.40 GHz = C-band.** Sentinel-1 is 5.405 GHz, a **0.09% difference**, so S1-trained backscatter behaviour transfers essentially directly. Look count answered too: MRS product reports `RangeLooks=2.0, AzimuthLooks=1.0`, `IncidenceAngle=37.8`, `NoOfPolarizations=2` (HH+HV), `OutputPixelSpacing=18.0 m`, 8 ScanSAR beams. FRS-1 sample is **quad-pol** (HH/HV/VH/VV). Residual risk: if ISRO instead uses RISAT-2B/2BR1 those are X-band; adaptive thresholds already cover that. | Geo lead | 2026-08-29 |
| 6 | Cartosat-2S MX band composition - confirm 4-band VNIR, no SWIR | **RESOLVED** | Read from the real sample's `BAND_META.txt` (2026-08-29): **`NoOfBands=4`, `BandNumbers=1234`** - 4-band VNIR, **no SWIR**. The original assumption holds, so MNDWI/NDBI stay unavailable and the SWIR-free fallback paths are the operative ones. Also learned: `SatID=CARTOSAT-2E` (sample is 2E, same MX sensor family), `PixelSpacing=1.6 m` (matches the assumed GSD), **`BitsPerPixel=11` in a uint16 container**, `ORTHORECTIFIED`, UTM zone 45N / EPSG:32645, scene 7687x7640 px. | Geo lead | 2026-08-29 |
| 7 | Newer ≤4B VLM beats Qwen2.5-VL-3B on the §2.1 criteria | **Open** | InternVL3-1B (used by the BigEarthNet.txt authors) flagged as the candidate to evaluate first. Not yet benchmarked. | ML lead | Pending — Phase 1 |
| 8 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | **Open** | Not yet checked. Fallback: run Stage A3 optical-only and document the limitation. | Geo lead | Pending — Phase 1 |
| 9 | Prescribed benchmark test splits downloadable (VRSBench, RSVQA, CDVQA) | **Open** | Not yet checked. Fallback: use published splits from the papers' own repos. | Eval lead | Pending — Phase 1 |
| 10 | SIH 2026 timeline: internal deadline, grand finale dates, submission format | **Open** | Not yet confirmed with team/organizers. Fallback: compress the phase plan in doc `04` proportionally once the real deadline is known. | Team lead | Pending |
| 11 | Bhoonidhi registration approved; Cartosat-2S + RISAT products downloaded | **RESOLVED** | **Registration completed and products downloaded 2026-08-29.** On disk: Cartosat-2E MX (5132611), EOS-04 FRS-1 (226981731, quad-pol), EOS-04 MRS x2 (226981721, 247111021). Items 5 and 6 were both closed from this data. The products are held out as the cross-sensor generalisation set per docs/03 section 4.3 and are never trained on. | Geo lead | 2026-08-29 |
| 12 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | **Open** | Not yet checked. Fallback: baseline against the un-finetuned base VLM only. | ML #2 | Pending — Phase 1 |

## Summary

- **Resolved: 5/12** - items 1 and 2 (dataset paper, 2026-08-27), and items 5, 6
  and 11, all closed 2026-08-29 by reading real Bhoonidhi product metadata.
- **Open: 7/12** (items 3, 4, 7, 8, 9, 10, 12). Every one has a documented, costed
  fallback per doc `03` section 6, and **none blocks Phase 2**.
- **The two plan-changing unknowns are gone.** Cartosat-2E MX is confirmed 4-band
  VNIR with no SWIR, so the SWIR-free fallback paths are the operative ones. EOS-04
  is confirmed C-band at 5.40 GHz, within 0.09% of Sentinel-1's 5.405 GHz, which
  means backscatter behaviour learned on S1 transfers to the target sensor almost
  directly. Both were assumptions; both now rest on primary evidence.

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
