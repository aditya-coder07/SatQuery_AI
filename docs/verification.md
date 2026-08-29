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
| 8 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | **RESOLVED — available, but the WRONG BAND for EOS-04** | **Accessible: yes, and confirmed by retrieval, not by reading a web page.** The Umbra bucket `s3://umbra-open-data-catalog` (us-west-2) serves anonymously — no AWS account, no credentials — and a real product's STAC metadata was pulled and read (2026-08-29, saved as `artifacts/verify/umbra.json`). **Licences are permissive:** Umbra and Capella open data are **CC BY 4.0**; SpaceNet 6 is **CC BY-SA 4.0** — note the **share-alike**, which matters because the PS lists model weights as a deliverable. **But all three are X-band, and item 5 established EOS-04 is C-band.** Measured from the Umbra product: `sar:frequency_band=X`, `sar:center_frequency=9.6924 GHz` (λ 3.09 cm) against EOS-04's 5.40 GHz (λ 5.55 cm) — **+79.5%, a 1.79× wavelength ratio**, where the Sentinel-1 match that made item 5 comfortable was 0.09%. Two further mismatches in the same file: `sar:polarizations=['HH']` (single-pol, so the VH/VV ratio and CoV the index engine computes are not derivable) and `view:incidence_angle=22.9°` against EOS-04's 37.8°. Resolution is genuinely superb (`sar:resolution_azimuth=0.125 m`, `range=0.50 m`, SPOTLIGHT). **Verdict: these sources trade a resolution gap for a frequency, polarisation and geometry gap.** Stage A3 therefore ran optical-only (task 3.2), which is the plan's own documented fallback — chosen on evidence rather than on unavailability. **If the team confirms RISAT-2B/2BR1 instead of EOS-04, those are X-band and this verdict inverts** — these become the right sources, so item 5's residual risk and this item must be re-read together. | Geo lead | 2026-08-29 |
| 9 | Prescribed benchmark test splits downloadable (VRSBench, RSVQA, CDVQA) | **Partly resolved** | VRSBench downloads freely from HF `xiang709/VRSBench` (6.1 GB, CC-BY-4.0) and includes the prescribed eval splits (`VRSBench_EVAL_vqa/Cap/referring.json`). **But it ships annotations only - zero imagery.** Its 142,390 train rows are LLaVA-style `conversations` referencing images that live in the separate DOTA and DIOR datasets, so it cannot train anything on its own. RSVQA-LR is available self-contained as `dmarsili/RSVQA-LR-2k` (174 MB, CC-BY-4.0, images embedded in the parquet) and was used for Track B v0. CDVQA not yet checked. | Eval lead | 2026-08-29 |
| 10 | SIH 2026 timeline: internal deadline, grand finale dates, submission format | **Open** | Not yet confirmed with team/organizers. Fallback: compress the phase plan in doc `04` proportionally once the real deadline is known. | Team lead | Pending |
| 11 | Bhoonidhi registration approved; Cartosat-2S + RISAT products downloaded | **RESOLVED** | **Registration completed and products downloaded 2026-08-29.** On disk: Cartosat-2E MX (5132611), EOS-04 FRS-1 (226981731, quad-pol), EOS-04 MRS x2 (226981721, 247111021). Items 5 and 6 were both closed from this data. The products are held out as the cross-sensor generalisation set per docs/03 section 4.3 and are never trained on. | Geo lead | 2026-08-29 |
| 12 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | **Open** | Not yet checked. Fallback: baseline against the un-finetuned base VLM only. | ML #2 | Pending — Phase 1 |

## Summary

- **Resolved: 6/12** - items 1 and 2 (dataset paper, 2026-08-27), items 5, 6 and
  11 (closed 2026-08-29 by reading real Bhoonidhi product metadata), and item 8
  (closed 2026-08-29 by pulling a real Umbra product's STAC metadata from the
  open bucket).
- **Open: 6/12** (items 3, 4, 7, 9, 10, 12). Every one has a documented, costed
  fallback per doc `03` section 6, and **none blocks Phase 2 or Phase 3**.
- **The two plan-changing unknowns are gone.** Cartosat-2E MX is confirmed 4-band
  VNIR with no SWIR, so the SWIR-free fallback paths are the operative ones. EOS-04
  is confirmed C-band at 5.40 GHz, within 0.09% of Sentinel-1's 5.405 GHz, which
  means backscatter behaviour learned on S1 transfers to the target sensor almost
  directly. Both were assumptions; both now rest on primary evidence.
- **Item 8 resolved to a "no", which is more useful than the "yes" it was
  looking for.** High-resolution SAR is freely available and permissively
  licensed - the question as written - but every source is **X-band** while the
  target sensor is **C-band**, a 1.79x wavelength difference against the 0.09%
  match that made item 5 comfortable. The plan offered "SpaceNet 6 / Umbra, or
  optical-only with the limitation documented" as if the fallback were the
  weaker branch; the measurement says optical-only is the **correct** branch for
  EOS-04, and Stage A3 (task 3.2) took it on that basis rather than on
  unavailability.

  **This inverts if the sensor changes.** Item 5 records a residual risk that
  ISRO may use RISAT-2B/2BR1, which *are* X-band. Under that sensor these three
  sources become exactly right, and Stage A3 should be redone against them. The
  two items must be read together, and whoever confirms the sensor should come
  back to this row.

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

1. ~~Download the free Cartosat-2S MX and EOS-04 sample products~~ — **done
   2026-08-29**, closing items 5, 6 and 11.
2. ~~Check SpaceNet 6 / Umbra / Capella~~ — **done 2026-08-29**, closing item 8.
   Reproduce with:
   `curl -s "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=8&prefix=sar-data/tasks/"`
   then fetch any `*.stac.v2.json` and read `sar:frequency_band` and
   `sar:center_frequency`. No AWS account needed.
3. **Confirm which SAR sensor ISRO/SAC will actually use** (item 5's residual
   risk). This is now the highest-value open question, not a footnote: EOS-04
   is C-band and makes item 8's X-band sources wrong, while RISAT-2B/2BR1 are
   X-band and make them right. The answer decides whether Stage A3 stays
   optical-only or is redone against 0.25 m SAR.
4. Confirm the SIH 2026 submission deadline and format (item 10) with the
   team/organizers — this affects how the W0–W14 phase plan in
   `docs/04-Implementation-Plan.md` should be paced.
5. Items 3, 4, 7, 9, 12 are model/dataset-availability checks with zero-cost
   fallbacks already specified — verify opportunistically rather than blocking
   on them.
