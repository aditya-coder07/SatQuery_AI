/**
 * Home-page content. Numbers are the measured official-split figures from
 * docs/research/sota_matrix.md (Phase 6, 2026-09-18); update them there
 * first, then here. Nothing on the home page is a target or a projection.
 */

export const SENSORS = [
  { name: 'Cartosat-2E MX', kind: 'optical · 4-band · 1.6 m' },
  { name: 'EOS-04 FRS-1', kind: 'SAR · C-band · HH' },
  { name: 'Sentinel-2', kind: 'multispectral · 12 bands used' },
  { name: 'Landsat 8/9', kind: 'multispectral · 30 m' },
  { name: 'DIOR / DOTA', kind: 'aerial · object grounding' },
  { name: 'BigEarthNet', kind: '19-class land cover' },
  { name: 'LEVIR-CD', kind: 'bi-temporal change' },
  { name: 'WHU-OPT-SAR', kind: 'optical–SAR pairs' },
];

export const CAPABILITIES = [
  {
    title: 'Understand a scene',
    lede: 'One shared 4-bit Qwen2.5-VL base with a LoRA adapter per task, so three tools cost one model in VRAM.',
    items: [
      {
        name: 'Visual question answering',
        tool: 'rs_vqa',
        metric: '91.2 % · RSVQA-LR',
        text: 'Presence, comparison, counting and rural/urban questions answered in the published convention.',
      },
      {
        name: 'Captioning',
        tool: 'caption',
        metric: 'BLEU-4 0.256 · RSICD',
        text: 'A grounded description of the scene; CIDEr-D 0.79 against five references.',
      },
      {
        name: 'Visual grounding',
        tool: 'grounding',
        metric: 'Acc@0.5 0.732 · DIOR-RSVG',
        text: 'Point at "the second ship from the left" and get a box - served at the 1024-px budget it was measured with.',
      },
    ],
  },
  {
    title: 'See what changed',
    lede: 'Two dates in, the difference out: a mask, a sentence, or an answer.',
    items: [
      {
        name: 'Change mask',
        tool: 'change_mask',
        metric: 'F1 0.904 · LEVIR-CD',
        text: 'Pixel-level change between two co-registered scenes, calibrated on the official validation split.',
      },
      {
        name: 'Change captioning',
        tool: 'change_caption',
        metric: 'BLEU-4 0.605 · LEVIR-CC',
        text: 'Describes what changed and where, from the images alone - no ground-truth mask at inference.',
      },
      {
        name: 'Change question answering',
        tool: 'change_vqa',
        metric: null,
        text: 'Ask about the change directly. Benchmarked on the licensed Landsat-SCD set (mIoU 0.623, SeK 0.509).',
      },
    ],
  },
  {
    title: 'Land, water and radar',
    lede: 'Multispectral heads that keep working with fewer bands, and a fusion head for when the optical scene is clouded.',
    items: [
      {
        name: 'Land cover',
        tool: 'landcover',
        metric: 'mAP 0.885 · BigEarthNet',
        text: '19 classes on the complete official split; 94 % of the score retained with only the four Cartosat bands.',
      },
      {
        name: 'Optical–SAR fusion',
        tool: 'optsar_fusion',
        metric: '+0.021 mIoU fused',
        text: 'Scene-disjoint WHU-OPT-SAR: the fused head beats optical alone, and degrades gracefully when one side is missing.',
      },
      {
        name: 'Index engine',
        tool: 'index_engine',
        metric: 'deterministic · 0 MB',
        text: 'NDVI, NDWI, NDBI and friends in pure numpy - the physics half of every answer, and the whole answer in lite mode.',
      },
    ],
  },
  {
    title: 'Trust the output',
    lede: 'The parts that are not models: what the system does so that a wrong answer is caught before you read it.',
    items: [
      {
        name: 'Entailment gate',
        tool: 'verifier',
        metric: 'per sentence',
        text: 'Each sentence of the answer is checked against the run payload; flagged and unverifiable ones are marked, not hidden.',
      },
      {
        name: 'Calibrated confidence',
        tool: 'confidence',
        metric: 'ECE 0.003 · land cover',
        text: 'Model score, cross-tool agreement and input quality, with registries fitted on validation - never on test.',
      },
      {
        name: 'Abstention',
        tool: 'router',
        metric: 'named trigger',
        text: 'Incompatible footprints, a missing band or a failed tool produce an abstain with its reason - not a confident guess.',
      },
    ],
  },
];

export const BENCHMARKS = [
  { task: 'Visual QA', value: '0.912', metric: 'accuracy', dataset: 'RSVQA-LR test · n 10,004', note: '95 % CI [0.905, 0.918]' },
  { task: 'Grounding', value: '0.732', metric: 'Acc@0.5', dataset: 'DIOR-RSVG test · n 7,500', note: 'VRSBench val 0.659' },
  { task: 'Change mask', value: '0.904', metric: 'F1', dataset: 'LEVIR-CD test', note: 'IoU 0.824 · independently re-scored' },
  { task: 'Land cover', value: '0.885', metric: 'micro mAP', dataset: 'BigEarthNet-S2 test · n 125,866', note: 'macro 0.792' },
  { task: 'Captioning', value: '0.256', metric: 'BLEU-4', dataset: 'RSICD test · n 1,093', note: 'CIDEr-D 0.793' },
  { task: 'Change caption', value: '0.605', metric: 'BLEU-4', dataset: 'LEVIR-CC test', note: 'CIDEr-D 1.29' },
];

export const METHOD = [
  {
    title: 'Ingest',
    line: 'Nothing runs on a scene that has not been checked.',
    points: [
      'Reads GeoTIFF, JP2, PNG - a multi-band vendor product counts as one scene',
      'CRS, nodata, cloud, size, GSD ratio, footprint overlap and date order checked',
      'A PNG is accepted for a visual question and its missing georeference is named',
    ],
  },
  {
    title: 'Route',
    line: 'The question picks the plan; the inputs decide what is legal.',
    points: [
      'The capability matrix says what is legal for these inputs',
      'A classifier picks the task; a tie goes to the language model',
      'The profile budget drops tools the card cannot hold',
    ],
  },
  {
    title: 'Execute',
    line: 'Specialists, not one model asked to do everything.',
    points: [
      'Specialist tools run in plan order, streaming each step',
      'VLM tools share one base and attach adapters by name',
      'Runtime, version and weight hashes are recorded per step',
    ],
  },
  {
    title: 'Verify',
    line: 'If the payload cannot support a sentence, the sentence says so.',
    points: [
      'Every sentence is checked against the measured payload',
      'Flagged and unverifiable sentences are marked in the answer',
      'Conflicts between tools are surfaced, not averaged away',
    ],
  },
  {
    title: 'Calibrate',
    line: 'A confidence you can act on, or an abstain with its reason.',
    points: [
      'Confidence from model score, agreement and input quality',
      'Registries fitted on official validation splits',
      'Below the bar, the run abstains with a named trigger',
    ],
  },
];

export const CARRIES = [
  { k: 'run_id', title: 'A permanent id', text: 'Every run is stored with its full trace and reopens at /runs/{id} - the same page a colleague sees.' },
  { k: 'checks', title: 'The ingest checks', text: 'Ten to twelve named checks with PASS, WARN or FAIL, so a bad input is diagnosed before a model touches it.' },
  { k: 'confidence', title: 'A confidence band', text: 'HIGH, MEDIUM or LOW with its components, and "uncalibrated" said out loud when it is not a probability.' },
  { k: 'weights', title: 'Weight hashes', text: 'The sha256 of every checkpoint that contributed, so the answer can be reproduced against the same model.' },
  { k: 'overlays', title: 'Georeferenced overlays', text: 'Masks and index rasters in the scene\'s own CRS, on a live basemap or exported.' },
  { k: 'report', title: 'A PDF report', text: 'The answer, the checks, the previews and the provenance on one page, for the people who were not watching.' },
];
