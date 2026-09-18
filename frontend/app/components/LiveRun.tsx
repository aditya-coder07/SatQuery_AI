'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * A replay of a real run in a terminal panel: the curl command types
 * itself, then the server-sent events arrive in the order and at roughly
 * the pace the API emits them. The lines are the actual event names and
 * fields of `POST /runs/stream` (run_started, ingest, routing, step,
 * verification, confidence, complete) with the numbers from a stored
 * bi-temporal run of the demo pair, so what plays is what the console
 * shows. Starts when scrolled into view, replays on re-entry, and shows the
 * finished transcript under reduced motion.
 */

type Line = { t: number; cls: string; text: string };

const COMMAND =
  'curl -N -F "query=What changed between these two scenes?" \\\n     -F "images=@levir_t1.tif" -F "images=@levir_t2.tif" \\\n     http://localhost:8000/runs/stream';

// t = ms after the command is sent.
const EVENTS: Line[] = [
  { t: 120, cls: 'ev', text: 'event: run_started    run_id=run_8ff64ac30953  images=2' },
  { t: 380, cls: 'ev', text: 'event: ingest         12 checks' },
  { t: 420, cls: 'ok', text: '  PASS  t1: CRS EPSG:32643 · 0.0% nodata · ~5% cloud · 256x256' },
  { t: 470, cls: 'ok', text: '  PASS  t2: CRS EPSG:32643 · 0.0% nodata · ~2% cloud · 256x256' },
  { t: 520, cls: 'ok', text: '  PASS  both in EPSG:32643 · GSD ratio 1.00x · footprint overlap 100%' },
  { t: 560, cls: 'ok', text: '  PASS  t1 2026-01-01 precedes t2 2026-06-01' },
  { t: 760, cls: 'ev', text: 'event: routing        selected_task=TEMPORAL_CHANGE_DESC  classifier=tfidf_logreg_v1 top1=0.997' },
  { t: 900, cls: 'ev', text: 'event: step           index_engine_v1     11 ms   EXPLICIT_CHANGE_LANGUAGE' },
  { t: 1180, cls: 'ev', text: 'event: step           change_mask_v1     208 ms   sha256:3d0c…9b2e' },
  { t: 3400, cls: 'ev', text: 'event: step           change_caption_v1 2164 ms   adapter=change_caption_vlm' },
  { t: 3600, cls: 'ev', text: 'event: verification   3 sentences · 0 flagged · 3 unverifiable · deterministic' },
  { t: 3700, cls: 'ev', text: 'event: confidence     0.9174  HIGH  model=0.88 agreement=1.00 input=0.85' },
  { t: 3850, cls: 'ev', text: 'event: complete       answer persisted · /runs/run_8ff64ac30953' },
  { t: 3950, cls: 'ans', text: '"many houses are built on both sides of the roads. No dominant land-cover class exceeded its detection threshold. The scene is centred at 18.0881° N, 75.0006° E and covers about 130 m by 130 m on the ground."' },
];

export default function LiveRun() {
  const root = useRef<HTMLDivElement>(null);
  const [typed, setTyped] = useState('');
  const [shown, setShown] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      setTyped(COMMAND);
      setShown(EVENTS.length);
      setDone(true);
      return;
    }
    let timers: number[] = [];
    const clear = () => {
      timers.forEach((id) => window.clearTimeout(id));
      timers = [];
    };
    const play = () => {
      clear();
      setTyped('');
      setShown(0);
      setDone(false);
      // Type the command at ~28 ms a character, then stream the events.
      for (let i = 1; i <= COMMAND.length; i++) {
        timers.push(window.setTimeout(() => setTyped(COMMAND.slice(0, i)), i * 22));
      }
      const t0 = COMMAND.length * 22 + 350;
      EVENTS.forEach((ev, i) => {
        timers.push(window.setTimeout(() => setShown(i + 1), t0 + ev.t));
      });
      timers.push(window.setTimeout(() => setDone(true), t0 + EVENTS[EVENTS.length - 1].t + 400));
    };
    const io = new IntersectionObserver(
      (es) => {
        for (const e of es) {
          if (e.isIntersecting) play();
          else clear();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => {
      clear();
      io.disconnect();
    };
  }, []);

  return (
    <div ref={root} className="liverun" aria-label="A replayed run">
      <div className="liverun-bar">
        <span className="liverun-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="liverun-title">POST /runs/stream · server-sent events</span>
        <span className={`liverun-state${done ? ' is-done' : ''}`}>{done ? 'complete · 2.62 s' : 'streaming'}</span>
      </div>
      <pre className="liverun-body">
        <code>
          <span className="cmd">
            <span className="prompt">$ </span>
            {typed}
            {typed.length < COMMAND.length && <span className="caret" />}
          </span>
          {EVENTS.slice(0, shown).map((ev, i) => (
            <span key={i} className={`line ${ev.cls}`}>
              {ev.text}
            </span>
          ))}
          {typed.length === COMMAND.length && !done && <span className="line ev"><span className="caret" /></span>}
        </code>
      </pre>
    </div>
  );
}
