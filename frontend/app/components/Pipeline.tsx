'use client';

/**
 * The pipeline board.
 *
 * The published design animated a canned seven-node run on a timer. On the
 * real page that would be a lie, so the board is driven by the SSE stream
 * instead: a node lights when its event arrives, and the clock is the actual
 * elapsed wall time since `run_started`. With no run yet it renders its own
 * skeleton with em-dashes rather than going blank — but it never invents a
 * number.
 *
 * Three corrections against the published board, all checked in the executor:
 *
 * 1. The kind is `STEP`, not `SPECIALIST` — `emit("step", ...)` at
 *    satquery/controller/executor.py:284.
 * 2. There is a `VERIFICATION` node between the steps and confidence —
 *    `emit("verification", ...)` at satquery/controller/executor.py:398. The
 *    published board had no such node, and it is the one that says whether
 *    the sentences in the answer are supported by the payload.
 * 3. The number of step nodes comes from the run. The executor loops over a
 *    plan, so a run may have one step or five, and they are *sequential* —
 *    the board chains them rather than fanning them out in parallel.
 */

import { motion, useReducedMotion } from 'framer-motion';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import type { TraceEvent } from '../lib/events';
import { isCalibrated } from '../lib/events';

const NODE_W = 196;
const NODE_H = 96;
const COL_GAP = 56;
const PAD = 16;
const LANE_Y = 152;
const TOP_Y = 36;
const BOTTOM_Y = 272;
const DESIGN_H = 400;

/**
 * Minimum time a node holds the baton before the next one takes it.
 *
 * The board is driven by the SSE stream, and a cached or trivial run can emit
 * every event inside about a tenth of a second. Bound straight to the events,
 * the whole graph flicked from empty to finished in one frame — technically
 * accurate and completely unreadable. The reveal is rate-limited instead, so
 * the board always plays through the pipeline in order and you can see which
 * stage produced what. It only ever lags a fast run; a slow one is still
 * governed by the events themselves, and the wall clock beside it is the
 * measured number either way.
 */
const REVEAL_MS = 280;

type NodeState = 'idle' | 'active' | 'done';

type BoardNode = {
  id: string;
  kind: string;
  title: string;
  value: string;
  sub: string;
  badge?: { text: string; tone: 'ok' | 'warn' | 'fail' };
  x: number;
  y: number;
  state: NodeState;
};

function ms(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

/**
 * Build the board from whatever events have arrived so far.
 *
 * Every value on a node is read out of an event payload. A node with no event
 * yet shows an em-dash: the board's shape is known in advance, its readings
 * are not.
 */
function buildBoard(
  events: TraceEvent[],
  running: boolean,
): { nodes: BoardNode[]; edges: [string, string][]; phase: string } {
  const first = (name: string) => events.find((e) => e.name === name)?.data;
  const ingest = first('ingest');
  const routing = first('routing');
  const steps = events.filter((e) => e.name === 'step').map((e) => e.data);
  const verification = first('verification');
  const confidence = first('confidence');
  const complete = first('complete');
  const failed = first('error');

  const nodes: BoardNode[] = [];
  const edges: [string, string][] = [];
  let col = 0;
  const at = (c: number) => PAD + c * (NODE_W + COL_GAP);

  const settle = (has: unknown, isNext: boolean): NodeState =>
    has ? 'done' : isNext && running ? 'active' : 'idle';

  // --- ingest -------------------------------------------------------------
  const checks: any[] = ingest?.checks ?? [];
  const counts = {
    pass: checks.filter((c) => c.status === 'PASS').length,
    warn: checks.filter((c) => c.status === 'WARN').length,
    fail: checks.filter((c) => c.status === 'FAIL').length,
  };
  const images: any[] = ingest?.images ?? [];
  nodes.push({
    id: 'ingest',
    kind: 'INGEST',
    title: 'Ingest',
    value: ingest ? `${images.length} scene${images.length === 1 ? '' : 's'} · ${checks.length} checks` : '—',
    sub: ingest
      ? [
          images.map((i) => i.role).join(' · ') || 'no roles',
          // Said on the node, not only in the trace: the whole board describes
          // a run on a crop when one was selected.
          images.some((i) => i.aoi_applied) ? 'cropped to area' : '',
        ]
          .filter(Boolean)
          .join(' · ')
      : '',
    badge: !ingest
      ? undefined
      : counts.fail
        ? { text: 'FAIL', tone: 'fail' }
        : counts.warn
          ? { text: 'WARN', tone: 'warn' }
          : { text: 'PASS', tone: 'ok' },
    x: at(col++),
    y: LANE_Y,
    state: settle(ingest, true),
  });

  // --- routing ------------------------------------------------------------
  nodes.push({
    id: 'route',
    kind: 'ROUTING',
    title: 'Route',
    value: routing?.selected_task ?? '—',
    sub: routing ? `confidence ${Number(routing.confidence ?? 0).toFixed(2)}` : '',
    badge: routing ? { text: 'DONE', tone: 'ok' } : undefined,
    x: at(col++),
    y: LANE_Y,
    state: settle(routing, Boolean(ingest)),
  });
  edges.push(['ingest', 'route']);

  // --- steps, in the order the executor ran them --------------------------
  // A run with no steps still gets one placeholder so the board keeps its
  // shape while the first step is in flight.
  const stepCount = Math.max(steps.length, 1);
  let previous = 'route';
  for (let i = 0; i < stepCount; i++) {
    const step = steps[i];
    const id = `step-${i}`;
    nodes.push({
      id,
      kind: 'STEP',
      title: step?.tool ?? `Step ${i + 1}`,
      value: step ? ms(step.runtime_ms) : '—',
      sub: step ? [step.rationale_tag, step.version].filter(Boolean).join(' · ') : '',
      badge: step ? { text: 'DONE', tone: 'ok' } : undefined,
      x: at(col++),
      y: LANE_Y,
      state: settle(step, i === steps.length && Boolean(routing)),
    });
    edges.push([previous, id]);
    previous = id;
  }

  // --- verification -------------------------------------------------------
  const gate = verification?.entailment_gate;
  const conflicts: string[] = verification?.conflicts ?? [];
  nodes.push({
    id: 'verify',
    kind: 'VERIFICATION',
    title: 'Verify',
    // `retained` alone reads as "verified", and the trace is explicit that it
    // is not: a sentence nothing in the payload speaks to is unverifiable,
    // neither supported nor contradicted. The board shows the split.
    value: gate ? `${gate.retained}/${gate.sentences} retained` : '—',
    sub: gate
      ? `${gate.flagged} flagged · ${gate.unverifiable ?? 0} unverifiable · ${gate.backend}`
      : '',
    badge: !verification
      ? undefined
      : conflicts.length || (gate?.flagged ?? 0) > 0
        ? { text: 'FLAGGED', tone: 'warn' }
        : { text: 'CLEAN', tone: 'ok' },
    x: at(col++),
    y: LANE_Y,
    state: settle(verification, steps.length > 0),
  });
  edges.push([previous, 'verify']);

  // --- confidence and the answer, side by side ----------------------------
  const lastCol = at(col);
  nodes.push({
    id: 'confidence',
    kind: 'CONFIDENCE',
    title: 'Confidence',
    value: confidence ? Number(confidence.final).toFixed(2) : '—',
    sub: confidence
      ? `${confidence.band} · ${isCalibrated(confidence) ? 'calibrated' : 'uncalibrated'}`
      : '',
    badge: confidence ? { text: confidence.band, tone: 'ok' } : undefined,
    x: lastCol,
    y: TOP_Y,
    state: settle(confidence, Boolean(verification)),
  });
  nodes.push({
    id: 'answer',
    kind: failed ? 'ERROR' : 'COMPLETE',
    title: failed ? 'Failed' : complete?.abstained ? 'Abstained' : 'Answer',
    value: failed
      ? 'run stopped'
      : complete
        ? complete.abstained
          ? 'no answer returned'
          : 'persisted'
        : '—',
    sub: complete?.abstain_trigger ?? (complete ? 'overlays ready' : ''),
    badge: failed
      ? { text: 'ERROR', tone: 'fail' }
      : complete
        ? complete.abstained
          ? { text: 'ABSTAIN', tone: 'warn' }
          : { text: 'DONE', tone: 'ok' }
        : undefined,
    x: lastCol,
    y: BOTTOM_Y,
    state: settle(complete ?? failed, Boolean(verification)),
  });
  edges.push(['verify', 'confidence'], ['verify', 'answer']);

  const phase = failed
    ? 'error'
    : complete
      ? complete.abstained
        ? 'abstained'
        : 'complete'
      : verification
        ? 'confidence'
        : steps.length
          ? 'step'
          : routing
            ? 'step'
            : ingest
              ? 'routing'
              : running
                ? 'ingest'
                : 'idle';

  return { nodes, edges, phase };
}

type Placed = BoardNode & { w: number; h: number };

function edgePath(a: Placed, b: Placed, portrait: boolean): string {
  if (portrait) {
    // Bottom-centre of the feeder to top-centre of the fed node.
    const x1 = a.x + a.w / 2;
    const y1 = a.y + a.h;
    const x2 = b.x + b.w / 2;
    const y2 = b.y;
    const dy = Math.max(18, (y2 - y1) * 0.55);
    return `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`;
  }
  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const x2 = b.x;
  const y2 = b.y + b.h / 2;
  const dx = Math.max(46, (x2 - x1) * 0.55);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

const ROW_GAP = 28;
const PAIR_GAP = 12;

/**
 * Place the board for the width it has.
 *
 * Landscape keeps the designed positions (a scaled transform fits them to
 * the container). Portrait - a phone - is the same graph turned on its
 * side: the lane runs down the screen, one node per row at full width, and
 * the two nodes that share the last column (confidence, answer) sit side by
 * side, so the wires still show the branch. It is a re-layout, not a list:
 * nothing about the graph is lost on a narrow screen, only its direction.
 */
function place(nodes: BoardNode[], portrait: boolean, width: number): { placed: Placed[]; w: number; h: number } {
  if (!portrait) {
    const placed = nodes.map((n) => ({ ...n, w: NODE_W, h: NODE_H }));
    return { placed, w: Math.max(...nodes.map((n) => n.x + NODE_W)) + PAD, h: DESIGN_H };
  }
  const inner = Math.max(200, width - 2 * PAD);
  const placed: Placed[] = [];
  let y = PAD;
  let i = 0;
  while (i < nodes.length) {
    const n = nodes[i];
    const pair = nodes[i + 1];
    // Two nodes on the same design column are a branch: one row, two halves.
    if (pair && pair.x === n.x) {
      const half = (inner - PAIR_GAP) / 2;
      placed.push({ ...n, x: PAD, y, w: half, h: NODE_H });
      placed.push({ ...pair, x: PAD + half + PAIR_GAP, y, w: half, h: NODE_H });
      i += 2;
    } else {
      placed.push({ ...n, x: PAD, y, w: inner, h: NODE_H });
      i += 1;
    }
    y += NODE_H + ROW_GAP;
  }
  return { placed, w: width, h: y - ROW_GAP + PAD };
}

export default function Pipeline({
  events,
  running,
  runId,
  startedAt,
}: {
  events: TraceEvent[];
  running: boolean;
  runId: string;
  startedAt: number | null;
}) {
  const reduce = useReducedMotion();
  const fitRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [portrait, setPortrait] = useState(false);
  const [boxWidth, setBoxWidth] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const [revealed, setRevealed] = useState(0);

  const { nodes, edges, phase } = buildBoard(events, running);
  const board = place(nodes, portrait, boxWidth);
  const byId = Object.fromEntries(board.placed.map((n) => [n.id, n]));
  const designWidth = board.w;
  const designHeight = board.h;

  // How far the events have actually got, against how far the board has been
  // allowed to show. The second chases the first, one node at a time.
  const reached = nodes.filter((n) => n.state !== 'idle').length;

  const displayState = (index: number): NodeState => {
    if (index < revealed) return 'done';
    if (index === revealed && (revealed < reached || running)) return 'active';
    return 'idle';
  };

  const progress = nodes.length ? Math.min(revealed, nodes.length) / nodes.length : 0;

  /* Advance one node per tick until the board has caught up with the stream.
     Under reduced motion there is no reveal to watch, so it jumps straight to
     wherever the events are. */
  useEffect(() => {
    if (revealed >= reached) return;
    if (reduce) {
      setRevealed(reached);
      return;
    }
    const id = window.setTimeout(() => setRevealed((r) => r + 1), REVEAL_MS);
    return () => window.clearTimeout(id);
  }, [revealed, reached, reduce]);

  /* A new run rewinds the board. */
  useEffect(() => {
    setRevealed(0);
  }, [startedAt]);

  /* The wall clock is measured, not scripted: it ticks while the run is open
     and freezes on the last reading when it closes. */
  useEffect(() => {
    if (!startedAt) return;
    if (!running) {
      setElapsed((e) => e);
      return;
    }
    const id = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 100);
    return () => window.clearInterval(id);
  }, [startedAt, running]);

  useEffect(() => {
    if (startedAt === null) setElapsed(0);
  }, [startedAt]);

  const fit = useCallback(() => {
    const box = fitRef.current;
    const stage = stageRef.current;
    if (!box || !stage) return;
    // Below this width the scaled landscape board would shrink its nodes
    // past legibility, so the graph is laid out down the screen instead.
    const narrow = box.clientWidth < 760;
    setPortrait(narrow);
    setBoxWidth(box.clientWidth);
    if (narrow) {
      stage.style.transform = '';
      stage.style.width = '100%';
      stage.style.height = `${designHeight}px`;
      box.style.height = `${designHeight}px`;
      return;
    }
    const scale = Math.min(box.clientWidth / designWidth, 1);
    stage.style.width = `${designWidth}px`;
    stage.style.height = `${DESIGN_H}px`;
    stage.style.transform = `scale(${scale})`;
    box.style.height = `${DESIGN_H * scale}px`;
  }, [designWidth, designHeight]);

  useLayoutEffect(() => {
    fit();
    window.addEventListener('resize', fit);
    return () => window.removeEventListener('resize', fit);
  }, [fit]);

  return (
    <section className="panel" aria-label="Pipeline">
      <div className="rig-head">
        <span className="live">
          <motion.span
            className={`pulse${running ? '' : ' idle'}`}
            animate={running && !reduce ? { opacity: [1, 0.25, 1], scale: [1, 1.35, 1] } : { opacity: 1, scale: 1 }}
            transition={running && !reduce ? { duration: 2.2, repeat: Infinity, ease: 'easeInOut' } : { duration: 0 }}
          />
          <span className="label">{running ? 'Live pipeline' : 'Pipeline'}</span>
        </span>
        <span className="clock">
          {runId ? runId : 'no run yet'} · <b>{elapsed.toFixed(2)} s</b>
        </span>
        <span className="rail">
          <motion.i
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: reduce ? 0 : 0.35, ease: [0.16, 0.9, 0.28, 1] }}
          />
        </span>
        <span className="label">{phase}</span>
      </div>

      <div className="stage-fit" ref={fitRef}>
        <div className={`stage${portrait ? ' portrait' : ''}`} ref={stageRef}>
          <svg
            className="wires"
            viewBox={`0 0 ${designWidth} ${designHeight}`}
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="flowGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#A16207" stopOpacity="0.22" />
                <stop offset="55%" stopColor="#E8C39E" stopOpacity="0.95" />
                <stop offset="100%" stopColor="#E8C39E" stopOpacity="0.4" />
              </linearGradient>
            </defs>
            {edges.map(([from, to]) => {
              const a = byId[from];
              const b = byId[to];
              if (!a || !b) return null;
              const d = edgePath(a, b, portrait);
              // An edge lights when the node it feeds does, so the flow runs
              // ahead of each node rather than all at once.
              const lit = displayState(nodes.findIndex((n) => n.id === b.id)) !== 'idle';
              return (
                <g key={`${from}-${to}`}>
                  <path className="edge-base" d={d} />
                  <motion.path
                    className="edge-flow"
                    d={d}
                    initial={false}
                    animate={{ pathLength: lit ? 1 : 0 }}
                    transition={{ duration: reduce ? 0 : 0.45, ease: [0.16, 0.9, 0.28, 1] }}
                  />
                </g>
              );
            })}
          </svg>

          {board.placed.map((node) => {
            const index = nodes.findIndex((n) => n.id === node.id);
            const state = displayState(index);
            return (
            <motion.div
              key={node.id}
              className={`node${state === 'idle' ? '' : ` ${state}`}`}
              style={{ left: node.x, top: node.y, width: node.w, height: node.h }}
              initial={false}
              // The one place a spring belongs: a node taking the baton. Two
              // states, not a keyframe array — a spring can only interpolate
              // between two values, and handing it three throws.
              animate={{ scale: state === 'active' && !reduce ? 1.025 : 1 }}
              transition={{ type: 'spring', stiffness: 260, damping: 14 }}
            >
              <span className="kind">{node.kind}</span>
              <span className="title">{node.title}</span>
              <span className="val">{state === 'idle' ? '—' : node.value}</span>
              <span className="sub">{node.sub}</span>
              {node.badge && state === 'done' && (
                <span
                  className={`badge${node.badge.tone === 'ok' ? '' : ` ${node.badge.tone}`}`}
                >
                  {node.badge.text}
                </span>
              )}
              <motion.span
                className="node-fill"
                initial={false}
                animate={{ width: state === 'idle' ? '0%' : '100%' }}
                transition={{
                  duration: reduce ? 0 : state === 'active' ? REVEAL_MS / 1000 : 0.4,
                  ease: state === 'active' ? 'linear' : [0.16, 0.9, 0.28, 1],
                }}
              />
            </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
