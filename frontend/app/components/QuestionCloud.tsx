'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Floating questions around a headline: the things people actually ask
 * of imagery, scattered across the section and drifting slowly, each with
 * a small counter and an arrow like a ticket. Positions are fixed per
 * phrase (seeded) so the layout is stable between renders; the drift is a
 * CSS keyframe with a per-item offset. Phrases fade in as the section
 * scrolls into view.
 */
const QUESTIONS: [string, number][] = [
  ['what changed here since the monsoon?', 41],
  ['is the new bypass finished?', 17],
  ['how many buildings are in this tile?', 63],
  ['cloud is hiding the site again', 28],
  ['which fields flooded last week?', 35],
  ['the SAR scene and the optical don\'t line up', 12],
  ['is that water or shadow?', 22],
  ['where exactly is the second ship?', 9],
  ['how much of this is built-up now?', 54],
  ['can I trust this number?', 77],
  ['describe this scene for the report', 31],
  ['did the reservoir shrink?', 19],
];

const SLOTS: [number, number][] = [
  [4, 6], [58, 4], [78, 14], [8, 22], [40, 18], [70, 30],
  [14, 62], [46, 70], [80, 58], [26, 84], [62, 86], [88, 78],
];

export default function QuestionCloud({ heading, sub }: { heading: string; sub: string }) {
  const ref = useRef<HTMLElement>(null);
  const [on, setOn] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) {
          setOn(true);
          io.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <section ref={ref} className={`qcloud${on ? ' is-on' : ''}`} id="questions">
      {QUESTIONS.map(([q, nn], i) => {
        const [x, y] = SLOTS[i % SLOTS.length];
        return (
          <span
            key={q}
            className="qcloud-item"
            style={{ left: `${x}%`, top: `${y}%`, animationDelay: `${-i * 1.7}s`, transitionDelay: `${i * 70}ms` }}
          >
            {q} <b>{nn}</b> <i>↗</i>
          </span>
        );
      })}
      <div className="qcloud-copy">
        <span className="eyebrow">[ the questions ]</span>
        <h2 className="display-l">{heading}</h2>
        <p>{sub}</p>
      </div>
    </section>
  );
}
