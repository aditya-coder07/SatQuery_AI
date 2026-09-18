'use client';

import { useEffect, useRef, useState } from 'react';

import Words from './Words';

/**
 * Floating questions around a headline: the things people actually ask
 * of imagery, scattered in a ring around the copy (never over it) with a
 * ticket-style count and an arrow. When the section arrives each phrase
 * drifts in from off-stage with a blur, its count ticks up from zero, and
 * from then on it keeps floating on its own slow orbit, leans a little
 * toward the pointer, and now and then one lights up as if it had just
 * been asked. The headline reveals word by word.
 */
const QUESTIONS: [string, number][] = [
  ['what changed here since the monsoon?', 41],
  ['is the new bypass finished?', 17],
  ['how many buildings are in this tile?', 63],
  ['cloud is hiding the site again', 28],
  ['which fields flooded last week?', 35],
  ["the SAR scene and the optical don't line up", 12],
  ['is that water or shadow?', 22],
  ['where exactly is the second ship?', 9],
  ['how much of this is built-up now?', 54],
  ['can I trust this number?', 77],
  ['describe this scene for the report', 31],
  ['did the reservoir shrink?', 19],
];

// Ring positions (% of the section), leaving the middle for the copy.
const SLOTS: [number, number][] = [
  [4, 8], [58, 5], [78, 16], [6, 28], [40, 12], [76, 34],
  [12, 84], [46, 92], [80, 76], [8, 60], [64, 88], [86, 54],
];

export default function QuestionCloud({ heading, sub }: { heading: string; sub: string }) {
  const ref = useRef<HTMLElement>(null);
  const [on, setOn] = useState(false);
  const [lit, setLit] = useState(-1);
  const [counts, setCounts] = useState<number[]>(() => QUESTIONS.map(() => 0));
  const [typed, setTyped] = useState('');
  const [typing, setTyping] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let timers: number[] = [];
    let lightTimer = 0;
    const io = new IntersectionObserver(
      (es) => {
        if (!es.some((e) => e.isIntersecting)) return;
        setOn(true);
        io.disconnect();
        if (reduced) {
          setCounts(QUESTIONS.map(([, n]) => n));
          setTyped(sub);
          return;
        }
        // The sub line types itself once the headline's words have landed.
        const headStart = 200 + heading.split(' ').length * 260 + 500;
        timers.push(window.setTimeout(() => setTyping(true), headStart));
        for (let i = 1; i <= sub.length; i++) {
          timers.push(window.setTimeout(() => setTyped(sub.slice(0, i)), headStart + i * 26));
        }
        timers.push(window.setTimeout(() => setTyping(false), headStart + sub.length * 26 + 600));
        // Counts tick up from zero, each on its own start.
        QUESTIONS.forEach(([, n], i) => {
          const start = 300 + i * 90;
          for (let k = 1; k <= 14; k++) {
            timers.push(
              window.setTimeout(() => {
                setCounts((c) => {
                  const next = c.slice();
                  next[i] = Math.round((n * k) / 14);
                  return next;
                });
              }, start + k * 55),
            );
          }
        });
        // Every few seconds one question lights up briefly.
        const light = () => {
          setLit(Math.floor(Math.random() * QUESTIONS.length));
          lightTimer = window.setTimeout(() => {
            setLit(-1);
            lightTimer = window.setTimeout(light, 1800 + Math.random() * 2200);
          }, 1400);
        };
        lightTimer = window.setTimeout(light, 2600);
      },
      { threshold: 0.15 },
    );
    io.observe(el);

    // Pointer parallax: the cloud leans a little toward the cursor.
    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width - 0.5) * 2;
      const y = ((e.clientY - r.top) / r.height - 0.5) * 2;
      el.style.setProperty('--px', x.toFixed(3));
      el.style.setProperty('--py', y.toFixed(3));
    };
    if (!reduced) el.addEventListener('pointermove', onMove, { passive: true });
    return () => {
      io.disconnect();
      timers.forEach((t) => window.clearTimeout(t));
      window.clearTimeout(lightTimer);
      el.removeEventListener('pointermove', onMove);
    };
  }, [heading, sub]);

  return (
    <section ref={ref} className={`qcloud${on ? ' is-on' : ''}`} id="questions">
      {QUESTIONS.map(([q, n], i) => {
        const [x, y] = SLOTS[i % SLOTS.length];
        // Enter from the side the slot is nearest to.
        const fromX = x < 50 ? -60 : 60;
        const fromY = y < 50 ? -30 : 30;
        const depth = 0.6 + ((i * 7) % 5) * 0.2; // parallax strength per item
        return (
          <span
            key={q}
            className={`qcloud-item${lit === i ? ' is-lit' : ''}`}
            style={
              {
                left: `${x}%`,
                top: `${y}%`,
                '--fx': `${fromX}px`,
                '--fy': `${fromY}px`,
                '--depth': depth,
                '--dur': `${7 + (i % 4) * 1.6}s`,
                animationDelay: `${-i * 1.7}s`,
                transitionDelay: `${120 + i * 90}ms`,
              } as React.CSSProperties
            }
          >
            <span className="qcloud-text">{q}</span>
            <b>{counts[i]}</b>
            <i>↗</i>
          </span>
        );
      })}
      <div className="qcloud-copy">
        <span className="eyebrow">[ the questions ]</span>
        <h2 className="display-l">
          <Words text={heading} stagger={260} delay={200} />
        </h2>
        <p className="qcloud-sub">
          {typed}
          {typing && <span className="qcloud-caret" />}
        </p>
      </div>
    </section>
  );
}
