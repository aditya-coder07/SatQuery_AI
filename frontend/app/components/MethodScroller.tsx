'use client';

import { useEffect, useRef, useState } from 'react';

import ParticleCloud from './ParticleCloud';

/**
 * The pinned, horizontally scrolling method section.
 *
 * The section is (phases + 1) viewports tall; a sticky viewport inside it
 * shows an intro slide and then one slide per phase, translated sideways
 * by scroll position, so the wheel drives the sequence and there is
 * nothing to click. Each slide carries a huge ghost number behind it, a
 * title whose letters reveal as the slide arrives, a one-line claim and
 * three points. A dot rail at the bottom shows where you are. The section
 * is light - the page floods to light as it arrives and back to dark as it
 * leaves, through the ink edges at its top and bottom.
 *
 * Driven by one scroll listener and requestAnimationFrame; no animation
 * library. Reduced motion lays the slides out vertically with no pinning.
 */

export type Phase = { title: string; line: string; points: string[] };

export default function MethodScroller({
  eyebrow,
  heading,
  intro,
  phases,
}: {
  eyebrow: string;
  heading: string;
  intro: string;
  phases: Phase[];
}) {
  const root = useRef<HTMLElement>(null);
  const [p, setP] = useState(0); // 0..1 through the whole section
  const [reduced, setReduced] = useState(false);
  const n = phases.length;

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    if (mq.matches) return;
    const el = root.current;
    if (!el) return;
    let raf = 0;
    const update = () => {
      raf = 0;
      const rect = el.getBoundingClientRect();
      const travel = rect.height - window.innerHeight;
      setP(travel > 0 ? Math.min(1, Math.max(0, -rect.top / travel)) : 0);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  // Slide index as a real number: 0 = intro, 1..n = phases. Each slide
  // dwells for the outer halves of its scroll segment and moves during the
  // middle half, so the wheel reads as "hold, slide, hold" rather than a
  // continuous drift.
  const raw = p * n;
  const seg = Math.floor(raw);
  const f = raw - seg;
  const ease = f < 0.25 ? 0 : f > 0.75 ? 1 : (() => { const x = (f - 0.25) / 0.5; return x * x * (3 - 2 * x); })();
  const pos = Math.min(n, seg + ease);
  const active = Math.min(n, Math.max(0, Math.round(pos)));

  // The stretch starts light and darkens across the last three slides, so
  // the final phase already sits on the page's own black and the section
  // simply continues into the dark below - no second ink edge.
  const dark = Math.min(1, Math.max(0, (pos - (n - 3)) / 2.4));
  const mixc = (a: number, b: number) => Math.round(a + (b - a) * dark);
  const bg = `rgb(${mixc(240, 23)}, ${mixc(240, 23)}, ${mixc(248, 23)})`;
  const fg = `rgb(${mixc(23, 240)}, ${mixc(23, 240)}, ${mixc(23, 248)})`;

  return (
    <section
      ref={root}
      className={`method-stage${reduced ? ' is-static' : ''}`}
      style={{ '--slides': n + 1, '--stage-bg': bg, '--stage-fg': fg, '--stage-dark': dark } as React.CSSProperties}
      id="method"
    >
      <div className="ink ink-top" aria-hidden="true" />
      <div className="method-pin">
        <ParticleCloud tone="dark" count={700} className="method-cloud" />
        <div className="method-track" style={{ transform: `translate3d(${-pos * 100}vw, 0, 0)` }}>
          <div className="method-slide method-intro" style={{ opacity: 1 - Math.min(1, pos * 1.6) }}>
            <span className="eyebrow">[ {eyebrow} ]</span>
            <h2 className="display-l">{heading}</h2>
            <p>{intro}</p>
            <p className="method-list">
              {phases.map((ph, i) => (
                <span key={ph.title} className={i === n - 1 ? 'accent' : undefined}>
                  {ph.title}
                </span>
              ))}
            </p>
          </div>

          {phases.map((ph, i) => {
            // -1 … 0 … 1: how far this slide is from centre, in slides.
            const d = pos - (i + 1);
            const reveal = Math.min(1, Math.max(0, 1 - Math.abs(d)));
            return (
              <div className="method-slide" key={ph.title} aria-hidden={Math.abs(d) > 0.5}>
                <span className="method-ghost" style={{ transform: `translate3d(${d * -18}vw, 0, 0)` }}>
                  0{i + 1}
                </span>
                <div className="method-body">
                  <h3 className="display-l method-title" style={{ opacity: 0.35 + 0.65 * reveal }}>
                    {Array.from(ph.title).map((ch, k) => (
                      <span
                        key={k}
                        style={{
                          opacity: reveal >= (k + 1) / (ph.title.length + 1) ? 1 : 0.18,
                          transition: 'opacity 160ms ease',
                        }}
                      >
                        {ch}
                      </span>
                    ))}
                    <i className="accent">.</i>
                  </h3>
                  <p className="method-line">{ph.line}</p>
                  <ul>
                    {ph.points.map((pt) => (
                      <li key={pt}>{pt}</li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>

        <div className="method-rail" aria-hidden="true">
          {phases.map((ph, i) => {
            const done = active > i + 1;
            const on = active === i + 1;
            return (
              <span key={ph.title} className={`rail-step${done ? ' done' : ''}${on ? ' on' : ''}`}>
                <b>{done || on ? ph.title[0] : '·'}</b>
                <em>{on ? ph.title : ''}</em>
              </span>
            );
          })}
        </div>
      </div>
    </section>
  );
}
