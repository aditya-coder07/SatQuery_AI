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
 * three points. A dot rail at the bottom shows where you are. The stretch
 * starts light and darkens across the last three slides, so the final
 * phase already sits on the page's own black.
 *
 * Nothing re-renders on scroll: one listener writes a handful of CSS
 * custom properties and a data attribute onto the section in a frame
 * callback, and every transform, opacity and colour is a CSS expression of
 * those. Reduced motion lays the slides out vertically with no pinning.
 */

export type Phase = { title: string; line: string; points: string[] };

function smooth(x: number): number {
  return x * x * (3 - 2 * x);
}

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
  const [reduced, setReduced] = useState(false);
  const n = phases.length;

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    if (mq.matches) return;
    const el = root.current;
    if (!el) return;
    let raf = 0;
    let lastActive = -1;
    const slides = Array.from(el.querySelectorAll<HTMLElement>('.method-slide:not(.method-intro)'));
    const update = () => {
      raf = 0;
      const rect = el.getBoundingClientRect();
      const travel = rect.height - window.innerHeight;
      const p = travel > 0 ? Math.min(1, Math.max(0, -rect.top / travel)) : 0;
      // Slide index as a real number: 0 = intro, 1..n = phases. Each slide
      // dwells for the outer quarters of its segment and moves through
      // the middle half: hold, slide, hold.
      const raw = p * n;
      const seg = Math.floor(raw);
      const f = raw - seg;
      const ease = f < 0.25 ? 0 : f > 0.75 ? 1 : smooth((f - 0.25) / 0.5);
      const pos = Math.min(n, seg + ease);
      const dark = Math.min(1, Math.max(0, (pos - (n - 3)) / 2.4));
      const st = el.style;
      st.setProperty('--pos', pos.toFixed(4));
      st.setProperty('--dark', dark.toFixed(4));
      slides.forEach((slide, i) => {
        const d = pos - (i + 1);
        slide.style.setProperty('--d', d.toFixed(4));
        slide.style.setProperty('--r', Math.min(1, Math.max(0, 1 - Math.abs(d))).toFixed(4));
      });
      const active = Math.min(n, Math.max(0, Math.round(pos)));
      if (active !== lastActive) {
        lastActive = active;
        el.dataset.active = String(active);
      }
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
  }, [n]);

  return (
    <section
      ref={root}
      className={`method-stage${reduced ? ' is-static' : ''}`}
      style={{ '--slides': n + 1, '--n': n } as React.CSSProperties}
      data-active="0"
      id="method"
    >
      <div className="method-pin">
        <ParticleCloud tone="dark" count={600} className="method-cloud" />
        <div className="method-track">
          <div className="method-slide method-intro">
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

          {phases.map((ph, i) => (
            <div className="method-slide" key={ph.title} style={{ '--i': i } as React.CSSProperties}>
              <span className="method-ghost">0{i + 1}</span>
              <div className="method-body">
                <h3 className="display-l method-title">
                  {Array.from(ph.title).map((ch, k) => (
                    <span key={k} style={{ '--k': ((k + 1) / (ph.title.length + 1)).toFixed(3) } as React.CSSProperties}>
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
          ))}
        </div>

        <div className="method-rail" aria-hidden="true">
          {phases.map((ph, i) => (
            <span key={ph.title} className="rail-step" style={{ '--i': i + 1 } as React.CSSProperties}>
              <b data-initial={ph.title[0]}>·</b>
              <em>{ph.title}</em>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
