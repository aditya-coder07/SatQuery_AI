'use client';

import { useEffect, useRef, type ReactNode } from 'react';

/**
 * The ink flood between two halves of the page.
 *
 * `before` is the dark part, `after` starts with the light stretch. Over
 * the last viewport of `before` sits a full-height layer of ink colour
 * that scales up from the bottom as `after` approaches: it is scrubbed to
 * scroll - a small blot the moment the light section's top enters the
 * viewport (2 % of the travel gives 18 % of the height), then a linear
 * rise until that top has passed 25 % above the viewport. The layer sits
 * inside an oversized wrapper carrying the turbulence filter, so its edge
 * is displaced into blots and the displacement never reaches the wrapper's
 * own edges. Same construction as the reference.
 *
 * The filter sits on a static element and only its unfiltered parent
 * moves, so the displaced edge is rasterised once and then composited -
 * re-running turbulence over a viewport every frame is what lags. The
 * transform is written straight to the element in a frame callback;
 * nothing re-renders on scroll.
 */
export default function InkReveal({
  before,
  after,
  color = '#f0f0f8',
}: {
  before: ReactNode;
  after: ReactNode;
  color?: string;
}) {
  const fill = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const f = fill.current;
    const t = trigger.current;
    if (!f || !t) return;
    let raf = 0;
    const update = () => {
      raf = 0;
      const vh = window.innerHeight;
      const top = t.getBoundingClientRect().top;
      // progress 0 when the trigger's top is at the viewport bottom,
      // 1 when it is 25 % of a viewport above the top.
      const p = Math.min(1, Math.max(0, (vh - top) / (vh * 1.25)));
      const s = p < 0.02 ? (p / 0.02) * 0.18 : 0.18 + ((p - 0.02) / 0.98) * 0.82;
      f.style.transform = `translate3d(0, ${((1 - s) * 100).toFixed(2)}%, 0)`;
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

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ position: 'relative' }}>
        {before}
        <div className="ink-well" aria-hidden="true">
          <div ref={fill} className="ink-mover">
            <div className="ink-filter">
              <div className="ink-fill" style={{ background: color }} />
            </div>
          </div>
        </div>
      </div>
      <div ref={trigger} style={{ position: 'relative' }}>
        {after}
      </div>
    </div>
  );
}
