'use client';

import { useEffect, useState } from 'react';

/**
 * The reading position, on the right edge: a short vertical line with a
 * tick that travels down it, and the percentage in mono. Blended with
 * `mix-blend-mode: difference` so it stays legible over both the dark and
 * the light stretches of the page.
 */
export default function ScrollProgress() {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setP(max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0);
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
    <div className="scroll-progress" aria-hidden="true">
      <span className="scroll-progress-line">
        <span className="scroll-progress-tick" style={{ top: `${p * 100}%` }} />
      </span>
      <span className="scroll-progress-pct">{String(Math.round(p * 100)).padStart(3, '0')}%</span>
    </div>
  );
}
