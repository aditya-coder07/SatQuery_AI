'use client';

import Lenis from 'lenis';
import { useEffect } from 'react';

/**
 * Inertial smooth scrolling (Lenis), the same driver the reference uses.
 * It moves the real scroll position, so every scroll listener on the page -
 * the ink flood, the pinned method, the progress tick - keeps working
 * unchanged; the difference is that wheel steps are eased over ~1 s
 * instead of jumping. Disabled under reduced motion.
 */
export default function SmoothScroll() {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const lenis = new Lenis({
      lerp: 0.085,
      wheelMultiplier: 0.9,
      smoothWheel: true,
      syncTouch: false,
    });
    let raf = 0;
    const loop = (time: number) => {
      lenis.raf(time);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
    };
  }, []);
  return null;
}
