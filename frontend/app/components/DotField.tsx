'use client';

import { useEffect, useRef } from 'react';

/**
 * The site background: a grid of 1 px dots on a 22 px pitch at 9 % white,
 * fixed behind every route. A click anywhere that is not a control sends a
 * ripple out through the grid - dots near the wavefront are pushed outward
 * and brighten, then settle - for 2.2 s. Up to six ripples at once. Nothing
 * runs while there is no ripple, so the idle cost is a static canvas.
 */
export default function DotField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const PITCH = 22;
    const LIFE = 2.2;
    type Ripple = { x: number; y: number; start: number };
    let ripples: Ripple[] = [];
    let w = 0, h = 0, dpr = window.devicePixelRatio || 1;
    let raf = 0;
    let running = false;

    const drawStill = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.09)';
      for (let y = 11; y < h; y += PITCH) {
        for (let x = 11; x < w; x += PITCH) {
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    const resize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawStill();
    };
    resize();

    const tick = () => {
      const now = performance.now() / 1000;
      ripples = ripples.filter((r) => now - r.start < LIFE);
      if (ripples.length === 0) {
        drawStill();
        running = false;
        return;
      }
      raf = requestAnimationFrame(tick);
      const waves = ripples.map((r) => {
        const age = now - r.start;
        const k = 1 - age / LIFE;
        return { x: r.x, y: r.y, radius: 380 * age, amplitude: 14 * k, brightness: 0.28 * k };
      });
      ctx.clearRect(0, 0, w, h);
      for (let y = 11; y < h; y += PITCH) {
        for (let x = 11; x < w; x += PITCH) {
          let px = x, py = y, alpha = 0.09;
          for (const wv of waves) {
            const dx = x - wv.x, dy = y - wv.y;
            const dist = Math.hypot(dx, dy);
            const off = dist - wv.radius;
            if (Math.abs(off) >= 80) continue;
            const c = Math.cos((off / 80) * Math.PI * 0.5);
            const inv = dist > 0.001 ? 1 / dist : 0;
            const push = c * wv.amplitude;
            px += dx * inv * push;
            py += dy * inv * push;
            alpha += wv.brightness * Math.max(0, c);
          }
          ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(1, alpha)})`;
          ctx.beginPath();
          ctx.arc(px, py, 1, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    const onClick = (e: MouseEvent) => {
      const target = e.target as Element | null;
      if (target?.closest?.('a, button, [role="button"], input, textarea, select, [data-cursor-hover]')) return;
      if (ripples.length >= 6) ripples.shift();
      ripples.push({ x: e.clientX, y: e.clientY, start: performance.now() / 1000 });
      if (!running) {
        running = true;
        raf = requestAnimationFrame(tick);
      }
    };
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) window.addEventListener('click', onClick);
    window.addEventListener('resize', resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('click', onClick);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <div aria-hidden="true" className="dotfield">
      <canvas ref={ref} />
    </div>
  );
}
