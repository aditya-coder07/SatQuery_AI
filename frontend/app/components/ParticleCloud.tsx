'use client';

import { useEffect, useRef } from 'react';

/**
 * A drifting cloud of particles on a 2D canvas: the globe's dots after they
 * have let go. Each dot spawns near the cloud's centre, drifts outward on
 * a slow, noisy path, fades, and respawns. `tone` picks dark dots for the
 * light stretch and light dots for the dark page. Sized to its container,
 * paused when off-screen, and a still frame under reduced motion.
 */
export default function ParticleCloud({
  tone = 'light',
  count = 900,
  className,
}: {
  tone?: 'light' | 'dark';
  count?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const rgb = tone === 'light' ? '240, 240, 248' : '23, 23, 23';
    type P = { x: number; y: number; vx: number; vy: number; life: number; max: number; r: number; seed: number };
    let w = 0, h = 0, dpr = 1;
    let raf = 0;
    let running = false;
    let visible = true;
    let t = 0;
    const ps: P[] = [];

    const spawn = (p: P, initial = false) => {
      // Spawn in a soft ellipse around the centre, biased upward.
      const a = Math.random() * Math.PI * 2;
      const rr = Math.pow(Math.random(), 0.6);
      p.x = w * 0.5 + Math.cos(a) * rr * w * 0.22;
      p.y = h * 0.55 + Math.sin(a) * rr * h * 0.28;
      const sp = 6 + Math.random() * 16;
      p.vx = Math.cos(a) * sp;
      p.vy = Math.sin(a) * sp - 6;
      p.max = 9 + Math.random() * 9;
      p.life = initial ? Math.random() * p.max : 0;
      p.r = 0.6 + Math.random() * 1.1;
      p.seed = Math.random() * 1000;
    };
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (ps.length === 0) {
        for (let i = 0; i < count; i++) {
          const p = { x: 0, y: 0, vx: 0, vy: 0, life: 0, max: 1, r: 1, seed: 0 };
          spawn(p, true);
          ps.push(p);
        }
      }
    };
    const draw = (dt: number) => {
      t += dt;
      ctx.clearRect(0, 0, w, h);
      for (const p of ps) {
        p.life += dt;
        if (p.life > p.max) spawn(p);
        const k = p.life / p.max;
        // Curl-ish wander from two sines; cheap and organic enough.
        const nx = Math.sin(t * 0.35 + p.seed) * 9;
        const ny = Math.cos(t * 0.27 + p.seed * 1.7) * 9;
        p.x += (p.vx + nx) * dt;
        p.y += (p.vy + ny) * dt;
        const alpha = Math.sin(Math.PI * k) * 0.55;
        ctx.fillStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    let last = performance.now();
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      draw(dt);
    };
    const sync = () => {
      if (visible && !reduced) {
        if (!running) { running = true; last = performance.now(); raf = requestAnimationFrame(frame); }
      } else {
        running = false;
        cancelAnimationFrame(raf);
      }
    };
    resize();
    if (reduced) {
      // A settled still: advance the field a little so it is not a ring.
      for (let i = 0; i < 60; i++) draw(0.05);
    }
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver((es) => { for (const e of es) visible = e.isIntersecting; sync(); });
    io.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, [tone, count]);

  return <canvas ref={ref} className={`pcloud${className ? ` ${className}` : ''}`} aria-hidden="true" />;
}
