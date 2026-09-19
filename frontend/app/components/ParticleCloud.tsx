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
  shape = 'cloud',
  tilt = 0,
  className,
}: {
  tone?: 'light' | 'dark';
  count?: number;
  /** `disc`: a thin elliptical ring of dots that slowly orbits, dense at the rim.
   *  `wave`: a band of dots streaming left to right on a slow sine swell. */
  shape?: 'cloud' | 'disc' | 'wave';
  /** Degrees the canvas is rotated by CSS (so pointer positions can be mapped back). */
  tilt?: number;
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
    type P = { x: number; y: number; vx: number; vy: number; life: number; max: number; r: number; seed: number; a: number; rad: number };
    let w = 0, h = 0, dpr = 1;
    let raf = 0;
    let running = false;
    let visible = true;
    let t = 0;
    const ps: P[] = [];
    // Pointer in canvas space (the canvas may be rotated by CSS; undo it).
    const ptr = { x: -1e4, y: -1e4, on: false };
    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const dx = e.clientX - cx, dy = e.clientY - cy;
      const a = (-tilt * Math.PI) / 180;
      const rx = dx * Math.cos(a) - dy * Math.sin(a);
      const ry = dx * Math.sin(a) + dy * Math.cos(a);
      ptr.x = w / 2 + rx;
      ptr.y = h / 2 + ry;
      ptr.on = true;
    };
    const onLeave = () => { ptr.on = false; ptr.x = -1e4; ptr.y = -1e4; };

    const spawn = (p: P, initial = false) => {
      if (shape === 'wave') {
        // One dense front that sweeps in from the left and out to the right,
        // then is gone. Each dot sits some distance behind the front (the
        // trail is thick near the front and thins out) in a lane across
        // the band's height, and drifts a little slower than the front.
        p.x = -Math.pow(Math.random(), 1.6) * w * 0.75; // offset behind the front
        p.rad = Math.random(); // lane 0..1
        p.vx = -(4 + Math.random() * 30); // slips back relative to the front
        p.vy = 0;
        p.max = 1e9;
        p.life = 0;
        p.r = 0.7 + Math.random() * 1.3;
        p.seed = Math.random() * 1000;
        p.a = Math.random() * Math.PI * 2;
        return;
      }
      if (shape === 'disc') {
        // Ring: radius peaks near the rim with a soft inner fill; each dot
        // keeps its own angle and orbits slowly.
        p.a = Math.random() * Math.PI * 2;
        const u = Math.random();
        p.rad = u < 0.65 ? 0.78 + Math.random() * 0.24 : 0.3 + Math.random() * 0.55;
        p.max = 12 + Math.random() * 10;
        p.life = initial ? Math.random() * p.max : 0;
        p.r = 0.5 + Math.random() * 1.0;
        p.seed = Math.random() * 1000;
        p.vx = (Math.random() - 0.5) * 4;
        p.vy = (Math.random() - 0.5) * 4;
        return;
      }
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
          const p = { x: 0, y: 0, vx: 0, vy: 0, life: 0, max: 1, r: 1, seed: 0, a: 0, rad: 0 };
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
        if (shape === 'wave') {
          // The front crosses the width in ~3.2 s and keeps going until the
          // whole trail has left the right edge.
          const front = -w * 0.05 + t * (w / 3.2);
          p.x += p.vx * dt;
          const x = front + p.x;
          if (x < -20 || x > w + 20) continue;
          const phase = (x / w) * Math.PI * 2;
          const swell = Math.sin(phase * 1.3 - t * 1.2) * h * 0.14 + Math.sin(phase * 2.8 + p.seed) * h * 0.04;
          const lane = (p.rad - 0.5) * h * 0.28;
          const flutter = Math.sin(t * 1.6 + p.seed) * 4;
          const y = h * 0.5 + swell + lane + flutter;
          // Brightest just behind the front, fading down the trail.
          const behind = Math.min(1, -p.x / (w * 0.75));
          const alpha = (0.65 - 0.5 * behind) * (0.7 + 0.3 * Math.sin(p.seed + t * 2));
          ctx.fillStyle = `rgba(${rgb}, ${Math.max(0, alpha).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(x, y, p.r, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        if (shape === 'disc') {
          p.a += dt * 0.06;
          const wob = Math.sin(t * 0.4 + p.seed) * 0.02;
          const rx = w * 0.46, ry = h * 0.46;
          const x = w * 0.5 + Math.cos(p.a) * (p.rad + wob) * rx + p.vx * k * 6;
          const y = h * 0.5 + Math.sin(p.a) * (p.rad + wob) * ry + p.vy * k * 6;
          let alpha = Math.sin(Math.PI * k) * (p.rad > 0.8 ? 0.75 : 0.4);
          let px = x, py = y, pr = p.r;
          // Near the pointer the dots are pushed outward and brighten, so
          // the disc parts around the cursor like a hand through smoke.
          if (ptr.on) {
            const ddx = x - ptr.x, ddy = y - ptr.y;
            const d2 = ddx * ddx + ddy * ddy;
            const R = Math.min(w, h) * 0.22;
            if (d2 < R * R) {
              const d = Math.sqrt(d2) || 1;
              const f = 1 - d / R;
              const push = f * f * R * 0.55;
              px += (ddx / d) * push;
              py += (ddy / d) * push;
              alpha = Math.min(1, alpha + f * 0.6);
              pr = p.r + f * 1.2;
            }
          }
          ctx.fillStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(px, py, pr, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
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
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      draw(dt);
      // A finished sweep (front + trail past the right edge) stops the loop.
      if (shape === 'wave' && t * (w / 3.2) > w * 1.85) { running = false; ctx.clearRect(0, 0, w, h); return; }
      raf = requestAnimationFrame(frame);
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
    if (shape === 'disc' && !reduced) {
      window.addEventListener('pointermove', onMove, { passive: true });
      window.addEventListener('pointerleave', onLeave);
      document.addEventListener('mouseleave', onLeave);
    }
    const io = new IntersectionObserver((es) => {
      for (const e of es) {
        visible = e.isIntersecting;
        // A wave replays from the left each time the section is entered.
        if (visible && shape === 'wave') { t = 0; for (const p of ps) spawn(p); }
      }
      sync();
    });
    io.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
      document.removeEventListener('mouseleave', onLeave);
    };
  }, [tone, count, shape, tilt]);

  return <canvas ref={ref} className={`pcloud${className ? ` ${className}` : ''}`} aria-hidden="true" />;
}
