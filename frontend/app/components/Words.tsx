'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';

/**
 * Word-by-word blur reveal for a headline. Each word starts blurred and
 * transparent and resolves in sequence once the element is on screen (or
 * immediately with `now`). The words are real text in the markup at full
 * opacity until JavaScript runs, so nothing is hidden from a slow client.
 * `accent` marks words to set in the accent colour.
 */
export default function Words({
  text,
  accent = [],
  now = false,
  stagger = 90,
  delay = 0,
}: {
  text: string;
  accent?: string[];
  now?: boolean;
  stagger?: number;
  delay?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [state, setState] = useState<'ssr' | 'wait' | 'go'>('ssr');

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (now) {
      setState('wait');
      const id = window.setTimeout(() => setState('go'), 30);
      return () => window.clearTimeout(id);
    }
    const el = ref.current;
    if (!el) return;
    setState('wait');
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) {
          setState('go');
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [now]);

  const words = text.split(' ');
  const out: ReactNode[] = [];
  words.forEach((w, i) => {
    const clean = w.replace(/[^\p{L}\p{N}]/gu, '');
    const isAccent = accent.includes(clean);
    out.push(
      <span
        key={i}
        className={`word${isAccent ? ' accent' : ''} is-${state}`}
        style={{ transitionDelay: `${delay + i * stagger}ms` }}
      >
        {w}
      </span>,
    );
    if (i < words.length - 1) out.push(' ');
  });
  return <span ref={ref} className="words">{out}</span>;
}
