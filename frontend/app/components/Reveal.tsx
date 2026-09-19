'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';

/**
 * Scroll-in reveal for the home page.
 *
 * The content is in the server HTML at full opacity; the class that hides
 * it is only added once JavaScript is running and the element is measured
 * off-screen, so a slow or blocked script still shows a complete page. Once
 * revealed it stays revealed - sections do not blink out when scrolled past.
 */
export default function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<'ssr' | 'hidden' | 'shown'>('ssr');

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.92) {
      setState('shown');
      return;
    }
    setState('hidden');
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setState('shown');
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.05 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`reveal is-${state}${className ? ` ${className}` : ''}`}
      style={{ transitionDelay: state === 'shown' ? `${delay}ms` : '0ms' }}
    >
      {children}
    </div>
  );
}
