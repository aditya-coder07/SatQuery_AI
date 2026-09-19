'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';

/**
 * A thin progress line under the top edge while a route change is in
 * flight. Starts the moment an internal link is clicked (before the router
 * has anything to show), creeps toward 90 %, and completes when the
 * pathname changes. Covers the second or two between click and the next
 * page's loading state, which otherwise reads as the click not working.
 */
export default function NavProgress() {
  const pathname = usePathname();
  const [width, setWidth] = useState(0);
  const [on, setOn] = useState(false);
  const timer = useRef(0);
  const started = useRef<string | null>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.('a[href]') as HTMLAnchorElement | null;
      if (!a || a.target === '_blank' || a.hasAttribute('download')) return;
      const url = new URL(a.href, window.location.href);
      if (url.origin !== window.location.origin) return;
      if (url.pathname === window.location.pathname) return; // same page (hash or focus)
      started.current = url.pathname;
      setOn(true);
      setWidth(12);
      window.clearInterval(timer.current);
      timer.current = window.setInterval(() => {
        setWidth((w) => (w < 90 ? w + (90 - w) * 0.12 : w));
      }, 120);
    };
    // Capture phase: next/link calls preventDefault in React's own root
    // listener, which runs before a bubbling document listener would.
    document.addEventListener('click', onClick, true);
    return () => {
      document.removeEventListener('click', onClick, true);
      window.clearInterval(timer.current);
    };
  }, []);

  // Pathname changed: finish and fade.
  useEffect(() => {
    if (!started.current) return;
    started.current = null;
    window.clearInterval(timer.current);
    setWidth(100);
    const id = window.setTimeout(() => {
      setOn(false);
      setWidth(0);
    }, 320);
    return () => window.clearTimeout(id);
  }, [pathname]);

  return (
    <div className={`nav-progress${on ? ' is-on' : ''}`} aria-hidden="true">
      <span style={{ width: `${width}%` }} />
    </div>
  );
}
