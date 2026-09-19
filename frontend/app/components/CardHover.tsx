'use client';

import { useEffect } from 'react';

/**
 * One delegated listener for every card on the page: writes the pointer's
 * position into `--mx`/`--my` on the card under it, so the CSS can draw a
 * soft highlight that follows the cursor. No per-card handlers, nothing
 * re-rendered; the stretch and lift themselves are pure CSS transitions.
 */
const SELECTOR = '.cap-card, .bench-card, .carry-card, .talk-card, .official-row, .deployed-row, .liverun';

export default function CardHover() {
  useEffect(() => {
    if (window.matchMedia('(hover: none)').matches) return;
    const onMove = (e: PointerEvent) => {
      const card = (e.target as Element | null)?.closest?.(SELECTOR) as HTMLElement | null;
      if (!card) return;
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', `${(((e.clientX - r.left) / r.width) * 100).toFixed(1)}%`);
      card.style.setProperty('--my', `${(((e.clientY - r.top) / r.height) * 100).toFixed(1)}%`);
    };
    document.addEventListener('pointermove', onMove, { passive: true });
    return () => document.removeEventListener('pointermove', onMove);
  }, []);
  return null;
}
