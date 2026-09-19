/**
 * Where the API lives.
 *
 * `NEXT_PUBLIC_API_URL` is inlined at build time, which is right for a
 * deployment whose backend has a fixed address and wrong for one whose
 * backend is a GPU session with a new public URL each time it starts (a
 * Kaggle notebook behind a tunnel, docs/deploy-free.md). So the build-time
 * value is a default, and the browser may override it:
 *
 *   1. `?api=https://host` on any page - saved, then used;
 *   2. the saved value (localStorage `satquery.api`);
 *   3. the build-time default.
 *
 * Module-level `const API = apiBase()` in a client component evaluates in
 * the browser, so the override applies without prop drilling; on the server
 * render it is the default, and nothing fetches during SSR.
 */

const BUILD_DEFAULT = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
export const API_STORAGE_KEY = 'satquery.api';

function normalise(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

export function apiBase(): string {
  if (typeof window === 'undefined') return BUILD_DEFAULT;
  try {
    const fromQuery = new URLSearchParams(window.location.search).get('api');
    if (fromQuery) {
      const value = normalise(fromQuery);
      window.localStorage.setItem(API_STORAGE_KEY, value);
      return value;
    }
    const saved = window.localStorage.getItem(API_STORAGE_KEY);
    if (saved) return normalise(saved);
  } catch {
    // Private mode or blocked storage: the default still works.
  }
  return BUILD_DEFAULT;
}

/** Save (or clear, with null) the override and reload so every module re-reads it. */
export function setApiBase(url: string | null): void {
  try {
    if (url && normalise(url)) window.localStorage.setItem(API_STORAGE_KEY, normalise(url));
    else window.localStorage.removeItem(API_STORAGE_KEY);
  } catch {
    return;
  }
  const clean = new URL(window.location.href);
  clean.searchParams.delete('api');
  window.location.replace(clean.toString());
}

export function isOverridden(): boolean {
  try {
    return typeof window !== 'undefined' && window.localStorage.getItem(API_STORAGE_KEY) != null;
  } catch {
    return false;
  }
}
