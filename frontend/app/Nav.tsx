'use client';

/**
 * Site navigation.
 *
 * Four routes existed - the query page, /models, /benchmarks and
 * /runs/[runId] - and nothing linked to any of them. The model registry and
 * the benchmark page were reachable only by typing the URL, which for a demo
 * means a judge cannot get to the evidence pages unless someone drives.
 *
 * `usePathname` marks the current route so the header says where you are;
 * `prefetch` is left at the Next default. /runs/[runId] is deliberately
 * absent: it needs an id, and the query page links to it once a run exists.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const ROUTES = [
  { href: '/', label: 'Query' },
  { href: '/models', label: 'Models' },
  { href: '/benchmarks', label: 'Benchmarks' },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="nav" aria-label="Main">
      <Link className="nav-brand" href="/">
        SatQuery&nbsp;AI
      </Link>
      <ul>
        {ROUTES.map((route) => {
          const active =
            route.href === '/' ? pathname === '/' : pathname?.startsWith(route.href);
          return (
            <li key={route.href}>
              <Link
                href={route.href}
                className={active ? 'active' : undefined}
                aria-current={active ? 'page' : undefined}
              >
                {route.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
