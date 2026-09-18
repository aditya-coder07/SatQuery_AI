'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

/**
 * The last few runs, from `GET /runs`.
 *
 * Every run is persisted with its answer and can be reopened at
 * `/runs/{id}`, but until now the only way to that page was the permalink on
 * the answer you had just watched. Anyone coming back later - or a second
 * person at the same deployment - had no way in. This is that way in: the
 * newest runs, what was asked, what came back, and a link. Refreshed after
 * each completed run so the one you just made is at the top.
 */

type RunRow = {
  run_id: string;
  created_utc: string;
  query: string;
  status: string;
  task: string | null;
  answer: string | null;
  confidence: number | null;
  band: string | null;
  abstained: number | boolean;
  error: string | null;
};

const LIMIT = 6;

function when(iso: string): string {
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return iso;
  const mins = Math.round((Date.now() - t.getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  return t.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function RecentRuns({ api, refreshKey }: { api: string; refreshKey: string }) {
  const [rows, setRows] = useState<RunRow[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`${api}/runs?limit=${LIMIT}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => {
        if (!cancelled) setRows((data.runs ?? []).slice(0, LIMIT));
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
    // refreshKey is the id of the run that just completed: a new value means
    // the list is stale.
  }, [api, refreshKey]);

  if (failed || (rows && rows.length === 0)) return null;

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 8v5l3 2" />
            <path d="M3.05 11a9 9 0 1 0 .5-4" />
            <path d="M3 3v5h5" />
          </svg>
          Recent runs
        </span>
        <span className="spacer" />
        <span className="meta">GET /runs · newest first</span>
      </div>

      {rows === null ? (
        <p className="answer empty">Loading…</p>
      ) : (
        <ul className="recent">
          {rows.map((row) => {
            const abstained = Boolean(row.abstained);
            const summary = row.error
              ? `failed · ${row.error}`
              : abstained
                ? 'abstained'
                : row.answer || (row.status === 'complete' ? '' : row.status);
            return (
              <li key={row.run_id} className="recent-row">
                <div className="recent-main">
                  <Link href={`/runs/${row.run_id}`} className="recent-query" title={row.query}>
                    {row.query}
                  </Link>
                  <div className="recent-answer" title={summary}>
                    {summary || '—'}
                  </div>
                </div>
                <div className="recent-side">
                  <span className="recent-task">{row.task || '—'}</span>
                  <span className="recent-conf">
                    {abstained || row.confidence == null
                      ? 'n/a'
                      : `${row.confidence.toFixed(2)} ${row.band ?? ''}`.trim()}
                  </span>
                  <span className="recent-when">{when(row.created_utc)}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
