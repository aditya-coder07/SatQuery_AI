'use client';

/**
 * Permalink for a stored run (plan task 1.6).
 *
 * The run store already keeps every completed trace, but nothing linked to
 * one - the map, the answer and the artifacts were reachable only in the tab
 * that happened to submit the query. This makes a run addressable, which is
 * also what lets the map viewer be checked against a run someone else made.
 */

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import MapView from '@/MapView';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function RunPage() {
  const params = useParams<{ runId: string }>();
  const runId = params?.runId;
  const [record, setRecord] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    fetch(`${API}/runs/${runId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => {
        // The store wraps the trace; a fresh POST returns it bare. Accept both
        // rather than assuming, and parse it if it came back as a string.
        let trace = d.trace ?? d;
        if (typeof trace === 'string') trace = JSON.parse(trace);
        setRecord(trace);
      })
      .catch((e) => setError(String(e)));
  }, [runId]);

  if (error) {
    return (
      <main className="page">
        <p className="load-error">
          Could not load run {runId}: {error}
        </p>
      </main>
    );
  }
  if (!record) {
    return (
      <main className="page">
        <p>Loading…</p>
      </main>
    );
  }

  const confidence = record.confidence;
  // An abstained run returned no answer, so a confidence *for that answer* is
  // not applicable (limitation L18). Rendering "HIGH 0.7937" directly above
  // "Abstained" reads as a contradiction even though the number is
  // arithmetically right - it abstained on input validation, not on low
  // confidence. Presentation only: the combiner is untouched.
  const abstained = Boolean(record.abstained);
  const failedChecks = (record.ingest?.checks ?? []).filter(
    (c: any) => c.status === 'FAIL',
  );

  return (
    <main className="page">
      <h1>Run {runId}</h1>
      <p className="note">{record.query}</p>

      <section className="card">
        <h3>{abstained ? 'Status: Abstained' : 'Answer'}</h3>
        {abstained ? (
          <>
            {failedChecks.length > 0 && (
              <ul className="checks">
                {failedChecks.map((c: any, i: number) => (
                  <li key={i} className="check-FAIL">
                    Input validation failed: <code>{c.name}</code> — {c.message}
                  </li>
                ))}
              </ul>
            )}
            <p className="caveat">
              Trigger: {record.abstain_trigger}. {record.abstain_resolving_input}
            </p>
          </>
        ) : (
          <p>{record.answer}</p>
        )}
      </section>

      {confidence && (
        <div className="stats">
          <div className="stat">
            <div className="label">confidence</div>
            <div className="value">
              {abstained ? '—' : Number(confidence.final).toFixed(4)}
            </div>
            <div className="sub">
              {abstained ? 'not applicable — the run abstained' : confidence.band}
            </div>
          </div>
          {/* Components stay visible even when abstained: they are the
              diagnosis of *why*, and input_quality is usually the low one. */}
          {Object.entries(confidence.components ?? {}).map(([k, v]) => (
            <div className="stat" key={k}>
              <div className="label">{k}</div>
              <div className="value">{Number(v).toFixed(3)}</div>
            </div>
          ))}
        </div>
      )}

      <h2>Map</h2>
      <MapView runId={String(runId)} />

      <p className="note" style={{ marginTop: 12 }}>
        <a href={`${API}/runs/${runId}/report.pdf`}>Download the PDF report</a>
      </p>
    </main>
  );
}
