'use client';

import { useCallback, useRef, useState } from 'react';

import Comparator from './Comparator';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type TraceEvent = { name: string; data: any };

type Confidence = {
  final: number;
  band: 'HIGH' | 'MEDIUM' | 'LOW';
  components: { model: number; agreement: number; input_quality: number };
};

type Check = { name: string; status: 'PASS' | 'WARN' | 'FAIL'; message: string };

/**
 * Parses an SSE byte stream into discrete events.
 *
 * EventSource cannot issue a POST with a file body, so the run is started with
 * fetch() and the response stream is decoded by hand. Events are separated by
 * a blank line; a partial event left in the buffer is carried to the next read.
 */
async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<TraceEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split: number;
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = 'message';
      const dataLines: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event: ')) name = line.slice(7);
        else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
      }
      if (dataLines.length === 0) continue;
      try {
        yield { name, data: JSON.parse(dataLines.join('\n')) };
      } catch {
        yield { name, data: dataLines.join('\n') };
      }
    }
  }
}

export default function Page() {
  const [query, setQuery] = useState('Describe this image.');
  const [files, setFiles] = useState<FileList | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [answer, setAnswer] = useState<string>('');
  const [task, setTask] = useState<string>('');
  const [confidence, setConfidence] = useState<Confidence | null>(null);
  const [checks, setChecks] = useState<Check[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>('');
  const [runId, setRunId] = useState<string>('');
  const [roles, setRoles] = useState<string[]>([]);
  const traceRef = useRef<HTMLDivElement>(null);

  const submit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!files || files.length === 0) {
        setError('Select one or two images first.');
        return;
      }

      setRunning(true);
      setError('');
      setEvents([]);
      setAnswer('');
      setTask('');
      setConfidence(null);
      setChecks([]);
      setRunId('');
      setRoles([]);

      const form = new FormData();
      form.append('query', query);
      Array.from(files).forEach((f) => form.append('images', f));

      try {
        const res = await fetch(`${API}/runs/stream`, { method: 'POST', body: form });
        if (!res.ok || !res.body) {
          throw new Error(`server returned ${res.status}`);
        }

        for await (const event of parseSSE(res.body)) {
          setEvents((prev) => [...prev, event]);

          if (event.name === 'run_started') {
            setRunId(event.data.run_id ?? '');
          } else if (event.name === 'ingest') {
            setChecks(event.data.checks ?? []);
            setRoles((event.data.images ?? []).map((i: any) => i.role));
          } else if (event.name === 'routing') {
            setTask(event.data.selected_task ?? '');
          } else if (event.name === 'confidence') {
            setConfidence(event.data);
          } else if (event.name === 'complete') {
            setAnswer(event.data.answer ?? '');
            if (event.data.abstained && event.data.abstain_reason) {
              setError(event.data.abstain_reason);
            }
          } else if (event.name === 'error') {
            setError(event.data.message ?? 'run failed');
          }

          // Keep the newest trace line in view as the run progresses.
          requestAnimationFrame(() => {
            if (traceRef.current) {
              traceRef.current.scrollTop = traceRef.current.scrollHeight;
            }
          });
        }
      } catch (err: any) {
        setError(err?.message ?? String(err));
      } finally {
        setRunning(false);
      }
    },
    [files, query]
  );

  return (
    <main className="wrap">
      <header>
        <h1>SatQuery AI</h1>
        <p>Interactive vision-language assistant for remote-sensing imagery</p>
      </header>

      <form className="panel" onSubmit={submit} style={{ marginBottom: 16 }}>
        <h2>Query</h2>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask something about the imagery"
        />
        <input
          type="file"
          accept=".tif,.tiff,.img,.jp2"
          multiple
          onChange={(e) => setFiles(e.target.files)}
        />
        <button type="submit" disabled={running}>
          {running ? 'Running…' : 'Run'}
        </button>
        {error && (
          <p className="error" style={{ marginBottom: 0 }}>
            {error}
          </p>
        )}
      </form>

      <div className="grid">
        <section className="panel">
          <h2>Answer</h2>
          {task && <span className="pill">{task}</span>}
          <p className="answer">{answer || (running ? 'Working…' : '—')}</p>

          {checks.length > 0 && (
            <>
              <h2 style={{ marginTop: 20 }}>Input checks</h2>
              <ul className="checks">
                {checks.map((c, i) => (
                  <li key={i} className={`check-${c.status}`}>
                    [{c.status}] {c.message}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        <section className="panel">
          <h2>Confidence</h2>
          {confidence ? (
            <>
              <div className={`confidence-value band-${confidence.band}`}>
                {(confidence.final * 100).toFixed(1)}%
                <span style={{ fontSize: 14, marginLeft: 8 }}>{confidence.band}</span>
              </div>
              <div className="components">
                <div className="component">
                  <div className="label">Model</div>
                  <div className="value">{confidence.components.model.toFixed(2)}</div>
                </div>
                <div className="component">
                  <div className="label">Agreement</div>
                  <div className="value">{confidence.components.agreement.toFixed(2)}</div>
                </div>
                <div className="component">
                  <div className="label">Input</div>
                  <div className="value">
                    {confidence.components.input_quality.toFixed(2)}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <p style={{ color: 'var(--muted)' }}>—</p>
          )}
        </section>
      </div>

      {runId && roles.length === 2 && (
        <div style={{ marginTop: 16 }}>
          <Comparator api={API} runId={runId} roles={roles} />
        </div>
      )}

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Live trace ({events.length} events)</h2>
        <div className="trace" ref={traceRef}>
          {events.length === 0 && <p style={{ color: 'var(--muted)' }}>—</p>}
          {events.map((e, i) => (
            <div className="event" key={i}>
              <span className="name">{e.name}</span>
              <pre>{JSON.stringify(e.data, null, 2).slice(0, 1200)}</pre>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
