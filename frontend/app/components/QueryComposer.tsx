'use client';

/**
 * The query composer.
 *
 * Structure follows the 21st.dev "Input Bar" pattern — one bounded composer,
 * a collapsing attachment row above the text, and a toolbar below with attach
 * on the left and send on the right — rebuilt on this project's own tokens
 * rather than installed, because the app has no shadcn/Radix layer and adding
 * one for a single control would be a bigger change than the control.
 *
 * It is deliberately the one bordered, filled object on the page. Everything
 * else is set with rules; this is the thing you are meant to touch, so it gets
 * the affordance and nothing else competes for it.
 *
 * What it fixes about the field it replaces:
 *
 * * The old "choose one or two scenes" was a mono label that looked like a
 *   caption. Attaching is now a button, and the composer is a real drop
 *   target — the previous dashed outline promised a drop that never existed.
 * * Files are listed with their sizes and can be removed one at a time.
 *   Picking again adds to the set instead of silently replacing it.
 * * A file over the API's per-file limit is refused here, with the number, in
 *   the composer — rather than after a 300 MB upload comes back a 413.
 * * "Scenes", not "files", because the API groups them: a Cartosat-2S MX
 *   product is four BAND*.tif plus a sidecar and counts as ONE scene.
 *   Counting files is what made the target sensor's own product format
 *   unuploadable (limitation L17), so the interface counts the way the
 *   backend does.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { QUERY_FIELD_ID } from '../lib/focusQuery';

export const ACCEPTED = '.tif,.tiff,.img,.jp2,.png,.jpg,.jpeg';

// Mirrors satquery/api/main.py: MAX_IMAGES, MAX_UPLOAD_FILES,
// MAX_UPLOAD_BYTES and MAX_TOTAL_UPLOAD_BYTES. Checking here is a courtesy —
// the server still enforces them, and it is the one that decides.
const MAX_FILES = 32;
const MAX_FILE_BYTES = 256 * 1024 * 1024;
const MAX_TOTAL_BYTES = 1024 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/** Two files are the same upload if the name, size and mtime all match. */
function sameFile(a: File, b: File): boolean {
  return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
}

export default function QueryComposer({
  query,
  onQueryChange,
  files,
  onFilesChange,
  running,
  onSubmit,
  error,
}: {
  query: string;
  onQueryChange: (value: string) => void;
  files: File[];
  onFilesChange: (files: File[]) => void;
  running: boolean;
  onSubmit: () => void;
  error: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [reject, setReject] = useState('');
  // Drag events fire for every child element, so a boolean flag flickers as
  // the pointer crosses the chips. A depth counter does not.
  const dragDepth = useRef(0);

  // Grow with the question, up to a ceiling, then scroll.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = '0';
    const next = Math.min(el.scrollHeight, 160);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > 160 ? 'auto' : 'hidden';
  }, [query]);

  const addFiles = useCallback(
    (incoming: FileList | File[] | null) => {
      if (!incoming) return;
      const next = [...files];
      const refused: string[] = [];

      for (const file of Array.from(incoming)) {
        if (next.some((existing) => sameFile(existing, file))) continue;
        if (file.size > MAX_FILE_BYTES) {
          refused.push(
            `${file.name} is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_FILE_BYTES)} per file`,
          );
          continue;
        }
        next.push(file);
      }

      if (next.length > MAX_FILES) {
        refused.push(`at most ${MAX_FILES} files per run`);
        next.length = MAX_FILES;
      }
      const total = next.reduce((sum, f) => sum + f.size, 0);
      if (total > MAX_TOTAL_BYTES) {
        refused.push(
          `${formatBytes(total)} in total — the limit is ${formatBytes(MAX_TOTAL_BYTES)}`,
        );
      }

      setReject(refused.join('. '));
      onFilesChange(next);
    },
    [files, onFilesChange],
  );

  const removeAt = useCallback(
    (index: number) => {
      setReject('');
      onFilesChange(files.filter((_, i) => i !== index));
    },
    [files, onFilesChange],
  );

  const submit = useCallback(() => {
    if (running) return;
    onSubmit();
  }, [running, onSubmit]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter runs, Shift+Enter writes a newline — the convention for a
    // composer. The Run button stays, so the shortcut is never the only way.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const total = files.reduce((sum, f) => sum + f.size, 0);
  const notice = reject || error;

  return (
    <form
      className={`composer${dragging ? ' dragging' : ''}`}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current -= 1;
        if (dragDepth.current <= 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      <div className="composer-head">
        <span className="label">Query</span>
        <span className="spacer" />
        <span className="meta">POST /runs/stream</span>
      </div>

      <div
        className="composer-box"
        onClick={(e) => {
          if (!(e.target as HTMLElement).closest('button, textarea')) {
            textareaRef.current?.focus();
          }
        }}
      >
        {/* The attachment rail collapses to nothing when there is nothing on
            it, so an empty composer is one clean field rather than a field
            with a permanently reserved gap above it. */}
        <div className={`composer-rail${files.length ? ' open' : ''}`}>
          <div className="composer-rail-inner">
            <ul className="attachments">
              {files.map((file, i) => (
                <li className="attachment" key={`${file.name}-${file.lastModified}-${i}`}>
                  <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 5h7l2 3h7v11H4z" />
                  </svg>
                  <span className="attachment-name" title={file.name}>
                    {file.name}
                  </span>
                  <span className="attachment-size">{formatBytes(file.size)}</span>
                  <button
                    type="button"
                    className="attachment-remove"
                    onClick={() => removeAt(i)}
                    aria-label={`Remove ${file.name}`}
                  >
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <label className="sr" htmlFor={QUERY_FIELD_ID}>
          Your question about the imagery
        </label>
        <textarea
          id={QUERY_FIELD_ID}
          ref={textareaRef}
          className="composer-input"
          rows={1}
          value={query}
          disabled={running}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="What changed between these two scenes?"
        />

        <div className="composer-tools">
          {/* Attach and its hint are one group, the keyboard hint and Run are
              another, so a narrow viewport wraps them as two blocks instead of
              stranding the Run button on a line of its own. */}
          <div className="composer-group">
            <button
              type="button"
              className="composer-attach"
              onClick={() => inputRef.current?.click()}
              disabled={running}
            >
              <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M21.4 11.1l-9.2 9.2a6 6 0 01-8.5-8.5l9.2-9.2a4 4 0 015.7 5.7l-9.2 9.2a2 2 0 01-2.8-2.9l8.5-8.4" />
              </svg>
              {files.length ? 'Add more' : 'Attach imagery'}
            </button>

            <span className="composer-hint">
              {files.length
                ? `${files.length} file${files.length === 1 ? '' : 's'} · ${formatBytes(total)}`
                : 'or drop files here · one or two scenes'}
            </span>
          </div>

          <div className="composer-group right">
            <kbd className="composer-kbd" title="Press Enter to run">
              ↵
            </kbd>
            <button className="btn btn-solid" type="submit" disabled={running}>
            {running ? (
              <>
                <span className="composer-spinner" aria-hidden="true" />
                Running…
              </>
            ) : (
              <>
                <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 3l14 9-14 9V3z" />
                </svg>
                Run
              </>
            )}
            </button>
          </div>
        </div>

        <input
          ref={inputRef}
          className="sr"
          type="file"
          accept={ACCEPTED}
          multiple
          onChange={(e) => {
            addFiles(e.target.files);
            // Reset, so picking the same file twice in a row still fires.
            e.target.value = '';
          }}
        />

        {dragging && (
          <div className="composer-drop" aria-hidden="true">
            Drop imagery to attach
          </div>
        )}
      </div>

      <div className="composer-foot">
        <span className="composer-formats">{ACCEPTED.replace(/,/g, '  ')}</span>
        <span className="spacer" />
        <span className="composer-note">
          A multi-band vendor product counts as one scene.
        </span>
      </div>

      {notice && (
        <p className="composer-error" role="alert">
          {notice}
        </p>
      )}
    </form>
  );
}
