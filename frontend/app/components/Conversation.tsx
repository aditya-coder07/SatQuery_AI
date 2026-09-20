'use client';

import type { Turn } from '../lib/events';

/**
 * The turns of this conversation, oldest first, above the current answer.
 *
 * Each turn is what the API received back as `history`: the question as
 * typed, the task it ran and the answer. Shown so a follow-up ("Only show
 * buildings.") reads in context, and so what the resolver had to work with
 * is visible rather than implied. Cleared when the scenes change and by
 * the button here.
 */
export default function Conversation({
  turns,
  current,
  onClear,
}: {
  turns: Turn[];
  current: string;
  onClear: () => void;
}) {
  if (turns.length === 0) return null;
  return (
    <div className="conversation">
      <div className="conversation-head">
        <span className="meta">
          conversation · {turns.length} earlier {turns.length === 1 ? 'turn' : 'turns'}
        </span>
        <span className="spacer" />
        <button type="button" className="conversation-clear" onClick={onClear}>
          new conversation
        </button>
      </div>
      <ol className="turns">
        {turns.map((t, i) => (
          <li className="turn" key={i}>
            <span className="turn-q">{t.query}</span>
            <span className="turn-task">{t.task}</span>
            <span className="turn-a">{t.answer.length > 180 ? `${t.answer.slice(0, 180)}…` : t.answer}</span>
          </li>
        ))}
        {current && (
          <li className="turn current">
            <span className="turn-q">{current}</span>
          </li>
        )}
      </ol>
    </div>
  );
}
