import type { AnswerDetails as Details } from '../lib/events';

/** The run's findings as labelled sections, each item read from a tool output. */
export default function AnswerDetails({ details }: { details: Details | null }) {
  if (!details || details.sections.length === 0) return null;
  return (
    <div className="answer-details">
      {details.sections.map((section) => (
        <section key={section.title} className="detail-section">
          <h3 className="detail-title">{section.title}</h3>
          <ul>
            {section.items.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
