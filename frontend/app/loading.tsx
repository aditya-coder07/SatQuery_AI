/**
 * Route-level loading state: shown by the App Router while the next page's
 * segment is being fetched or, in development, compiled. The dot grid and
 * nav stay where they are; this fills the content area with the same
 * language as the rest of the site - a pulsing accent dot, a mono label
 * and a sweeping hairline - rather than a blank frame.
 */
export default function Loading() {
  return (
    <div className="route-loading" role="status" aria-live="polite">
      <div className="route-loading-inner">
        <span className="route-loading-dot" />
        <span className="route-loading-label">loading</span>
        <span className="route-loading-bar">
          <span />
        </span>
      </div>
    </div>
  );
}
