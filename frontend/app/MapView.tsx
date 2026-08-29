'use client';

/**
 * Georeferenced map viewer (plan task 1.6) with a hybrid basemap.
 *
 * ## The hybrid, and why it is not just `navigator.onLine`
 *
 * Task 3.9 requires the system to boot and answer with **no network**, and a
 * map viewer wants tiles from the internet. The team's decision is hybrid:
 * live basemap when the internet is reachable, local rendering when it is not.
 *
 * `navigator.onLine` cannot implement that. It reports whether the machine has
 * *a* network interface, not whether the tile server is reachable - it is true
 * on a venue wifi that captive-portals every request, which is exactly the
 * demo-day failure mode. So the basemap is decided by **actually fetching one
 * tile** with a short timeout, and the result is cached for the session.
 *
 * The overlay never depends on the outcome. It is served by our own API,
 * already reprojected to EPSG:3857, so the scene, the mask and the indices all
 * render identically offline - only the backdrop changes. Offline draws a
 * neutral graticule instead, and the badge says which mode is active rather
 * than leaving a blank map looking broken.
 *
 * ## Why the server reprojects
 *
 * OpenLayers would need proj4 plus a registry entry for every UTM zone an
 * Indian scene can land in (the Cartosat sample alone is 45N). The API returns
 * EPSG:3857 with the extent in a header, so the client places the image with
 * no projection maths and no extra dependency.
 */

import { useEffect, useRef, useState } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import ImageLayer from 'ol/layer/Image';
import XYZ from 'ol/source/XYZ';
import Static from 'ol/source/ImageStatic';
import { Graticule } from 'ol/layer';
import { Stroke } from 'ol/style';
import 'ol/ol.css';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

/**
 * The basemap tile source, as an XYZ template.
 *
 * Configurable rather than hardcoded, because "offline" and "no basemap" are
 * not the same thing: a venue with no internet can still run a LOCAL tile
 * server, and pointing this at it gives a real basemap with no external
 * network. Setting it to an empty string forces the offline rendering path,
 * which is also how the offline branch gets tested without unplugging
 * anything.
 */
const BASEMAP_URL =
  process.env.NEXT_PUBLIC_BASEMAP_URL ??
  'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

// Attribution is required by the ODbL for OSM tiles and is wrong to show for
// someone else's source, so it travels with the URL.
const BASEMAP_ATTRIBUTION =
  process.env.NEXT_PUBLIC_BASEMAP_ATTRIBUTION ?? '';

// One small tile, with a short timeout. A captive portal answers, but not with
// an image, so a successful fetch alone is not proof - the payload has to
// decode.
const PROBE_URL = BASEMAP_URL.replace('{z}', '0')
  .replace('{x}', '0')
  .replace('{y}', '0');
const PROBE_TIMEOUT_MS = 2500;

type Basemap = 'probing' | 'online' | 'offline';

async function basemapReachable(): Promise<boolean> {
  // An explicitly empty basemap is a deliberate choice, not a failure.
  if (!BASEMAP_URL) return false;
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    // A definite "no" is trustworthy; a "yes" is not, so only short-circuit
    // the negative case.
    return false;
  }
  try {
    const image = new Image();
    const loaded = new Promise<boolean>((resolve) => {
      image.onload = () => resolve(image.width > 0);
      image.onerror = () => resolve(false);
    });
    const timeout = new Promise<boolean>((resolve) =>
      setTimeout(() => resolve(false), PROBE_TIMEOUT_MS),
    );
    image.crossOrigin = 'anonymous';
    image.src = `${PROBE_URL}?probe=${Date.now()}`;
    return await Promise.race([loaded, timeout]);
  } catch {
    return false;
  }
}

type Overlay = { key: string; available: boolean };

export default function MapView({ runId }: { runId: string }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const [basemap, setBasemap] = useState<Basemap>('probing');
  const [overlays, setOverlays] = useState<Overlay[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isOpenStreetMap = BASEMAP_URL.includes('openstreetmap.org');

  useEffect(() => {
    let cancelled = false;
    basemapReachable().then((ok) => {
      if (!cancelled) setBasemap(ok ? 'online' : 'offline');
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    fetch(`${API}/runs/${runId}/overlays`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => {
        const usable: Overlay[] = (d.overlays ?? []).filter(
          (o: Overlay) => o.available,
        );
        setOverlays(usable);
        setActive((current) => current ?? usable[0]?.key ?? null);
      })
      .catch((e) => setError(String(e)));
  }, [runId]);

  useEffect(() => {
    if (!container.current || basemap === 'probing' || mapRef.current) return;

    const base =
      basemap === 'online'
        ? new TileLayer({
            source: new XYZ({
              url: BASEMAP_URL,
              attributions: BASEMAP_ATTRIBUTION || undefined,
              crossOrigin: 'anonymous',
            }),
          })
        : // No tiles offline. A graticule gives the eye a coordinate frame so
          // the overlay is not floating in a void.
          new Graticule({
            strokeStyle: new Stroke({ color: 'rgba(120,140,160,0.35)', width: 1 }),
            showLabels: true,
            wrapX: false,
          });

    mapRef.current = new Map({
      target: container.current,
      layers: [base],
      view: new View({ center: [0, 0], zoom: 2 }),
    });

    return () => {
      mapRef.current?.setTarget(undefined);
      mapRef.current = null;
    };
  }, [basemap]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !active) return;
    let cancelled = false;

    (async () => {
      try {
        const response = await fetch(`${API}/runs/${runId}/overlay/${active}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        // The extent travels in a header because the payload is the PNG
        // itself; the server already reprojected both to EPSG:3857.
        const header = response.headers.get('X-Extent');
        if (!header) throw new Error('overlay is missing its X-Extent header');
        const extent = header.split(',').map(Number) as [
          number, number, number, number,
        ];
        const url = URL.createObjectURL(await response.blob());
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }

        map
          .getLayers()
          .getArray()
          .filter((l) => l.get('overlay'))
          .forEach((l) => map.removeLayer(l));

        const layer = new ImageLayer({
          source: new Static({ url, imageExtent: extent, projection: 'EPSG:3857' }),
          opacity: 0.85,
        });
        layer.set('overlay', true);
        map.addLayer(layer);
        map.getView().fit(extent, { padding: [24, 24, 24, 24], maxZoom: 17 });
        setError(null);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [runId, active, basemap]);

  return (
    <section className="mapview">
      <div className="mapview-bar">
        <span className={`mapview-badge ${basemap}`}>
          {basemap === 'probing'
            ? 'checking for a basemap…'
            : basemap === 'online'
              ? 'live basemap'
              : 'offline — local rendering'}
        </span>
        {overlays.map((o) => (
          <button
            key={o.key}
            className={`mapview-layer ${active === o.key ? 'active' : ''}`}
            onClick={() => setActive(o.key)}
          >
            {o.key}
          </button>
        ))}
        {overlays.length === 0 && !error && (
          <span className="mapview-note">no georeferenced artifacts</span>
        )}
      </div>
      <div ref={container} className="mapview-canvas" />
      {basemap === 'online' && isOpenStreetMap && (
        // Required by the ODbL, and shown only when OSM tiles are the source -
        // attributing OSM for someone else's tile server would be wrong.
        <div className="mapview-attribution">
          © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>{' '}
          contributors
        </div>
      )}
      {basemap === 'online' && !isOpenStreetMap && BASEMAP_ATTRIBUTION && (
        <div className="mapview-attribution">{BASEMAP_ATTRIBUTION}</div>
      )}
      {error && <p className="mapview-note error">{error}</p>}
    </section>
  );
}
