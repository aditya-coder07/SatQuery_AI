/**
 * One-click sample inputs for the composer.
 *
 * A first-time user rarely has a GeoTIFF to hand, and a query console that
 * needs one before it shows anything is a console nobody tries. These are the
 * synthetic scenes from `data/demo_bundle` (small, licence-free, made for the
 * rehearsal script), served from `public/samples/` and turned into `File`
 * objects on the client, so the run goes through exactly the same upload path
 * as a user's own imagery - nothing is special-cased on the server.
 */

export type Sample = {
  key: string;
  /** Chip label. */
  title: string;
  /** What the run shows off, one clause. */
  blurb: string;
  query: string;
  files: string[];
};

export const SAMPLES: Sample[] = [
  {
    key: 'describe',
    title: 'Describe a scene',
    blurb: 'PNG tile · caption + land cover',
    query: 'Describe the land-cover and major objects visible in this image.',
    files: ['benchmark_tile.png'],
  },
  {
    key: 'count',
    title: 'Ask a question',
    blurb: 'GeoTIFF · visual QA',
    query: 'Are there buildings in this image, and roughly how many?',
    files: ['optical_t1.tif'],
  },
  {
    key: 'change',
    title: 'What changed?',
    blurb: 'bi-temporal pair · change mask + caption',
    query: 'What changed between these two scenes?',
    files: ['levir_t1.tif', 'levir_t2.tif'],
  },
];

function mime(name: string): string {
  if (/\.png$/i.test(name)) return 'image/png';
  if (/\.jpe?g$/i.test(name)) return 'image/jpeg';
  return 'image/tiff';
}

/** Fetch a sample's scenes as File objects, in order. Throws on any 404. */
export async function loadSample(sample: Sample): Promise<File[]> {
  return Promise.all(
    sample.files.map(async (name) => {
      const res = await fetch(`/samples/${name}`);
      if (!res.ok) throw new Error(`sample ${name} is missing (${res.status})`);
      const blob = await res.blob();
      return new File([blob], name, { type: mime(name) });
    }),
  );
}
