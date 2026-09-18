'use client';

/**
 * The hero globe: a particle Earth in the register of the reference site.
 *
 * ~50k points on a Fibonacci sphere, kept where a rasterised Natural Earth
 * 1:110m land mask says there is land; ground stations as larger points, the
 * two hubs in the accent colour and pulsing; great-circle arcs from the hubs
 * to the stations, drawn in after the globe forms, each with its own light
 * packet on its own period so they never synchronise. On load the particles
 * fly in from the screen edges and the camera dollies in; afterwards the
 * globe turns slowly, breathes, and can be dragged (with inertia). A rim
 * sphere gives it a limb without any lighting.
 *
 * Raw three.js in one effect rather than react-three-fiber: everything here
 * is buffers and uniforms updated per frame, and a component tree adds
 * nothing but reconciliation. Rendering pauses when the hero is off-screen
 * or the tab is hidden. Reduced motion shows the formed globe, static.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { feature } from 'topojson-client';
import type { GeometryCollection, Topology } from 'topojson-specification';
import land110 from 'world-atlas/land-110m.json';

const FG = '#F0F0F8';
const ACCENT = '#E4007C';
const HALF_FOV_TAN = Math.tan((40 * Math.PI) / 360);

type Site = { name: string; lat: number; lng: number; hub?: boolean };

// Where the imagery comes down and where it is used. The two hubs are the
// national EO data centre and the space agency's headquarters; the rest are
// receiving stations, agency centres and the international archives the
// benchmarks came from.
const SITES: Site[] = [
  { name: 'Shadnagar', lat: 17.03, lng: 78.18, hub: true },
  { name: 'Bengaluru', lat: 12.97, lng: 77.59, hub: true },
  { name: 'Ahmedabad', lat: 23.02, lng: 72.57 },
  { name: 'Dehradun', lat: 30.32, lng: 78.03 },
  { name: 'Thiruvananthapuram', lat: 8.52, lng: 76.95 },
  { name: 'Sriharikota', lat: 13.72, lng: 80.23 },
  { name: 'Shillong', lat: 25.58, lng: 91.89 },
  { name: 'Jodhpur', lat: 26.24, lng: 73.02 },
  { name: 'Lucknow', lat: 26.85, lng: 80.95 },
  { name: 'Kolkata', lat: 22.57, lng: 88.36 },
  { name: 'Delhi', lat: 28.61, lng: 77.21 },
  { name: 'Mumbai', lat: 19.08, lng: 72.88 },
  { name: 'Nagpur', lat: 21.15, lng: 79.09 },
  { name: 'Port Blair', lat: 11.62, lng: 92.73 },
  { name: 'Svalbard', lat: 78.23, lng: 15.39 },
  { name: 'Frascati', lat: 41.81, lng: 12.67 },
  { name: 'Greenbelt', lat: 38.99, lng: -76.85 },
  { name: 'Sioux Falls', lat: 43.55, lng: -96.73 },
  { name: 'Tsukuba', lat: 36.08, lng: 140.08 },
  { name: 'Canberra', lat: -35.28, lng: 149.13 },
  { name: 'Kiruna', lat: 67.86, lng: 20.23 },
  { name: 'Pretoria', lat: -25.75, lng: 28.19 },
  { name: 'São Paulo', lat: -23.55, lng: -46.63 },
  { name: 'Singapore', lat: 1.35, lng: 103.82 },
  { name: 'Dubai', lat: 25.2, lng: 55.27 },
  { name: 'Mauritius', lat: -20.35, lng: 57.55 },
];
const ARCS: [string, string][] = [
  ['Shadnagar', 'Ahmedabad'], ['Shadnagar', 'Dehradun'], ['Shadnagar', 'Shillong'], ['Shadnagar', 'Jodhpur'],
  ['Shadnagar', 'Lucknow'], ['Shadnagar', 'Kolkata'], ['Shadnagar', 'Delhi'], ['Shadnagar', 'Mumbai'],
  ['Shadnagar', 'Nagpur'], ['Shadnagar', 'Port Blair'], ['Shadnagar', 'Svalbard'], ['Shadnagar', 'Mauritius'],
  ['Bengaluru', 'Thiruvananthapuram'], ['Bengaluru', 'Sriharikota'], ['Bengaluru', 'Frascati'], ['Bengaluru', 'Greenbelt'],
  ['Bengaluru', 'Sioux Falls'], ['Bengaluru', 'Tsukuba'], ['Bengaluru', 'Canberra'], ['Bengaluru', 'Kiruna'],
  ['Bengaluru', 'Pretoria'], ['Bengaluru', 'São Paulo'], ['Bengaluru', 'Singapore'], ['Bengaluru', 'Dubai'],
];

function expoOut(x: number): number {
  return x >= 1 ? 1 : 1 - Math.pow(2, -10 * x);
}
function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
function toXYZ(lat: number, lng: number, r: number): [number, number, number] {
  const a = (lat * Math.PI) / 180;
  const b = (lng * Math.PI) / 180;
  return [r * Math.cos(a) * Math.cos(b), r * Math.sin(a), -r * Math.cos(a) * Math.sin(b)];
}

/** Rasterise the land polygons once; returns an (lat, lng) -> land test. */
function landMask(): ((lat: number, lng: number) => boolean) | null {
  const topo = land110 as unknown as Topology<{ land: GeometryCollection }>;
  const geo = feature(topo, topo.objects.land) as unknown as { features?: { geometry: any }[]; geometry?: any };
  const geoms = geo.features ? geo.features.map((f) => f.geometry) : [geo.geometry];
  const W = 2048, H = 1024;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d', { willReadFrequently: true });
  if (!g) return null;
  g.fillStyle = '#000';
  g.fillRect(0, 0, W, H);
  g.fillStyle = '#fff';
  for (const geom of geoms) {
    const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.type === 'MultiPolygon' ? geom.coordinates : [];
    for (const poly of polys) {
      g.beginPath();
      for (const ring of poly) {
        for (let i = 0; i < ring.length; i++) {
          const x = ((ring[i][0] + 180) / 360) * W;
          const y = ((90 - ring[i][1]) / 180) * H;
          if (i === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.closePath();
      }
      g.fill('evenodd');
    }
  }
  const data = g.getImageData(0, 0, W, H).data;
  return (lat, lng) => {
    const x = clamp(Math.floor(((lng + 180) / 360) * W), 0, W - 1);
    const y = clamp(Math.floor(((90 - lat) / 180) * H), 0, H - 1);
    return data[(y * W + x) * 4] > 127;
  };
}

type ArcSpec = { a: number[]; b: number[]; ang: number; sinAng: number; offset: number; freq: number };

function arcSegments(arcs: ArcSpec[]) {
  const position: number[] = [], arcT: number[] = [], offset: number[] = [], speed: number[] = [];
  for (const { a, b, ang, sinAng, offset: off, freq } of arcs) {
    const n = Math.max(32, Math.round(64 * ang));
    let px = 0, py = 0, pz = 0, pt = 0;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const s0 = Math.sin((1 - t) * ang) / sinAng;
      const s1 = Math.sin(t * ang) / sinAng;
      const lift = 1 + 0.3 * Math.sin(Math.PI * t);
      const x = (a[0] * s0 + b[0] * s1) * lift;
      const y = (a[1] * s0 + b[1] * s1) * lift;
      const z = (a[2] * s0 + b[2] * s1) * lift;
      if (i > 0) {
        position.push(px, py, pz, x, y, z);
        arcT.push(pt, t);
        offset.push(off, off);
        speed.push(freq, freq);
      }
      px = x; py = y; pz = z; pt = t;
    }
  }
  return {
    position: Float32Array.from(position), arcT: Float32Array.from(arcT),
    offset: Float32Array.from(offset), speed: Float32Array.from(speed),
  };
}

const POINT_VERT = `
  attribute vec3 aScatter; attribute float aDelay; attribute float aSize; attribute float aPhase; attribute float aAccent;
  uniform float uTime; uniform float uProgress; uniform float uPixelRatio; uniform float uScale; uniform float uCamZ;
  uniform float uMotion; uniform float uRotY; uniform float uRotX;
  varying float vAlpha; varying float vAccent;
  float expoOut(float x) { return x >= 1.0 ? 1.0 : 1.0 - pow(2.0, -10.0 * x); }
  vec3 rotYX(vec3 v) {
    float cy = cos(uRotY); float sy = sin(uRotY);
    vec3 r = vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    float cx = cos(uRotX); float sx = sin(uRotX);
    return vec3(r.x, cx * r.y - sx * r.z, sx * r.y + cx * r.z);
  }
  void main() {
    vec3 tw = rotYX(position);
    float p = clamp((uProgress - aDelay) / 0.45, 0.0, 1.0);
    float e = expoOut(p);
    vec3 pos = mix(aScatter, tw, e);
    pos += tw * sin(uTime * 0.6 + aPhase * 6.2831) * 0.004 * e * uMotion;
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    float front = clamp((mv.z + uCamZ + 1.0) * 0.5, 0.0, 1.0);
    float pulse = 1.0 + aAccent * 0.3 * sin(uTime * 2.2 + aPhase) * uMotion;
    float size = aSize * pulse * mix(0.72, 1.0, front);
    gl_PointSize = size * uScale * uPixelRatio / max(-mv.z, 0.001);
    float twinkle = 0.86 + 0.14 * sin(uTime * 1.4 + aPhase * 12.566) * uMotion;
    vAlpha = smoothstep(0.0, 0.12, uProgress) * twinkle * mix(0.16, 1.0, front);
    vAccent = aAccent;
    gl_Position = projectionMatrix * mv;
  }`;
const POINT_FRAG = `
  uniform vec3 uColor; uniform vec3 uAccentColor; varying float vAlpha; varying float vAccent;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float disc = 1.0 - smoothstep(0.14, 0.5, d);
    if (disc < 0.004) discard;
    gl_FragColor = vec4(mix(uColor, uAccentColor, vAccent), disc * vAlpha);
  }`;
const ARC_VERT = `
  attribute float aArcT; attribute float aOffset; attribute float aSpeed;
  uniform float uRotY; uniform float uRotX; uniform float uCamZ;
  varying float vT; varying float vOff; varying float vSpeed; varying float vFront;
  void main() {
    float cy = cos(uRotY); float sy = sin(uRotY);
    vec3 t = vec3(cy * position.x + sy * position.z, position.y, -sy * position.x + cy * position.z);
    float cx = cos(uRotX); float sx = sin(uRotX);
    t = vec3(t.x, cx * t.y - sx * t.z, sx * t.y + cx * t.z);
    vec4 mv = modelViewMatrix * vec4(t, 1.0);
    vFront = clamp((mv.z + uCamZ + 1.0) * 0.5, 0.0, 1.0);
    vT = aArcT; vOff = aOffset; vSpeed = aSpeed;
    gl_Position = projectionMatrix * mv;
  }`;
const ARC_FRAG = `
  uniform float uTime; uniform float uArcProgress; uniform float uMotion; uniform vec3 uColor; uniform float uBase;
  varying float vT; varying float vOff; varying float vSpeed; varying float vFront;
  void main() {
    float reveal = clamp((uArcProgress - vT * 0.85) / 0.15, 0.0, 1.0);
    float cyc = fract(uTime * vSpeed + vOff);
    float run = step(cyc, 0.20);
    float tr = clamp(cyc / 0.20, 0.0, 1.0);
    float tt = tr * tr * (3.0 - 2.0 * tr);
    float d = tt - vT;
    float beam = (d >= 0.0 ? exp(-d * 40.0) : 0.0) * run;
    float gate = smoothstep(0.85, 1.0, uArcProgress) * uMotion;
    float a = (uBase + beam * 0.42 * gate) * reveal * mix(0.28, 1.0, vFront);
    gl_FragColor = vec4(uColor, a);
  }`;
const RIM_VERT = `
  varying vec3 vNormal; varying vec3 vView;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
const RIM_FRAG = `
  uniform vec3 uColor; uniform float uIntensity; varying vec3 vNormal; varying vec3 vView;
  void main() {
    float rim = pow(1.0 - abs(dot(vView, normalize(vNormal))), 3.5);
    gl_FragColor = vec4(uColor, rim * uIntensity);
  }`;

export default function ParticleWorld({ status }: { status?: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const el = wrap.current;
    const cv = canvas.current;
    if (!el || !cv) return;
    // ?static shows the formed globe without the intro or motion - the same
    // path reduced-motion takes - for screenshots and slow machines.
    const reduced =
      window.matchMedia('(prefers-reduced-motion: reduce)').matches ||
      new URLSearchParams(window.location.search).has('static');
    const mobile = window.matchMedia('(max-width: 767px)').matches;
    const fine = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    let disposed = false;
    let raf = 0;
    let running = false;
    let visible = true;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: cv, alpha: true, antialias: false, powerPreference: 'high-performance' });
    } catch {
      return;
    }
    renderer.setClearColor(0, 0);

    const rimMat = new THREE.ShaderMaterial({
      vertexShader: RIM_VERT, fragmentShader: RIM_FRAG,
      uniforms: { uColor: { value: new THREE.Color(FG) }, uIntensity: { value: 0 } },
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    });
    const rimGeo = new THREE.SphereGeometry(1.045, 48, 48);
    scene.add(new THREE.Mesh(rimGeo, rimMat));

    const U = {
      uTime: { value: 0 }, uProgress: { value: reduced ? 1 : 0 }, uArcProgress: { value: reduced ? 1 : 0 },
      uPixelRatio: { value: 1 }, uScale: { value: 1 }, uCamZ: { value: 3 }, uMotion: { value: reduced ? 0 : 1 },
      uRotY: { value: 0 }, uRotX: { value: 0 }, uColor: { value: new THREE.Color(FG) }, uAccentColor: { value: new THREE.Color(ACCENT) },
    };

    // ---- sizing: the globe is min(66vmin, 512px) across, centred, with its
    // centre at max(38svh, radius + 120px) from the top of the hero.
    let w = 1, h = 1, camZ = 3, ndcY = 0.24;
    const resize = () => {
      w = Math.max(1, Math.round(el.clientWidth));
      h = Math.max(1, Math.round(el.clientHeight));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      const size = Math.min(512, 0.66 * Math.min(w, h));
      camZ = h / (HALF_FOV_TAN * size);
      ndcY = 1 - (Math.max(0.38 * h, size / 2 + 120) / h) * 2;
      U.uPixelRatio.value = dpr;
      U.uScale.value = (0.5 * h) / HALF_FOV_TAN;
    };
    resize();
    camera.position.z = reduced ? camZ : 1.145 * camZ;
    camera.position.y = -ndcY * HALF_FOV_TAN * camera.position.z;
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    // ---- rotation: start with the first hub facing the camera.
    const byName = Object.fromEntries(SITES.map((s) => [s.name, s]));
    const hubs = SITES.filter((s) => s.hub);
    let rotY = 0;
    {
      const [x, , z] = toXYZ(hubs[0].lat, hubs[0].lng, 1);
      rotY = -Math.atan2(x, z);
    }
    let rotX = 0;
    let dragging = false, touchPending = false, isTouch = false;
    let sx = 0, sy = 0, lx = 0, ly = 0;
    const idleSpin = reduced ? 0 : 0.055;
    let spin = idleSpin;
    const onDown = (e: PointerEvent) => {
      if (dragging || touchPending) return;
      isTouch = e.pointerType === 'touch';
      sx = lx = e.clientX; sy = ly = e.clientY; spin = 0;
      if (isTouch) touchPending = true;
      else { dragging = true; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; }
    };
    const onMove = (e: PointerEvent) => {
      if (touchPending && !dragging) {
        const dx = e.clientX - sx, dy = e.clientY - sy;
        if (Math.abs(dx) > 8 && Math.abs(dx) > Math.abs(dy)) { dragging = true; lx = e.clientX; ly = e.clientY; }
        else { if (Math.abs(dy) > 8) touchPending = false; return; }
      }
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      rotY += 0.0045 * dx;
      spin = clamp(0.0045 * dx * 60, -2.2, 2.2);
      if (!isTouch) {
        const room = 1 - Math.min(1, Math.abs(rotX) / 0.55);
        rotX = clamp(rotX + 0.0035 * dy * (0.35 + 0.65 * room), -0.55, 0.55);
      }
    };
    const onUp = (e: PointerEvent) => {
      touchPending = false;
      if (dragging) {
        dragging = false;
        if (!isTouch) { try { cv.releasePointerCapture(e.pointerId); } catch {} cv.style.cursor = 'grab'; }
      }
    };
    cv.style.touchAction = 'pan-y';
    if (fine) cv.style.cursor = 'grab';
    cv.addEventListener('pointerdown', onDown);
    cv.addEventListener('pointermove', onMove);
    cv.addEventListener('pointerup', onUp);
    cv.addEventListener('pointercancel', onUp);

    // ---- frame loop
    const clock = new THREE.Clock();
    let t = reduced ? 2.4 : 0;
    let built = false;
    let shown = false;
    const frame = () => {
      if (disposed) return;
      raf = requestAnimationFrame(frame);
      const dt = Math.min(clock.getDelta(), 0.05);
      t = Math.min(t + dt, 4.4);
      U.uTime.value += dt;
      U.uProgress.value = Math.min(t / 2.4, 1);
      U.uArcProgress.value = reduced ? 1 : clamp((t - 2.65) / 1.3, 0, 1);
      camera.position.z = reduced ? camZ : camZ + camZ * (1.145 - 1) * (1 - expoOut(Math.min(t / 3, 1)));
      camera.position.y = -ndcY * HALF_FOV_TAN * camera.position.z;
      U.uCamZ.value = camera.position.z;
      rimMat.uniforms.uIntensity.value = 0.32 * expoOut(Math.min(t / 2.4, 1));
      if (!dragging) {
        spin += (idleSpin - spin) * (1 - Math.exp(-1.6 * dt));
        rotY += spin * dt;
        rotX += (0 - rotX) * (1 - Math.exp(-0.9 * dt));
      }
      U.uRotY.value = rotY;
      U.uRotX.value = rotX;
      renderer.render(scene, camera);
      if (!shown) { shown = true; setReady(true); }
    };
    const stop = () => { running = false; cancelAnimationFrame(raf); };
    const sync = () => {
      // The browser throttles rAF in hidden tabs itself; gating on
      // document.hidden as well left embedded/preview panes blank.
      if (visible && built) {
        if (!running) { running = true; clock.getDelta(); raf = requestAnimationFrame(frame); }
      } else stop();
    };
    const io = new IntersectionObserver((entries) => { for (const en of entries) visible = en.isIntersecting; sync(); }, { threshold: 0 });
    io.observe(el);

    // ---- geometry
    const disposables: { dispose(): void }[] = [rimGeo, rimMat];
    const build = () => {
      const isLand = landMask();
      if (!isLand || disposed) return;
      const count = mobile ? 22000 : 52000;
      const pos: number[] = [], scatter: number[] = [], delay: number[] = [], size: number[] = [], phase: number[] = [], accent: number[] = [];
      const aspect = w / h;
      const startZ = 1.145 * camZ;
      const startY = -ndcY * HALF_FOV_TAN * startZ;
      // A random point just outside the viewport, at a random depth in
      // front of the globe, for the fly-in.
      const scatterPoint = () => {
        const u = 2 * Math.random() - 1;
        const r = 1.06 + 0.28 * Math.random();
        let x: number, y: number;
        if (Math.random() < w / (w + h)) { x = u * r; y = Math.random() < 0.5 ? r : -r; }
        else { x = Math.random() < 0.5 ? r : -r; y = u * r; }
        const dz = startZ - 1.2 + 2.8 * Math.random();
        const half = dz * HALF_FOV_TAN;
        scatter.push(x * half * aspect, y * half + startY, startZ - dz);
      };
      const golden = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < count; i++) {
        const y = 1 - (i / (count - 1)) * 2;
        const rad = Math.sqrt(Math.max(0, 1 - y * y));
        const th = golden * i;
        const x = Math.cos(th) * rad, z = Math.sin(th) * rad;
        const lng = (180 * Math.atan2(-z, x)) / Math.PI;
        const lat = (180 * Math.asin(y)) / Math.PI;
        if (!isLand(lat, lng)) continue;
        pos.push(x, y, z);
        scatterPoint();
        delay.push(((lng + 180) / 360) * 0.33 + 0.22 * Math.random());
        size.push(0.011 + 0.006 * Math.random());
        phase.push(Math.random());
        accent.push(0);
      }
      for (const s of SITES) {
        const [x, y, z] = toXYZ(s.lat, s.lng, 1);
        pos.push(x, y, z);
        scatterPoint();
        delay.push(s.hub ? 0.05 : ((s.lng + 180) / 360) * 0.33 + 0.22 * Math.random());
        size.push(s.hub ? 0.026 : 0.017);
        phase.push(Math.random());
        accent.push(s.hub ? 1 : 0);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(pos), 3));
      geo.setAttribute('aScatter', new THREE.BufferAttribute(Float32Array.from(scatter), 3));
      geo.setAttribute('aDelay', new THREE.BufferAttribute(Float32Array.from(delay), 1));
      geo.setAttribute('aSize', new THREE.BufferAttribute(Float32Array.from(size), 1));
      geo.setAttribute('aPhase', new THREE.BufferAttribute(Float32Array.from(phase), 1));
      geo.setAttribute('aAccent', new THREE.BufferAttribute(Float32Array.from(accent), 1));
      const mat = new THREE.ShaderMaterial({
        vertexShader: POINT_VERT, fragmentShader: POINT_FRAG, uniforms: U,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      });
      const points = new THREE.Points(geo, mat);
      points.frustumCulled = false;
      scene.add(points);
      disposables.push(geo, mat);

      const arcs: ArcSpec[] = [];
      for (const [from, to] of ARCS) {
        const a = byName[from], b = byName[to];
        if (!a || !b) continue;
        const pa = toXYZ(a.lat, a.lng, 1), pb = toXYZ(b.lat, b.lng, 1);
        const ang = Math.acos(clamp(pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2], -1, 1));
        const sinAng = Math.sin(ang);
        if (sinAng < 1e-4) continue;
        arcs.push({ a: pa, b: pb, ang, sinAng, offset: Math.random(), freq: 1 / (18 + 12 * Math.random()) });
      }
      const seg = arcSegments(arcs);
      const ageo = new THREE.BufferGeometry();
      ageo.setAttribute('position', new THREE.BufferAttribute(seg.position, 3));
      ageo.setAttribute('aArcT', new THREE.BufferAttribute(seg.arcT, 1));
      ageo.setAttribute('aOffset', new THREE.BufferAttribute(seg.offset, 1));
      ageo.setAttribute('aSpeed', new THREE.BufferAttribute(seg.speed, 1));
      const amat = new THREE.ShaderMaterial({
        vertexShader: ARC_VERT, fragmentShader: ARC_FRAG,
        uniforms: {
          uTime: U.uTime, uArcProgress: U.uArcProgress, uRotY: U.uRotY, uRotX: U.uRotX, uCamZ: U.uCamZ,
          uMotion: U.uMotion, uColor: U.uColor, uBase: { value: 0.3 },
        },
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      });
      const lines = new THREE.LineSegments(ageo, amat);
      lines.frustumCulled = false;
      scene.add(lines);
      disposables.push(ageo, amat);
      el.dataset.particles = String(pos.length / 3);
      built = true;
      sync();
    };
    // Off the first paint: the mask rasterisation and 50k random points are
    // a few tens of milliseconds the headline should not wait for.
    const timer = window.setTimeout(build, 0);

    return () => {
      disposed = true;
      window.clearTimeout(timer);
      stop();
      io.disconnect();
      ro.disconnect();
      cv.removeEventListener('pointerdown', onDown);
      cv.removeEventListener('pointermove', onMove);
      cv.removeEventListener('pointerup', onUp);
      cv.removeEventListener('pointercancel', onUp);
      for (const d of disposables) d.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div ref={wrap} className="world" style={{ opacity: ready ? 1 : 0 }}>
      <canvas ref={canvas} className="world-canvas" aria-hidden="true" />
      {ready && status && (
        <div className="world-status-frame" aria-hidden="true">
          <span className="world-status">
            <span className="world-status-dot" />
            {status}
          </span>
        </div>
      )}
    </div>
  );
}
