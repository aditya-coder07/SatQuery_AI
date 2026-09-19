'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import { apiBase } from '../lib/api';

/**
 * The hero globe with a badge that says what the API is actually running
 * on: `/device` reports the CUDA device and free VRAM (or "cpu"). If the
 * API is unreachable the badge says so - it never claims to be online on
 * its own authority.
 */
const API = apiBase();

const ParticleWorld = dynamic(() => import('./ParticleWorld'), {
  ssr: false,
  loading: () => <div className="world" />,
});

export default function HeroWorld() {
  const [status, setStatus] = useState('connecting to the api');
  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/device`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => {
        if (cancelled) return;
        const free = d.vram_free_bytes != null ? ` · ${(d.vram_free_bytes / 1024 ** 3).toFixed(1)} GB free` : '';
        setStatus(`online · ${String(d.device).toLowerCase()}${free}`);
      })
      .catch(() => {
        if (!cancelled) setStatus('api offline');
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return <ParticleWorld status={status} />;
}
