import Link from 'next/link';

import HeroWorld from './components/HeroWorld';
import InkReveal from './components/InkReveal';
import LiveRun from './components/LiveRun';
import MethodScroller from './components/MethodScroller';
import ParticleCloud from './components/ParticleCloud';
import QuestionCloud from './components/QuestionCloud';
import Reveal from './components/Reveal';
import ScrollProgress from './components/ScrollProgress';
import Words from './components/Words';
import { BENCHMARKS, CAPABILITIES, CARRIES, METHOD, SENSORS } from './lib/site';

/**
 * The home page. The layout follows the reference: a particle globe centred
 * above a centred monospace headline, a marquee of the imagery the system
 * is built for, then card sections over the fixed dot grid - capabilities,
 * measured evidence, the five-phase method, what an answer carries, how it
 * deploys - a talk-to-it band, and a four-column footer.
 * The query console is its own route (/query).
 *
 * Every number here is a measured official-split figure from
 * docs/research/sota_matrix.md, the same ones /benchmarks reads from the
 * API. This page is a server component; only the globe and the reveal
 * wrappers run on the client.
 */


export default function Home() {
  return (
    <>
      <svg width="0" height="0" aria-hidden="true" style={{ position: 'absolute', pointerEvents: 'none' }}>
        <defs>
          {/* The ink edge: turbulence displaces a straight edge into blots,
              then the alpha is cut hard so it reads as ink, not fog. */}
          <filter id="ink-edge" x="-10%" y="-10%" width="120%" height="120%">
            <feTurbulence type="fractalNoise" baseFrequency="0.015 0.02" numOctaves="3" seed="7" result="noise" />
            <feDisplacementMap in="SourceGraphic" in2="noise" scale="220" xChannelSelector="R" yChannelSelector="G" result="displaced" />
            <feGaussianBlur in="displaced" stdDeviation="1.8" result="presmooth" />
            <feComponentTransfer in="presmooth" result="cut">
              <feFuncA type="discrete" tableValues="0 0 0 0 0 1 1 1 1 1" />
            </feComponentTransfer>
            <feGaussianBlur in="cut" stdDeviation="0.4" />
          </filter>
        </defs>
      </svg>
      <ScrollProgress />

      <main className="home">
        <InkReveal
          before={
            <>
        {/* ---------------------------------------------------------- hero */}
        <section className="home-hero" id="top">
          <HeroWorld />

          <div className="home-hero-copy">
            <h1 className="display-xl">
              <Words text="Ask the imagery. Watch it reason." now delay={900} stagger={220} />
            </h1>
            <p className="home-sub">
              <Words
                text="A vision-language assistant for remote sensing. One question in plain language, routed to the right specialist model - and answered with its evidence, its checks and its calibrated doubt attached."
                now
                delay={2400}
                stagger={60}
              />
            </p>
            <div className="home-ctas hero-late" style={{ animationDelay: '4700ms' }}>
              <Link href="/query" className="pill pill-solid">
                open the query console
              </Link>
              <Link href="/benchmarks" className="pill pill-line">
                see the benchmarks
              </Link>
            </div>
            <p className="home-stats hero-late" style={{ animationDelay: '5100ms' }}>
              <span>9 specialist tools</span>
              <span>8 official benchmarks</span>
              <span>RSVQA-LR 91.2 %</span>
              <span>LEVIR-CD F1 0.904</span>
              <span>abstains when it should</span>
            </p>
          </div>

          <div className="home-strip" aria-label="Imagery the system is built for">
            <div className="home-strip-track">
              {[...SENSORS, ...SENSORS].map((s, i) => (
                <span key={i} className="home-strip-item">
                  <b>{s.name}</b>
                  <em>{s.kind}</em>
                </span>
              ))}
            </div>
          </div>
        </section>

        {/* -------------------------------------------------- capabilities */}
        <section className="home-section" id="capabilities">
          <Reveal>
            <div className="home-section-head">
              <span className="eyebrow">capabilities</span>
              <h2 className="display-m">Nine tools, one question box.</h2>
              <p>
                The router reads the question and the scenes and picks the plan; the tools do
                one job each and report what they measured. Every learned tool here has an
                official-split number next to it.
              </p>
            </div>
          </Reveal>
          <div className="cap-groups">
            {CAPABILITIES.map((group, gi) => (
              <Reveal key={group.title} delay={gi * 80}>
                <div className="cap-group">
                  <div className="cap-group-head">
                    <h3>{group.title}</h3>
                    <p className="cap-group-lede">{group.lede}</p>
                  </div>
                  <ul className="cap-cards">
                    {group.items.map((item) => (
                      <li key={item.name} className="cap-card">
                        <span className="cap-tool">
                          {item.metric ? <i className="accent">✦</i> : null}
                          {item.tool}
                        </span>
                        <span className="cap-name">{item.name}</span>
                        <p>{item.text}</p>
                        {item.metric && <span className="cap-metric">{item.metric}</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              </Reveal>
            ))}
          </div>
          <p className="home-note">
            <i className="accent">✦</i> measured on the official test split
          </p>
        </section>

        {/* ---------------------------------------------------- benchmarks */}
        <section className="home-section ink-before" id="evidence">
          <Reveal>
            <div className="home-section-head">
              <span className="eyebrow">evidence</span>
              <h2 className="display-m">Measured on the official test splits.</h2>
              <p>
                No leaked images, no cherry-picked subsets. Each figure is from the published
                test split with a 95 % interval, and the run that produced it is in the
                experiment registry.
              </p>
            </div>
          </Reveal>
          <div className="bench-grid">
            {BENCHMARKS.map((b, i) => (
              <Reveal key={b.task} delay={i * 60}>
                <div className="bench-card">
                  <span className="bench-task">{b.task}</span>
                  <span className="bench-value">{b.value}</span>
                  <span className="bench-metric">{b.metric}</span>
                  <span className="bench-set">{b.dataset}</span>
                  {b.note && <span className="bench-note">{b.note}</span>}
                </div>
              </Reveal>
            ))}
          </div>
          <Reveal>
            <p className="home-more">
              <Link href="/benchmarks" className="arrow-link">
                all benchmarks, with intervals and the runs behind them →
              </Link>
            </p>
          </Reveal>
        </section>

            </>
          }
          after={
            <>
        {/* -------------------------------------------------------- method */}
        <MethodScroller
          eyebrow="the run"
          heading="The 5-phase run."
          intro="The same pipeline answers a one-line question and a two-scene change query. What changes is which tools the plan calls - never whether the checks run."
          phases={METHOD}
        />

        {/* ------------------------------------------------------- carries */}
        <section className="home-section" id="answer">
          <Reveal>
            <div className="home-section-head">
              <span className="eyebrow">the answer</span>
              <h2 className="display-m">What every answer carries.</h2>
              <p>
                An answer without its provenance is an opinion. Each run leaves with the
                things you need to check it, re-open it, or hand it to someone else.
              </p>
            </div>
          </Reveal>
          <div className="carry-row">
            {CARRIES.map((c, i) => (
              <Reveal key={c.title} delay={i * 60}>
                <div className="carry-card">
                  <span className="carry-k">{c.k}</span>
                  <h3>{c.title}</h3>
                  <p>{c.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </section>

        {/* -------------------------------------------------------- deploy */}
        <section className="home-section" id="deploy">
          <div className="deploy">
            <Reveal>
              <div className="deploy-copy">
                <span className="eyebrow">run it anywhere</span>
                <h2 className="display-m">One container, one GPU, no cloud dependency.</h2>
                <p>
                  The API and the web console ship as two Docker images. Weights are mounted,
                  not downloaded; fonts are self-hosted; the API answers with the network cable
                  pulled, and the map falls back to the scene footprint when no basemap is
                  reachable. A 6 GB card runs every tool - this page was tested on one.
                </p>
                <ul className="deploy-list">
                  <li>
                    <b>API-first.</b> <code>POST /runs</code> answers synchronously,{' '}
                    <code>POST /runs/stream</code> streams the same run as server-sent events.
                  </li>
                  <li>
                    <b>Stored runs.</b> Every run is persisted with its trace and can be
                    reopened, compared, or exported as a PDF report.
                  </li>
                  <li>
                    <b>Profiles.</b> <code>full</code> loads everything; <code>lite</code>{' '}
                    keeps the deterministic index engine and still answers, degraded but never
                    failing.
                  </li>
                </ul>
              </div>
            </Reveal>
            <Reveal delay={120}>
              <LiveRun />
            </Reveal>
          </div>
        </section>

        {/* ----------------------------------------------------- questions */}
        <QuestionCloud
          heading="What can't you get out of your imagery yet?"
          sub="Every one of these is a question the console answers today - with the checks, the confidence and the run id attached."
        />

        {/* ---------------------------------------------------------- talk */}
        <section className="home-cta">
          <Reveal>
            <div className="home-cta-inner">
              <span className="eyebrow">[ talk to it ]</span>
              <div className="talk-cards">
                <Link href="/query" className="talk-card">
                  <span className="talk-k">query console</span>
                  <span className="talk-v">attach a scene, ask in plain language →</span>
                </Link>
                <a href={`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/docs`} className="talk-card" rel="noreferrer" target="_blank">
                  <span className="talk-k">api</span>
                  <span className="talk-v">POST /runs · stream over SSE →</span>
                </a>
              </div>
            </div>
          </Reveal>
        </section>
            </>
          }
        />
      </main>

      <footer className="home-foot">
        <div className="home-foot-hero">
          <div className="home-foot-disc" aria-hidden="true">
            <ParticleCloud tone="light" count={3400} shape="disc" tilt={-14} className="home-foot-cloud" />
          </div>
          <Reveal>
            <span className="eyebrow">[ the close ]</span>
            <h2 className="display-xl home-foot-title">
              <Words text="And every reading deserves to be checked." accent={['checked']} stagger={140} />
            </h2>
            <div className="home-ctas">
              <Link href="/query" className="pill pill-solid">
                ask the imagery
              </Link>
              <Link href="/benchmarks" className="pill pill-line">
                see the evidence
              </Link>
            </div>
          </Reveal>
        </div>
        <div className="home-foot-grid">
          <div className="home-foot-brand">
            <span className="brand">
              SatQuery<span className="accent">AI</span>
            </span>
            <p>
              Interactive vision-language assistant for remote sensing. Built for Indian
              Earth-observation products - Cartosat-2E MX, EOS-04 SAR - and the public
              benchmarks the field measures itself on.
            </p>
          </div>
          <div>
            <span className="eyebrow">product</span>
            <ul>
              <li><Link href="/query">query console</Link></li>
              <li><Link href="/models">models</Link></li>
              <li><Link href="/benchmarks">benchmarks</Link></li>
            </ul>
          </div>
          <div>
            <span className="eyebrow">explore</span>
            <ul>
              <li><a href="#capabilities">capabilities</a></li>
              <li><a href="#evidence">evidence</a></li>
              <li><a href="#method">method</a></li>
              <li><a href="#deploy">deployment</a></li>
            </ul>
          </div>
          <div>
            <span className="eyebrow">project</span>
            <ul>
              <li>
                <a href="https://github.com/aditya-coder07/SatQuery_AI" rel="noreferrer" target="_blank">
                  source on github
                </a>
              </li>
              <li>
                <a href={`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/docs`} rel="noreferrer" target="_blank">
                  api reference
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="home-foot-base">
          <span>© 2026 SatQuery AI · built to be checked</span>
          <span>every reading on this site comes from the run that produced it</span>
        </div>
      </footer>
    </>
  );
}
