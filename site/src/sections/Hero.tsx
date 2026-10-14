import { motion, useReducedMotion } from 'framer-motion';
import { hero } from '../lib/content';
import { Reveal } from '../lib/Reveal';

function HeroCode() {
  const reduced = useReducedMotion();
  return (
    <div className="surface light:surface-light p-5 sm:p-6">
      <div className="flex items-center gap-1.5 pb-4">
        <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
        <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
        <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
        <span className="ml-3 text-xs text-ink-400 light:text-ink-500">kvfold · quickstart</span>
      </div>
      <pre className="code overflow-x-auto text-[13px] leading-7 text-ink-200 light:text-ink-800">
        <code>
          {hero.codeLines.map((tok, i) => {
            if (tok.tone === 'kw')
              return (
                <span key={i} className="text-accent-300 light:text-accent-600">
                  {tok.text}
                </span>
              );
            if (tok.tone === 'str')
              return (
                <span key={i} className="text-emerald-300/90 light:text-emerald-600">
                  {tok.text}
                </span>
              );
            if (tok.tone === 'fn')
              return (
                <span key={i} className="text-amber-300/90 light:text-amber-600">
                  {tok.text}
                </span>
              );
            if (tok.tone === 'num')
              return (
                <span key={i} className="text-sky-300/90 light:text-sky-600">
                  {tok.text}
                </span>
              );
            return (
              <span key={i} className="text-ink-200 light:text-ink-800">
                {tok.text}
              </span>
            );
          })}
        </code>
      </pre>
      <motion.div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-ink-900/80 to-transparent light:from-white/80"
        initial={reduced ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.6 }}
      />
    </div>
  );
}

export function Hero() {
  const reduced = useReducedMotion();
  return (
    <section id="top" className="relative overflow-hidden pt-32 pb-24 sm:pt-40 sm:pb-32">
      {/* Background grid + glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 grid-bg opacity-60 light:grid-bg-light"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 -z-10 h-[640px] w-[1100px] -translate-x-1/2 bg-radial-fade"
      />
      <div className="container-x">
        <Reveal>
          <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
            <span className="chip light:chip-light">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              {hero.eyebrow}
            </span>
            <h1 className="mt-7 text-balance text-display-1 font-semibold tracking-tightest">
              {hero.title}
              <span className="text-accent-400 light:text-accent-500">.</span>
            </h1>
            <p className="mt-6 max-w-2xl text-balance text-base text-ink-300 light:text-ink-600 sm:text-lg">
              {hero.subtitle}
            </p>
            <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
              <a href={hero.primaryCta.href} className="btn-primary light:btn-light-primary">
                {hero.primaryCta.label}
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M13 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </a>
              <a
                href={hero.secondaryCta.href}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost light:btn-light-ghost"
              >
                {hero.secondaryCta.label}
              </a>
            </div>
            <p className="mt-4 text-xs text-ink-500 light:text-ink-500">
              Python 3.11–3.13 · Apache-2.0 · independent paper implementation
            </p>
          </div>
        </Reveal>

        <Reveal delay={0.15} className="mx-auto mt-16 max-w-3xl sm:mt-20">
          <div className="relative">
            <div
              aria-hidden
              className="pointer-events-none absolute -inset-x-10 -inset-y-6 -z-10 rounded-[2.5rem] bg-gradient-to-b from-accent-500/15 via-accent-500/5 to-transparent blur-2xl"
            />
            <HeroCode />
          </div>
        </Reveal>

        {/* Animated metrics row */}
        <Reveal delay={0.3} className="mx-auto mt-14 max-w-4xl">
          <div className="relative overflow-hidden rounded-2xl bg-white/[0.02] ring-1 ring-white/[0.08] light:bg-white light:ring-ink-900/[0.06]">
            <div
              aria-hidden
              className="pointer-events-none absolute inset-y-0 left-0 w-1/2 bg-gradient-to-r from-accent-500/[0.06] to-transparent"
            />
            <dl className="relative grid grid-cols-2 divide-x divide-white/[0.06] light:divide-ink-900/[0.06] sm:grid-cols-4">
              {[
                { k: '4×', v: 'max compression' },
                { k: '400+', v: 'tests in repo' },
                { k: 'CPU', v: 'reference benchmarks' },
                { k: '0', v: 'model surgery' },
              ].map((m) => (
                <div
                  key={m.v}
                  className="flex flex-col items-center gap-1 px-4 py-5 text-center sm:py-6"
                >
                  <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-ink-500">
                    {m.v}
                  </dt>
                  <dd className="text-2xl font-semibold tracking-tightest text-ink-50 light:text-ink-900 sm:text-3xl">
                    {m.k}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </Reveal>

        {!reduced && (
          <motion.div
            aria-hidden
            className="mx-auto mt-12 flex w-max items-center gap-2 text-xs text-ink-500 light:text-ink-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2, duration: 0.8 }}
          >
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
            v0.2.0 just shipped · rebrand + architectural refactor
          </motion.div>
        )}
      </div>
    </section>
  );
}
