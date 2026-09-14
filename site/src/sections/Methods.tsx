import { Reveal } from '../lib/Reveal';
import { methods } from '../lib/content';

export function Methods() {
  return (
    <section id="methods" className="relative py-24 sm:py-32">
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">Methods</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              One interface. Every strategy.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              Pick the trade-off that matches your workload. Mix and match — the allocator handles
              the per-cell decisions.
            </p>
          </div>
        </Reveal>

        <div className="mx-auto mt-14 grid max-w-6xl grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {methods.map((m, i) => (
            <Reveal key={m.name} delay={i * 0.05}>
              <article
                className={`group relative flex h-full flex-col overflow-hidden rounded-3xl p-6 transition-all duration-500 ease-out-expo ${
                  m.recommended
                    ? 'bg-gradient-to-b from-accent-500/[0.08] to-transparent ring-1 ring-accent-500/20 light:from-accent-500/[0.06]'
                    : 'bg-white/[0.02] ring-1 ring-white/[0.06] hover:ring-white/10 light:bg-white light:ring-ink-900/[0.06] light:hover:ring-ink-900/10'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <span className="inline-flex h-9 items-center rounded-lg bg-white/[0.04] px-2.5 font-mono text-xs text-ink-100 ring-1 ring-white/10 light:bg-ink-900/[0.04] light:text-ink-900 light:ring-ink-900/10">
                      {m.name}
                    </span>
                    {m.recommended && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-accent-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-accent-200 ring-1 ring-accent-500/30 light:text-accent-700">
                        <span className="h-1 w-1 rounded-full bg-accent-300" />
                        recommended
                      </span>
                    )}
                  </div>
                  <span className="font-mono text-xs text-ink-500 light:text-ink-500">
                    {m.ratio}
                  </span>
                </div>
                <h3 className="mt-5 text-xl font-semibold tracking-tight light:text-ink-900">
                  {m.label}
                </h3>
                <p className="mt-2 flex-1 text-sm leading-relaxed text-ink-400 light:text-ink-600">
                  {m.description}
                </p>
                <div className="mt-6 flex items-center justify-between border-t border-white/[0.05] pt-4 text-[11px] uppercase tracking-wider text-ink-500 light:border-ink-900/[0.06]">
                  <span>fidelity</span>
                  <span className="font-mono normal-case tracking-normal text-ink-300 light:text-ink-700">
                    {m.quality}
                  </span>
                </div>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2}>
          <div className="mx-auto mt-10 max-w-3xl rounded-2xl bg-white/[0.02] p-5 ring-1 ring-white/[0.06] light:bg-ink-900/[0.03] light:ring-ink-900/[0.06]">
            <pre className="code overflow-x-auto text-ink-200 light:text-ink-800">
              <code>
                <span className="text-accent-300 light:text-accent-600">enable_compression</span>(model, method=
                <span className="text-emerald-300 light:text-emerald-600">"flash"</span>, ratio=
                <span className="text-amber-300 light:text-amber-600">3.0</span>)
                {'\n'}
                <span className="text-accent-300 light:text-accent-600">enable_compression</span>(model, method=
                <span className="text-emerald-300 light:text-emerald-600">"int4"</span>, per_channel=
                <span className="text-amber-300 light:text-amber-600">True</span>)
              </code>
            </pre>
          </div>
        </Reveal>
      </div>
    </section>
  );
}