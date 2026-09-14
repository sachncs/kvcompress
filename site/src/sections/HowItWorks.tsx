import { Reveal } from '../lib/Reveal';
import { howSteps, codeSnippets } from '../lib/content';

export function HowItWorks() {
  return (
    <section id="how" className="relative py-24 sm:py-32">
      <div aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 grid-bg opacity-30 light:grid-bg-light"
      />
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">How it works</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              Four lines of code. Four concepts.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              The whole surface area is small by design. If you know how transformers.cache works,
              you already know how kvfold works.
            </p>
          </div>
        </Reveal>

        <div className="mx-auto mt-16 grid max-w-6xl grid-cols-1 gap-10 lg:grid-cols-2 lg:gap-16">
          {/* Steps */}
          <ol className="space-y-3">
            {howSteps.map((s, i) => (
              <Reveal key={s.n} delay={i * 0.06}>
                <li className="group relative grid grid-cols-[auto_1fr] gap-5 rounded-2xl p-5 ring-1 ring-white/[0.05] transition-all duration-500 hover:bg-white/[0.02] hover:ring-white/10 light:ring-ink-900/[0.05] light:hover:bg-white light:hover:ring-ink-900/10">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] font-mono text-xs text-ink-300 ring-1 ring-white/10 light:bg-ink-900/[0.04] light:text-ink-700 light:ring-ink-900/10">
                    {s.n}
                  </div>
                  <div>
                    <h3 className="text-base font-semibold tracking-tight light:text-ink-900">
                      {s.title}
                    </h3>
                    <p className="mt-1.5 text-sm leading-relaxed text-ink-400 light:text-ink-600">
                      {s.body}
                    </p>
                  </div>
                </li>
              </Reveal>
            ))}
          </ol>

          {/* Code */}
          <Reveal delay={0.15}>
            <div className="sticky top-24 surface light:surface-light p-5 sm:p-6">
              <div className="flex items-center justify-between pb-4">
                <div className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
                  <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
                  <span className="h-2.5 w-2.5 rounded-full bg-ink-700/80 light:bg-ink-200" />
                </div>
                <span className="text-xs text-ink-500">end_to_end.py</span>
              </div>
              <pre className="code overflow-x-auto text-[13px] leading-7 text-ink-200 light:text-ink-800">
                <code>{codeSnippets.quickstart}</code>
              </pre>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}