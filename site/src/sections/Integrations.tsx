import { Reveal } from '../lib/Reveal';
import { integrations } from '../lib/content';

export function Integrations() {
  return (
    <section id="integrations" className="relative py-24 sm:py-32">
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">Integrations</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              Plays nicely with your stack.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              Three opinionated paths — choose the one that matches how you ship.
            </p>
          </div>
        </Reveal>

        <div className="mx-auto mt-14 grid max-w-5xl grid-cols-1 gap-4 md:grid-cols-3">
          {integrations.map((it, i) => (
            <Reveal key={it.name} delay={i * 0.07}>
              <article className="group relative h-full overflow-hidden rounded-3xl bg-white/[0.02] p-7 ring-1 ring-white/[0.06] transition-all duration-500 hover:bg-white/[0.04] hover:ring-white/10 light:bg-white light:ring-ink-900/[0.06] light:hover:ring-ink-900/10">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold tracking-tight light:text-ink-900">
                    {it.name}
                  </h3>
                  <span className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
                    {it.badge}
                  </span>
                </div>
                <p className="mt-4 text-sm leading-relaxed text-ink-400 light:text-ink-600">
                  {it.description}
                </p>
                <div className="mt-6 h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent light:via-ink-900/10" />
                <div className="mt-4 flex items-center justify-between text-[11px] uppercase tracking-wider text-ink-500">
                  <span>verified</span>
                  <span className="inline-flex items-center gap-1 normal-case tracking-normal text-emerald-300 light:text-emerald-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    end-to-end
                  </span>
                </div>
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}