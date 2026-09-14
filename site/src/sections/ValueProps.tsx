import { Reveal } from '../lib/Reveal';
import { valueProps } from '../lib/content';

export function ValueProps() {
  return (
    <section className="relative py-24 sm:py-32">
      <div className="container-x">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <span className="chip light:chip-light">Why kvfold</span>
            <h2 className="mt-5 text-balance text-display-2 font-semibold tracking-tighter">
              Built for the way you actually run inference.
            </h2>
            <p className="mt-5 text-base text-ink-300 light:text-ink-600">
              A library, not a model. A contract, not a convention. Drop it in, set a target, ship.
            </p>
          </div>
        </Reveal>

        <div className="mx-auto mt-16 grid max-w-5xl grid-cols-1 gap-4 sm:grid-cols-2">
          {valueProps.map((p, i) => (
            <Reveal key={p.title} delay={i * 0.06}>
              <div className="group relative h-full overflow-hidden rounded-3xl bg-white/[0.02] p-7 ring-1 ring-white/[0.06] transition-all duration-500 ease-out-expo hover:bg-white/[0.04] hover:ring-white/10 light:bg-white light:ring-ink-900/[0.06] light:hover:ring-ink-900/10">
                <div
                  aria-hidden
                  className="pointer-events-none absolute -right-20 -top-20 h-40 w-40 rounded-full bg-accent-500/10 blur-3xl transition-opacity duration-700 group-hover:opacity-100 opacity-60"
                />
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-500/10 text-accent-300 ring-1 ring-accent-500/20 light:bg-accent-500/10 light:text-accent-600">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6">
                    <path d="M5 12l4 4L19 6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <h3 className="mt-5 text-lg font-semibold tracking-tight light:text-ink-900">
                  {p.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-400 light:text-ink-600">
                  {p.description}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}