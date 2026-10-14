const setup = `git clone https://github.com/sachncs/kvcompress
cd kvcompress
uv sync --locked --extra dev --extra bench --extra docs`;

const checks = `uv run pytest -m "not slow and not integration and not gpu"
uv run pytest -m integration
uv run ruff check kvfold tests examples scripts
uv run mypy kvfold`;

export function OnboardingPage() {
  return (
    <main className="page-shell">
      <section className="container-x pt-32 sm:pt-40">
        <div className="max-w-3xl">
          <p className="eyebrow">Developer onboarding</p>
          <h1 className="page-title">A clean path from clone to contribution.</h1>
          <p className="page-lede">This is the working agreement for the repository: reproducible setup, explicit scope, small changes, and evidence before claims.</p>
        </div>
      </section>

      <section className="container-x mt-16 grid gap-5 md:grid-cols-3">
        <OnboardingCard number="01" title="Set up" body="Python 3.11–3.13, PyTorch ≥2.5, uv, and a clean editable environment." />
        <OnboardingCard number="02" title="Understand" body="Start with the cache adapter, Pool, compressor interface, and device boundaries." />
        <OnboardingCard number="03" title="Prove" body="Run the narrowest relevant tests first, then the full local gate before opening a PR." />
      </section>

      <section className="container-x mt-20 grid gap-8 lg:grid-cols-2" id="setup">
        <CodePanel label="terminal · setup" code={setup} />
        <CodePanel label="terminal · checks" code={checks} />
      </section>

      <section className="container-x mt-24 grid gap-12 border-t border-white/[0.08] pb-24 pt-16 light:border-ink-900/[0.08] lg:grid-cols-2" id="contributing">
        <div>
          <p className="eyebrow">Contribution loop</p>
          <h2 className="page-section-title">Make the change easy to review.</h2>
          <ol className="mt-8 space-y-6">
            <Step n="01" title="State the contract" body="Describe the user-visible behavior and the device or framework boundary it affects." />
            <Step n="02" title="Add the smallest proof" body="Pair new behavior with a focused unit, integration, or residency test. Do not inflate claims from synthetic numbers." />
            <Step n="03" title="Run the gates" body="Format, lint, type-check, test, inspect the diff, and update the product/docs surface if semantics changed." />
            <Step n="04" title="Call out pending hardware" body="CUDA/Triton, real vLLM, and long-context claims require the matching execution environment." />
          </ol>
        </div>
        <div id="security">
          <p className="eyebrow">Repository map</p>
          <h2 className="page-section-title">Where to work.</h2>
          <div className="mt-8 grid gap-3 sm:grid-cols-2">
            <MapCard path="kvfold/api.py" body="Public API and lifecycle handle." />
            <MapCard path="kvfold/adapter/" body="Framework integration boundaries." />
            <MapCard path="kvfold/core/" body="Compressor implementations and allocation." />
            <MapCard path="kvfold/store/" body="Compressed cache storage and metadata." />
            <MapCard path="tests/" body="Unit, integration, MPS, and property evidence." />
            <MapCard path="site/src/" body="Product surface and standalone docs." />
          </div>
          <div className="mt-8 rounded-2xl bg-white/[0.04] p-5 text-sm leading-6 text-ink-300 ring-1 ring-white/[0.08] light:bg-ink-900/[0.04] light:text-ink-700 light:ring-ink-900/[0.08]">
            Security reports should use the repository’s private reporting process. Do not include model weights, prompts, generated text, or cache contents in public issues.
          </div>
        </div>
      </section>
    </main>
  );
}

function CodePanel({ label, code }: { label: string; code: string }) {
  return <div className="code-panel"><div className="code-panel-head"><span>{label}</span><span className="text-ink-500">copy into your shell</span></div><pre><code>{code}</code></pre></div>;
}

function OnboardingCard({ number, title, body }: { number: string; title: string; body: string }) {
  return <div className="surface p-6 light:surface-light"><span className="font-mono text-xs text-blue-300 light:text-blue-700">{number}</span><h2 className="mt-8 text-lg font-semibold">{title}</h2><p className="mt-2 text-sm leading-6 text-ink-400 light:text-ink-600">{body}</p></div>;
}

function Step({ n, title, body }: { n: string; title: string; body: string }) {
  return <li className="flex gap-4"><span className="font-mono text-xs text-blue-300 light:text-blue-700">{n}</span><div><h3 className="font-medium">{title}</h3><p className="mt-1 text-sm leading-6 text-ink-400 light:text-ink-600">{body}</p></div></li>;
}

function MapCard({ path, body }: { path: string; body: string }) {
  return <div className="rounded-2xl border border-white/[0.08] p-4 light:border-ink-900/[0.08]"><code className="text-xs text-blue-300 light:text-blue-700">{path}</code><p className="mt-2 text-xs leading-5 text-ink-400 light:text-ink-600">{body}</p></div>;
}
