import { LinkCard } from './LinkCard';

const code = `from transformers import AutoModelForCausalLM
from kvfold import enable_compression

model = AutoModelForCausalLM.from_pretrained("gpt2")
handle = enable_compression(
    model,
    method="flash",
    target_memory=0.25,
)

output = model.generate(input_ids, max_new_tokens=64)
print(handle.stats_dict())
handle.disable()`;

export function DocsPage() {
  return (
    <main className="page-shell">
      <section className="container-x pt-32 sm:pt-40">
        <div className="max-w-3xl">
          <p className="eyebrow">Documentation · v0.2</p>
          <h1 className="page-title">Compress the cache. Keep the model.</h1>
          <p className="page-lede">
            A practical guide to installing kvfold, choosing a method, reading the memory contract,
            and validating it on your own workload. This page is maintained with the product.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a className="btn-primary light:btn-light-primary" href="?page=onboarding">Developer onboarding <span aria-hidden>→</span></a>
            <a className="btn-ghost light:btn-light-ghost" href="?page=home#install">Jump to install <span aria-hidden>↓</span></a>
          </div>
        </div>
      </section>

      <section className="container-x mt-20 grid gap-6 lg:grid-cols-[0.75fr_1.25fr]" id="start">
        <div className="space-y-4">
          <p className="eyebrow">Start here</p>
          <h2 className="page-section-title">The 60-second path</h2>
          <p className="body-copy">Use a numeric memory fraction. For example, <code>0.25</code> asks the allocator to target roughly one quarter of the original cache.</p>
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <LinkCard number="01" title="Install" body="pip install kvfold" />
            <LinkCard number="02" title="Enable" body="Call enable_compression once per model." />
            <LinkCard number="03" title="Measure" body="Inspect stats and validate quality before rollout." />
          </div>
        </div>
        <CodeBlock code={code} label="quickstart.py" />
      </section>

      <section className="container-x mt-24 grid gap-12 border-t border-white/[0.08] pt-16 light:border-ink-900/[0.08] lg:grid-cols-2" id="api">
        <div>
          <p className="eyebrow">Core concepts</p>
          <h2 className="page-section-title">Choose the trade-off deliberately.</h2>
          <div className="mt-8 space-y-5">
            <DocRow title="flash" body="Randomized-SVD path; the default starting point for CPU/MPS experimentation." />
            <DocRow title="jolt" body="Reference Tucker + JL residual path; slower, useful for comparisons." />
            <DocRow title="low" body="Rank-controlled low-rank baseline." />
            <DocRow title="int4 / int8 / fp8" body="Quantized or cast payloads for predictable memory savings." />
            <DocRow title="pass" body="Identity baseline for integration and regression checks." />
          </div>
        </div>
        <div>
          <p className="eyebrow">Runtime contract</p>
          <h2 className="page-section-title">What is verified today?</h2>
          <ul className="mt-8 space-y-4 text-sm leading-7 text-ink-300 light:text-ink-700">
            <li><span className="status-dot" />CPU and Apple MPS cache residency tests pass.</li>
            <li><span className="status-dot" />GPT-2 generation integration passes locally.</li>
            <li><span className="status-dot pending" />CUDA/Triton execution is pending a real NVIDIA runner.</li>
            <li><span className="status-dot pending" />Real vLLM execution and long-context stress are pending.</li>
          </ul>
          <div className="mt-8 rounded-2xl bg-amber-400/[0.08] p-5 text-sm leading-6 text-amber-100 ring-1 ring-amber-300/20 light:text-amber-900">
            Compression is approximate for lossy methods. Compare task quality and latency on your model before treating a target as a production setting.
          </div>
        </div>
      </section>

      <section className="container-x mt-24 border-t border-white/[0.08] pb-24 pt-16 light:border-ink-900/[0.08]" id="benchmarks">
        <p className="eyebrow">Benchmarks</p>
        <h2 className="page-section-title">Reproduce the numbers, then replace them.</h2>
        <p className="mt-5 max-w-2xl body-copy">The homepage numbers are synthetic CPU reference measurements. Use the repository benchmark scripts for deeper experiments, but treat the output as workload-specific evidence.</p>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <CommandCard title="Memory" command="python -m scripts.run_memory_benchmark --T 1024 --dh 128 --ratios 2 3 4 5 8" />
          <CommandCard title="Speed" command="python -m scripts.run_speed_benchmark --T 1024 --dh 128 --ratio 3" />
          <CommandCard title="Fidelity" command="python -m scripts.run_table2_reconstruction --T 1024 --ratio 2.0" />
        </div>
      </section>
    </main>
  );
}

function CodeBlock({ code, label }: { code: string; label: string }) {
  return <div className="code-panel"><div className="code-panel-head"><span>{label}</span><span className="text-ink-500">Python</span></div><pre><code>{code}</code></pre></div>;
}

function DocRow({ title, body }: { title: string; body: string }) {
  return <div className="border-b border-white/[0.08] pb-4 light:border-ink-900/[0.08]"><h3 className="font-mono text-sm text-blue-300 light:text-blue-700">{title}</h3><p className="mt-1 text-sm leading-6 text-ink-300 light:text-ink-700">{body}</p></div>;
}

function CommandCard({ title, command }: { title: string; command: string }) {
  return <div className="surface p-5 light:surface-light"><p className="eyebrow">{title}</p><code className="mt-4 block break-words text-xs leading-6 text-ink-200 light:text-ink-700">{command}</code></div>;
}
