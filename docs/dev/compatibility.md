# Compatibility matrix

This matrix separates what is tested in the repository from what is only
expected to work. A supported dependency range is not a substitute for an
end-to-end test on the target runtime.

| Surface | Declared range | Local verification | CI verification | Status |
| --- | --- | --- | --- | --- |
| Python | 3.11–3.13 | 3.12.14 | Ubuntu and macOS matrix | Supported |
| PyTorch | `>=2.5,<3` | 2.11.0 CPU + Apple MPS | CPU install; MPS integration lane is available locally | Supported for CPU/MPS paths |
| Transformers | `>=4.45,<5` | 4.57.6 + GPT-2 integration | Fast tests | Supported for tested cache API |
| vLLM offload | `>=0.19,<1.0` API | Not installed locally | No GPU runner | Untested end-to-end |
| CUDA/Triton | Optional | No CUDA device | Manual self-hosted lane defined | Pending CUDA hardware validation |
| Browser | Chromium | Playwright Chromium | Playwright Chromium | Smoke-tested |

## Release policy

- A release must record the exact Python, PyTorch, Transformers, CUDA, GPU,
  vLLM, and browser versions used for integration or benchmark evidence.
- vLLM support is opt-in. `Offload` validates the imported handler surface and
  rejects a known unsupported vLLM major/minor version before use.
- CPU and Apple MPS claims are backed by local test runs. CUDA/Triton and
  vLLM claims remain provisional until a real CUDA hardware report exists.
- Adding a dependency range requires either a test matrix entry or an
  explicit documented reason why the range cannot be exercised in CI.
