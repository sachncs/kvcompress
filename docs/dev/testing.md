# Testing

## Layout

- `tests/unit/` — fast tests, no network, no model downloads.
- `tests/integration/` — tests that download real HF models (GPT-2).
- `tests/property/` — Hypothesis property tests.
- `tests/fixtures/` — shared fixtures.

## Conventions

- File naming: `<module>_test.py` for unit tests;
  `test_<integration_target>.py` for integration tests.
- One assertion concept per test.
- Mark slow / integration / GPU tests with `@pytest.mark.<marker>`.
- Coverage of `kvcompress/` is targeted at ≥ 95%.

## Running

```bash
# Unit only — fast, run on every save.
pytest -m "not slow and not integration and not gpu"

# All tests.
pytest

# Coverage report.
pytest --cov=kvcompress --cov-report=term-missing
```

## Property tests

We use [Hypothesis](https://hypothesis.readthedocs.io/) for property
testing. Common patterns:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=8), st.integers(min_value=4, max_value=16))
def test_int_quant_roundtrip_property(bits, last):
    torch.manual_seed(0)
    x = torch.randn(8, last) * 4.0
    q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
    packed, scale, zp = q.quantize(x)
    x_hat = q.dequantize(packed, scale, zp)
    err = (x - x_hat).abs().max().item()
    bin_size = (x.abs().amax().item() / q._qmax)
    assert err <= bin_size + 1e-3
```

Property tests are best for invariants like round-trip bounds; use unit
tests for specific behaviours and edge cases.

## Fixtures

Session-scoped fixtures live in `tests/integration/conftest.py` and
download the model once per test run.

```python
@pytest.fixture(scope="session")
def gpt2_model_with_pad():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tok, model
```

## Mocking Hugging Face

For unit tests that exercise the adapter without a real model, use the
`_FakeModel` pattern from `tests/unit/adapter_test.py`:

```python
class _FakeModel:
    def __init__(self, model_type="llama"):
        class _Config: pass
        self.config = _Config()
        self.config.model_type = model_type
        class _GenConfig: cache_implementation = None
        self.generation_config = _GenConfig()
```

## Coverage

The CI gate fails if `coverage report --fail-under=80` exits non-zero.
We aim higher (95%+) but don't gate on it to keep velocity.