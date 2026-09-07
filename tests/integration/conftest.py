"""Integration test fixtures.

These tests require network access to download small HF models. They
are marked @pytest.mark.integration so the default ``pytest -m 'not
integration'`` excludes them.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 (small, 124M params) for end-to-end tests."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tok, model


@pytest.fixture(scope="session")
def gpt2_model_with_pad():
    """Load GPT-2 with a pad token set."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tok, model
