"""A retired LLM_MODEL (e.g. left in a deployment's secrets) must not break every question."""
from types import SimpleNamespace

import groq
import httpx
import pytest

from docmind import config
from docmind.llm import LLM


def not_found():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.NotFoundError("model not found", response=httpx.Response(404, request=request), body=None)


class FakeCompletions:
    def __init__(self, dead_models):
        self.dead, self.calls = set(dead_models), []

    def create(self, model, **kwargs):
        self.calls.append(model)
        if model in self.dead:
            raise not_found()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer from {model} [1]"))])


def llm_with(model, dead):
    llm = LLM("gsk_test", model)
    completions = FakeCompletions(dead)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


CHUNKS = [{"text": "Gift cards never expire.", "source": "a.pdf", "page": 1}]


def test_retired_model_switches_to_fallback_once():
    llm, calls = llm_with("llama-3.3-70b-versatile", {"llama-3.3-70b-versatile"})
    assert llm.answer("Do gift cards expire?", CHUNKS) == f"answer from {config.FALLBACK_MODEL} [1]"
    assert llm.model == config.FALLBACK_MODEL  # later questions go straight to the working model
    llm.answer("Again?", CHUNKS)
    assert calls.calls == ["llama-3.3-70b-versatile", config.FALLBACK_MODEL, config.FALLBACK_MODEL]


def test_error_is_raised_when_fallback_is_also_missing():
    llm, _ = llm_with("old-model", {"old-model", config.FALLBACK_MODEL})
    with pytest.raises(groq.NotFoundError):
        llm.answer("Do gift cards expire?", CHUNKS)


def test_working_model_is_used_as_is():
    llm, calls = llm_with("qwen/qwen3.8-27b", set())
    llm.answer("Do gift cards expire?", CHUNKS)
    assert calls.calls == ["qwen/qwen3.8-27b"]
