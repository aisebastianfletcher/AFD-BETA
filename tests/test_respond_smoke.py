import os
import pytest
from afd_ami_core import AFDInfinityAMI

def test_respond_smoke(monkeypatch):
    """Smoke test for AFDInfinityAMI.respond that avoids heavy model downloads by patching LLM methods."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)

    # Patch neutralizer and renderer to avoid network/model downloads
    monkeypatch.setattr(ami, "_neutralize_with_hf", lambda prompt: "neutral: " + prompt)
    monkeypatch.setattr(ami, "_render_with_hf", lambda neutral, directives: "Rendered based on: " + neutral)

    # Patch sentiment analyzer to return a stable neutral score
    monkeypatch.setattr(ami, "sentiment_analyzer", lambda text: [{"label": "NEUTRAL", "score": 0.5}])

    resp, coherence, reflection = ami.respond("Test prompt")

    assert isinstance(resp, str)
    assert isinstance(coherence, float)
    assert isinstance(reflection, str)