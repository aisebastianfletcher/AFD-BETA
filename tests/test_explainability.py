"""
Test explainability features in AFDInfinityAMI.
"""
import pytest
from afd_ami_core import AFDInfinityAMI


def test_explainability_snapshot(monkeypatch):
    """Test that explainability snapshot is created and contains all required fields."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)
    
    # Patch neutralizer and renderer to avoid network/model downloads
    monkeypatch.setattr(ami, "_neutralize_with_hf", lambda prompt: "neutral: " + prompt)
    monkeypatch.setattr(ami, "_render_with_hf", lambda neutral, directives: "Rendered based on: " + neutral)
    
    # Patch sentiment analyzer to return a stable neutral score
    monkeypatch.setattr(ami, "_ensure_sentiment_analyzer", lambda: lambda text: [{"label": "NEUTRAL", "score": 0.5}])
    
    # Call respond
    resp, coherence, reflection = ami.respond("Test prompt for explainability")
    
    # Check explainability snapshot exists and has all required fields
    explainability = ami.get_last_explainability()
    
    assert explainability, "Explainability snapshot should not be empty"
    assert "timestamp" in explainability
    assert "original_prompt" in explainability
    assert "neutral_prompt" in explainability
    assert "neutralizer_system" in explainability
    assert "renderer_system" in explainability
    assert "renderer_used" in explainability
    assert "renderer_prompt" in explainability
    assert "sentiment_label" in explainability
    assert "sentiment_score" in explainability
    assert "state" in explainability
    assert "action" in explainability
    assert "s_prime" in explainability
    assert "interp_s" in explainability
    assert "afd_metrics" in explainability
    assert "coherence" in explainability
    assert "coefficients" in explainability
    assert "final_text" in explainability
    
    # Check that values are correct types
    assert isinstance(explainability["original_prompt"], str)
    assert explainability["original_prompt"] == "Test prompt for explainability"
    assert isinstance(explainability["sentiment_score"], float)
    assert isinstance(explainability["state"], list)
    assert isinstance(explainability["action"], list)
    assert isinstance(explainability["afd_metrics"], dict)
    assert isinstance(explainability["coefficients"], dict)


def test_get_last_explainability_empty():
    """Test that get_last_explainability returns empty dict before any respond() call."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)
    explainability = ami.get_last_explainability()
    assert explainability == {}


def test_generate_reflection():
    """Test that generate_reflection creates a human-readable summary."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)
    
    reflection = ami.generate_reflection(
        prompt="Test prompt",
        neutral_prompt="neutral test prompt",
        sentiment_label="POSITIVE",
        sentiment_score=0.85,
        coherence=0.75,
        metrics={"harmony": 0.5, "info_gradient": 0.3, "oscillation": 0.1, "potential": 0.2},
        final_text="This is the final response"
    )
    
    assert isinstance(reflection, str)
    assert len(reflection) > 0
    assert "POSITIVE" in reflection
    assert "0.850" in reflection or "0.85" in reflection
    assert "Coherence" in reflection
    assert "0.75" in reflection or "0.7500" in reflection


def test_build_renderer_prompt():
    """Test that _build_renderer_prompt formats correctly."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)
    
    neutral_input = "neutral test input"
    afd_directives = "AFD metrics: coherence=0.5"
    
    renderer_prompt = ami._build_renderer_prompt(neutral_input, afd_directives)
    
    assert isinstance(renderer_prompt, str)
    assert "neutral test input" in renderer_prompt
    assert "AFD metrics: coherence=0.5" in renderer_prompt
    assert "Neutral input:" in renderer_prompt
    assert "AFD directives:" in renderer_prompt


def test_reflection_log_updated(monkeypatch):
    """Test that reflection log gets updated with new reflection on each respond() call."""
    ami = AFDInfinityAMI(use_openai=False, openai_api_key=None)
    
    # Patch neutralizer and renderer
    monkeypatch.setattr(ami, "_neutralize_with_hf", lambda prompt: "neutral: " + prompt)
    monkeypatch.setattr(ami, "_render_with_hf", lambda neutral, directives: "Rendered")
    monkeypatch.setattr(ami, "_ensure_sentiment_analyzer", lambda: lambda text: [{"label": "NEUTRAL", "score": 0.5}])
    
    initial_log_size = len(ami.reflection_log)
    
    # Call respond
    ami.respond("Test prompt")
    
    # Check that reflection log was updated
    assert len(ami.reflection_log) > initial_log_size
    latest_reflection = ami.get_latest_reflection()
    assert isinstance(latest_reflection, str)
    assert len(latest_reflection) > 0
