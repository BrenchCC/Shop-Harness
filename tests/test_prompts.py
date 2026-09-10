"""Central prompt package regression tests."""

from shopharness.prompts import (
    AFTERSALE_PROMPT,
    BASE_PROMPT,
    EXTRACT_PROMPT,
    PROPOSE_PROMPT,
    RESEARCH_PROMPT,
    SUMMARY_PROMPT,
)


def test_prompt_templates_are_centralized():
    """Verify every model prompt is available from shopharness.prompts."""
    prompts = (
        AFTERSALE_PROMPT,
        BASE_PROMPT,
        EXTRACT_PROMPT,
        PROPOSE_PROMPT,
        RESEARCH_PROMPT,
        SUMMARY_PROMPT,
    )
    assert all(prompt.strip() for prompt in prompts)
