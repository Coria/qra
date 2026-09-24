"""OpenAI-compatible LLM call used for the human-review draft.

The model receives structured facts and interprets them.  It is not expected
to recalculate metrics, and its output is always routed through human review.
"""

from __future__ import annotations

from openai import OpenAI


def summarize_with_llm(
    facts: dict[str, object],
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 300.0,
) -> str:
    """Generate a structured review draft with a bounded request timeout."""
    if timeout <= 0:
        raise ValueError("llm timeout must be positive")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    system_prompt = (
        "You are a rigorous quant backtest reviewer. Use only the supplied facts. "
        "If metrics could not be recomputed, say so explicitly. "
        "Do not enumerate every metric. Quote at most two or three key figures only when they support a judgement. "
        "Produce detailed Chinese output with sections: "
        "人审清单, 交易风格特征, 异常与风险, 改进方向, 特征解读, 总体结论, 总体评价."
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=6400,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": facts["prompt"]},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=timeout,
    )
    return response.choices[0].message.content or ""
