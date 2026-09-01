"""
Thin wrapper around the Groq API that always returns timing, token, and
cost numbers alongside the answer -- since instrumentation is the actual
point of this project, every LLM call goes through here rather than
being called ad hoc from each variant.
"""

import time
from dataclasses import dataclass

from groq import Groq

from . import config

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it as an environment variable "
                "(locally: export GROQ_API_KEY=... ; on Render: set it in "
                "the service's Environment tab). Never hardcode it in code."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


@dataclass
class LLMCallResult:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a B.Tech Computer Science (AI/ML specialization) curriculum, using ONLY the context provided below. If the context does not contain enough information to answer, say so plainly rather than guessing.

Context:
{context}

Question: {question}

Answer concisely and accurately, grounded only in the context above."""


def call_llm(question: str, context_chunks: list) -> LLMCallResult:
    """
    context_chunks: list of chunk text strings (already retrieved).
    Returns an LLMCallResult with the answer plus every instrumentation
    field this project needs. Token counts come from Groq's own API
    response usage field -- the actual count the model saw, not an
    estimate -- so cost numbers are exact, not approximated.
    """
    context = "\n\n---\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature: this is a factual-QA benchmark,
        # not a creative task, and low temperature makes answers more
        # reproducible across repeated runs of the same question.
        reasoning_effort="low",  # openai/gpt-oss-20b is a reasoning model
        # and defaults to reasoning_effort="medium", which spends extra
        # output tokens on an internal reasoning pass before answering --
        # those reasoning tokens are billed as completion tokens, so left
        # at the default they would inflate this benchmark's cost/latency
        # numbers for what is a simple factual-QA-over-provided-context
        # task that doesn't need deep multi-step reasoning. "low" keeps
        # the cost/latency profile representative of a cheap small-model
        # workload, which is what this project is actually benchmarking.
        include_reasoning=False,  # don't bother returning the reasoning
        # trace in the response at all -- we're not displaying or logging
        # it, so there's no reason to pay the (small) response-payload
        # and parsing overhead of including it.
    )
    latency = time.perf_counter() - start

    answer = response.choices[0].message.content
    usage = response.usage
    if usage is None:
        # Groq's SDK types this as Optional -- shouldn't happen for a
        # standard (non-streaming) call like this one, but if it ever
        # does, fail loudly rather than silently logging wrong (zero)
        # cost/token numbers, since accurate cost tracking is the whole
        # point of this project.
        raise RuntimeError(
            "Groq API response had no usage data -- cannot compute accurate "
            "token counts or cost for this call. Check the Groq API/SDK version."
        )
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    cost = (
        input_tokens * config.GROQ_PRICE_PER_INPUT_TOKEN
        + output_tokens * config.GROQ_PRICE_PER_OUTPUT_TOKEN
    )

    return LLMCallResult(
        answer=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        cost_usd=cost,
    )
