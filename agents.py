"""The agents: the real OpenAI agent and a simulated agent used for testing.

Both have the same method: ask(prompt) returns a dictionary with the answer
text and the token counts. The rest of the program does not know which one it uses.
"""

import random
import re
import threading
import time

import settings


class BudgetExceeded(Exception):
    """Raised when the estimated cost reaches the limit."""


class CostTracker:
    """Adds up tokens and estimated cost. Safe to use from several threads."""

    def __init__(self, limit_usd):
        self.limit = limit_usd
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0  # includes hidden reasoning tokens
        self.reasoning_tokens = 0
        self.latency_total = 0.0
        self._lock = threading.Lock()

    @property
    def cost(self):
        return (
            self.prompt_tokens * settings.PRICE_INPUT_PER_M
            + self.completion_tokens * settings.PRICE_OUTPUT_PER_M
        ) / 1_000_000

    def add(self, prompt_tokens, completion_tokens, reasoning_tokens, latency):
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.latency_total += latency

    def check(self):
        if self.cost >= self.limit:
            raise BudgetExceeded(
                f"estimated cost ${self.cost:.2f} reached the limit ${self.limit:.2f}"
            )


class OpenAIAgent:
    """Calls the OpenAI chat completions endpoint."""

    def __init__(self, model):
        import openai  # imported here so that tests do not need the package

        self._openai = openai
        self.model = model
        self.client = openai.OpenAI(
            max_retries=0, timeout=settings.REQUEST_TIMEOUT_S
        )

    def ask(self, prompt):
        openai = self._openai
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        if settings.REASONING_EFFORT:
            kwargs["reasoning_effort"] = settings.REASONING_EFFORT

        for attempt in range(settings.MAX_API_RETRIES + 1):
            started = time.time()
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ):
                if attempt == settings.MAX_API_RETRIES:
                    raise
                time.sleep(min(60, 2**attempt) + random.random())
        latency = time.time() - started

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        choice = response.choices[0]
        return {
            "text": choice.message.content or "",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": reasoning,
            "latency": latency,
            "finish_reason": choice.finish_reason,
            "model_returned": response.model,
        }


class MockAgent:
    """A simulated agent for testing the program without any cost.
    It answers at random, and sometimes gives an unreadable answer."""

    model = "mock"

    def __init__(self, seed=1, bad_answer_rate=0.04):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.bad_answer_rate = bad_answer_rate

    def ask(self, prompt):
        with self._lock:
            bad = self._rng.random() < self.bad_answer_rate
            pick_a = self._rng.random() < 0.5
            amount = self._rng.random()
        if bad:
            text = "I am not sure what to choose."
        elif '"contribution"' in prompt:
            top = int(re.search(r"from 0 to (\d+)", prompt).group(1))
            text = '{"contribution": %d, "reason": "test"}' % round(amount * top)
        else:
            text = '{"choice": "%s", "reason": "test"}' % ("A" if pick_a else "B")
        return {
            "text": text,
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": 40,
            "reasoning_tokens": 0,
            "latency": 0.01,
            "finish_reason": "stop",
            "model_returned": "mock",
        }
