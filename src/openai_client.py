"""OpenAI client for GPT-5 nano, GPT-5-mini, and other commercial models."""
import logging
import json
import re
from typing import Dict, Any, List, Union, TYPE_CHECKING
from openai import AsyncOpenAI
from .config import settings
from . import prompt_builder

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI commercial models."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        enable_web_search: bool = False,
        reasoning_effort: str = "minimal",
    ):
        if model is None:
            model = "gpt-5-nano"

        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.enable_web_search = enable_web_search

        if "gpt-5.1" in model:
            self.reasoning_effort = "medium"
        else:
            self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(f"Initialized OpenAI client - Model: {self.model}")
        logger.info("Prompt caching enabled")
        if self.enable_web_search:
            logger.warning("Web search enabled")
        if "gpt-5" in self.model:
            logger.info(f"Reasoning effort: {self.reasoning_effort}")

    def _extract_json_manually(self, content: str) -> Dict[str, Any]:
        """Manually extract JSON values when parsing fails."""
        result = {
            "support": "no",
            "sentences": [],
            "reasoning": "Manual extraction fallback",
            "subject_mentioned": False,
            "object_mentioned": False,
        }

        support_match = re.search(
            r'"support":\s*"(yes|no|maybe)"', content, re.IGNORECASE
        )
        if support_match:
            result["support"] = support_match.group(1).lower()
        else:
            supported_match = re.search(
                r'"is_supported":\s*(true|false)', content, re.IGNORECASE
            )
            if supported_match:
                result["support"] = (
                    "yes" if supported_match.group(1).lower() == "true" else "no"
                )

        sentences_match = re.search(
            r'"sentences":\s*\[(.*?)\]', content, re.DOTALL
        )
        if sentences_match:
            raw = sentences_match.group(1)
            sents = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
            result["sentences"] = [
                s.replace('\\"', '"') for s in sents if s.strip()
            ]
        else:
            sent_match = re.search(
                r'"supporting_sentence":\s*"((?:[^"\\]|\\.)*)"',
                content,
                re.DOTALL,
            )
            if sent_match:
                s = sent_match.group(1).replace('\\"', '"')
                if s.strip():
                    result["sentences"] = [s.strip()]

        reasoning_match = re.search(
            r'"reasoning":\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL
        )
        if reasoning_match:
            r = reasoning_match.group(1).replace('\\"', '"')
            result["reasoning"] = r if r.strip() else "Manual extraction"

        subject_match = re.search(
            r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE
        )
        if subject_match:
            result["subject_mentioned"] = (
                subject_match.group(1).lower() == "true"
            )

        object_match = re.search(
            r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE
        )
        if object_match:
            result["object_mentioned"] = (
                object_match.group(1).lower() == "true"
            )

        return result

    async def generate_response(
        self, prompt: str, cacheable_prefix: str = None
    ) -> str:
        """Generate response using OpenAI API with optional prompt caching."""
        try:
            use_caching = bool(cacheable_prefix)

            if use_caching:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an expert medical researcher with a "
                                    "strong understanding of medical and biological "
                                    "semantics."
                                ),
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": cacheable_prefix,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert medical researcher with a strong "
                            "understanding of medical and biological semantics."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

            api_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 7000,
            }

            if "gpt-5" in self.model or "gpt-4" in self.model:
                api_params["response_format"] = {"type": "json_object"}

            if "gpt-5" in self.model:
                api_params["reasoning_effort"] = self.reasoning_effort

            if self.enable_web_search:
                api_params["tools"] = [{"type": "web_search"}]

            response = await self.client.chat.completions.create(**api_params)

            message = response.choices[0].message
            if hasattr(message, 'refusal') and message.refusal:
                raise Exception(f"OpenAI refused the request: {message.refusal}")

            content = message.content or ""

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                logger.info(
                    f"Token usage: prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}, "
                    f"total_tokens={total_tokens}"
                )

                cached_tokens = 0
                if (
                    hasattr(usage, 'prompt_tokens_details')
                    and usage.prompt_tokens_details
                ):
                    details = usage.prompt_tokens_details
                    cached_tokens = getattr(details, 'cached_tokens', 0)
                    if cached_tokens > 0:
                        logger.info(
                            f"Prompt cache hit: {cached_tokens} cached tokens"
                        )

            return content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")

    async def evaluate_triple_support(
        self, triple: Union[List[str], 'TripleData'], abstract: str
    ) -> Dict[str, Any]:
        """Evaluate if an abstract supports a given triple."""
        cacheable_prefix, unique_content = prompt_builder.build_evaluation_prompt_parts(
            triple, abstract
        )

        try:
            content = await self.generate_response(unique_content, cacheable_prefix)

            if not content or not content.strip():
                return {
                    "support": "no",
                    "sentences": [],
                    "reasoning": "Empty response from OpenAI",
                    "subject_mentioned": False,
                    "object_mentioned": False,
                }

            try:
                evaluation = json.loads(content)
                return evaluation
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse OpenAI response as JSON: {e}"
                )
                return self._extract_json_manually(content)

        except Exception as e:
            logger.error(f"Error during OpenAI evaluation: {e}")
            raise
