"""vLLM client via OpenAI-compatible API."""
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


class VLLMClient:
    """Client for vLLM-served models via OpenAI-compatible API."""

    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, model: str = None, base_url: str = None):
        if model is None:
            model = "hermes4-vllm"

        self.model = model
        self.base_url = (base_url or settings.vllm_base_url).rstrip('/')

        if not self.base_url.endswith('/v1'):
            self.base_url = self.base_url + '/v1'

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            timeout=settings.request_timeout,
        )

        self.server_max_model_len = self._query_max_model_len()

        logger.info(f"Initialized vLLM client - Model: {self.model}")
        logger.info(f"vLLM server: {self.base_url}")
        logger.info(f"Server max_model_len: {self.server_max_model_len}")

    def _query_max_model_len(self) -> int:
        """Query the vLLM server for its max_model_len at startup."""
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=10)
            data = resp.json()
            for m in data.get("data", []):
                if m.get("id") == self.model:
                    return m.get("max_model_len", 0)
            if data.get("data"):
                return data["data"][0].get("max_model_len", 0)
        except Exception as e:
            logger.warning(f"Could not query server max_model_len: {e}")
        return 0

    def _fix_json_formatting(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses."""
        json_str = json_str.strip()

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

        json_str = re.sub(r"'([^']*)'(\s*:\s*)", r'"\1"\2', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)

        return json_str

    def _extract_json_manually(self, content: str) -> Dict[str, Any]:
        """Manually extract JSON values when parsing fails."""
        result = {
            "support": "no",
            "sentences": [],
            "reasoning": "Manual extraction fallback",
            "subject_mentioned": False,
            "object_mentioned": False,
        }

        support_match = re.search(r'"support":\s*"(yes|no|maybe)"', content, re.IGNORECASE)
        if support_match:
            result["support"] = support_match.group(1).lower()
        else:
            supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
            if supported_match:
                result["support"] = "yes" if supported_match.group(1).lower() == "true" else "no"

        sentences_match = re.search(r'"sentences":\s*\[(.*?)\]', content, re.DOTALL)
        if sentences_match:
            raw = sentences_match.group(1)
            sents = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
            result["sentences"] = [s.replace('\\"', '"') for s in sents if s.strip()]
        else:
            sent_match = re.search(r'"supporting_sentence":\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
            if sent_match:
                s = sent_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                if s.strip():
                    result["sentences"] = [s.strip()]

        reasoning_match = re.search(r'"reasoning":\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
        if reasoning_match:
            r = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            result["reasoning"] = r if r.strip() else "Manual extraction"

        subject_match = re.search(r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if subject_match:
            result["subject_mentioned"] = subject_match.group(1).lower() == 'true'

        object_match = re.search(r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if object_match:
            result["object_mentioned"] = object_match.group(1).lower() == 'true'

        return result

    def _estimate_token_count(self, text: str) -> int:
        """Rough token estimate (~4 chars per token for English text)."""
        return len(text) // 4

    async def generate_response(self, prompt: str) -> str:
        """Generate response using vLLM's OpenAI-compatible API."""
        try:
            system_msg = (
                "You are an expert medical researcher with a strong "
                "understanding of medical and biological semantics."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]

            max_tokens = self.DEFAULT_MAX_TOKENS
            if self.server_max_model_len > 0:
                est_input = self._estimate_token_count(system_msg + prompt)
                available = self.server_max_model_len - est_input - 64
                max_tokens = max(512, min(max_tokens, available))

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
            )

            content = response.choices[0].message.content or ""

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                logger.info(
                    f"Token usage: prompt={prompt_tokens}, "
                    f"completion={completion_tokens}, total={total_tokens}"
                )

            return content

        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise Exception(f"vLLM API error: {e}")

    async def evaluate_triple_support(
        self, triple: Union[List[str], 'TripleData'], abstract: str
    ) -> Dict[str, Any]:
        """Evaluate if an abstract supports a given triple."""
        prompt_text = prompt_builder.build_evaluation_prompt(triple, abstract)

        preamble = (
            "Analyze the abstract carefully and provide your final answer as a JSON object.\n"
            "Use <think></think> tags to systematically reason through the evaluation "
            "before providing your final JSON response.\n\n"
        )

        full_prompt = preamble + prompt_text

        try:
            content = await self.generate_response(full_prompt)

            if not content or not content.strip():
                return {
                    "support": "no",
                    "sentences": [],
                    "reasoning": "Empty response from vLLM",
                    "subject_mentioned": False,
                    "object_mentioned": False,
                }

            # Handle thinking tags
            if "</think>" in content:
                think_end = content.find("</think>") + 8
                content = content[think_end:].strip()

            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_match = re.search(
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                    else:
                        return self._extract_json_manually(content)

            if json_str:
                json_str = self._fix_json_formatting(json_str)
                try:
                    evaluation = json.loads(json_str)
                except json.JSONDecodeError:
                    evaluation = self._extract_json_manually(content)
            else:
                raise ValueError("Empty JSON content extracted.")

            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            try:
                return self._extract_json_manually(content)
            except Exception:
                return {
                    "support": "no",
                    "sentences": [],
                    "reasoning": f"LLM response not valid JSON: {content[:200]}...",
                }
        except Exception as e:
            logger.error(f"Error during vLLM evaluation: {e}")
            raise
