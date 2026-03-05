"""Ollama client for Hermes 3 and GPT-OSS models."""
import logging
import json
import re
from typing import Dict, Any, List, Union, TYPE_CHECKING
import ollama
from .config import settings
from . import prompt_builder

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama-served models."""

    def __init__(self, model: str = None, base_url: str = None):
        if model is None:
            model = settings.default_model

        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')

        logger.info(f"Initialized Ollama client - Model: {self.model}")
        logger.info(f"Ollama server: {self.base_url}")

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
            # Legacy fallback
            supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
            if supported_match:
                result["support"] = "yes" if supported_match.group(1).lower() == "true" else "no"

        # Extract sentences array
        sentences_match = re.search(r'"sentences":\s*\[(.*?)\]', content, re.DOTALL)
        if sentences_match:
            raw = sentences_match.group(1)
            sents = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
            result["sentences"] = [s.replace('\\"', '"') for s in sents if s.strip()]
        else:
            # Legacy: single supporting_sentence
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

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Ollama."""
        client = ollama.AsyncClient(host=self.base_url, timeout=settings.request_timeout)

        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "top_p": 0.95 if "hermes4" in self.model else 0.9,
                    "top_k": 20 if "hermes4" in self.model else -1,
                    "num_predict": 800,
                },
            )
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

    async def evaluate_triple_support(
        self, triple: Union[List[str], 'TripleData'], abstract: str
    ) -> Dict[str, Any]:
        """Evaluate if an abstract supports a given triple."""
        prompt_text = prompt_builder.build_evaluation_prompt(triple, abstract)

        # Add model-specific reasoning instructions
        if "gpt-oss" in self.model.lower():
            preamble = (
                "You are an expert medical researcher. "
                "After your analysis, provide a final JSON response.\n"
                "FINAL_ANSWER: <your JSON>\n\n"
            )
        else:
            preamble = (
                "You are an expert medical researcher. "
                "Use <think></think> tags for reasoning, then provide your JSON answer.\n\n"
            )

        full_prompt = preamble + prompt_text

        try:
            response_data = await self.generate_response(full_prompt)

            # Handle Pydantic and dict response formats
            if hasattr(response_data, 'message'):
                message = response_data.message
                content = message.content if hasattr(message, 'content') and message.content else ""
                thinking_content = message.thinking if hasattr(message, 'thinking') and message.thinking else ""
            else:
                message = response_data["message"]
                content = message.get("content", "") or ""
                thinking_content = message.get("thinking", "") or ""

            # Handle FINAL_ANSWER pattern (GPT-OSS)
            if "FINAL_ANSWER:" in content:
                answer_start = content.find("FINAL_ANSWER:") + 13
                content = content[answer_start:].strip()

            elif not content.strip() and thinking_content:
                logger.info("Using thinking field as model put response there")
                if "FINAL_ANSWER:" in thinking_content:
                    answer_start = thinking_content.find("FINAL_ANSWER:") + 13
                    content = thinking_content[answer_start:].strip()
                else:
                    content = thinking_content
                    if '{' not in thinking_content:
                        content = '{"support": "no", "sentences": [], "reasoning": "Based on model reasoning in thinking field"}'

            elif not content.strip() and not thinking_content.strip():
                raise ValueError("Model returned empty response")

            # Strip think tags
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
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
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

        except json.JSONDecodeError:
            try:
                return self._extract_json_manually(content)
            except Exception:
                return {
                    "support": "no",
                    "sentences": [],
                    "reasoning": f"LLM response not valid JSON: {content[:200]}...",
                }
        except Exception as e:
            logger.error(f"Error during Ollama evaluation: {e}")
            raise
