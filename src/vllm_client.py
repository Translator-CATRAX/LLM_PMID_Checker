"""vLLM client via OpenAI-compatible API.

"""
import logging
from typing import Dict, Any, List, Union, TYPE_CHECKING
import json
import re
from openai import AsyncOpenAI
from .config import settings

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for vLLM-served models via OpenAI-compatible API."""
    
    def __init__(self, 
                 model: str = None,
                 base_url: str = None):
        """Initialize vLLM client.
        
        Args:
            model: Model name as served by vLLM (e.g., 'hermes4-vllm')
            base_url: vLLM server URL (e.g., 'http://localhost:8000/v1')
        """
        if model is None:
            model = "hermes4-vllm"
        
        self.model = model
        self.base_url = (base_url or settings.vllm_base_url).rstrip('/')
        
        # Ensure base_url ends with /v1 for OpenAI-compatible API
        if not self.base_url.endswith('/v1'):
            self.base_url = self.base_url + '/v1'
        
        # vLLM doesn't require an API key, but openai library needs one
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",  # vLLM doesn't use API keys
            timeout=settings.request_timeout
        )
        
        logger.info(f"Initialized vLLM client - Model: {self.model}")
        logger.info(f"vLLM server: {self.base_url}")
    
    def _fix_json_formatting(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses."""
        json_str = json_str.strip()
        
        # Extract JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
        # Replace single quotes with double quotes for JSON keys and string values
        json_str = re.sub(r"'([^']*)'(\s*:\s*)", r'"\1"\2', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Handle boolean and null values
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str

    def _extract_json_manually(self, content: str) -> Dict[str, Any]:
        """Manually extract JSON values when parsing fails."""
        result = {
            "is_supported": False,
            "evidence_category": "not_supported",
            "supporting_sentence": None,
            "reasoning": "Manual extraction fallback",
            "subject_mentioned": False,
            "object_mentioned": False
        }
        
        supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
        if supported_match:
            result["is_supported"] = supported_match.group(1).lower() == 'true'
        
        category_match = re.search(r'"evidence_category":\s*"([^"]*)"', content)
        if category_match:
            result["evidence_category"] = category_match.group(1)
        
        sentence_match = re.search(
            r'"supporting_sentence":\s*"((?:[^"\\]|\\.)*)"\s*[,}]', content, re.DOTALL
        )
        if sentence_match:
            sentence = sentence_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            result["supporting_sentence"] = sentence if sentence.strip() else None
        
        reasoning_match = re.search(
            r'"reasoning":\s*"((?:[^"\\]|\\.)*)"\s*[,}]', content, re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            result["reasoning"] = reasoning if reasoning.strip() else "Manual extraction"
        
        subject_match = re.search(r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if subject_match:
            result["subject_mentioned"] = subject_match.group(1).lower() == 'true'
        
        object_match = re.search(r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if object_match:
            result["object_mentioned"] = object_match.group(1).lower() == 'true'
        
        logger.debug("Manual extraction completed successfully")
        return result

    async def generate_response(self, prompt: str) -> str:
        """Generate response using vLLM's OpenAI-compatible API.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            String response content from the model
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert medical researcher with a strong understanding of medical and biological semantics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
            )
            
            content = response.choices[0].message.content
            
            # Log token usage if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                logger.info(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            
            return content
            
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise Exception(f"vLLM API error: {e}")

    def _get_common_matching_rules(self) -> str:
        """Returns common matching rules shared between evaluation and verification prompts."""
        return (
            "**MATCHING RULES**:\n"
            "1. **Entity Matching**: Start with equivalent names from the provided lists above\n"
            "   - You MAY use your knowledge to recognize common abbreviations and variants of the equivalent names (e.g., 'AAA' from 'AAA gene/protein')\n"
            "   - BUT any alternative name you use MUST be clearly related to names in the provided list\n"
            "2. **Relationship Matching**: Match by semantic meaning:\n"
            "   - 'inhibits' = 'suppresses' = 'reduces' = 'blocks' = 'decreases activity/expression/abundance'\n"
            "   - 'activates' = 'stimulates' = 'promotes' = 'increases' = 'upregulates' = 'enhances' = 'increases activity/expression/abundance'\n"
            "   - 'expression', 'abundance', and 'levels' are interchangeable\n\n"
            "**CRITICAL REQUIREMENTS**:\n"
            "• Relationship MUST be between the SUBJECT and OBJECT from the triple\n"
            "• Do NOT confuse relationships involving other entities\n"
            "• SUPPORTING SENTENCE MUST EXPLICITLY MENTION BOTH SUBJECT and OBJECT (or EQUIVALENT NAMES)\n"
            "**CRITICAL LOGIC RULES**:\n"
            "1. **Correlation ≠ Causation**: 'A correlates with B' does NOT support 'A causes B'\n"
            "2. **No Transitive Reasoning**: If X→A and X→B, does NOT mean A→B (must be direct)\n"
            "3. **Opposite = Contradiction**: Inverse relationship → use 'opposite_assertion'\n"
            "4. **Both Entities Required**: Supporting sentence must mention BOTH entities (names from the equivalent names list or via common abbreviations and variants of the equivalent names)\n\n"
            "**CATEGORIES**:\n"
            "• 'direct_support': Explicit causal/regulatory statement (e.g., 'X activates Y', 'X is a regulator of Y', 'X treatment increased Y')\n"
            "• 'opposite_assertion': Direct relationship explicitly contradicts the triple\n"
            "• 'not_supported': Missing entities, correlation only, transitive/indirect reasoning, or no evidence\n\n"
        )

    async def evaluate_triple_support(self, triple: Union[List[str], 'TripleData'], abstract: str) -> Dict[str, Any]:
        """Evaluate if an abstract supports a given triple.
        
        Args:
            triple: List [subject, predicate, object] or TripleData object with equivalent names
            abstract: Abstract text to evaluate
            
        Returns:
            Dict with evaluation results
        """
        # Handle both list and TripleData formats
        if hasattr(triple, 'subject'):
            subject = triple.subject
            predicate = triple.predicate
            obj = triple.object
            subject_names = getattr(triple, 'subject_names', [subject])
            object_names = getattr(triple, 'object_names', [obj])
            qualified_predicate = getattr(triple, 'qualified_predicate', None)
            qualified_object_aspect = getattr(triple, 'qualified_object_aspect', None)
            qualified_object_direction = getattr(triple, 'qualified_object_direction', None)
            has_qualifiers = getattr(triple, 'has_qualifiers', lambda: False)()
        else:
            subject, predicate, obj = triple
            subject_names = [subject]
            object_names = [obj]
            qualified_predicate = None
            qualified_object_aspect = None
            qualified_object_direction = None
            has_qualifiers = False
        
        # Hermes 4 uses <think></think> tags for reasoning
        reasoning_prompt = (
            "Analyze the abstract carefully and provide your final answer as a JSON object.\n\n"
            "Use <think></think> tags to systematically reason through the evaluation before providing your final JSON response.\n\n"
        )
        
        # Format equivalent names
        subject_names_list = subject_names if subject_names else [subject]
        object_names_list = object_names if object_names else [obj]
        
        subject_names_text = "\n**SUBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
        for i, name in enumerate(subject_names_list, 1):
            subject_names_text += f"  {i}. {name}\n"
        
        object_names_text = "\n**OBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
        for i, name in enumerate(object_names_list, 1):
            object_names_text += f"  {i}. {name}\n"
        
        # Build triple description with qualifiers
        if has_qualifiers:
            qualified_parts = []
            if qualified_object_direction:
                qualified_parts.append(qualified_object_direction)
            if qualified_object_aspect:
                qualified_parts.append(qualified_object_aspect)
            
            qualified_description = " ".join(qualified_parts)
            triple_description = f"'{subject}' {qualified_predicate} {qualified_description} of '{obj}'"
            
            qualifier_guidance = (
                f"**QUALIFIERS TO CHECK**:\n"
                f"- Predicate: {qualified_predicate}\n"
                f"- Direction: {qualified_object_direction or 'any'}\n"
                f"- Aspect: {qualified_object_aspect or 'any'}\n\n"
                f"**QUALIFIER MATCHING**:\n"
                f"• If direction='{qualified_object_direction}': Abstract must show {qualified_object_direction} effect (e.g., {'increases/activates/upregulates' if qualified_object_direction=='increased' else 'decreases/inhibits/downregulates' if qualified_object_direction=='decreased' else qualified_object_direction})\n"
                f"• If aspect='activity_or_abundance': Abstract can mention activity OR abundance OR both (any one is sufficient)\n"
                f"• Match semantic meaning: 'inhibitor' = 'causes decreased activity', 'upregulates expression' = 'causes increased abundance'\n\n"
                f"**CATEGORIES**:\n"
                f"• 'wrong_qualifier': Direction conflicts (e.g., abstract says increases but qualifier wants decreased)\n"
                f"• 'missing_qualifier': Subject/object found but no info about the qualifiers\n\n"
            )
        else:
            triple_description = f"'{subject}' {predicate} '{obj}'"
            qualifier_guidance = ""
        
        prompt = (
            f"{reasoning_prompt}"
            f"**TASK**: Determine if this triple is supported by the abstract:\n"
            f"**TRIPLE**: {triple_description}\n\n"
            f"{subject_names_text}"
            f"{object_names_text}\n"
            f"**ABSTRACT**: {abstract}\n\n"
            f"{self._get_common_matching_rules()}"
            f"{qualifier_guidance if has_qualifiers else ''}"
            f"**EVALUATION STEPS**:\n"
            f"1. **Check entity mentions** (subject_mentioned / object_mentioned):\n"
            f"   - Set to TRUE if the entity appears ANYWHERE in abstract (use equivalent names lists or any common abbreviations or variants of the equivalent names)\n"
            f"   - Entity mention is COMPLETELY INDEPENDENT from whether the triple is supported\n"
            f"2. **Find sentence describing DIRECT relationship** between subject and object:\n"
            f"   - The sentence MUST mention BOTH entities\n"
            f"   - The sentence MUST describe a DIRECT causal link\n"
            f"3. **Verify relationship matches predicate**:\n"
            f"   - 'stimulates': Explicit causal/activation language OR strong implication from experimental manipulation\n"
            f"   - 'inhibits': Explicit inhibition/suppression language OR strong implication from experimental evidence\n"
            f"4. **Determine category**:\n"
            f"   - 'direct_support': Explicit causal statement or clear experimental evidence ('X activates Y', 'X treatment increased Y')\n"
            f"   - 'opposite_assertion': Relationship explicitly contradicts triple\n"
            f"   - 'not_supported': Correlation, co-occurrence, transitive reasoning, or no evidence\n\n"
            f"**OUTPUT** (JSON only, no other text):\n"
            f"{{\n"
            f"  \"is_supported\": true/false,\n"
            f"  \"evidence_category\": \"direct_support\"/\"opposite_assertion\"/\"wrong_qualifier\"/\"missing_qualifier\"/\"not_supported\",\n"
            f"  \"supporting_sentence\": \"exact quote from abstract\" or null,\n"
            f"  \"reasoning\": \"Subject: [found/not found] as '[name found in abstract]' (equivalent to '[original name or name from list]'). Object: [found/not found] as '[name found in abstract]' (equivalent to '[original name or name from list]'). Relationship analysis: [detailed and professional explanation based on the abstract content]. Evidence category: [category] because [detailed reason based on category definition]. Conclusion: [supported/not supported].\",\n"
            f"  \"subject_mentioned\": true/false,\n"
            f"  \"object_mentioned\": true/false\n"
            f"}}\n\n"
            f"**CONSISTENCY RULES**:\n"
            f"• If no supporting_sentence → is_supported=false\n"
            f"• The logic of the reasoning must align with is_supported value\n"
            f"• Supporting sentence must align with is_supported value\n\n"
        )

        try:
            content = await self.generate_response(prompt)
            
            # Handle Hermes 4 reasoning tags
            if "</think>" in content:
                think_end = content.find("</think>") + 8
                content = content[think_end:].strip()
            
            # Extract JSON from response
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
                        logger.warning("No JSON found in response, using manual extraction")
                        return self._extract_json_manually(content)
            
            # Clean and parse JSON
            if json_str:
                json_str = self._fix_json_formatting(json_str)
                try:
                    evaluation = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.debug("JSON parse failed, using manual extraction")
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
                    "is_supported": False,
                    "supporting_sentence": None,
                    "reasoning": f"LLM response not valid JSON: {content[:200]}..."
                }
        except Exception as e:
            logger.error(f"Error during vLLM evaluation: {e}")
            raise

    async def verify_evaluation(self, triple: Union[List[str], 'TripleData'], abstract: str, 
                                original_evaluation: Dict[str, Any], checker_model: str = None) -> Dict[str, Any]:
        """Verify an evaluation result using a checker model.
        
        Args:
            triple: The research triple (TripleData or list)
            abstract: The abstract text
            original_evaluation: The original evaluation result from another LLM
            checker_model: The model to use for verification (unused for vLLM, uses self)
            
        Returns:
            Dict with verification results and potentially corrected evaluation
        """
        # Handle both list and TripleData formats
        if hasattr(triple, 'subject'):
            subject = triple.subject
            predicate = triple.predicate
            obj = triple.object
            subject_names = getattr(triple, 'subject_names', [subject])
            object_names = getattr(triple, 'object_names', [obj])
            qual_pred = getattr(triple, 'qualified_predicate', None)
            qual_aspect = getattr(triple, 'qualified_object_aspect', None)
            qual_direction = getattr(triple, 'qualified_object_direction', None)
        else:
            subject, predicate, obj = triple
            subject_names = [subject]
            object_names = [obj]
            qual_pred = None
            qual_aspect = None
            qual_direction = None
        
        # Build triple description
        triple_desc = f"'{subject}' {predicate} '{obj}'"
        if qual_pred:
            qualifier_parts = [qual_pred]
            if qual_direction:
                qualifier_parts.append(qual_direction)
            if qual_aspect:
                qualifier_parts.append(qual_aspect)
            triple_desc += f" (with qualifiers: {', '.join(qualifier_parts)})"
        
        # Build verification prompt
        verification_prompt = (
            f"You are an expert medical researcher with a strong understanding of medical and biological semantics.\n"
            f"Your job is to carefully verify and check another expert medical researcher's evaluation result.\n\n"
            f"Use <think></think> tags to systematically reason through the verification.\n\n"
            f"TRIPLE BEING EVALUATED:\n{triple_desc}\n\n"
            f"ORIGINAL EVALUATION:\n"
            f"- subject_mentioned: {original_evaluation.get('subject_mentioned', False)}\n"
            f"- object_mentioned: {original_evaluation.get('object_mentioned', False)}\n"
            f"- is_supported: {original_evaluation.get('is_supported', False)}\n"
            f"- evidence_category: {original_evaluation.get('evidence_category', 'not_supported')}\n"
            f"- reasoning: {original_evaluation.get('reasoning', '')}\n\n"
            f"SUBJECT EQUIVALENT NAMES:\n"
        )
        
        for i, name in enumerate(subject_names, 1):
            verification_prompt += f"  {i}. {name}\n"
        
        verification_prompt += f"\nOBJECT EQUIVALENT NAMES:\n"
        for i, name in enumerate(object_names, 1):
            verification_prompt += f"  {i}. {name}\n"
        
        verification_prompt += (
            f"\n**ABSTRACT**: {abstract}\n\n"
            f"**VERIFICATION TASK**: Check if the original evaluation is correct.\n\n"
            f"{self._get_common_matching_rules()}"
            f"**VERIFICATION STEPS**:\n"
            f"1. **Check entity mentions**: Set TRUE if entity appears ANYWHERE\n"
            f"2. **Find sentence describing DIRECT relationship** between subject and object\n"
            f"3. **VALIDATE SUPPORTING SENTENCE**: Does it mention BOTH entities?\n"
            f"4. **Verify relationship matches predicate**\n"
            f"5. **LOGIC**: Entity mentions are INDEPENDENT from support\n"
            f"6. Compare with original evaluation\n"
            f"7. Set is_correct=true if original is right, false if wrong\n\n"
            f"**COMMON ERRORS TO CHECK**:\n"
            f"• Entity mentions marked incorrectly\n"
            f"• Confusing mention with support\n"
            f"• Supporting sentence missing one entity\n"
            f"• Correlation marked as causation\n"
            f"• Transitive reasoning accepted\n"
            f"• Opposite labeled as support\n\n"
            f"**OUTPUT** (JSON only):\n"
            f"{{\n"
            f'  "is_correct": true/false,\n'
            f'  "corrected_subject_mentioned": true/false,\n'
            f'  "corrected_object_mentioned": true/false,\n'
            f'  "corrected_is_supported": true/false,\n'
            f'  "corrected_evidence_category": "direct_support|opposite_assertion|wrong_qualifier|missing_qualifier|not_supported",\n'
            f'  "corrected_supporting_sentence": "quote from abstract" or null,\n'
            f'  "corrected_reasoning": "detailed explanation"\n'
            f"}}\n"
        )
        
        try:
            content = await self.generate_response(verification_prompt)
            
            # Handle thinking tags
            if "</think>" in content:
                think_end = content.find("</think>") + 8
                content = content[think_end:].strip()
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = self._fix_json_formatting(json_str)
                verification_result = json.loads(json_str)
                
                # If verification found errors, return corrected evaluation
                if not verification_result.get("is_correct", True):
                    logger.info("Verification found issues, returning corrected evaluation")
                    corrected_eval = original_evaluation.copy()
                    
                    corrected_eval["subject_mentioned"] = verification_result.get(
                        "corrected_subject_mentioned", original_evaluation.get("subject_mentioned"))
                    corrected_eval["object_mentioned"] = verification_result.get(
                        "corrected_object_mentioned", original_evaluation.get("object_mentioned"))
                    
                    if "corrected_is_supported" in verification_result:
                        corrected_eval["is_supported"] = verification_result["corrected_is_supported"]
                    if "corrected_evidence_category" in verification_result:
                        corrected_eval["evidence_category"] = verification_result["corrected_evidence_category"]
                    if "corrected_supporting_sentence" in verification_result:
                        corrected_eval["supporting_sentence"] = verification_result["corrected_supporting_sentence"]
                    
                    new_reasoning = verification_result.get("corrected_reasoning", "").strip()
                    if new_reasoning:
                        corrected_eval["reasoning"] = f"[VERIFIED & CORRECTED] {new_reasoning}"
                    else:
                        corrected_eval["reasoning"] = original_evaluation.get("reasoning", "") + " [Verification attempted but no corrected reasoning provided]"
                    
                    return corrected_eval
                else:
                    logger.info("Verification confirmed original evaluation is correct")
                    return original_evaluation
            else:
                logger.warning("Could not parse verification response, returning original evaluation")
                return original_evaluation
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return original_evaluation
