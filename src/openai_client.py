"""OpenAI client for GPT-5 nano, GPT-5-mini, and other commercial models."""
import logging
from typing import Dict, Any, List, Union, TYPE_CHECKING
import json
from openai import AsyncOpenAI
from .config import settings

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for OpenAI commercial models."""
    
    def __init__(self, 
                 model: str = None,
                 api_key: str = None,
                 enable_web_search: bool = False,
                 reasoning_effort: str = "minimal"):
        """Initialize OpenAI client.
        
        Args:
            model: OpenAI model name (e.g., gpt-5-nano, gpt-5-mini)
            api_key: OpenAI API key (if None, uses settings.openai_api_key)
            enable_web_search: Enable OpenAI's built-in web_search tool
            reasoning_effort: Control thinking depth for GPT-5 models 
                             ("minimal", "low", "medium", "high"). Default: "minimal"
        """
        if model is None:
            model = "gpt-5-nano"
        
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.enable_web_search = enable_web_search
        self.reasoning_effort = reasoning_effort
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI client - Model: {self.model}")
        logger.info("Prompt caching enabled")
        if self.enable_web_search:
            logger.warning("Web search enabled")
        if "gpt-5" in self.model:
            logger.info(f"Reasoning effort set to: {self.reasoning_effort}")
    
    def _extract_json_manually(self, content: str) -> Dict[str, Any]:
        """Manually extract JSON values when parsing fails.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Dictionary with extracted values
        """
        import re
        
        # Default values
        result = {
            "is_supported": False,
            "evidence_category": "not_supported",
            "supporting_sentence": None,
            "reasoning": "Manual extraction fallback",
            "subject_mentioned": False,
            "object_mentioned": False
        }
        
        # Extract is_supported
        supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
        if supported_match:
            result["is_supported"] = supported_match.group(1).lower() == 'true'
        
        # Extract evidence_category
        category_match = re.search(r'"evidence_category":\s*"([^"]*)"', content)
        if category_match:
            result["evidence_category"] = category_match.group(1)
        
        # Extract supporting_sentence (handle quotes carefully)
        sentence_match = re.search(r'"supporting_sentence":\s*"([^"]*(?:\\.[^"]*)*)"', content)
        if sentence_match:
            sentence = sentence_match.group(1).replace('\\"', '"')
            result["supporting_sentence"] = sentence if sentence.strip() else None
        
        # Extract reasoning (handle quotes carefully)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', content)
        if reasoning_match:
            reasoning = reasoning_match.group(1).replace('\\"', '"')
            result["reasoning"] = reasoning if reasoning.strip() else "Manual extraction"
        
        # Extract subject_mentioned
        subject_mentioned_match = re.search(r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if subject_mentioned_match:
            result["subject_mentioned"] = subject_mentioned_match.group(1).lower() == 'true'
        
        # Extract object_mentioned
        object_mentioned_match = re.search(r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if object_mentioned_match:
            result["object_mentioned"] = object_mentioned_match.group(1).lower() == 'true'
        
        return result

    async def generate_response(self, prompt: str, cacheable_prefix: str = None) -> str:
        """Generate response using OpenAI API with optional prompt caching.
        
        Args:
            prompt: Input prompt for the model (unique content)
            cacheable_prefix: Optional cacheable content (instructions, rules, etc.)
            
        Returns:
            String response from the model
        """
        try:
            # Prepare messages with prompt caching
            # Note: Prompt caching requires >1024 tokens in cached section for optimal performance
            use_caching = bool(cacheable_prefix)
            
            if use_caching:
                # Use structured format with cache_control (all models)
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an expert medical researcher with a strong understanding of medical and biological semantics.",
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": cacheable_prefix,
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            else:
                # Standard format (no cacheable prefix provided)
                messages = [
                    {"role": "system", "content": "You are an expert medical researcher with a strong understanding of medical and biological semantics."},
                    {"role": "user", "content": prompt}
                ]
            
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 7000
            }
            
            # Add response_format for JSON mode
            if "gpt-5" in self.model or "gpt-4" in self.model:
                api_params["response_format"] = {"type": "json_object"}
            
            # Add reasoning_effort for GPT-5 thinking models
            if "gpt-5" in self.model:
                api_params["reasoning_effort"] = self.reasoning_effort
                logger.debug(f"Using reasoning_effort={self.reasoning_effort} for {self.model}")
            
            # Add web_search tool if enabled
            if self.enable_web_search:
                api_params["tools"] = [{"type": "web_search"}]
                logger.info("Web search tool enabled for this request")
            
            response = await self.client.chat.completions.create(**api_params)
            
            # Check for refusals
            message = response.choices[0].message
            if hasattr(message, 'refusal') and message.refusal:
                logger.error(f"OpenAI refused the request: {message.refusal}")
                raise Exception(f"OpenAI refused the request: {message.refusal}")
            
            content = message.content
            
            # Log token usage statistics
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                
                # Log in parseable format for scripts
                logger.info(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
                
                # Log cache statistics if available
                cached_tokens = 0
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    details = usage.prompt_tokens_details
                    cached_tokens = getattr(details, 'cached_tokens', 0)
                    if cached_tokens > 0:
                        logger.info(f"Prompt cache hit: {cached_tokens} cached tokens (saved ~${cached_tokens * 0.000025:.6f})")
                    else:
                        logger.debug("Prompt cache miss - first request with this prompt structure")
                else:
                    # No prompt_tokens_details in response - model may not support caching
                    logger.debug(f"No prompt_tokens_details in API response - caching may not be supported for {self.model}")
            else:
                # No usage data in response - model may not support token usage reporting
                logger.warning(f"No token usage data in API response for model {self.model}. This model may not report token usage.")
            
            # Log if web search was actually used
            if self.enable_web_search and hasattr(response.choices[0].message, 'tool_calls'):
                if response.choices[0].message.tool_calls:
                    logger.info(f"Web search tool was invoked: {len(response.choices[0].message.tool_calls)} time(s)")
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")

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
            # Get qualifier information
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
        
        # Format equivalent names
        subject_names_list = subject_names if subject_names else [subject]
        object_names_list = object_names if object_names else [obj]
        
        subject_names_text = "\n**SUBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
        for i, name in enumerate(subject_names_list, 1):
            subject_names_text += f"  {i}. {name}\n"
        
        object_names_text = "\n**OBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
        for i, name in enumerate(object_names_list, 1):
            object_names_text += f"  {i}. {name}\n"
        
        # Build the triple description based on whether qualifiers are present
        if has_qualifiers:
            # Build qualified description
            qualified_parts = []
            if qualified_object_direction:
                qualified_parts.append(qualified_object_direction)
            if qualified_object_aspect:
                qualified_parts.append(qualified_object_aspect)
            
            qualified_description = " ".join(qualified_parts)
            triple_description = f"'{subject}' {qualified_predicate} {qualified_description} of '{obj}'"
            
            # Add qualifier-specific guidance
            provided_qualifiers = [f"predicate '{qualified_predicate}'"]
            if qualified_object_direction:
                provided_qualifiers.append(f"direction '{qualified_object_direction}'")
            if qualified_object_aspect:
                provided_qualifiers.append(f"aspect '{qualified_object_aspect}'")
            
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
        
        # Separate cacheable content (instructions, rules, format) from unique content
        cacheable_prefix = (
            f"Analyze the abstract carefully and provide your answer as a JSON object.\n\n"
            f"**TASK**: Determine if this triple is supported by the abstract.\n\n"
            f"{self._get_common_matching_rules()}"
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
            f"**OUTPUT** (must be valid JSON, no other text):\n"
            f"{{\n"
            f'  "is_supported": true/false,\n'
            f'  "evidence_category": "direct_support"/"opposite_assertion"/"wrong_qualifier"/"missing_qualifier"/"not_supported",\n'
            f'  "supporting_sentence": "exact quote from abstract" or null,\n'
            f'  "reasoning": "Subject: [found/not found] as \'[name found in abstract]\' (equivalent to \'[original name or name from list]\'). Object: [found/not found] as \'[name found in abstract]\' (equivalent to \'[original name or name from list]\'). Relationship analysis: [detailed and professional explanation based on the abstract content]. Evidence category: [category] because [detailed reason based on category definition]. Conclusion: [supported/not supported].",\n'
            f'  "subject_mentioned": true/false,\n'
            f'  "object_mentioned": true/false\n'
            f"}}\n\n"
            f"**CONSISTENCY RULES**:\n"
            f"• If no supporting_sentence → is_supported=false\n"
            f"• The logic of the reasoning must align with is_supported value\n"
            f"• Supporting sentence must align with is_supported value\n"
        )
        
        # Unique content per request (triple, abstract, names)
        unique_prompt = (
            f"**TRIPLE**: {triple_description}\n\n"
            f"{subject_names_text}"
            f"{object_names_text}\n"
            f"**ABSTRACT**: {abstract}\n\n"
            f"{qualifier_guidance if has_qualifiers else ''}"
        )
        
        try:
            content = await self.generate_response(unique_prompt, cacheable_prefix)
            
            # Parse JSON response (OpenAI's json_object format ensures valid JSON)
            try:
                evaluation = json.loads(content)
                logger.debug(f"Successfully parsed JSON response: {evaluation}")
                return evaluation
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON. Content: {content[:500]}... Error: {e}")
                # Try manual extraction as fallback
                evaluation = self._extract_json_manually(content)
                logger.warning(f"Used manual extraction fallback: {evaluation}")
                return evaluation
                
        except Exception as e:
            logger.error(f"Error during OpenAI evaluation: {e}")
            raise
    
    async def verify_evaluation(self, triple: Union[List[str], 'TripleData'], abstract: str, 
                                original_evaluation: Dict[str, Any], checker_model: str = None) -> Dict[str, Any]:
        """Verify an evaluation result using OpenAI, specifically checking name recognition.
        
        Args:
            triple: The research triple (TripleData or list)
            abstract: The abstract text
            original_evaluation: The original evaluation result from another LLM
            checker_model: The model to use for verification (if None, uses self.model)
            
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
            # Get qualifier information if available
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
        
        # Build triple description with qualifiers if available
        triple_desc = f"'{subject}' {predicate} '{obj}'"
        if qual_pred:
            qualifier_parts = [qual_pred]
            if qual_direction:
                qualifier_parts.append(qual_direction)
            if qual_aspect:
                qualifier_parts.append(qual_aspect)
            triple_desc += f" (with qualifiers: {', '.join(qualifier_parts)})"
        
        # Separate cacheable content (verification instructions, rules) from unique content
        cacheable_prefix = (
            f"You are an expert medical researcher with a strong understanding of medical and biological semantics.\n"
            f"Your job is to carefully verify and check another expert medical researcher's evaluation result.\n\n"
            f"**VERIFICATION TASK**: Check if the original evaluation is correct.\n\n"
            f"{self._get_common_matching_rules()}"
            f"**VERIFICATION STEPS**:\n"
            f"1. **Check entity mentions**:\n"
            f"   - Set subject_mentioned=TRUE if subject appears ANYWHERE\n"
            f"   - Set object_mentioned=TRUE if object appears ANYWHERE\n"
            f"   - Use equivalent names list or common abbreviations\n"
            f"2. **Find sentence describing DIRECT relationship** between subject and object:\n"
            f"   - Sentence MUST mention BOTH entities\n"
            f"   - Sentence MUST show DIRECT causal link (not correlation, not transitive)\n"
            f"3. **VALIDATE SUPPORTING SENTENCE**: Does the original supporting_sentence mention BOTH subject and object?\n"
            f"   - If NO → Find correct sentence and provide in corrected_supporting_sentence\n"
            f"4. **Verify relationship matches predicate** (use semantic equivalents)\n"
            f"5. **LOGIC**:\n"
            f"• Entity mentions are INDEPENDENT: can be mentioned=TRUE but still not_supported (not correctly match to the triple relationship)\n"
            f"• If no direct relationship → is_supported=false\n"
            f"• Distinguish positive/negative/neutral → 'does not decrease/inhibit' is NEUTRAL (not an increase)\n\n"
            f"6. Compare with original: Did it correctly identify entities and relationship?\n"
            f"7. Set is_correct=true if original is right, false if wrong (with corrections)\n\n"
            f"**COMMON ERRORS TO CHECK**:\n"
            f"• **Entity mentions marked incorrectly**: Entity appears in abstract but marked as not mentioned\n"
            f"• **Confusing mention with support**: Entity mentioned but not correctly match to the triple relationship → should be mentioned=TRUE but not_supported\n"
            f"• Supporting sentence mentions different entity than the target subject or object\n"
            f"• Supporting sentence describes relationship with wrong subject or object (e.g., says 'X inhibits Y' but triple is about 'Z inhibits Y')\n"
            f"• Missing correct evidence that actually exists in the abstract\n"
            f"• **CRITICAL**: Confusing neutral ('does not decrease') with positive ('increases') - these are NOT equivalent\n"
            f"• **Correlation as causation**: 'X correlates with Y' marked as supporting 'X stimulates Y' (WRONG - correlation ≠ causation)\n"
            f"• **Transitive reasoning**: 'Z affects both X and Y' marked as supporting 'X affects Y' (WRONG - no direct relationship)\n"
            f"• **Opposite labeled as support**: Abstract contradicts triple but marked as 'direct_support' instead of 'opposite_assertion'\n"
            f"• **Inverse relationship**: 'Increases A while decreasing B' marked as supporting 'B stimulates A' (WRONG)\n"
            f"• **Missing entity in supporting sentence**: Supporting sentence DOES NOT mention BOTH subject and object explicitly\n\n"
            f"**OUTPUT** (JSON only):\n"
            f"{{\n"
            f'  "is_correct": true/false,\n'
            f'  "corrected_subject_mentioned": true/false,\n'
            f'  "corrected_object_mentioned": true/false,\n'
            f'  "corrected_is_supported": true/false,\n'
            f'  "corrected_evidence_category": "direct_support|opposite_assertion|wrong_qualifier|missing_qualifier|not_supported",\n'
            f'  "corrected_supporting_sentence": "quote from abstract" or null,\n'
            f'  "corrected_reasoning": "Subject: [found/not found] as \'[name found in abstract]\' (equivalent to \'[original name or name from list]\'). Object: [found/not found] as \'[name found in abstract]\' (equivalent to \'[original name or name from list]\'). Relationship: [matches/conflicts/missing] because [detailed and professional explanation based on the abstract content]. Evidence category: [category] because [detailed reason based on category definition]. Conclusion: [supported/not supported]."\n'
            f"}}\n"
        )
        
        # Build unique content (triple, names, abstract, original evaluation)
        subject_names_text = "SUBJECT EQUIVALENT NAMES:\n"
        for i, name in enumerate(subject_names, 1):
            subject_names_text += f"  {i}. {name}\n"
        
        object_names_text = "\nOBJECT EQUIVALENT NAMES:\n"
        for i, name in enumerate(object_names, 1):
            object_names_text += f"  {i}. {name}\n"
        
        unique_prompt = (
            f"TRIPLE BEING EVALUATED:\n{triple_desc}\n\n"
            f"ORIGINAL EVALUATION:\n"
            f"- subject_mentioned: {original_evaluation.get('subject_mentioned', False)}\n"
            f"- object_mentioned: {original_evaluation.get('object_mentioned', False)}\n"
            f"- is_supported: {original_evaluation.get('is_supported', False)}\n"
            f"- evidence_category: {original_evaluation.get('evidence_category', 'not_supported')}\n"
            f"- reasoning: {original_evaluation.get('reasoning', '')}\n\n"
            f"{subject_names_text}"
            f"{object_names_text}\n"
            f"**ABSTRACT**: {abstract}\n"
        )
        
        try:
            # Use checker_model if specified, otherwise use self.model
            if checker_model and checker_model != self.model:
                verifier = OpenAIClient(
                    model=checker_model, 
                    api_key=self.api_key,
                    enable_web_search=self.enable_web_search,
                    reasoning_effort=self.reasoning_effort
                )
            else:
                verifier = self
            
            content = await verifier.generate_response(unique_prompt, cacheable_prefix)
            
            # Parse JSON response
            try:
                verification_result = json.loads(content)
                
                # If verification found errors, return corrected evaluation
                if not verification_result.get("is_correct", True):
                    logger.info("Verification found issues, returning corrected evaluation")
                    corrected_eval = original_evaluation.copy()
                    
                    # Update name mentions
                    corrected_eval["subject_mentioned"] = verification_result.get("corrected_subject_mentioned", 
                                                                                  original_evaluation.get("subject_mentioned"))
                    corrected_eval["object_mentioned"] = verification_result.get("corrected_object_mentioned",
                                                                                 original_evaluation.get("object_mentioned"))
                    
                    # Update is_supported and evidence_category if provided
                    if "corrected_is_supported" in verification_result:
                        corrected_eval["is_supported"] = verification_result["corrected_is_supported"]
                    
                    if "corrected_evidence_category" in verification_result:
                        corrected_eval["evidence_category"] = verification_result["corrected_evidence_category"]
                    
                    # Update supporting_sentence if provided
                    if "corrected_supporting_sentence" in verification_result:
                        corrected_eval["supporting_sentence"] = verification_result["corrected_supporting_sentence"]
                    
                    # Use the corrected reasoning with [VERIFIED & CORRECTED] label
                    new_reasoning = verification_result.get("corrected_reasoning", "").strip()
                    if new_reasoning:
                        corrected_eval["reasoning"] = f"[VERIFIED & CORRECTED] {new_reasoning}"
                    else:
                        logger.warning("Verification did not provide corrected_reasoning, keeping original")
                        corrected_eval["reasoning"] = original_evaluation.get("reasoning", "") + " [Verification attempted but no corrected reasoning provided]"
                    
                    return corrected_eval
                else:
                    logger.info("Verification confirmed original evaluation is correct")
                    return original_evaluation
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse verification response: {content}. Error: {e}")
                return original_evaluation
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            # If verification fails, return original evaluation
            return original_evaluation

