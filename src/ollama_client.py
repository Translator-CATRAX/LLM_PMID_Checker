"""Ollama client for Hermes 3 and GPT-OSS models."""
import logging
from typing import Dict, Any, List, Union, TYPE_CHECKING
import json
import ollama
from .config import settings

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for Ollama-served models."""
    
    def __init__(self, 
                 model: str = None,
                 base_url: str = None):
        """Initialize Ollama client.
        
        Args:
            model: Ollama model name (e.g., any available Ollama model from .env)
                  If None, uses settings.default_model
            base_url: Ollama server URL
        """
        if model is None:
            model = settings.default_model

        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        
        logger.info(f"Initialized Ollama client - Model: {self.model}")
        logger.info(f"Ollama server: {self.base_url}")
        
        # Log model info
        if "hermes4" in model:
            if model.endswith("iq4-xs"):
                logger.info("Using Hermes 4 70B (IQ4_XS - fastest, lowest VRAM)")
            elif model.endswith("q4-s"):
                logger.info("Using Hermes 4 70B (Q4_S - balanced)")
            elif model.endswith("q4-m"):
                logger.info("Using Hermes 4 70B (Q4_M - highest quality)")
        elif "gpt-oss" in model:
            logger.info("Using GPT-OSS 20B")

    def _fix_json_formatting(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses.
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Cleaned JSON string with proper formatting
        """
        import re
        
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        
        # Handle case where LLM includes explanatory text before/after JSON
        # Look for the actual JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
        # Replace single quotes with double quotes for JSON keys and string values
        # Handle keys first
        json_str = re.sub(r"'([^']*)'(\s*:\s*)", r'"\1"\2', json_str)
        
        # Handle string values - be more careful about nested quotes
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Handle boolean and null values (ensure they're lowercase)
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str

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
        
        # Extract supporting_sentence - improved to handle embedded quotes
        # Look for the field, then extract until the next field or end of object
        sentence_match = re.search(
            r'"supporting_sentence":\s*"((?:[^"\\]|\\.)*)"\s*[,}]', 
            content, 
            re.DOTALL
        )
        if sentence_match:
            sentence = sentence_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            result["supporting_sentence"] = sentence if sentence.strip() else None
        else:
            # Fallback: try to get anything between quotes after supporting_sentence
            sentence_match2 = re.search(
                r'"supporting_sentence":\s*"([^"]*)"', 
                content
            )
            if sentence_match2:
                result["supporting_sentence"] = sentence_match2.group(1) if sentence_match2.group(1).strip() else None
        
        # Extract reasoning - improved to handle embedded quotes  
        reasoning_match = re.search(
            r'"reasoning":\s*"((?:[^"\\]|\\.)*)"\s*[,}]', 
            content,
            re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            result["reasoning"] = reasoning if reasoning.strip() else "Manual extraction"
        else:
            # Fallback: try to get anything between quotes after reasoning
            reasoning_match2 = re.search(
                r'"reasoning":\s*"([^"]*)"',
                content
            )
            if reasoning_match2:
                result["reasoning"] = reasoning_match2.group(1) if reasoning_match2.group(1).strip() else "Manual extraction"
        
        # Extract subject_mentioned
        subject_mentioned_match = re.search(r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if subject_mentioned_match:
            result["subject_mentioned"] = subject_mentioned_match.group(1).lower() == 'true'
        
        # Extract object_mentioned
        object_mentioned_match = re.search(r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if object_mentioned_match:
            result["object_mentioned"] = object_mentioned_match.group(1).lower() == 'true'
        
        logger.debug(f"Manual extraction completed successfully")
        return result

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Ollama.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Dictionary with response in native Ollama format
        """
        client = ollama.AsyncClient(host=self.base_url, timeout=settings.request_timeout)
        
        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "top_p": 0.95 if "hermes4" in self.model else 0.9,
                    "top_k": 20 if "hermes4" in self.model else -1,
                    "num_predict": 800
                }
            )
            
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

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
        
        # Use model-specific prompting approaches
        if "gpt-oss" in self.model.lower():
            # GPT-OSS uses thinking field for reasoning, so we need explicit instructions for final output
            reasoning_prompt = (
                "You are an expert medical researcher with a strong understanding of medical and biological semantics. Analyze the abstract carefully and provide your final answer as a JSON object.\n\n"
                "IMPORTANT: After your analysis, you MUST provide a final JSON response in this exact format:\n"
                "FINAL_ANSWER: {\"is_supported\": true/false, \"supporting_sentence\": \"quote\" or null, \"reasoning\": \"brief explanation\"}\n\n"
            )
        else:
            # Hermes 4 and other models use <think></think> tags
            reasoning_prompt = (
                "You are an expert medical researcher with a strong understanding of medical and biological semantics. Analyze the abstract carefully and provide your final answer as a JSON object.\n\n"
                "Use <think></think> tags to systematically reason through the evaluation before providing your final JSON response.\n\n"
            )
        
        # Format them clearly so the LLM knows exactly what to check for
        subject_names_list = subject_names if subject_names else [subject]
        object_names_list = object_names if object_names else [obj]
        
        subject_names_text = f"\n**SUBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
        for i, name in enumerate(subject_names_list, 1):
            subject_names_text += f"  {i}. {name}\n"
        
        object_names_text = f"\n**OBJECT EQUIVALENT NAMES** (check for ANY of these + common abbreviations in the abstract):\n"
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
            # Build list of provided qualifiers for clearer instructions
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
            f"  \"reasoning\": \"Subject: [found/not found] as '[name found in abstract]' (equivalent to '[original name or name from list]'). Object: [found/not found] as '[name found in abstract]' (equivalent to '[original name or name from list]'). Relationship analysis: [detailed and professionalexplanation based on the abstract content]. Evidence category: [category] because [detailed reason based on category definition]. Conclusion: [supported/not supported].\",\n"
            f"  \"subject_mentioned\": true/false,\n"
            f"  \"object_mentioned\": true/false\n"
            f"}}\n\n"
            f"**CONSISTENCY RULES**:\n"
            f"• If no supporting_sentence → is_supported=false\n"
            f"• The logic of the reasoning must align with is_supported value\n"
            f"• Supporting sentence must align with is_supported value\n\n"

        )

        try:
            response_data = await self.generate_response(prompt)
            
            # Handle both Pydantic objects (newer ollama) and dict format (older versions)
            if hasattr(response_data, 'message'):
                # Pydantic object format
                message = response_data.message
                content = message.content if hasattr(message, 'content') and message.content is not None else ""
                thinking_content = message.thinking if hasattr(message, 'thinking') and message.thinking is not None else ""
            else:
                # Dictionary format (legacy)
                message = response_data["message"]
                content = message.get("content", "") or ""
                thinking_content = message.get("thinking", "") or ""
            
            logger.debug(f"Content field: {content[:200]}...")
            logger.debug(f"Thinking field: {thinking_content[:200]}...")
            
            # Handle FINAL_ANSWER: pattern in content (for GPT-OSS)
            if "FINAL_ANSWER:" in content:
                # Extract JSON after FINAL_ANSWER:
                answer_start = content.find("FINAL_ANSWER:") + 13
                json_part = content[answer_start:].strip()
                content = json_part
                logger.info("Found FINAL_ANSWER pattern in content field, extracted JSON")
            
            # Fallback: Handle models that put response in thinking field (legacy support)
            elif not content.strip() and thinking_content:
                logger.info("Using thinking field as model put response there instead of content field")
                
                # Look for FINAL_ANSWER: pattern in thinking content
                if "FINAL_ANSWER:" in thinking_content:
                    # Extract JSON after FINAL_ANSWER:
                    answer_start = thinking_content.find("FINAL_ANSWER:") + 13
                    json_part = thinking_content[answer_start:].strip()
                    content = json_part
                    logger.info("Found FINAL_ANSWER pattern in thinking field")
                else:
                    # Fallback: look for any JSON-like content in the thinking field
                    content = thinking_content
                    logger.debug(f"Thinking field content (first 500 chars): {thinking_content[:500]}")
                    
                    # If no JSON found in thinking, try to extract the decision and create JSON
                    if '{' not in thinking_content:
                        logger.warning("No JSON found in thinking field, attempting to create JSON from reasoning")
                        # Try to determine if it's supported based on keywords in the reasoning
                        is_supported = any(keyword in thinking_content.lower() for keyword in [
                            'supported', 'supports', 'evidence', 'indicates', 'suggests', 'shows', 'demonstrates'
                        ])
                        # Create a basic JSON response
                        content = f'{{"is_supported": {str(is_supported).lower()}, "supporting_sentence": null, "reasoning": "Based on model reasoning in thinking field"}}'
                        logger.info(f"Created JSON from reasoning: {content}")
            
            # New fallback: If content is empty but we have thinking, try to use the thinking content directly
            elif not content.strip() and not thinking_content.strip():
                logger.error("Both content and thinking fields are empty")
                raise ValueError("Model returned empty response in both content and thinking fields")
            
            # Handle Hermes 4 reasoning tags first
            if "</think>" in content:
                # Extract content after </think> tag
                think_end = content.find("</think>") + 8
                content = content[think_end:].strip()
            
            # Extract JSON from response - handle markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # Look for JSON object in content - handle multiline JSON
                import re
                # Find the first { and the last } that would complete a valid JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to simple brace matching
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start and json_end != 0:
                        json_str = content[json_start:json_end]
                    else:
                        logger.error(f"Could not find JSON braces in content. Content: {content[:1000]}")
                        # If JSON is incomplete, try manual extraction
                        logger.warning("Attempting manual extraction from incomplete JSON")
                        evaluation = self._extract_json_manually(content)
                        return evaluation
            
            # Clean and parse JSON
            if json_str:
                # Fix common JSON formatting issues from LLM responses
                json_str = self._fix_json_formatting(json_str)
                
                # Try to parse JSON, with fallback handling
                try:
                    evaluation = json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    # Try to fix common quote issues in JSON strings
                    logger.debug(f"Initial JSON parse failed, attempting to fix quotes: {parse_error}")
                    
                    import re
                    
                    # Strategy 1: Fix unescaped quotes in string fields more comprehensively
                    # Match the field name, opening quote, content (may have quotes), closing quote
                    def fix_field_quotes(field_name, text):
                        """Fix quotes within a specific JSON field."""
                        # Pattern: "field_name": "...content with possible "quotes"..."
                        pattern = rf'("{field_name}":\s*")([^"]*(?:"[^"]*)*")(\s*[,}}])'
                        
                        def replace_fn(match):
                            field_start = match.group(1)  # "field_name": "
                            content = match.group(2)      # content with quotes
                            field_end = match.group(3)    # , or }
                            
                            # Escape all internal quotes in content
                            # Content ends with ", so we need to handle it carefully
                            if content.endswith('"'):
                                actual_content = content[:-1]  # Remove trailing "
                                escaped_content = actual_content.replace('"', '\\"')
                                return field_start + escaped_content + '"' + field_end
                            else:
                                escaped_content = content.replace('"', '\\"')
                                return field_start + escaped_content + field_end
                        
                        return re.sub(pattern, replace_fn, text, flags=re.DOTALL)
                    
                    # Fix quotes in supporting_sentence and reasoning fields
                    json_str = fix_field_quotes('supporting_sentence', json_str)
                    json_str = fix_field_quotes('reasoning', json_str)
                    
                    try:
                        evaluation = json.loads(json_str)
                        logger.debug("Successfully parsed JSON after quote fixing")
                    except json.JSONDecodeError as e2:
                        # Strategy 2: More aggressive - extract field by field
                        logger.debug(f"Advanced quote fixing failed: {e2}, using manual extraction")
                        evaluation = self._extract_json_manually(content)
            else:
                raise ValueError("Empty JSON content extracted.")
            return evaluation
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {content}. Error: {e}")
            # Try manual extraction as final fallback
            try:
                evaluation = self._extract_json_manually(content)
                logger.info("Successfully extracted JSON manually")
                return evaluation
            except Exception as manual_error:
                logger.error(f"Manual extraction also failed: {manual_error}")
                return {
                    "is_supported": False,
                    "supporting_sentence": None,
                    "reasoning": f"LLM response not valid JSON: {content[:200]}..."
                }
        except Exception as e:
            logger.error(f"Error during Ollama evaluation: {e}")
            raise
    
    async def verify_evaluation(self, triple: Union[List[str], 'TripleData'], abstract: str, 
                                original_evaluation: Dict[str, Any], checker_model: str = "gpt-oss:20b") -> Dict[str, Any]:
        """Verify an evaluation result using a checker model, specifically checking name recognition.
        
        Args:
            triple: The research triple (TripleData or list)
            abstract: The abstract text
            original_evaluation: The original evaluation result from another LLM
            checker_model: The model to use for verification (default: gpt-oss:20b)
            
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
        
        # Build verification prompt
        verification_prompt = (
            f"You are an expert medical researcher with a strong understanding of medical and biological semantics.\n"
            f"Your job is to carefully verify and check another expert medical researcher's evaluation result.\n\n"
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
            f"1. **Check entity mentions**:\n"
            f"   - Set subject_mentioned=TRUE if subject  appears ANYWHERE\n"
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
        
        try:
            # Use specified checker model for verification
            verifier = OllamaClient(model=checker_model, base_url=self.base_url)
            response_data = await verifier.generate_response(verification_prompt)
            
            # Extract content
            if hasattr(response_data, 'message'):
                content = response_data.message.content if hasattr(response_data.message, 'content') else ""
            else:
                content = response_data.get("message", {}).get("content", "")
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = self._fix_json_formatting(json_str)
                verification_result = json.loads(json_str)
                
                # If verification found errors, return corrected evaluation
                if not verification_result.get("is_correct", True):
                    logger.info(f"Verification found issues, returning corrected evaluation")
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
            else:
                logger.warning("Could not parse verification response, returning original evaluation")
                return original_evaluation
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            # If verification fails, return original evaluation
            return original_evaluation
