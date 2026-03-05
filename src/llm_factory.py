"""Factory for creating LLM clients based on configuration."""
import logging
from typing import Any
from .config import settings
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .vllm_client import VLLMClient

logger = logging.getLogger(__name__)

def create_llm_client(model_name: str) -> Any:
    """Create LLM client based on model name.
    
    Args:
        model_name: Model name to use (must be in settings.available_models)
                   Can also use aliases like 'hermes4-fast' or 'hermes4-ultrafast'
                 
    Returns:
        OllamaClient or OpenAIClient instance configured for the specified model
        
    Raises:
        ValueError: If model_name is not in available_models
    """
    
    model_aliases = {
        "hermes4": settings.default_model if 'hermes4' in settings.default_model else "hermes4:70b-q4-m",
        "hermes4-fast": "hermes4:70b-q4-s",
        "hermes4-ultrafast": "hermes4:70b-iq4-xs",
        "gpt-oss": "gpt-oss:20b",
        "gpt-oss-20b": "gpt-oss-20b-vllm",
        "gpt-oss-120b": "gpt-oss-120b-vllm",
    }
    
    # Resolve alias if provided
    model_lower = model_name.lower()
    if model_lower in model_aliases:
        model_name = model_aliases[model_lower]
    
    # Validate model is in available models
    try:
        validated_model = settings.validate_model(model_name)
    except ValueError as e:
        logger.error(str(e))
        raise
    
    # Route to appropriate client based on model type
    if settings.is_openai_model(validated_model):
        logger.info(f"Using OpenAI client - Model: {validated_model}")
        if settings.openai_enable_web_search:
            logger.warning("OpenAI web_search tool enabled - Additional costs apply (~$10-50 per 1,000 searches)")
        return OpenAIClient(
            model=validated_model,
            api_key=settings.openai_api_key,
            enable_web_search=settings.openai_enable_web_search
        )
    elif settings.is_vllm_model(validated_model):
        vllm_url = settings.get_vllm_url(validated_model)
        logger.info(f"Using vLLM client - Model: {validated_model}, URL: {vllm_url}")
        return VLLMClient(
            model=validated_model,
            base_url=vllm_url
        )
    else:
        logger.info(f"Using Ollama client - Model: {validated_model}")
        return OllamaClient(
            model=validated_model,
            base_url=settings.ollama_base_url
        )