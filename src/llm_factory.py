"""Factory for creating LLM clients based on configuration."""
import logging
from typing import Any
from .config import settings

logger = logging.getLogger(__name__)

def create_llm_client(model_name: str) -> Any:
    """Create a vLLM client for the specified model.
    
    Args:
        model_name: Model name to use (must be in settings.available_models)
                   Can also use aliases like 'gpt-oss-20b' or 'gpt-oss-120b'
                 
    Returns:
        VLLMClient instance configured for the specified model
        
    Raises:
        ValueError: If model_name is not in available_models
    """
    
    model_aliases = {
        "gpt-oss-20b": "gpt-oss-20b-vllm",
        "gpt-oss-120b": "gpt-oss-120b-vllm",
    }
    
    model_lower = model_name.lower()
    if model_lower in model_aliases:
        model_name = model_aliases[model_lower]
    
    try:
        validated_model = settings.validate_model(model_name)
    except ValueError as e:
        logger.error(str(e))
        raise
    
    from .vllm_client import VLLMClient
    vllm_url = settings.get_vllm_url(validated_model)
    logger.info(f"Using vLLM client - Model: {validated_model}, URL: {vllm_url}")
    return VLLMClient(
        model=validated_model,
        base_url=vllm_url
    )
