"""Configuration settings for the LLM PMID Checker system."""
import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


DEFAULT_VLLM_MODELS = [
    "gpt-oss-20b-vllm",
    "gpt-oss-120b-vllm",
]

class Settings(BaseModel):
    """Application settings."""
    
    # vLLM configuration
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    vllm_model_urls: Dict[str, str] = Field(default_factory=dict)
    
    # Available vLLM models (comma-separated list from .env)
    available_vllm_models: List[str] = Field(default_factory=list)
    
    # Combined list of all available models
    available_models: List[str] = Field(default_factory=list)
    
    # Default model (first model in available_models)
    default_model: str = ""
    
    # NCBI E-utilities configuration
    ncbi_email: Optional[str] = os.getenv("NCBI_EMAIL")
    ncbi_api_key: Optional[str] = os.getenv("NCBI_API_KEY")
    
    # Request settings
    max_retries: int = 3
    request_timeout: int = 180

    # Batch processing settings
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

    def __init__(self, **data):
        super().__init__(**data)

        # Load available vLLM models from environment
        vllm_models_env = os.getenv("AVAILABLE_VLLM_MODELS")
        if not self.available_vllm_models:
            if vllm_models_env:
                parsed_models = [model.strip() for model in vllm_models_env.split(",") if model.strip()]
                self.available_vllm_models = parsed_models if parsed_models else list(DEFAULT_VLLM_MODELS)
            else:
                self.available_vllm_models = list(DEFAULT_VLLM_MODELS)
        
        # Parse per-model vLLM URLs: "model1=url1,model2=url2"
        vllm_urls_env = os.getenv("VLLM_MODEL_URLS", "")
        if vllm_urls_env and not self.vllm_model_urls:
            for pair in vllm_urls_env.split(","):
                pair = pair.strip()
                if "=" in pair:
                    name, url = pair.split("=", 1)
                    self.vllm_model_urls[name.strip()] = url.strip()
        
        self.available_models = list(self.available_vllm_models)
        
        # Set default model to first available model
        self.default_model = self.available_models[0] if self.available_models else ""
    
    def get_vllm_url(self, model: str) -> str:
        """Get the base URL for a specific vLLM model.
        
        Falls back to the default vllm_base_url if no per-model URL is configured.
        """
        return self.vllm_model_urls.get(model, self.vllm_base_url)
    
    def validate_model(self, model: str) -> str:
        """Validate that a model name is in the available models list.
        
        Args:
            model: Model name to validate
            
        Returns:
            The validated model name
            
        Raises:
            ValueError: If model is not in available_models
        """
        if not model:
            raise ValueError("Model name cannot be empty")
        
        if model not in self.available_models:
            raise ValueError(
                f"Model '{model}' is not available. "
                f"Available vLLM models: {', '.join(self.available_vllm_models)}. "
                f"Please check your AVAILABLE_VLLM_MODELS environment variable in .env or update your model selection."
            )
        
        return model

# Global settings instance
settings = Settings()
