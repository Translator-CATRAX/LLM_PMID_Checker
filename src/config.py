"""Configuration settings for the LLM PMID Checker system."""
import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


DEFAULT_OLLAMA_MODELS = [
    "hermes4:70b-q4-m",
    "gpt-oss:20b",
]

DEFAULT_OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5"
]

DEFAULT_VLLM_MODELS = [
    "hermes4-vllm",
    "gpt-oss-20b-vllm",
    "gpt-oss-120b-vllm",
]

class Settings(BaseModel):
    """Application settings."""
    
    # Ollama configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # vLLM configuration
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    vllm_model_urls: Dict[str, str] = Field(default_factory=dict)
    
    # Available Ollama models (comma-separated list from .env)
    available_ollama_models: List[str] = Field(default_factory=list)
    
    # OpenAI configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # OpenAI web search tool (adds ~$10-50 per 1,000 searches)
    openai_enable_web_search: bool = os.getenv("OPENAI_ENABLE_WEB_SEARCH", "false").lower() == "true"
    
    # Available OpenAI models (comma-separated list from .env)
    available_openai_models: List[str] = Field(default_factory=list)
    
    # Available vLLM models (comma-separated list from .env)
    available_vllm_models: List[str] = Field(default_factory=list)
    
    # Combined list of all available models
    available_models: List[str] = Field(default_factory=list)
    
    # Default model (first model in available_models)
    default_model: str = ""
    
    # NCBI E-utilities configuration
    ncbi_email: Optional[str] = os.getenv("NCBI_EMAIL")
    ncbi_api_key: Optional[str] = os.getenv("NCBI_API_KEY")
    
    # UMLS configuration
    umls_api_key: Optional[str] = os.getenv("UMLS_API_KEY")
    use_umls: bool = os.getenv("USE_UMLS", "true").lower() == "true"
    
    # Request settings
    max_retries: int = 3
    request_timeout: int = 180

    # Batch processing settings
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Ollama batch processing environment variables
    # These should be set at the system level when running Ollama server
    ollama_num_parallel: Optional[str] = os.getenv("OLLAMA_NUM_PARALLEL")
    ollama_max_loaded_models: Optional[str] = os.getenv("OLLAMA_MAX_LOADED_MODELS")
    ollama_max_queue: Optional[str] = os.getenv("OLLAMA_MAX_QUEUE")
    ollama_keep_alive: Optional[str] = os.getenv("OLLAMA_KEEP_ALIVE")

    def __init__(self, **data):
        super().__init__(**data)

        # Load available Ollama models from environment
        ollama_models_env = os.getenv("AVAILABLE_OLLAMA_MODELS")
        if not self.available_ollama_models:
            if ollama_models_env:
                parsed_models = [model.strip() for model in ollama_models_env.split(",") if model.strip()]
                self.available_ollama_models = parsed_models if parsed_models else list(DEFAULT_OLLAMA_MODELS)
            else:
                self.available_ollama_models = list(DEFAULT_OLLAMA_MODELS)
        
        # Load available OpenAI models from environment
        openai_models_env = os.getenv("AVAILABLE_OPENAI_MODELS")
        if not self.available_openai_models:
            if openai_models_env:
                parsed_models = [model.strip() for model in openai_models_env.split(",") if model.strip()]
                self.available_openai_models = parsed_models if parsed_models else list(DEFAULT_OPENAI_MODELS)
            else:
                self.available_openai_models = list(DEFAULT_OPENAI_MODELS)
        
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
        
        # Combine all model lists into available_models
        self.available_models = (
            self.available_ollama_models + 
            self.available_openai_models + 
            self.available_vllm_models
        )
        
        # Set default model to first available model
        self.default_model = self.available_models[0] if self.available_models else ""
    
    def is_openai_model(self, model: str) -> bool:
        """Check if a model is an OpenAI model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is an OpenAI model, False otherwise
        """
        return model in self.available_openai_models
    
    def is_ollama_model(self, model: str) -> bool:
        """Check if a model is an Ollama model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is an Ollama model, False otherwise
        """
        return model in self.available_ollama_models
    
    def is_vllm_model(self, model: str) -> bool:
        """Check if a model is a vLLM model."""
        return model in self.available_vllm_models
    
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
                f"Available Ollama models: {', '.join(self.available_ollama_models)}. "
                f"Available OpenAI models: {', '.join(self.available_openai_models)}. "
                f"Available vLLM models: {', '.join(self.available_vllm_models)}. "
                f"Please check your AVAILABLE_OLLAMA_MODELS, AVAILABLE_OPENAI_MODELS, and AVAILABLE_VLLM_MODELS environment variables in .env or update your model selection."
            )
        
        return model

# Global settings instance
settings = Settings()