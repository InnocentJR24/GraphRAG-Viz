"""
Configuration module for GraphRAG-Viz pipeline.
Provides centralized configuration for all pipeline components.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    """Configuration for the GraphRAG pipeline with Glass Box implementation."""
    
    # LLM Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Text Processing
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Graph Configuration
    max_communities: int = int(os.getenv("MAX_COMMUNITIES", "10"))
    
    # Interpretability Settings
    enable_logging: bool = True
    save_intermediate_results: bool = True
    output_dir: str = "output"
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return True


# Global configuration instance
config = PipelineConfig()
