from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, Optional, Dict

@dataclass
class OpenAIGeneratorConfig:
    """OpenAI generator configuration."""
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: int | None = None
    system_prompt: str | None = None
    timeout: float = 60.0
    use_schema: bool = False


@dataclass
class VLLMGeneratorConfig:
    """
    vLLM generator (local OpenAI-compatible /v1/chat/completions client).
    These fields configure the HTTP client, not the vLLM server launch arguments.
    """
    model_name: str = "Qwen2.5-7B-Instruct"   
    api_key: Optional[str] = "empty"          
    base_url: str = "http://localhost:8000/v1"
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0
    use_schema: bool = False