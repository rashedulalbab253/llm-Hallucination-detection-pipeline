"""
Custom LLM model wrappers for DeepEval benchmark evaluation.

DeepEval requires models to inherit from DeepEvalBaseLLM and implement
the `generate`, `a_generate`, `load_model`, and `get_model_name` methods.
This module provides wrappers for OpenAI GPT models.
"""

import asyncio
from typing import Optional
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM
from src.config import OPENAI_API_KEY


class GPTModel(DeepEvalBaseLLM):
    """
    Wrapper around OpenAI GPT models for DeepEval benchmarking.
    
    Supports both synchronous and asynchronous generation,
    which is required by DeepEval's benchmark evaluation engine.
    
    Parameters
    ----------
    model_name : str
        OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini").
    temperature : float
        Sampling temperature (0.0 = deterministic, higher = more random).
    max_tokens : int
        Maximum tokens in the response.
    api_key : str, optional
        OpenAI API key. Falls back to environment variable.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or OPENAI_API_KEY

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in your .env "
                "file or pass api_key to GPTModel()."
            )

        self._client = OpenAI(api_key=self._api_key)
        self._async_client = AsyncOpenAI(api_key=self._api_key)

    def load_model(self):
        """Return the underlying OpenAI client."""
        return self._client

    def generate(self, prompt: str) -> str:
        """
        Synchronously generate a response from the model.
        
        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
            
        Returns
        -------
        str
            The model's text response.
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the model.
        
        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
            
        Returns
        -------
        str
            The model's text response.
        """
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self.model_name

    def __repr__(self) -> str:
        return (
            f"GPTModel(model_name='{self.model_name}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
