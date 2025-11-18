"""Grok API client for ATS Brain.

This module provides a thin wrapper around the Grok 4 Fast chat completions API
for use as the LLM-powered scoring engine.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import requests
from loguru import logger

from core.config import settings


class GrokClient:
    """Client for interacting with the Grok 4 Fast chat completions API."""

    _BASE_URL: str = "https://api.x.ai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key: str | None = api_key or settings.GROK_API_KEY
        self.model: str = model or getattr(settings, "MODEL_NAME", "grok-4-fast-reasoning")

        if not self.api_key:
            logger.error("GROK_API_KEY is not configured. GrokClient will not be able to send requests.")
            raise ValueError("GROK_API_KEY is required for GrokClient")

    def run(self, prompt: str) -> str:
        """Run a chat completion request against Grok 4 Fast.

        Parameters
        ----------
        prompt: str
            The prompt to send to the model.

        Returns
        -------
        str
            The message content returned by the model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
        }

        try:
            response = requests.post(self._BASE_URL, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error(f"Error while calling Grok API: {exc}")
            raise RuntimeError("Failed to call Grok API") from exc

        logger.debug(f"Grok raw response text: {response.text}")

        try:
            data: Dict[str, Any] = response.json()
        except ValueError as exc:
            logger.error("Failed to parse Grok API response as JSON. Raw text: {}", response.text)
            raise RuntimeError("Invalid response from Grok API") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Grok API response structure: {data}")
            raise RuntimeError("Unexpected Grok API response structure") from exc
