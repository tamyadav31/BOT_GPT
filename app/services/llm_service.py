"""
LLM service for chat completions.

Integrates with external LLM providers (e.g., Groq, HuggingFace, OpenAI-compatible).
Handles:
- API communication
- Error handling and retries
- Timeout management
- Response validation
"""
import logging
from typing import List, Dict, Optional

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""
    pass


class LLMService:
    """
    Service for interacting with LLM APIs.
    
    Supports OpenAI-compatible endpoints (Groq, HuggingFace, etc.).
    """

    def __init__(self) -> None:
        """
        Initialize LLM service with configuration.
        
        Raises:
            ValueError: If critical configuration is invalid.
        """
        settings = get_settings()  # Load fresh settings each time
        self.api_key = settings.LLM_API_KEY
        self.api_base_url = settings.LLM_API_BASE_URL
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT

        if not self.api_key:
            logger.warning("LLM_API_KEY not set. LLM calls will fail.")
        
        logger.info(f"LLM Service initialized with model: {self.model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Call the LLM API for chat completion.
        
        Synchronous method (use for blocking calls).
        For async, wrap with asyncio.to_thread or use async HTTP client.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Expected roles: "system", "user", "assistant".
            temperature: Sampling temperature (0.0 to 2.0). Default: 0.7.
            max_tokens: Maximum tokens in response. Default: 1000.

        Returns:
            Assistant response text (stripped).

        Raises:
            LLMServiceError: If API call fails or response is invalid.
            ValueError: If messages list is empty.
        """
        if not messages:
            raise ValueError("messages list cannot be empty")
        
        if not self.api_key:
            raise LLMServiceError("LLM_API_KEY is not configured")

        # Validate temperature range
        if not (0.0 <= temperature <= 2.0):
            logger.warning(f"Temperature {temperature} outside recommended range [0.0, 2.0]")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = self._parse_response(response.json())
            logger.info(f"LLM call successful. Model: {self.model}")
            return result

        except requests.exceptions.Timeout:
            logger.error(f"LLM API timeout after {self.timeout}s")
            raise LLMServiceError(f"LLM API timeout after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text if hasattr(e.response, 'text') else 'No error details'
            logger.error(f"LLM API HTTP error: {status_code}")
            logger.error(f"Error response: {error_text}")
            if status_code == 401:
                raise LLMServiceError("LLM API authentication failed (invalid API key)")
            elif status_code == 429:
                raise LLMServiceError("LLM API rate limit exceeded")
            elif status_code == 400:
                raise LLMServiceError(f"LLM API bad request: {error_text}")
            elif status_code >= 500:
                raise LLMServiceError(f"LLM API server error ({status_code})")
            else:
                raise LLMServiceError(f"LLM API error: {status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request error: {str(e)}")
            raise LLMServiceError(f"LLM API request failed: {str(e)}")

    def _parse_response(self, response_json: Dict) -> str:
        """
        Parse and validate LLM API response.
        
        Args:
            response_json: JSON response from LLM API.
            
        Returns:
            Extracted assistant message content.
            
        Raises:
            LLMServiceError: If response format is invalid.
        """
        try:
            # Validate response structure
            if "choices" not in response_json or not response_json["choices"]:
                raise LLMServiceError("Invalid LLM response: missing or empty 'choices'")
            
            choice = response_json["choices"][0]
            if "message" not in choice:
                raise LLMServiceError("Invalid LLM response: missing 'message' in choice")
            
            message = choice["message"]
            if "content" not in message:
                raise LLMServiceError("Invalid LLM response: missing 'content' in message")
            
            content = message["content"]
            if not isinstance(content, str):
                raise LLMServiceError("Invalid LLM response: 'content' is not a string")
            
            return content.strip()

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise LLMServiceError(f"Unexpected LLM response format: {str(e)}")


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get or create LLM service instance (singleton).
    
    Returns:
        LLMService: Singleton instance.
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
