from abc import ABC, abstractmethod
import requests
import logging
import time
import json
import re
from typing import Optional, Dict, Any


class BaseAgent(ABC):
    """Base class for all AI agents using Gemini"""

    def __init__(self, api_key: str, base_url: str, model: str, reasoning_tokens: Optional[float] = None, reasoning_effort: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.reasoning_tokens = reasoning_tokens
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(self.__class__.__name__)
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 1.0,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Call LLM with retry logic and error handling using HTTPS API"""

        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if self.reasoning_tokens is not None:
                    payload["reasoning"] = {"max_tokens": int(self.reasoning_tokens), "enabled": True}
                    
                if self.reasoning_effort is not None:
                    payload["reasoning"] = {"effort": self.reasoning_effort, "enabled": True}

                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                url = f"{self.base_url}/chat/completions"
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                response.raise_for_status()
                response_data = response.json()

                if "usage" in response_data:
                    self.prompt_tokens += response_data["usage"].get("prompt_tokens", 0)
                    self.completion_tokens += response_data["usage"].get("completion_tokens", 0)
                    self.total_tokens += response_data["usage"].get("total_tokens", 0)

                content = response_data["choices"][0]["message"]["content"]
                self.logger.info(f"LLM call successful (attempt {attempt + 1})")

                return content

            except Exception as e:
                self.logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        f"LLM call failed after {max_retries} attempts: {e}"
                    )

    def _call_llm_structured(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 1.0,
        max_retries: int = 3,
    ) -> Dict[Any, Any]:
        """Call LLM and return parsed JSON response"""

        response = self._call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_retries=max_retries,
            json_mode=True,
        )

        # Sanitize JSON output before parsing
        sanitized_response = self._sanitize_json_output(response)

        try:
            return json.loads(sanitized_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {response[:200]}...")

    def get_token_usage(self) -> Dict[str, int]:
        """Return token usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _sanitize_json_output(self, text: str) -> str:
        """Remove ```json ``` code blocks from LLM output"""
        # Remove ```json``` blocks
        sanitized = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        sanitized = re.sub(r'```\s*$', '', sanitized)
        return sanitized.strip()

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the agent's main task - to be implemented by subclasses"""
        pass
