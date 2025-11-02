import json
import os
import ssl
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from asyncio import BoundedSemaphore
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Set up basic logging to see retries and errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Default URL from environment variable or fallback
DEFAULT_LLM_ROUTER_URL = os.getenv("LLM_ROUTER_URL", None)
if DEFAULT_LLM_ROUTER_URL is None:
    raise ValueError("LLM_ROUTER_URL environment variable is not set")


class SimpleLLMCaller:
    """
    A simplified, independent LLM caller based on the provided BaseService.
    
    This class handles async requests, rate limiting (semaphore),
    and retries.
    """

    def __init__(self, **kwargs):
        """
        Initializes the caller.
        
        Args:
            **kwargs: Configuration options, including:
                url (str): The endpoint URL to call.
                semaphore (int): Max concurrent requests. Default: 30
                timeout (int): Request timeout in seconds. Default: 60
                max_retries (int): Max retry attempts. Default: 2
                backoff_multiplier (int): Multiplier for retry delay. Default: 1
                headers (dict): Default request headers.
        """
        self.url = kwargs.get("url", DEFAULT_LLM_ROUTER_URL)
        self.proxy = kwargs.get("proxy")
        
        # Concurrency and retry settings from BaseService
        self.semaphore = BoundedSemaphore(kwargs.get("semaphore", 30))
        self.timeout = ClientTimeout(total=kwargs.get("timeout", 60))
        self.max_retries = kwargs.get("max_retries", 4)
        self.backoff_multiplier = kwargs.get("backoff_multiplier", 1)
        self.initial_delay = kwargs.get("delay", 1) # From BaseService
        
        self.headers = kwargs.get("headers", {"Content-Type": "application/json"})
        
        # Re-use the session parameter logic for TLS/SSL from BaseService
        self.client_session_parameters = self._get_client_session_parameters()

    def _get_client_session_parameters(self) -> Dict:
        """
        Gets parameters for the ClientSession, including SSL context
        if required by environment variables.
        (Copied from BaseService)
        """
        parameters = {}
        if os.getenv("TLS", False):
            certificate_authority_file = os.getenv("CACERT")
            ssl_context = ssl.create_default_context(
                cafile=certificate_authority_file
            )
            connector = TCPConnector(ssl_context=ssl_context)
            parameters["connector"] = connector
        return parameters

    def _build_payload(self, 
                       prompt: str, 
                       model: str, 
                       system_prompt: Optional[str] = None,
                       generation_params: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        """
        Builds the chat completion request payload in the format
        expected by the router.
        """
        # Start with base generation parameters
        gen_params = generation_params.copy() if generation_params else {}
        
        # Add required model and other parameters from your files
        gen_params.update({
            "model": model,
            "temperature": gen_params.get("temperature", 0),
            "client_identifier": gen_params.get("client_identifier", "simple-llm-caller"),
            "provider": gen_params.get("provider", "OPEN_AI"),
            "max_tokens": gen_params.get("max_tokens", 4096)
        })

        # Build the final payload
        payload = gen_params
        payload["messages"] = []
        
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
            
        payload["messages"].append({"role": "user", "content": prompt})
        
        return payload

    async def call(self,
                 prompt: str, 
                 model: str, 
                 system_prompt: Optional[str] = None,
                 generation_params: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
        """
        Makes a single, retriable call to the LLM.

        Args:
            prompt: The user prompt text.
            model: The model name (e.g., "gpt-4.1-mini").
            system_prompt: An optional system prompt.
            generation_params: Optional extra parameters for the model.

        Returns:
            The full JSON response dictionary from the server.
            
        Raises:
            Exception: If all retry attempts fail.
        """
        request_payload = self._build_payload(
            prompt, model, system_prompt, generation_params
        )
        
        kwargs = {
            "url": self.url,
            "data": json.dumps(request_payload), # Use standard json.dumps
            "proxy": self.proxy,
            "timeout": self.timeout,
            "headers": self.headers
        }

        attempts = 0
        delay = self.initial_delay
        last_exception = None

        while attempts <= self.max_retries:
            attempts += 1
            try:
                # Create a new session for each attempt to avoid issues
                async with ClientSession(**self.client_session_parameters) as client_session:
                    # Wait for semaphore
                    async with self.semaphore:
                        async with client_session.post(**kwargs) as response:
                            if response.status == 200:
                                response_payload = await response.json()
                                # Basic validation
                                if response_payload.get("success") is False:
                                    raise Exception(f"API indicated failure: {response_payload}")
                                return response_payload
                            else:
                                raise Exception(
                                    f"Failed to get response from server [Status Code {response.status}]"
                                )
                                
            except Exception as e:
                last_exception = e
                logging.warning(
                    f"Attempt {attempts}/{self.max_retries + 1} failed: {e}"
                )
                if attempts > self.max_retries:
                    logging.error(
                        f"All retries failed for prompt: {prompt[:50]}..."
                    )
                    raise last_exception  # Re-raise the last exception
                
                # Wait before retrying
                await asyncio.sleep(delay)
                delay *= self.backoff_multiplier
        
        # This line should not be reachable, but as a fallback
        raise last_exception or Exception("LLM call failed after all retries.")


    async def call_and_get_text(self,
                                prompt: str, 
                                model: str, 
                                system_prompt: Optional[str] = None,
                                generation_params: Optional[Dict[str, Any]] = None
                                ) -> str:
        """
        A helper method that calls the LLM and extracts just the
        response text.

        Returns:
            The string content from the LLM's response.
            Returns an empty string if parsing fails.
        """
        try:
            response_payload = await self.call(
                prompt, model, system_prompt, generation_params
            )
            # Extract text from the common chat completion format
            return response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Could not parse text from response: {e}")
            return ""
        except Exception as e:
            logging.error(f"API call failed entirely: {e}")
            return ""


# --- Example Usage ---
async def main():
    """
    Demonstrates how to use the SimpleLLMCaller.
    """
    
    # Initialize the caller
    # You can override defaults here, e.g., max_retries=5
    caller = SimpleLLMCaller(max_retries=1) 
    
    # --- Example 1: Get text response directly ---
    print("--- Running Example 1 (Text Response) ---")
    prompt_1 = "Hello! Who are you and what can you do?"
    model_1 = "gpt-4.1-mini" # Using the TEST_MODEL from your script
    
    response_text = await caller.call_and_get_text(prompt_1, model_1)
    
    print(f"Prompt: {prompt_1}")
    print(f"Model: {model_1}")
    print(f"Response:\n{response_text}\n")
    
    # --- Example 2: Get full JSON response with system prompt ---
    print("--- Running Example 2 (Full JSON Response) ---")
    prompt_2 = "List three facts about the Roman Empire."
    model_2 = "gpt-4.1" # Using the JUDGE_MODEL from your script
    system_2 = "You are a helpful assistant that speaks like a pirate."
    
    try:
        response_json = await caller.call(prompt_2, model_2, system_prompt=system_2)
        print(f"Prompt: {prompt_2}")
        print(f"Model: {model_2}")
        print(f"System Prompt: {system_2}")
        print("Full JSON Response:")
        # Pretty-print the JSON response
        print(json.dumps(response_json, indent=2))
        
    except Exception as e:
        print(f"Example 2 failed after retries: {e}")


if __name__ == "__main__":
    # This script is runnable directly
    # Note: Requires 'aiohttp' to be installed (pip install aiohttp)
    asyncio.run(main())