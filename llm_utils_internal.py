import concurrent.futures
from tqdm import tqdm
import json
import logging
import os
import requests
import random
import time
import ast
import re

JSON_RE = r"""\s*\[[\S\n\t\v ]+\]\s*"""

# LLM Proxy
API_URL = os.getenv('LLMPROXY_ENDPOINT_URL', 'http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy')
# Wait time parameters if throttled
LLMPROXY_THROTTLE_WAIT_TIME = int(os.getenv('LLMPROXY_THROTTLE_WAIT_TIME', "15"))
LLMPROXY_THROTTLE_EXPBACKOFF = 2
RETRY_CODES = {429, 502, 529}
GUARDRAIL_CODES = {400, 451, 551}
MAX_RETRIES = int(os.getenv('LLMPROXY_MAX_RETRIES', "8"))


def is_response_an_error(response, is_transparent_proxy=False):
    response_json = ""
    # If response OK
    if response.status_code == 200:
        try:
            response_json = response.json()
        # JSON parsing error
        except Exception:
            logging.error(f"Error converting json response: {response.text}. Retrying")
            return True
        else:
            # 'error' is raised in the response
            if 'error' in response_json:
                error_code = int(response_json['error']['code'])

                # Handle Retry Codes
                if error_code in RETRY_CODES:
                    logging.warning(f"Error {error_code}. Will retry")
                # Handle Guardrail Codes
                if error_code in GUARDRAIL_CODES:
                    logging.error(f"Guardrails error: {error_code}. "
                                  f"Will retry. Response JSON: {response_json}")
                else:
                    logging.error(f"Unhandled error code {error_code}")
                    logging.error(f"Failing with message {response_json}")
                return True
            if is_transparent_proxy:
                return False
            else:
                if 'chunk' in response.json():
                    return False
                else:
                    # 'chunk' not found in the response
                    logging.error(f"No chunk found: {response.json()}")
                    return True
    else:
        logging.error(f"Unknown response code: {response}. Retrying.")
        return True


def generate_from_messages_transparent_proxy(
        messages,
        model: str = "gpt-4o",
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.7,
        top_p: int = 1,
        max_tokens: int = 4096,
        is_test: bool = False,
        **kwargs
):
    API_URL = os.getenv('LLMPROXY_TRANSPARENT_ENDPOINT_URL',
                        'http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1/chat/completions')
    headers = {
        "Content-Type": "text/plain",
        "X-LLM-Proxy-Calling-Service": "vivek.kulkarni@grammarly.com",
    }
    data = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        **kwargs
    }
    retry_delay = RetryWithBackOff(
        max_retries=max_retries,
        initial_delay=LLMPROXY_THROTTLE_WAIT_TIME,
        exp_factor=LLMPROXY_THROTTLE_EXPBACKOFF)

    while retry_delay.retry():
        response = None
        with requests.Session() as session:
            response = session.post(API_URL,
                                    headers=headers,
                                    data=json.dumps(data))
        if not is_response_an_error(response, is_transparent_proxy=True):
            return response.json()
    # Failure after retrying
    raise ValueError("Failed after retries")


def generate_from_messages(
        messages,
        model: str = "oai_chat_gpt4o_2024_05_13_standard_research",
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.7,
        top_p: int = 1,
        max_tokens: int = 4096,
        is_test: bool = False,
        **kwargs
):
    if 'llama3' in model:
        # TODO: Remove this temporary fix once LLM Proxy team updates the generation parameter names for Llama models.
        # This is needed since currently Llama3 has a different parameter name for max tokens and has a upper limit on it.
        tok_length_param = 'max_gen_len'
        max_tokens = min(max_tokens, 2048)
    else:
        tok_length_param = 'max_tokens'

    headers = {
        "Content-Type": "text/plain",
    }
    data = {
        "tracking_id": "1234567890",
        "messages": messages,
        "llm_backend": model,
        "metadata": {"json": "{}"},
        "tags": {
            "client_name": "postman",
        },
        "generation_parameters": {
            "json": json.dumps(
                {tok_length_param: max_tokens, "temperature": temperature, "top_p": top_p, **kwargs}
                # Pass in other arguments that caller has set.
            ),
        },
        "moderation": {
            "state": "disabled"
        }
    }

    # Use token-less proxy for testing
    if is_test:
        data['tags']['calling_service'] = "service_name"
    else:
        # comma-separated list of OpenAI keys to use.
        API_KEYS = os.environ["LLM_PROXY_KEY"].split(',')
        data["api_token"] = random.choice(API_KEYS)

    retry_delay = RetryWithBackOff(
        max_retries=max_retries,
        initial_delay=LLMPROXY_THROTTLE_WAIT_TIME,
        exp_factor=LLMPROXY_THROTTLE_EXPBACKOFF)

    while retry_delay.retry():
        response = None
        with requests.Session() as session:
            response = session.post(API_URL,
                                    headers=headers,
                                    data=json.dumps(data))
        if not is_response_an_error(response, is_transparent_proxy=False):
            return response.json()['chunk']['text']
    # Failure after retrying
    raise ValueError("Failed after retries")


def generate(
        prompt,
        model: str = "oai_chat_gpt4o_2024_05_13_standard_research",
        temperature: float = 0.7,
        top_p: int = 1,
        max_tokens: int = 4096,
        max_retries: int = MAX_RETRIES,
        is_test: bool = False,
        use_transparent_proxy=False,
        **kwargs
) -> str:
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt
    # logging.info(f"Executing prompt for model: {model}")
    if use_transparent_proxy:
        return generate_from_messages_transparent_proxy(messages, model, max_retries, temperature, top_p, max_tokens,
                                                        **kwargs)
    else:
        return generate_from_messages(messages, model, max_retries, temperature, top_p, max_tokens, is_test, **kwargs)


def generate_wrapper(args):
    prompt, model, temperature, max_output_tokens, top_p = args
    return generate(prompt=prompt, model=model, temperature=temperature,
                    top_p=top_p, max_tokens=max_output_tokens)


def generate_batched(
        prompts: list[str],
        batch_size: int,
        model: str = "oai_chat_gpt4o_2024_05_13_standard_research",
        temperature: float = 0.7,
        top_p: int = 1,
        max_tokens: int = 4096,
        verbose: bool = True,
        **kwargs
) -> list[str]:
    output_texts = []
    inputs = [(prompt, model, temperature, max_tokens, top_p) for prompt in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        results = list(tqdm(executor.map(generate_wrapper, inputs), total=len(inputs),
                            desc=f'Processing prompts by {model}', disable=not verbose))
        output_texts.extend(results)

    return output_texts


class RetryWithBackOff:
    """
    Exponential backoff for reties with LLM Proxy API
    """

    def __init__(self, max_retries: int = MAX_RETRIES,
                 initial_delay: float = LLMPROXY_THROTTLE_WAIT_TIME,
                 exp_factor: float = LLMPROXY_THROTTLE_EXPBACKOFF):
        self.max_retries = max(1, max_retries)
        self.retry_delay = initial_delay
        self.exp_factor = exp_factor
        self.attempts = 0

    def retry(self) -> bool:
        if self.attempts == 0:
            self.attempts += 1
            return True
        if self.attempts < self.max_retries:
            logging.info(f"Retrying after {self.retry_delay} seconds")
            time.sleep(self.retry_delay)
            self.retry_delay *= self.exp_factor * (1 + random.random())
            self.attempts += 1
            return True
        else:
            return False