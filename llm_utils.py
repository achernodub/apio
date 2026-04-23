from llm_utils_external import *


def llm_single(use_llm_proxy: bool, prompt: str, model_name: str, temperature: str, max_output_tokens: int,
               top_p: float, **kwargs) -> str:
    if use_llm_proxy:
        raise NotImplementedError("LLM proxy backend is not available in this build; set use_llm_proxy: False.")
    if model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini']:
        return llm_openai(prompt, model_name, temperature, max_output_tokens, top_p)
    elif model_name == 'gemini-1.5-flash-002':
        return llm_gemini_15(prompt, model_name, temperature, max_output_tokens, **kwargs)
    elif model_name == 'gemini-exp-1206':
        return llm_gemini_1206(prompt, model_name, temperature, max_output_tokens, **kwargs)
    else:
        return llm_lmstudio(prompt, model_name, temperature, max_output_tokens, top_p, **kwargs)


def llm_batch(use_llm_proxy: bool, prompts: list[str], model_name: str, temperature: float, max_output_tokens: int,
              batch_size: int, top_p: float, verbose: bool = True, **kwargs) -> list[str]:
    if use_llm_proxy:
        raise NotImplementedError("LLM proxy backend is not available in this build; set use_llm_proxy: False.")
    if model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini']:
        return llm_openai_batched(prompts=prompts, model_name=model_name, temperature=temperature,
                                  max_output_tokens=max_output_tokens, batch_size=batch_size, top_p=top_p,
                                  verbose=verbose, **kwargs)
    else:
        return llm_lmstudio_batched(prompts=prompts, model_name=model_name, temperature=temperature,
                                    max_output_tokens=max_output_tokens, batch_size=batch_size, top_p=top_p,
                                    verbose=verbose, **kwargs)
