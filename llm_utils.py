from llm_utils_external import *
from llm_utils_internal import *

llm_mapping = {"gpt-4o": "openai_direct_chat_gpt4o", #"oai_chat_gpt4o_2024_05_13_standard_research",
               "gpt-4o-mini": "openai_direct_chat_gpt4o_mini",
               "gpt-4": "openai_direct_chat_gpt4_0125",
               "llama31-70b": "aws_bedrock_llama31_70b",
               "llama3-405b": "aws_bedrock_llama31_405b"
            }
def llm_single(use_llm_proxy: bool, prompt: str, model_name: str, temperature: str, max_output_tokens: int,
               top_p: float, **kwargs) -> str:
    if use_llm_proxy:
        top_p = int(top_p)
        temperature = float(temperature)
        if model_name not in llm_mapping:
            raise Exception(f"Model {model_name} is not supported. Only Supported models are {llm_mapping.keys()}")
        response = generate(prompt=prompt, model=llm_mapping[model_name], temperature=temperature,
                        top_p=top_p, max_tokens=max_output_tokens, **kwargs)
        return response
    else:
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
        top_p = int(top_p)
        if model_name not in llm_mapping:
            raise Exception(f"Model {model_name} is not supported. Only Supported models are {llm_mapping.keys()}")
        response = generate_batched(prompts=prompts, batch_size=batch_size,
                                model=llm_mapping[model_name], temperature=temperature,
                                top_p=top_p, max_tokens=max_output_tokens, **kwargs)
        # print("Batched Response: ", response)
        return response

    else:
        if model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini']:
            return llm_openai_batched(prompts=prompts, model_name=model_name, temperature=temperature,
                                      max_output_tokens=max_output_tokens, batch_size=batch_size, top_p=top_p,
                                      verbose=verbose, **kwargs)
        else:
            return llm_lmstudio_batched(prompts=prompts, model_name=model_name, temperature=temperature,
                                        max_output_tokens=max_output_tokens, batch_size=batch_size, top_p=top_p,
                                        verbose=verbose, **kwargs)
