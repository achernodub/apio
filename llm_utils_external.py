import concurrent.futures
import litellm
import math
import time
import vertexai
from openai import OpenAI
from tqdm import tqdm
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

def llm_lmstudio(prompt: str, model_name: str, temperature: float, max_output_tokens: int, top_p: float,
                 **kwargs) -> str:
    client = OpenAI(base_url='http://localhost:1234/v1', api_key='lm-studio')
    completion = client.chat.completions.create(model=model_name,
                                                messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                                                          {'role': 'user', 'content': prompt}],
                                                temperature=temperature,
                                                max_tokens=max_output_tokens,
                                                top_p=top_p,
                                                **kwargs)

    output_text = completion.choices[0].message.content
    return output_text


def llm_repack_parameters_wrapper(args):
    prompt, model_name, temperature, max_output_tokens, top_p = args

    if model_name in ['gpt-4o', 'gpt-4o-mini', 'o1-mini']:
        return llm_openai(prompt, model_name, temperature, max_output_tokens, top_p)
    else:
        return llm_lmstudio(prompt, model_name, temperature, max_output_tokens, top_p)


def llm_lmstudio_batched(prompts: list[str], model_name: str, temperature: float, max_output_tokens: int,
                         batch_size: int, top_p: float, verbose: bool=True, **kwargs) -> list[str]:
    output_texts = []
    prompts_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    for prompt_batch in tqdm(prompts_batches, total=len(prompts_batches), disable=not verbose):
        inputs = []
        for i, prompt in enumerate(prompt_batch):
            curr_model_name = f'{model_name}:{i+1}' if i != 0 else model_name
            inputs.append((prompt, curr_model_name, temperature, max_output_tokens, top_p))

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(llm_repack_parameters_wrapper, inputs))
            output_texts.extend(results)

    return output_texts


def llm_gemini_15(prompt: str, model_name: str, temperature: str, max_output_tokens: int, **kwargs) -> str:
    vertexai.init(project='mystic-castle-444910-a0', location='europe-west1')
    model = GenerativeModel(model_name)
    generation_config = {
        'max_output_tokens': max_output_tokens,
        'temperature': temperature,
        'top_p': 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    output_text = ''.join(response.text for response in responses)
    return output_text


def llm_gemini_1206(prompt: str, model_name: str, temperature: str, max_output_tokens: int, **kwargs) -> str:
    vertexai.init(project="mystic-castle-444910-a0", location="us-central1")
    model = GenerativeModel(
        "gemini-exp-1206",
    )

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": 1,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]

    chat = model.start_chat()
    try:
        response = chat.send_message([prompt], generation_config=generation_config, safety_settings=safety_settings)
        output_text = response.candidates[0].content.parts[0].text
    except:
        output_text = 'Error'
    return output_text


def llm_gemini_1206_get_chat():
    vertexai.init(project="mystic-castle-444910-a0", location="us-central1")
    model = GenerativeModel("gemini-exp-1206")
    chat = model.start_chat()
    return chat


def llm_gemini_1206_send_message(chat, prompt: str, temperature: str, max_output_tokens: int):
    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": 1,
    }
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]
    response = chat.send_message([prompt], generation_config=generation_config, safety_settings=safety_settings)
    return response.candidates[0].content.parts[0].text


def llm_openai(prompt: str, model_name: str, temperature: str, max_output_tokens: int, top_p: float, **kwargs) -> str:
    litellm.set_verbose = False
    litellm.turn_off_message_logging = True
    max_retries = 10
    retry_no = 0
    messages = [{'role': 'user', 'content': prompt}]
    succeed = False
    output_text = 'N/A'
    while not succeed:
        try:
            #response = await litellm.acompletion(model='openai/gpt-4o', messages=messages, temperature=temperature, max_tokens=max_output_tokens)
            if model_name not in ['o1-mini']:
                response = litellm.completion(model=f'openai/{model_name}',
                                              messages=messages,
                                              temperature=temperature,
                                              max_tokens=max_output_tokens,
                                              top_p=top_p,
                                              )
            else:
                response = litellm.completion(model=f'openai/{model_name}',
                                              messages=messages,
                                              temperature=temperature,
                                              max_tokens=max_output_tokens
                                              )

            output_text = response.choices[0].message.content
            succeed = True
        except Exception as e:
            retry_no += 1
            print(f'Retry # {retry_no}/{max_retries}, {e}')
            time.sleep(math.pow(2, retry_no))
            #litellm.set_verbose = True
            if retry_no > max_retries:
                output_text = str(e)
                succeed = True
    return output_text


def llm_openai_batched(prompts: list[str], model_name: str, temperature: float, max_output_tokens: int,
                       batch_size: int, top_p: float, verbose: bool=True, **kwargs) -> list[str]:
    output_texts = []
    inputs = [(prompt, model_name, temperature, max_output_tokens, top_p) for prompt in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        results = list(tqdm(executor.map(llm_repack_parameters_wrapper, inputs), total=len(inputs),
                            desc=f'Processing prompts by {model_name}', disable=not verbose))
        output_texts.extend(results)

    return output_texts