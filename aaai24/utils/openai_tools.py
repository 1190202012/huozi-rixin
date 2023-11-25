import json
import time
from typing import Union, List, Dict, Optional, Tuple

import openai
from tqdm import tqdm

import atexit

openai.api_key = "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"
openai_log_file = "openai_usage.jsonl"
usage_log = open(openai_log_file, "a", encoding="UTF-8")

OPENAI_MODEL_LIST = [
    "whisper-1",
    "babbage",
    "davinci",
    "text-davinci-edit-001",
    "babbage-code-search-code",
    "text-similarity-babbage-001",
    "code-davinci-edit-001",
    "text-davinci-001",
    "ada",
    "babbage-code-search-text",
    "babbage-similarity",
    "code-search-babbage-text-001",
    "text-curie-001",
    "code-search-babbage-code-001",
    "text-ada-001",
    "text-similarity-ada-001",
    "curie-instruct-beta",
    "ada-code-search-code",
    "ada-similarity",
    "code-search-ada-text-001",
    "text-search-ada-query-001",
    "davinci-search-document",
    "ada-code-search-text",
    "text-search-ada-doc-001",
    "davinci-instruct-beta",
    "text-similarity-curie-001",
    "code-search-ada-code-001",
    "ada-search-query",
    "text-search-davinci-query-001",
    "curie-search-query",
    "davinci-search-query",
    "babbage-search-document",
    "ada-search-document",
    "text-search-curie-query-001",
    "text-search-babbage-doc-001",
    "curie-search-document",
    "text-search-curie-doc-001",
    "babbage-search-query",
    "text-babbage-001",
    "text-search-davinci-doc-001",
    "text-embedding-ada-002",
    "text-search-babbage-query-001",
    "curie-similarity",
    "curie",
    "text-similarity-davinci-001",
    "text-davinci-002",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "davinci-similarity",
    "gpt-3.5-turbo-0301",
]

OPENAI_MODELS = {
    "4": "gpt-4",
    "4-long": "gpt-4-32k",
    "turbo": "gpt-3.5-turbo",
    "3": "text-davinci-003"
}

COMPLETION_MODEL_LIST = [
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "davinci",
    "curie",
    "babbage",
    "ada"
]

CHAT_MODEL_LIST = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301"
]

MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "gpt-4": 8192,
    "gpt-4-32k": 32768
}

MAX_BATCH_SIZE = 20


def prompt_openai_api(model_name: str,
                      messages: Union[str, List[str], List[Dict]],
                      customized_model_name: str = None,
                      batch_size: int = 1,
                      temperature: float = 1,
                      top_p: float = 1,
                      n: int = 1,
                      logprobs: Optional[int] = None,  #
                      echo: bool = False,  #
                      stop=None,
                      max_tokens: Optional[int] = None,
                      try_times: int = 5,
                      wait_time: int = 2,
                      ) -> Tuple[List, List]:
    """
    control the batch calling of openai api

    Parameters:
    model: str - ID of the model to use.
    messages: List[str] or List[List[Dict]], first dimension is the batch_size m, for chat completion,
        batch size is 1.
        completion: str, array of strs, array of tokens, or array of token arrays - The prompt(s) to generate
            completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        chat: array of dict - A list of messages describing the conversation so far.
            "role": required - The role of the author of this message. One of system, user, or assistant.
            "content": required - The contents of the message.
            "name": The name of the author of this message. May contain a-z, A-Z, 0-9, and underscores,
            with a maximum length of 64 characters.
    customized_model_name: self-finetuned model name given by "openai api fine_tunes.list", if customized_model_name, model_name should be one of finetunable model "ada, curie, babbage, davinci"
    batch_size: batch size used for each API calling. 1 for chat style
    temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
        output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or top_p but not both.
    top_p: An alternative to sampling with temperature, called nucleus sampling, where the model
        considers the results of the tokens with top_p probability mass.
        So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    n: How many chat completion choices to generate for each input message.
    max_tokens: max_output_tokens
    stop: Up to 4 sequences where the API will stop generating further tokens
    logprobs: (completion only)Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response. The maximum value for logprobs is 5.
    echo: if true, return the whole the prompt logprob in addition to the completion
    try_times: maximum times to try openAI API
    wait_time: time to wait if API failed

    Returns: responses: List[str], List[List[str]] - first dimension size is the batch size m, second dimension size
    is n - the number of choices to generate for each input message. if n is 1, will output List[str] instead ot
    List[List[]]

    finish_reasons: same size of responses

    logprobs:OPTIONAL(only return if completion with logprobs not None) -  a dict with key 
        "tokens":List[str] - all tokens
        "token_logprobs": List[float] -  the logprob of token in "tokens"
        "top_logprobs": List[dict] - each dict is the top-k token(key) with corresponding log probability(value). if echo=True, the first entry in the list is null.
        "text_offset" : the start position of the token
    """
    global usage_log

    if customized_model_name is None:
        model = model_name
    else:
        model = customized_model_name

    batch_size = batch_size if batch_size < MAX_BATCH_SIZE else MAX_BATCH_SIZE

    if isinstance(messages, str):
        messages = [messages]

    if isinstance(messages, list) and isinstance(messages[0], dict):
        messages = [messages]

    if model_name in COMPLETION_MODEL_LIST:
        assert isinstance(messages[0], str), f"prompt for completion model should be str, array of strs."
        if logprobs is not None:  #
            assert isinstance(logprobs,
                              int) and logprobs <= 5, f'the maximum value for logprobs is 5, but {logprobs} is given'  #

        responses = []
        finish_reasons = []
        logps = []
        for index in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[index:index + batch_size]
            response, finish_reason, logp = call_openai_api(model_name, model, batch_messages, temperature=temperature,
                                                            top_p=top_p, logprobs=logprobs, echo=echo, n=n,
                                                            max_tokens=max_tokens, try_times=try_times, stop=stop,
                                                            wait_time=wait_time)
            responses.extend(response)
            finish_reasons.extend(finish_reason)
            logps.extend(logp)

        if logprobs is not None:
            return responses, finish_reasons, logps
        else:
            return responses, finish_reasons

    elif model_name in CHAT_MODEL_LIST:
        assert isinstance(messages[0], list) and isinstance(messages[0][0], dict), \
            "messages for chat completion should be arrary of dict"

        responses = []
        finish_reasons = []

        for message in tqdm(messages):
            response, finish_reason = call_openai_api(model_name, model, message, temperature=temperature, top_p=top_p,
                                                      n=n,
                                                      max_tokens=max_tokens, try_times=try_times, wait_time=wait_time,
                                                      stop=stop)
            responses.extend(response)
            finish_reasons.extend(finish_reason)

        return responses, finish_reasons


def call_openai_api(model_name: str,
                    model: str,
                    messages: Union[str, List[str], List[Dict]],
                    temperature: float = 1,
                    top_p: float = 1,
                    n: int = 1,
                    max_tokens: Optional[int] = None,
                    logprobs: Optional[int] = None,  #
                    echo: bool = False,
                    stop=None,  #
                    try_times: int = 5,
                    wait_time: int = 2) -> Tuple[List, List]:
    """
    Parameters:
    model: str - ID of the model to use.
    messages: List[str] or List[List[Dict]], first dimension is the batch_size m, for chat completion, batch size is 1.
        completion: str, array of strs, array of tokens, or array of token arrays - The prompt(s) to generate completions for,
            encoded as a string, array of strings, array of tokens, or array of token arrays.
        chat: array of dict - A list of messages describing the conversation so far.
            "role": required - The role of the author of this message. One of system, user, or assistant.
            "content": required - The contents of the message.
            "name": The name of the author of this message. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
    temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower
        values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
    top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens
        with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    n: How many chat completion choices to generate for each input message.
    max_tokens: max_output_tokens
    try_times: maximum times to try openAI API
    wait_time: time to wait if API failed

    Returns:
    responses: List[List[str]] - first dimension size is the batch size m, second dimionsion size is n - the number of choices to generate for each input message

    finish_reasons: same size of responses
    """

    if model_name in COMPLETION_MODEL_LIST:

        for try_time in range(try_times):

            try:

                if max_tokens is None:
                    replys = openai.Completion.create(model=model,
                                                      prompt=messages,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      logprobs=logprobs,
                                                      echo=echo,
                                                      stop=stop,
                                                      n=n)
                else:
                    replys = openai.Completion.create(model=model,
                                                      prompt=messages,
                                                      temperature=temperature,
                                                      max_tokens=max_tokens,
                                                      top_p=top_p,
                                                      logprobs=logprobs,
                                                      echo=echo,
                                                      stop=stop,
                                                      n=n)
                responses = [sequence["text"] for sequence in replys["choices"]]
                finish_reasons = [sequence["finish_reason"] for sequence in replys["choices"]]
                usage_log.write(json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                        "model": replys["model"],
                        "completion_tokens": replys["usage"]["completion_tokens"],
                        "prompt_tokens": replys["usage"]["prompt_tokens"],
                        "total_tokens": replys["usage"]["total_tokens"],
                        "created": replys["created"],
                        "id": replys["id"],
                        "object": replys["object"]
                    },
                ) + "\n")
                if logprobs is not None:
                    logprob = [sequence["logprobs"] for sequence in replys["choices"]]
                    return reshape_sequences(responses, n), reshape_sequences(finish_reasons, n), reshape_sequences(
                        logprob, n)
                else:
                    return reshape_sequences(responses, n), reshape_sequences(finish_reasons, n), [""]

            except Exception as e:
                print(f"API failed {try_time + 1} times due to: {e} ")
                time.sleep(wait_time)
                continue

        raise RuntimeError('openai.error.RateLimitError')

    elif model_name in CHAT_MODEL_LIST:

        for try_time in range(try_times):

            try:

                if max_tokens is None:
                    replys = openai.ChatCompletion.create(model=model,
                                                          messages=messages,
                                                          temperature=temperature,
                                                          top_p=top_p,
                                                          stop=stop,
                                                          n=n)
                else:
                    replys = openai.ChatCompletion.create(model=model,
                                                          messages=messages,
                                                          temperature=temperature,
                                                          max_tokens=max_tokens,
                                                          top_p=top_p,
                                                          stop=stop,
                                                          n=n)
                responses = [sequence["message"]["content"] for sequence in replys["choices"]]
                finish_reasons = [sequence["finish_reason"] for sequence in replys["choices"]]
                usage_log.write(json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                        "model": replys["model"],
                        "completion_tokens": replys["usage"]["completion_tokens"],
                        "prompt_tokens": replys["usage"]["prompt_tokens"],
                        "total_tokens": replys["usage"]["total_tokens"],
                        "created": replys["created"],
                        "id": replys["id"],
                        "object": replys["object"]
                    }
                ) + "\n")
                return [responses], [finish_reasons]

            except Exception as e:
                print(f"API failed {try_time + 1} times due to: {e} ")
                time.sleep(wait_time)
                continue

        raise RuntimeError('openai.error.RateLimitError')


color_prefix_by_role = {
    "system": "\033[0m",  # gray
    "user": "\033[0m",  # gray
    "assistant": "\033[92m",  # green
}


def reshape_sequences(sequences, n):
    """
    reshape input sequences List[] with len m*n to List[List[]],each sublist is with len n
    """
    prompts = []
    assert len(
        sequences) % n == 0, f"length of sequences should be a multiple of {n}, but the length of given sequences is {len(sequences)}"
    m = int(len(sequences) / n)
    start_id = 0
    for _ in range(m):
        end_id = start_id + n
        prompts.append(sequences[start_id: end_id])
        start_id = end_id
    return prompts


@atexit.register
def clean():
    global usage_log
    usage_log.close()
