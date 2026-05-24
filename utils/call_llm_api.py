"""
LLM API Calling Module

Provides unified LLM and Embedding API calling interfaces.
"""
import requests
import json
import time


API_BASE_URL = ""
API_KEY = ""


def call_llm_general(messages=[], model="", stream=False, temperature=0.9, top_p=0.95, max_retry=5, app_name="experiment_on_prompt_optimization"):
    """
    General LLM API calling function, supports text and multimodal models.

    Args:
        messages: message list, format [{"role": "user"/"assistant"/"system", "content": ...}]
        model: model name
        stream: whether to use streaming output
        temperature: temperature parameter
        top_p: top-p sampling parameter
        max_retry: maximum retry count
        app_name: application name (for logging)

    Returns:
        model-generated text response
    """
    def stream_chat(url, headers, data):
        with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as resp:
            resp.encoding = "utf-8"
            for line in resp.iter_lines():
                if line and line.startswith(b'data:'):
                    payload = line[len(b'data:'):].strip()
                    chunk = json.loads(payload)

                    if chunk['object'] == 'chat.completion.chunk' and not chunk['choices'][0].get('finish_reason'):
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        yield delta
                    elif chunk['object'] == "error":
                        print(chunk)
                        raise RuntimeError("【Error】: " + chunk['type'] + " " + chunk['message'])
                    elif chunk['object'] == 'chat.completion.chunk' and chunk['choices'][0].get('finish_reason') == "stop":
                        break

    def full_chat(url, headers, data):
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=(30, 300))
        r = response.json()
        if r.get("object", "") == "chat.completion":
            return r["choices"][0].get("message", {}).get("content", "")
        elif r.get("object", "") == "error":
            print(r)
            raise RuntimeError("【Error】: " + r['type'] + " " + r['message'])

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }

    cur_try = 0
    error_message = ""

    while cur_try < max_retry:
        cur_try += 1

        try:
            if stream:
                return stream_chat(url, headers, data)
            else:
                return full_chat(url, headers, data)

        except requests.exceptions.ConnectionError as e:
            error_message = f"【Error】connection error: {e}"
            if cur_try < max_retry:
                time.sleep(1)
            continue
        except requests.exceptions.Timeout:
            error_message = f"【Error】timeout"
            if cur_try < max_retry:
                time.sleep(1)
            continue
        except RuntimeError as e:
            raise e
        except Exception as e:
            error_message = f"【Error】: {e}"
            if cur_try < max_retry:
                time.sleep(1)
            continue

    return error_message


def call_embedding_general(texts=[], model="", max_retry=5, app_name="experiment_on_prompt_optimization"):
    """
    General text Embedding API calling function.

    Args:
        texts: texts to embed (string or list of strings)
        model: embedding model name
        max_retry: maximum retry count
        app_name: application name (for logging)

    Returns:
        dict: dict containing embedding results
        {
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
            "model": "text-embedding-xxx"
        }
    """
    def parse_embedding_response(response):
        """Parse embedding API response"""
        r = response.json()
        if r.get("object", "") == "list":
            embeddings = [item["embedding"] for item in r.get("data", [])]
            return {
                "embeddings": embeddings,
                "usage": r.get("usage", {}),
                "model": r.get("model", model)
            }
        elif r.get("object", "") == "error":
            print(r)
            raise RuntimeError("【Error】Embedding: " + r.get('type', '') + " " + r.get('message', ''))
        else:
            raise RuntimeError("【Error】not known")

    if isinstance(texts, str):
        texts = [texts]

    url = f"{API_BASE_URL}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": model,
        "input": texts
    }

    cur_try = 0
    error_message = ""

    while cur_try < max_retry:
        cur_try += 1

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=(30, 300))
            return parse_embedding_response(response)

        except requests.exceptions.ConnectionError as e:
            error_message = f"【Error】embedding connection error: {e}"
            if cur_try < max_retry:
                time.sleep(1)
            continue
        except requests.exceptions.Timeout:
            error_message = f"【Error】Embedding timeout"
            if cur_try < max_retry:
                time.sleep(1)
            continue
        except RuntimeError as e:
            raise e
        except Exception as e:
            error_message = f"【Error】Embedding: {e}"
            if cur_try < max_retry:
                time.sleep(1)
            continue

    return {"error": error_message}
