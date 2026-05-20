# OpenAI APIs - Vision

SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/guides/vision).
This tutorial covers the vision APIs for vision language models.

SGLang supports various vision language models such as Llama 3.2, LLaVA-OneVision, Qwen2.5-VL, Gemma3 and [more](../supported_models/text_generation/multimodal_language_models.md).

As an alternative to the OpenAI API, you can also use the [SGLang offline engine](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py).

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

example_image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
logo_image_url = (
    "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
)

vision_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=vision_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-20 23:51:24] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-20 23:51:28] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:03,  1.03it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.11it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.13it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:03<00:00,  1.15it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.46it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.28it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-20 23:51:40,989 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 23:51:40] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Using cURL

Once the server is up, you can send test requests using curl or requests.


```python
import subprocess

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{
            "type": "text",
            "text": "What’s in this image?"
          }},
          {{
            "type": "image_url",
            "image_url": {{
              "url": "{example_image_url}"
            }}
          }}
        ]
      }}
    ],
    "max_tokens": 300
  }}'
"""

response = subprocess.check_output(curl_command, shell=True).decode()
print_highlight(response)


response = subprocess.check_output(curl_command, shell=True).decode()
print_highlight(response)
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:877: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>{"id":"82fdf693f11e44e7a78ff20cc5f48770","object":"chat.completion","created":1779321107,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing clothes. The taxi is parked on a city street, and there are other taxis visible in the background. The man appears to be balancing on the tailgate while ironing a blue shirt. The setting looks like an urban environment with buildings and flags in the background.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":377,"completion_tokens":70,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>



<strong style='color: #00008B;'>{"id":"bbd63b04eecf42d08352d0a2ce27ab46","object":"chat.completion","created":1779321108,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing a blue shirt. The taxi is parked on a city street with other vehicles and buildings in the background. The man appears to be balancing on the tailgate while performing this task. The scene seems to be set in an urban environment, possibly New York City, given the style of the taxi and the architecture visible in the background.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":390,"completion_tokens":83,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


## Using Python Requests


```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": example_image_url},
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print_highlight(response.text)
```


<strong style='color: #00008B;'>{"id":"965d6ee4aa2249e3982d3c0006455e0d","object":"chat.completion","created":1779321109,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing a blue shirt. The taxi is parked on a city street with other vehicles and buildings in the background. The man appears to be balancing on the tailgate while performing this task. The scene suggests an unusual or humorous situation, as ironing clothes outdoors is not a typical activity.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":380,"completion_tokens":73,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


## Using OpenAI Python Client


```python
from openai import OpenAI

client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": example_image_url},
                },
            ],
        }
    ],
    max_tokens=300,
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>The image shows a man standing on the back of a yellow taxi, ironing a blue shirt. The taxi is parked on a city street, and there are other taxis visible in the background. The man appears to be balancing on the tailgate while ironing, which is an unusual and humorous scene. The setting suggests it might be in a busy urban area with tall buildings and flags in the background.</strong>


## Multiple-Image Inputs

The server also supports multiple images and interleaved text and images if the model supports it.


```python
from openai import OpenAI

client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": example_image_url,
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": logo_image_url,
                    },
                },
                {
                    "type": "text",
                    "text": "I have two very different images. They are not related at all. "
                    "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                },
            ],
        }
    ],
    temperature=0,
)

print_highlight(response.choices[0].message.content)
```

    [2026-05-20 23:51:53] [load_mm_data(simple)] error loading IMAGE data at index=1
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 284, in _read_status
        line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
      File "/usr/lib/python3.10/socket.py", line 705, in readinto
        return self._sock.recv_into(b)
      File "/usr/lib/python3.10/ssl.py", line 1303, in recv_into
        return self.read(nbytes, buffer)
      File "/usr/lib/python3.10/ssl.py", line 1159, in read
        return self._sslobj.read(len, buffer)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 645, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 770, in reraise
        raise value
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 470, in _make_request
        self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 358, in _raise_timeout
        raise ReadTimeoutError(
    urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 530, in _load_single_item
        img, _ = load_image(data, cls.gpu_image_decode)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 906, in load_image
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 874, in _load_image
        image_bytes = get_image_bytes(image_file)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 933, in get_image_bytes
        response = requests.get(image_file, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 592, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 706, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 691, in send
        raise ReadTimeout(e, request=request)
    requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 849, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 545, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png', detail='auto', max_dynamic_patch=None, preprocess_kwargs=None): HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    [2026-05-20 23:51:53] Error in request: An exception occurred while loading IMAGE data at index 1: Error while loading data ImageData(url='https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png', detail='auto', max_dynamic_patch=None, preprocess_kwargs=None): HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 284, in _read_status
        line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
      File "/usr/lib/python3.10/socket.py", line 705, in readinto
        return self._sock.recv_into(b)
      File "/usr/lib/python3.10/ssl.py", line 1303, in recv_into
        return self.read(nbytes, buffer)
      File "/usr/lib/python3.10/ssl.py", line 1159, in read
        return self._sslobj.read(len, buffer)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 645, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 770, in reraise
        raise value
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 470, in _make_request
        self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 358, in _raise_timeout
        raise ReadTimeoutError(
    urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 530, in _load_single_item
        img, _ = load_image(data, cls.gpu_image_decode)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 906, in load_image
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 874, in _load_image
        image_bytes = get_image_bytes(image_file)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 933, in get_image_bytes
        response = requests.get(image_file, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 592, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 706, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 691, in send
        raise ReadTimeout(e, request=request)
    requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 849, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 545, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png', detail='auto', max_dynamic_patch=None, preprocess_kwargs=None): HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py", line 107, in handle_request
        return await self._handle_non_streaming_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py", line 1095, in _handle_non_streaming_request
        ret = await self.tokenizer_manager.generate_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 555, in generate_request
        tokenized_obj = await self._tokenize_one_request(obj)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 788, in _tokenize_one_request
        mm_inputs = await self.mm_processor.process_mm_data_async(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py", line 508, in process_mm_data_async
        base_output = self.load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 786, in load_mm_data
        return self.fast_load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 856, in fast_load_mm_data
        raise RuntimeError(
    RuntimeError: An exception occurred while loading IMAGE data at index 1: Error while loading data ImageData(url='https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png', detail='auto', max_dynamic_patch=None, preprocess_kwargs=None): HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Read timed out. (read timeout=3)
    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:129: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return self.create_error_response(



<strong style='color: #00008B;'>The first image shows a man ironing clothes on the back of a taxi in an urban setting. The second image is a stylized logo featuring the letters "SGL" with a book and a computer icon incorporated into the design.</strong>



```python
terminate_process(vision_process)
```
