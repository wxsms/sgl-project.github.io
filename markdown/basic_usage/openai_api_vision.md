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

vision_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
```

    [2026-02-12 17:30:04] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-12 17:30:04] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-12 17:30:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-12 17:30:09] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-12 17:30:09] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-12 17:30:09] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-12 17:30:11] INFO server_args.py:1813: Attention backend not specified. Use fa3 backend by default.
    [2026-02-12 17:30:11] INFO server_args.py:2821: Set soft_watchdog_timeout since in CI


    [2026-02-12 17:30:14] Ignore import error when loading sglang.srt.multimodal.processors.glm4v: No module named 'transformers.models.glm_ocr'
    [2026-02-12 17:30:14] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-02-12 17:30:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-12 17:30:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-12 17:30:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-12 17:30:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-12 17:30:15] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-12 17:30:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-12 17:30:21] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-12 17:30:21] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-12 17:30:21] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.35it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:01,  1.61it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.03it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:03<00:00,  1.23it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.57it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.40it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.44 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.44 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.55it/s]Capturing batches (bs=2 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.55it/s]Capturing batches (bs=1 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.55it/s]Capturing batches (bs=1 avail_mem=61.42 GB): 100%|██████████| 3/3 [00:00<00:00,  4.20it/s]



<strong style='color: #00008B;'><br><br>                    NOTE: Typically, the server runs in a separate terminal.<br>                    In this notebook, we run the server and notebook code together, so their outputs are combined.<br>                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>                    To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>                    We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>                    </strong>


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
              "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
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

    [2026-02-12 17:30:37] [load_mm_data(simple)] error loading IMAGE data at index=0
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 419, in connect
        self.sock = ssl_wrap_socket(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/ssl_.py", line 458, in ssl_wrap_socket
        ssl_sock = _ssl_wrap_socket_impl(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/ssl_.py", line 502, in _ssl_wrap_socket_impl
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
      File "/usr/lib/python3.10/ssl.py", line 513, in wrap_socket
        return self.sslsocket_class._create(
      File "/usr/lib/python3.10/ssl.py", line 1100, in _create
        self.do_handshake()
      File "/usr/lib/python3.10/ssl.py", line 1371, in do_handshake
        self._sslobj.do_handshake()
    TimeoutError: _ssl.c:990: The handshake operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 770, in reraise
        raise value
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 407, in _make_request
        self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 358, in _raise_timeout
        raise ReadTimeoutError(
    urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 690, in send
        raise ReadTimeout(e, request=request)
    requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    [2026-02-12 17:30:37] Error in request: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 419, in connect
        self.sock = ssl_wrap_socket(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/ssl_.py", line 458, in ssl_wrap_socket
        ssl_sock = _ssl_wrap_socket_impl(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/ssl_.py", line 502, in _ssl_wrap_socket_impl
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
      File "/usr/lib/python3.10/ssl.py", line 513, in wrap_socket
        return self.sslsocket_class._create(
      File "/usr/lib/python3.10/ssl.py", line 1100, in _create
        self.do_handshake()
      File "/usr/lib/python3.10/ssl.py", line 1371, in do_handshake
        self._sslobj.do_handshake()
    TimeoutError: _ssl.c:990: The handshake operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 770, in reraise
        raise value
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 407, in _make_request
        self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 358, in _raise_timeout
        raise ReadTimeoutError(
    urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 690, in send
        raise ReadTimeout(e, request=request)
    requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py", line 120, in handle_request
        return await self._handle_non_streaming_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py", line 869, in _handle_non_streaming_request
        ret = await self.tokenizer_manager.generate_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 519, in generate_request
        tokenized_obj = await self._tokenize_one_request(obj)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 725, in _tokenize_one_request
        mm_inputs: Dict = await self.mm_data_processor.process(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 99, in process
        return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
      File "/usr/lib/python3.10/asyncio/tasks.py", line 445, in wait_for
        return fut.result()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 70, in _invoke
        return await self._proc_async(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py", line 318, in process_mm_data_async
        base_output = self.load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 677, in load_mm_data
        return self.fast_load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 747, in fast_load_mm_data
        raise RuntimeError(
    RuntimeError: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)



<strong style='color: #00008B;'>{"object":"error","message":"Internal server error: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=3)","type":"InternalServerError","param":null,"code":500}</strong>



<strong style='color: #00008B;'>{"id":"1b4f56492ef94a49ba924c35a0aa50da","object":"chat.completion","created":1770917441,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a pair of blue jeans. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing the jeans.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":371,"completion_tokens":64,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


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
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print_highlight(response.text)
```


<strong style='color: #00008B;'>{"id":"d78858c50e3c458fb90e0bf3211243d8","object":"chat.completion","created":1770917448,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a pair of blue jeans. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing the jeans.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":371,"completion_tokens":64,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


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
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print_highlight(response.choices[0].message.content)
```

    [2026-02-12 17:30:51] INFO _base_client.py:1071: Retrying request to /chat/completions in 0.376583 seconds


    [2026-02-12 17:30:51] [load_mm_data(simple)] error loading IMAGE data at index=0
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 174, in _new_conn
        conn = connection.create_connection(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 95, in create_connection
        raise err
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 85, in create_connection
        sock.connect(sa)
    TimeoutError: timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 363, in connect
        self.sock = conn = self._new_conn()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 179, in _new_conn
        raise ConnectTimeoutError(
    urllib3.exceptions.ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)')
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 594, in increment
        raise MaxRetryError(_pool, url, error or ResponseError(cause))
    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 665, in send
        raise ConnectTimeout(e, request=request)
    requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    [2026-02-12 17:30:51] Error in request: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 174, in _new_conn
        conn = connection.create_connection(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 95, in create_connection
        raise err
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 85, in create_connection
        sock.connect(sa)
    TimeoutError: timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 363, in connect
        self.sock = conn = self._new_conn()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 179, in _new_conn
        raise ConnectTimeoutError(
    urllib3.exceptions.ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)')
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 594, in increment
        raise MaxRetryError(_pool, url, error or ResponseError(cause))
    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 665, in send
        raise ConnectTimeout(e, request=request)
    requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py", line 120, in handle_request
        return await self._handle_non_streaming_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py", line 869, in _handle_non_streaming_request
        ret = await self.tokenizer_manager.generate_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 519, in generate_request
        tokenized_obj = await self._tokenize_one_request(obj)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 725, in _tokenize_one_request
        mm_inputs: Dict = await self.mm_data_processor.process(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 99, in process
        return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
      File "/usr/lib/python3.10/asyncio/tasks.py", line 445, in wait_for
        return fut.result()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 70, in _invoke
        return await self._proc_async(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py", line 318, in process_mm_data_async
        base_output = self.load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 677, in load_mm_data
        return self.fast_load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 747, in fast_load_mm_data
        raise RuntimeError(
    RuntimeError: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359cb80>, 'Connection to github.com timed out. (connect timeout=3)'))


    [2026-02-12 17:30:55] INFO _base_client.py:1071: Retrying request to /chat/completions in 0.957793 seconds


    [2026-02-12 17:30:55] [load_mm_data(simple)] error loading IMAGE data at index=0
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 174, in _new_conn
        conn = connection.create_connection(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 95, in create_connection
        raise err
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 85, in create_connection
        sock.connect(sa)
    TimeoutError: timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 363, in connect
        self.sock = conn = self._new_conn()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 179, in _new_conn
        raise ConnectTimeoutError(
    urllib3.exceptions.ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)')
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 594, in increment
        raise MaxRetryError(_pool, url, error or ResponseError(cause))
    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 665, in send
        raise ConnectTimeout(e, request=request)
    requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    [2026-02-12 17:30:55] Error in request: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 174, in _new_conn
        conn = connection.create_connection(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 95, in create_connection
        raise err
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/connection.py", line 85, in create_connection
        sock.connect(sa)
    TimeoutError: timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 404, in _make_request
        self._validate_conn(conn)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 1061, in _validate_conn
        conn.connect()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 363, in connect
        self.sock = conn = self._new_conn()
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connection.py", line 179, in _new_conn
        raise ConnectTimeoutError(
    urllib3.exceptions.ConnectTimeoutError: (<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)')
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 594, in increment
        raise MaxRetryError(_pool, url, error or ResponseError(cause))
    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 426, in _load_single_item
        img, _ = load_image(data)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 908, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 665, in send
        raise ConnectTimeout(e, request=request)
    requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 740, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 436, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py", line 120, in handle_request
        return await self._handle_non_streaming_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py", line 869, in _handle_non_streaming_request
        ret = await self.tokenizer_manager.generate_request(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 519, in generate_request
        tokenized_obj = await self._tokenize_one_request(obj)
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 725, in _tokenize_one_request
        mm_inputs: Dict = await self.mm_data_processor.process(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 99, in process
        return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
      File "/usr/lib/python3.10/asyncio/tasks.py", line 445, in wait_for
        return fut.result()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 70, in _invoke
        return await self._proc_async(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py", line 318, in process_mm_data_async
        base_output = self.load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 677, in load_mm_data
        return self.fast_load_mm_data(
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 747, in fast_load_mm_data
        raise RuntimeError(
    RuntimeError: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x782ee359d300>, 'Connection to github.com timed out. (connect timeout=3)'))



<strong style='color: #00008B;'>The image shows a man standing on the back of a yellow taxi, using an iron to iron a piece of clothing. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing, which is an unusual and somewhat humorous scene.</strong>


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
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png",
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


<strong style='color: #00008B;'>The first image shows a man ironing clothes on the back of a yellow taxi in an urban setting. The second image is a stylized logo featuring the letters "SGL" in orange with a book and a computer icon as part of the design.</strong>



```python
terminate_process(vision_process)
```
