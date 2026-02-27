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

vision_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=vision_process)
```

    [2026-02-27 09:51:07] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-02-27 09:51:07] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-02-27 09:51:07] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-27 09:51:13] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-27 09:51:13] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-27 09:51:13] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-02-27 09:51:16] INFO server_args.py:1864: Attention backend not specified. Use fa3 backend by default.
    [2026-02-27 09:51:16] INFO server_args.py:2934: Set soft_watchdog_timeout since in CI


    [2026-02-27 09:51:19] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-02-27 09:51:22] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-27 09:51:22] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-27 09:51:22] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-02-27 09:51:22] INFO utils.py:148: Note: detected 112 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-02-27 09:51:22] INFO utils.py:151: Note: NumExpr detected 112 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-02-27 09:51:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-02-27 09:51:30] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-02-27 09:51:30] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-02-27 09:51:30] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:03,  1.20it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.17it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.17it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:03<00:00,  1.18it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.53it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.35it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.37 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.37 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.65it/s]Capturing batches (bs=2 avail_mem=61.34 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.65it/s]Capturing batches (bs=1 avail_mem=61.33 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.65it/s]Capturing batches (bs=1 avail_mem=61.33 GB): 100%|██████████| 3/3 [00:00<00:00,  4.45it/s]



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


<strong style='color: #00008B;'>{"id":"f50c5bcb379d4b74acc853b8dbe5b7c9","object":"chat.completion","created":1772185905,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a pair of blue jeans. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing the jeans.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":371,"completion_tokens":64,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


    [2026-02-27 09:51:45] [load_mm_data(simple)] error loading IMAGE data at index=0
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 292, in _read_status
        raise RemoteDisconnected("Remote end closed connection without"
    http.client.RemoteDisconnected: Remote end closed connection without response
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 769, in reraise
        raise value.with_traceback(tb)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 292, in _read_status
        raise RemoteDisconnected("Remote end closed connection without"
    urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 433, in _load_single_item
        img, _ = load_image(data)
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 909, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 659, in send
        raise ConnectionError(err, request=request)
    requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 747, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 443, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    [2026-02-27 09:51:45] Error in request: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 292, in _read_status
        raise RemoteDisconnected("Remote end closed connection without"
    http.client.RemoteDisconnected: Remote end closed connection without response
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 644, in send
        resp = conn.urlopen(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 802, in urlopen
        retries = retries.increment(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/util/retry.py", line 552, in increment
        raise six.reraise(type(error), error, _stacktrace)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/packages/six.py", line 769, in reraise
        raise value.with_traceback(tb)
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 716, in urlopen
        httplib_response = self._make_request(
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 468, in _make_request
        six.raise_from(e, None)
      File "<string>", line 3, in raise_from
      File "/usr/local/lib/python3.10/dist-packages/urllib3/connectionpool.py", line 463, in _make_request
        httplib_response = conn.getresponse()
      File "/usr/lib/python3.10/http/client.py", line 1395, in getresponse
        response.begin()
      File "/usr/lib/python3.10/http/client.py", line 323, in begin
        version, status, reason = self._read_status()
      File "/usr/lib/python3.10/http/client.py", line 292, in _read_status
        raise RemoteDisconnected("Remote end closed connection without"
    urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 433, in _load_single_item
        img, _ = load_image(data)
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 909, in load_image
        response = requests.get(image_file, stream=True, timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 73, in get
        return request("get", url, params=params, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/api.py", line 59, in request
        return session.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/requests/adapters.py", line 659, in send
        raise ConnectionError(err, request=request)
    requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 747, in fast_load_mm_data
        result = future.result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 458, in result
        return self.__get_result()
      File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
        raise self._exception
      File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
        result = self.fn(*self.args, **self.kwargs)
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 443, in _load_single_item
        raise RuntimeError(f"Error while loading data {data}: {e}")
    RuntimeError: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py", line 102, in handle_request
        return await self._handle_non_streaming_request(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py", line 874, in _handle_non_streaming_request
        ret = await self.tokenizer_manager.generate_request(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 503, in generate_request
        tokenized_obj = await self._tokenize_one_request(obj)
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 714, in _tokenize_one_request
        mm_inputs: Dict = await self.mm_data_processor.process(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 99, in process
        return await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
      File "/usr/lib/python3.10/asyncio/tasks.py", line 445, in wait_for
        return fut.result()
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/managers/async_mm_data_processor.py", line 70, in _invoke
        return await self._proc_async(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py", line 320, in process_mm_data_async
        base_output = self.load_mm_data(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 684, in load_mm_data
        return self.fast_load_mm_data(
      File "/public_sglang_ci/runner-l3-9gzh5-gpu-23/_work/sglang/sglang/python/sglang/srt/multimodal/processors/base_processor.py", line 754, in fast_load_mm_data
        raise RuntimeError(
    RuntimeError: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))



<strong style='color: #00008B;'>{"object":"error","message":"Internal server error: An exception occurred while loading IMAGE data at index 0: Error while loading data ImageData(url='https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true', detail='auto', max_dynamic_patch=None): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))","type":"InternalServerError","param":null,"code":500}</strong>


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


<strong style='color: #00008B;'>{"id":"59743cfb077d4eeb82002c84602b97e1","object":"chat.completion","created":1772185906,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, using an iron to iron a piece of clothing. The taxi is parked on a city street, and there are other taxis and buildings in the background. The man appears to be balancing on the taxi's rear bumper while ironing, which is an unusual and somewhat humorous scene.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":377,"completion_tokens":70,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


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
