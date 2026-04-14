# Query VLM with Offline Engine

This tutorial demonstrates how to use SGLang's **offline Engine API** to query VLMs. We will demonstrate usage with Qwen2.5-VL and Llama 4. This section demonstrates three different calling approaches:

1. **Basic Call**: Directly pass images and text.
2. **Processor Output**: Use HuggingFace processor for data preprocessing.
3. **Precomputed Embeddings**: Pre-calculate image features to improve inference efficiency.

## Understanding the Three Input Formats

SGLang supports three ways to pass visual data, each optimized for different scenarios:

### 1. **Raw Images** - Simplest approach
- Pass PIL Images, file paths, URLs, or base64 strings directly
- SGLang handles all preprocessing automatically
- Best for: Quick prototyping, simple applications

### 2. **Processor Output** - For custom preprocessing
- Pre-process images with HuggingFace processor
- Pass the complete processor output dict with `format: "processor_output"`
- Best for: Custom image transformations, integration with existing pipelines
- Requirement: Must use `input_ids` instead of text prompt

### 3. **Precomputed Embeddings** - For maximum performance
- Pre-calculate visual embeddings using the vision encoder
- Pass embeddings with `format: "precomputed_embedding"`
- Best for: Repeated queries on same images, caching, high-throughput serving
- Performance gain: Avoids redundant vision encoder computation (30-50% speedup)

**Key Rule**: Within a single request, use only one format for all images. Don't mix formats.

The examples below demonstrate all three approaches with both Qwen2.5-VL and Llama 4 models.

## Querying Qwen2.5-VL Model


```python
import nest_asyncio

nest_asyncio.apply()

import sglang.test.doc_patch  # noqa: F401

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
chat_template = "qwen2-vl"
example_image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
```


```python
from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Generated prompt text:")
print(conv.get_prompt())
print(f"\nImage size: {image.size}")
image
```

    Generated prompt text:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What's shown here: <|vision_start|><|image_pad|><|vision_end|>?<|im_end|>
    <|im_start|>assistant
    
    
    Image size: (570, 380)





    
![png](vlm_query_files/vlm_query_4_1.png)
    



### Basic Offline Engine API Call


```python
from sglang import Engine

llm = Engine(model_path=model_path, chat_template=chat_template, log_level="warning")
```

    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Temporarily Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Temporarily Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/hf_quant_config.json' (Amz CF ID: MHhGOernhSUzywktMmhv2hWyJlxuwocDvF7XZvrRqDFziP3A8BjZoQ==)
    For more information check: https://httpstatuses.com/503
    retry() failed once (0th try, maximum 2 retries). Will delay 0.84s and retry. Error: Server error '503 Service Temporarily Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/hf_quant_config.json' (Amz CF ID: MHhGOernhSUzywktMmhv2hWyJlxuwocDvF7XZvrRqDFziP3A8BjZoQ==)
    For more information check: https://httpstatuses.com/503


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    retry() failed once (1th try, maximum 2 retries). Will delay 1.92s and retry. Error: The read operation timed out


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-3B-Instruct: retry() exceed maximum number of retries.


    [2026-04-14 15:03:32] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json


    [2026-04-14 15:03:59] HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json


    Retrying in 1s [Retry 1/5].


    [2026-04-14 15:03:59] Retrying in 1s [Retry 1/5].


    HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json


    [2026-04-14 15:04:10] HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json


    Retrying in 2s [Retry 2/5].


    [2026-04-14 15:04:10] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json
    [2026-04-14 15:04:57] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-14 15:04:57] Retrying in 1s [Retry 1/5].


    [2026-04-14 15:05:15] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-14 15:05:20] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-14 15:05:20] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-14 15:05:20] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-14 15:05:20] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    [2026-04-14 15:05:26] retry() failed once (0th try, maximum 2 retries). Will delay 0.76s and retry. Error: The read operation timed out


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-04-14 15:05:35] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 15:05:35] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 15:05:35] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-14 15:05:35] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  2.02it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.51it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.57it/s]


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    [2026-04-14 15:05:47] retry() failed once (0th try, maximum 2 retries). Will delay 0.86s and retry. Error: The read operation timed out


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    [2026-04-14 15:05:58] retry() failed once (1th try, maximum 2 retries). Will delay 1.93s and retry. Error: The read operation timed out


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 10, in map_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 28, in read
        return self._sock.recv(max_bytes)
      File "/usr/lib/python3.10/ssl.py", line 1288, in recv
        return self.read(buflen)
      File "/usr/lib/python3.10/ssl.py", line 1161, in read
        return self._sslobj.read(len)
    TimeoutError: The read operation timed out
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 60, in map_httpcore_exceptions
        yield
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 218, in handle_request
        resp = self._pool.handle_request(req)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 253, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection_pool.py", line 237, in handle_request
        response = connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/connection.py", line 90, in handle_request
        return self._connection.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 112, in handle_request
        raise exc
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 91, in handle_request
        ) = self._receive_response_headers(**kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 155, in _receive_response_headers
        event = self._receive_event(timeout=timeout)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_sync/http11.py", line 191, in _receive_event
        data = self._network_stream.read(
      File "/usr/local/lib/python3.10/dist-packages/httpcore/backends/sync.py", line 26, in read
        with map_exceptions(exc_map):
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpcore/_exceptions.py", line 14, in map_exceptions
        raise to_exc(exc)
    httpcore.ReadTimeout: The read operation timed out
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 755, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 685, in _httpx_follow_relative_redirects_with_backoff
        response = http_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 559, in http_backoff
        return next(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 467, in _http_backoff_base
        response = client.request(method=method, url=url, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 821, in request
        return self.send(request, auth=auth, follow_redirects=follow_redirects)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 908, in send
        response = self._send_handling_auth(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 936, in _send_handling_auth
        response = self._send_handling_redirects(
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 973, in _send_handling_redirects
        response = self._send_single_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_client.py", line 1009, in _send_single_request
        response = transport.handle_request(request)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 217, in handle_request
        with map_httpcore_exceptions():
      File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
        self.gen.throw(typ, value, traceback)
      File "/usr/local/lib/python3.10/dist-packages/httpx/_transports/default.py", line 77, in map_httpcore_exceptions
        raise mapped_exc(message) from exc
    httpx.ReadTimeout: The read operation timed out
    [2026-04-14 15:06:10] Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-3B-Instruct: retry() exceed maximum number of retries.


    HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json
    [2026-04-14 15:06:42] HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-14 15:06:42] Retrying in 1s [Retry 1/5].


    HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.jinja
    [2026-04-14 15:07:33] HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-14 15:07:33] Retrying in 1s [Retry 1/5].


    HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.jinja
    [2026-04-14 15:07:44] HTTP Error 504 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.jinja
    Retrying in 2s [Retry 2/5].
    [2026-04-14 15:07:44] Retrying in 2s [Retry 2/5].



```python
out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Model response:")
print(out["text"])
```

    2026-04-14 15:08:24,508 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 15:08:24] Unexpected error during package walk: cutlass.cute.experimental


    Model response:
    The image depicts a scene where a person is ironing clothes on the back of a standing blue and white flag attached to a small table. The scene appears to be taking place on a city street, with two yellow taxis in the background. There is also a banner hanging on a building behind, but its contents are not clearly readable. The setting suggests it might be an event or an art installation.


### Call with Processor Output

Using a HuggingFace processor to preprocess text and images, and passing the `processor_output` directly into `Engine.generate`.


```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out["text"])
```

    Response using processor output:
    The image shows a street scene with a yellow taxi cab parked on the road. The taxi is pulling a ladder off of it. On the ladder, there are various pieces of luggage, including a blue backpack and other items. The background includes a building with various flags and a store. There are also several streetlights and more vehicles visible.


### Call with Precomputed Embeddings

You can pre-calculate image features to avoid repeated visual encoding processes.


```python
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()
vision = model.model.visual.cuda()
```


    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/824 [00:00<?, ?it/s]



```python
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

precomputed_embeddings = vision(
    processor_output["pixel_values"].cuda(), processor_output["image_grid_thw"].cuda()
)
precomputed_embeddings = precomputed_embeddings.pooler_output

multi_modal_item = dict(
    processor_output,
    format="precomputed_embedding",
    feature=precomputed_embeddings,
)

out = llm.generate(input_ids=input_ids, image_data=[multi_modal_item])
print("Response using precomputed embeddings:")
print(out["text"])

llm.shutdown()
```

    Response using precomputed embeddings:
    The image shows a scene with a yellow taxi cab and a person standing next to it. The taxi appears to be in motion on a city street. The background includes some buildings and flags, suggesting an urban setting. The person standing next to the taxi seems to be preparing or adjusting something, possibly a banner or a sign for a parade or event.


## Querying Llama 4 Vision Model

```python
model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
chat_template = "llama-4"

from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

# Download the same example image
image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Llama 4 generated prompt text:")
print(conv.get_prompt())
print(f"Image size: {image.size}")

image
```

### Llama 4 Basic Call

Llama 4 requires more computational resources, so it's configured with multi-GPU parallelism (tp_size=4) and larger context length.

```python
llm = Engine(
    model_path=model_path,
    enable_multimodal=True,
    attention_backend="fa3",
    tp_size=4,
    context_length=65536,
)

out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Llama 4 response:")
print(out["text"])
```

### Call with Processor Output

Using HuggingFace processor to preprocess data can reduce computational overhead during inference.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out)
```

### Call with Precomputed Embeddings

```python
from transformers import AutoProcessor
from transformers import Llama4ForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto"
).eval()

vision = model.vision_model.cuda()
multi_modal_projector = model.multi_modal_projector.cuda()

print(f'Image pixel values shape: {processor_output["pixel_values"].shape}')
input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

# Process image through vision encoder
image_outputs = vision(
    processor_output["pixel_values"].to("cuda"), 
    aspect_ratio_ids=processor_output["aspect_ratio_ids"].to("cuda"),
    aspect_ratio_mask=processor_output["aspect_ratio_mask"].to("cuda"),
    output_hidden_states=False
)
image_features = image_outputs.last_hidden_state

# Flatten image features and pass through multimodal projector
vision_flat = image_features.view(-1, image_features.size(-1))
precomputed_embeddings = multi_modal_projector(vision_flat)

# Build precomputed embedding data item
mm_item = dict(
    processor_output, 
    format="precomputed_embedding", 
    feature=precomputed_embeddings
)

# Use precomputed embeddings for efficient inference
out = llm.generate(input_ids=input_ids, image_data=[mm_item])
print("Llama 4 precomputed embedding response:")
print(out["text"])
```
