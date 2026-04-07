# Sending Requests
This notebook provides a quick-start guide to use SGLang in chat completions after installation. Once your server is running, API documentation is available at `http://localhost:30000/docs` (Swagger UI), `http://localhost:30000/redoc` (ReDoc), or `http://localhost:30000/openapi.json` (OpenAPI spec, useful for AI agents). Replace `30000` with your port if using a different one.

- For Vision Language Models, see [OpenAI APIs - Vision](openai_api_vision.ipynb).
- For Embedding Models, see [OpenAI APIs - Embedding](openai_api_embeddings.ipynb) and [Encode (embedding model)](native_api.html#Encode-(embedding-model)).
- For Reward Models, see [Classify (reward model)](native_api.html#Classify-(reward-model)).

## Launch A Server


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
 --host 0.0.0.0 --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


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
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3431, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1587, in get_hf_file_metadata
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
    retry() failed once (0th try, maximum 2 retries). Will delay 0.89s and retry. Error: The read operation timed out


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:172: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    2026-04-07 22:11:37.737 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:11:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:11:37.737 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:11:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:11:37.737 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:11:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:11:37.737 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:11:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-07 22:11:37.737 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-07 22:11:37] Persistent cache disabled, using in-memory JIT cache


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
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3431, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1587, in get_hf_file_metadata
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
    [2026-04-07 22:11:38] retry() failed once (0th try, maximum 2 retries). Will delay 0.78s and retry. Error: The read operation timed out


    [2026-04-07 22:11:39] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-07 22:11:39] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-07 22:11:39] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-07 22:11:39] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.34it/s]


    2026-04-07 22:11:40,247 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 22:11:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.01it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.17it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.10it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:01, 24.57it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 33.90it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 38.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 33.46it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  31%|███       | 18/58 [00:00<00:01, 33.46it/s] Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:00<00:01, 35.33it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]

    Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.49it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.23it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.23it/s]

    Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  60%|██████    | 35/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.99it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.99it/s]

    Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.22it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]

    Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.18it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.18it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 36.04it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Using cURL



```python
import subprocess, json

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "qwen/qwen2.5-0.5b-instruct", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
"""

response = json.loads(subprocess.check_output(curl_command, shell=True))
print_highlight(response)
```


<strong style='color: #00008B;'>{'id': '171d7a28939b4edcbdcb49b302ab7f27', 'object': 'chat.completion', 'created': 1775599914, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


## Using Python Requests


```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'id': '7da96f48fd9049da9ae3d269330866b0', 'object': 'chat.completion', 'created': 1775599914, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


## Using OpenAI Python Client


```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print_highlight(response)
```


<strong style='color: #00008B;'>ChatCompletion(id='4e47b6d74b6646e4bb3083fa0ed5d5f5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1775599914, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


### Streaming


```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# Use stream=True for streaming responses
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
    stream=True,
)

# Handle the streaming output
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

    Sure

    ,

     here

     are

     three

     countries

     and

     their

     respective

     capitals

    :
    


    1

    .

     **

    United

     States

    **

     -

     Washington

     D

    .C

    .


    2

    .

     **

    Canada

    **

     -

     Ottawa

    


    3

    .

     **

    Australia

    **

     -

     Canberra

## Using Native Generation APIs

You can also use the native `/generate` endpoint with requests, which provides more flexibility. An API reference is available at [Sampling Parameters](sampling_params.md).


```python
import requests

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '2b6f5f4ac8304137ae5e4dd5725eb8f4', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2561388360336423, 'response_sent_to_client_ts': 1775599915.556652}}</strong>


### Streaming


```python
import requests, json

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"]
        print(output[prev:], end="", flush=True)
        prev = len(output)
```

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     second

     largest

     city

     in

     the

     world

    .

     It

     is

     located

     in

     the

     south

     of

     France

    ,

     on

     the

     banks

     of

     the


```python
terminate_process(server_process)
```
