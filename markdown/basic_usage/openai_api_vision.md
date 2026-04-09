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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a13-48bb8a123a7afade378e740b;498abfba-b139-4721-8989-2adfd5034923)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    retry() failed once (0th try, maximum 2 retries). Will delay 0.79s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a13-48bb8a123a7afade378e740b;498abfba-b139-4721-8989-2adfd5034923)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a14-137379800edd452418f4c88b;c4a2187d-9054-4847-93c2-fb24348e97c7)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    retry() failed once (1th try, maximum 2 retries). Will delay 1.51s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a14-137379800edd452418f4c88b;c4a2187d-9054-4847-93c2-fb24348e97c7)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a15-1ada1f464e6a4dd523e099cb;4d3eeaa5-4849-4284-b3a4-26ebc8601796)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a17-51d6c8e3417f480475d571fa;512dbad2-8e87-4958-bfa0-19ba4415f3f0)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:15] retry() failed once (0th try, maximum 2 retries). Will delay 0.81s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a17-51d6c8e3417f480475d571fa;512dbad2-8e87-4958-bfa0-19ba4415f3f0)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a17-4bb47ce763cfd71822dfe330;a0cb8df6-5ede-4ebd-ab2c-888863e590ab)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:15] retry() failed once (1th try, maximum 2 retries). Will delay 1.62s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a17-4bb47ce763cfd71822dfe330;a0cb8df6-5ede-4ebd-ab2c-888863e590ab)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a19-25805b4857bc979673bdd943;53062a02-31ed-442d-8bd1-8f841e427c86)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:17] Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.


    [2026-04-09 10:06:19] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:19] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:20] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:20] Retrying in 2s [Retry 2/5].


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a1e-3cf9096008666f3840e90e70;a1399129-ea85-4d28-bbd8-178e1ad4d583)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:22] retry() failed once (0th try, maximum 2 retries). Will delay 0.84s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a1e-3cf9096008666f3840e90e70;a1399129-ea85-4d28-bbd8-178e1ad4d583)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:22] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:22] Retrying in 4s [Retry 3/5].


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a1f-5a8962474268107012c5be8d;f1081440-fe93-4a4f-b6b2-71d57bb41b63)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:23] retry() failed once (1th try, maximum 2 retries). Will delay 1.79s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a1f-5a8962474268107012c5be8d;f1081440-fe93-4a4f-b6b2-71d57bb41b63)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d77a21-588bc9f17185be881b2a87ff;cebf92db-04c1-4db2-b481-de6534320312)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:06:25] Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:25] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:25] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:26] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:26] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:28] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:28] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:32] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:32] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:34] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:34] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:40] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:40] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:42] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:42] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:42] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:43] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:43] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:46] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:48] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:48] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:48] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:49] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:49] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:50] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:50] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:51] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:51] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:55] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:55] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:58] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:58] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:04] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:06] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:07] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:07] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:09] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:09] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:12] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:12] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:12] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:13] Retrying in 2s [Retry 2/5].
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:13] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:15] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:15] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:19] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:21] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:21] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:27] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:27] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:29] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:29] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:29] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:31] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:31] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:33] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:33] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:35] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:36] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:36] Retrying in 2s [Retry 2/5].
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:37] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:37] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:39] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:39] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:43] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:43] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:45] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:45] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:51] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:51] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:53] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:53] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:53] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:54] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:54] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:56] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:56] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:59] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:59] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:59] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:00] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:00] Retrying in 2s [Retry 2/5].
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:00] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:00] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:02] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:02] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:06] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:08] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:08] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:14] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:14] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:16] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:16] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:16] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:17] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:17] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:19] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:08:22] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:22] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:22] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:23] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:23] Retrying in 2s [Retry 2/5].
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:24] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:24] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:25] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:25] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:30] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:30] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:32] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:32] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:38] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:38] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:40] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:40] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:40] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:41] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:41] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:43] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:43] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:46] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:47] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:47] Retrying in 2s [Retry 2/5].
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:47] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:47] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:49] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:49] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:53] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:53] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:55] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:55] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:01] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:09:01] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:03] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:04] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:05] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:09:05] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:07] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:09:07] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:09] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:10] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:10] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:11] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:09:11] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:11] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:11] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:09:13] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:09:13] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:15] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:09:15] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:17] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:09:17] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:09:19] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:26] Retrying in 1s [Retry 1/5].


    [2026-04-09 10:09:28] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:28] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:28] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:28] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.89it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:01,  1.53it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.45it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.34it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.65it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.57it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-09 10:09:38,558 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 10:09:38] Unexpected error during package walk: cutlass.cute.experimental



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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:799: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>{"id":"dd1fa1785b584186b154924b99cfd400","object":"chat.completion","created":1775729385,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing a blue shirt. The taxi is parked on a city street with other vehicles and buildings in the background. The man appears to be balancing on the tailgate while performing this task. The scene suggests an unusual or humorous situation, as ironing clothes outdoors is not a typical activity.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":380,"completion_tokens":73,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>



<strong style='color: #00008B;'>{"id":"e56b03091abd4db3b8898e23715131a1","object":"chat.completion","created":1775729386,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing a blue shirt. The taxi is parked on a city street with other vehicles and buildings in the background. The man appears to be balancing on the tailgate while performing the task. The scene suggests an unusual or humorous situation, as it is not typical for someone to iron clothes from the back of a moving vehicle.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":387,"completion_tokens":80,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


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


<strong style='color: #00008B;'>{"id":"3d6d4b63cfab45fe9d0959320c2a43d7","object":"chat.completion","created":1775729387,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows a man standing on the back of a yellow taxi, ironing a piece of clothing. The taxi is parked on a city street, and there are other taxis visible in the background. The man appears to be balancing on the tailgate while ironing, which is an unusual and humorous scene. The setting suggests an urban environment with buildings and flags in the background.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":307,"total_tokens":384,"completion_tokens":77,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}</strong>


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


<strong style='color: #00008B;'>The image shows a man standing on the back of a yellow taxi, ironing a piece of clothing. The taxi is parked on a city street, and there are other taxis visible in the background. The man appears to be balancing on the tailgate while ironing, which is an unusual and humorous scene. The setting suggests it might be in a busy urban area with tall buildings and flags in the background.</strong>


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


<strong style='color: #00008B;'>The first image shows a man ironing clothes on the back of a taxi in an urban setting. The second image is a stylized logo featuring the letters "SGL" with a book and a computer icon incorporated into the design.</strong>



```python
terminate_process(vision_process)
```
