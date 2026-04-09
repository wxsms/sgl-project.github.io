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
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/hf_quant_config.json'
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
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779ae-7e72ffff021a023d1b430b27;dedb31a0-9212-4496-986f-dab9124bb9f3)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:04:30] retry() failed once (0th try, maximum 2 retries). Will delay 0.79s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779ae-7e72ffff021a023d1b430b27;dedb31a0-9212-4496-986f-dab9124bb9f3)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-04-09 10:04:40] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:41] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:41] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:41] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.60it/s]


    2026-04-09 10:04:42,029 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 10:04:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.76it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.19it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.19it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.19it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 17.48it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.05it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 28.70it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 36.15it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 42.17it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 42.17it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 42.17it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 42.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.47 GB):   3%|▎         | 2/58 [00:00<00:03, 14.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.47 GB):   3%|▎         | 2/58 [00:00<00:03, 14.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.47 GB):   3%|▎         | 2/58 [00:00<00:03, 14.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.47 GB):   7%|▋         | 4/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.47 GB):   7%|▋         | 4/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.46 GB):   7%|▋         | 4/58 [00:00<00:03, 15.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.46 GB):  10%|█         | 6/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.46 GB):  10%|█         | 6/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.46 GB):  10%|█         | 6/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.46 GB):  10%|█         | 6/58 [00:00<00:02, 17.42it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=118.46 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.46 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.45 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.45 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.45 GB):  21%|██        | 12/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.44 GB):  21%|██        | 12/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.44 GB):  21%|██        | 12/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.44 GB):  21%|██        | 12/58 [00:00<00:02, 22.30it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=118.44 GB):  21%|██        | 12/58 [00:00<00:02, 22.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.44 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.43 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.42 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.42 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.40 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.95it/s]Capturing num tokens (num_tokens=960 avail_mem=118.42 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.95it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=118.41 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.95it/s]Capturing num tokens (num_tokens=832 avail_mem=118.41 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.95it/s]Capturing num tokens (num_tokens=832 avail_mem=118.41 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=768 avail_mem=118.41 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=704 avail_mem=118.40 GB):  41%|████▏     | 24/58 [00:00<00:01, 30.17it/s]Capturing num tokens (num_tokens=640 avail_mem=118.11 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.17it/s]

    Capturing num tokens (num_tokens=576 avail_mem=117.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.17it/s]Capturing num tokens (num_tokens=576 avail_mem=117.24 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=512 avail_mem=117.23 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.11it/s]

    Capturing num tokens (num_tokens=480 avail_mem=117.25 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.11it/s]Capturing num tokens (num_tokens=448 avail_mem=117.25 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.11it/s]

    Capturing num tokens (num_tokens=448 avail_mem=117.25 GB):  53%|█████▎    | 31/58 [00:01<00:01, 16.73it/s]Capturing num tokens (num_tokens=416 avail_mem=117.24 GB):  53%|█████▎    | 31/58 [00:01<00:01, 16.73it/s]Capturing num tokens (num_tokens=384 avail_mem=117.24 GB):  53%|█████▎    | 31/58 [00:01<00:01, 16.73it/s]Capturing num tokens (num_tokens=352 avail_mem=117.24 GB):  53%|█████▎    | 31/58 [00:01<00:01, 16.73it/s]

    Capturing num tokens (num_tokens=352 avail_mem=117.24 GB):  59%|█████▊    | 34/58 [00:01<00:01, 14.29it/s]Capturing num tokens (num_tokens=320 avail_mem=117.23 GB):  59%|█████▊    | 34/58 [00:01<00:01, 14.29it/s]Capturing num tokens (num_tokens=288 avail_mem=117.23 GB):  59%|█████▊    | 34/58 [00:01<00:01, 14.29it/s]Capturing num tokens (num_tokens=288 avail_mem=117.23 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.17it/s]Capturing num tokens (num_tokens=256 avail_mem=117.23 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.17it/s]

    Capturing num tokens (num_tokens=240 avail_mem=117.22 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.17it/s]Capturing num tokens (num_tokens=240 avail_mem=117.22 GB):  66%|██████▌   | 38/58 [00:02<00:01, 12.55it/s]Capturing num tokens (num_tokens=224 avail_mem=117.22 GB):  66%|██████▌   | 38/58 [00:02<00:01, 12.55it/s]Capturing num tokens (num_tokens=208 avail_mem=117.22 GB):  66%|██████▌   | 38/58 [00:02<00:01, 12.55it/s]

    Capturing num tokens (num_tokens=208 avail_mem=117.22 GB):  69%|██████▉   | 40/58 [00:02<00:01, 11.60it/s]Capturing num tokens (num_tokens=192 avail_mem=117.22 GB):  69%|██████▉   | 40/58 [00:02<00:01, 11.60it/s]Capturing num tokens (num_tokens=176 avail_mem=117.21 GB):  69%|██████▉   | 40/58 [00:02<00:01, 11.60it/s]

    Capturing num tokens (num_tokens=176 avail_mem=117.21 GB):  72%|███████▏  | 42/58 [00:02<00:01, 10.99it/s]Capturing num tokens (num_tokens=160 avail_mem=117.21 GB):  72%|███████▏  | 42/58 [00:02<00:01, 10.99it/s]Capturing num tokens (num_tokens=144 avail_mem=117.20 GB):  72%|███████▏  | 42/58 [00:02<00:01, 10.99it/s]Capturing num tokens (num_tokens=144 avail_mem=117.20 GB):  76%|███████▌  | 44/58 [00:02<00:01, 10.98it/s]Capturing num tokens (num_tokens=128 avail_mem=117.20 GB):  76%|███████▌  | 44/58 [00:02<00:01, 10.98it/s]

    Capturing num tokens (num_tokens=112 avail_mem=117.20 GB):  76%|███████▌  | 44/58 [00:02<00:01, 10.98it/s]Capturing num tokens (num_tokens=112 avail_mem=117.20 GB):  79%|███████▉  | 46/58 [00:02<00:01, 10.89it/s]Capturing num tokens (num_tokens=96 avail_mem=117.19 GB):  79%|███████▉  | 46/58 [00:02<00:01, 10.89it/s] Capturing num tokens (num_tokens=80 avail_mem=117.19 GB):  79%|███████▉  | 46/58 [00:03<00:01, 10.89it/s]

    Capturing num tokens (num_tokens=80 avail_mem=117.19 GB):  83%|████████▎ | 48/58 [00:03<00:00, 10.79it/s]Capturing num tokens (num_tokens=64 avail_mem=117.19 GB):  83%|████████▎ | 48/58 [00:03<00:00, 10.79it/s]Capturing num tokens (num_tokens=48 avail_mem=117.19 GB):  83%|████████▎ | 48/58 [00:03<00:00, 10.79it/s]Capturing num tokens (num_tokens=48 avail_mem=117.19 GB):  86%|████████▌ | 50/58 [00:03<00:00, 10.86it/s]Capturing num tokens (num_tokens=32 avail_mem=117.18 GB):  86%|████████▌ | 50/58 [00:03<00:00, 10.86it/s]

    Capturing num tokens (num_tokens=28 avail_mem=117.18 GB):  86%|████████▌ | 50/58 [00:03<00:00, 10.86it/s]Capturing num tokens (num_tokens=28 avail_mem=117.18 GB):  90%|████████▉ | 52/58 [00:03<00:00, 10.71it/s]Capturing num tokens (num_tokens=24 avail_mem=117.18 GB):  90%|████████▉ | 52/58 [00:03<00:00, 10.71it/s]Capturing num tokens (num_tokens=20 avail_mem=117.17 GB):  90%|████████▉ | 52/58 [00:03<00:00, 10.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=117.17 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.06it/s]Capturing num tokens (num_tokens=16 avail_mem=117.17 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.06it/s]Capturing num tokens (num_tokens=12 avail_mem=117.17 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.06it/s]Capturing num tokens (num_tokens=12 avail_mem=117.17 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.08it/s]Capturing num tokens (num_tokens=8 avail_mem=117.16 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.08it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=117.16 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.08it/s]Capturing num tokens (num_tokens=4 avail_mem=117.16 GB): 100%|██████████| 58/58 [00:04<00:00, 11.40it/s]Capturing num tokens (num_tokens=4 avail_mem=117.16 GB): 100%|██████████| 58/58 [00:04<00:00, 14.25it/s]


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


<strong style='color: #00008B;'>{'id': '6a973babcaf84632b8f38db245a08043', 'object': 'chat.completion', 'created': 1775729098, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'b9074275c90f47588f467fc082230efa', 'object': 'chat.completion', 'created': 1775729098, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='f20e5567facb4862bd9ffae35142b1f5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1775729099, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '8eb7a279c5884625aa6dc525c957d81c', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24671756196767092, 'response_sent_to_client_ts': 1775729100.0432115}}</strong>


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
