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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.14it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.12it/s]


    2026-04-27 20:42:37,106 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 20:42:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=5632):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]

    Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:31,  1.65it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s]Compiling num tokens (num_tokens=960):  22%|██▏       | 13/58 [00:05<00:10,  4.37it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:04,  8.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.55it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=121.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=121.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=121.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=121.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=121.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=121.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=121.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=121.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=121.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=121.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=121.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=121.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=121.63 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]

    Capturing num tokens (num_tokens=960 avail_mem=121.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s] Capturing num tokens (num_tokens=896 avail_mem=121.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.82it/s]Capturing num tokens (num_tokens=896 avail_mem=121.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=832 avail_mem=121.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=768 avail_mem=121.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=704 avail_mem=121.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=640 avail_mem=121.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=576 avail_mem=121.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=576 avail_mem=121.63 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=512 avail_mem=121.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=480 avail_mem=121.63 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]

    Capturing num tokens (num_tokens=448 avail_mem=121.63 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=416 avail_mem=121.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=384 avail_mem=121.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.48it/s]Capturing num tokens (num_tokens=384 avail_mem=121.62 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=352 avail_mem=121.62 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=320 avail_mem=121.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=288 avail_mem=121.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=256 avail_mem=121.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=240 avail_mem=121.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 41.95it/s]Capturing num tokens (num_tokens=240 avail_mem=121.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=224 avail_mem=121.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]

    Capturing num tokens (num_tokens=208 avail_mem=121.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=192 avail_mem=121.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=176 avail_mem=121.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=160 avail_mem=121.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=160 avail_mem=121.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=144 avail_mem=121.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=128 avail_mem=121.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=112 avail_mem=121.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=96 avail_mem=121.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s] Capturing num tokens (num_tokens=80 avail_mem=121.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.42it/s]

    Capturing num tokens (num_tokens=80 avail_mem=121.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=64 avail_mem=121.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=48 avail_mem=121.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=32 avail_mem=121.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=28 avail_mem=121.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=24 avail_mem=121.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.31it/s]Capturing num tokens (num_tokens=24 avail_mem=121.56 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=20 avail_mem=121.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=16 avail_mem=121.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=12 avail_mem=121.55 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=8 avail_mem=121.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=121.54 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=4 avail_mem=121.54 GB): 100%|██████████| 58/58 [00:01<00:00, 43.98it/s]Capturing num tokens (num_tokens=4 avail_mem=121.54 GB): 100%|██████████| 58/58 [00:01<00:00, 39.39it/s]


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


<strong style='color: #00008B;'>{'id': '310b46291b694b9d937663035db23f0c', 'object': 'chat.completion', 'created': 1777322572, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '620606e1c0ac45f982828582f26d7ffe', 'object': 'chat.completion', 'created': 1777322572, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='b70f2a1c3b3f4deb9b3d05f552c3d37a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1777322572, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '61b285c03f0f4d2aae1b745b789d6021', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.270369078963995, 'response_sent_to_client_ts': 1777322573.3204153}}</strong>


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
