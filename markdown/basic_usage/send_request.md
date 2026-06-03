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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:32,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.67it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.67it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.67it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.24it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.92it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 20.72it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.44it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.66it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.31it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.00it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.00it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.00it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.00it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.00it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.00it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.96it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.92it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.74it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.57it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.39it/s]


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


<strong style='color: #00008B;'>{'id': 'f682f338d49d4736bf1c01f828cb03a2', 'object': 'chat.completion', 'created': 1780447319, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '8d7997c0c2a94718ac4fef4203dd8410', 'object': 'chat.completion', 'created': 1780447319, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='ff749f5bd1cb400280e86c56fdbbe456', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1780447319, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '4343519bace34a04a145d78b5385d808', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.232062466442585, 'response_sent_to_client_ts': 1780447320.1951122}}</strong>


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
