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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.04it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.37it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.62it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.77it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.77it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.26 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.25 GB):   3%|▎         | 2/58 [00:00<00:03, 17.85it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.23 GB):   9%|▊         | 5/58 [00:00<00:02, 22.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.23 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.22 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.80it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.22 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.22 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.22 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.21 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.21 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.21 GB):  21%|██        | 12/58 [00:00<00:01, 30.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.18 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.90it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=58.18 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s]Capturing num tokens (num_tokens=960 avail_mem=58.19 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s] Capturing num tokens (num_tokens=896 avail_mem=58.16 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s]Capturing num tokens (num_tokens=832 avail_mem=58.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s]Capturing num tokens (num_tokens=768 avail_mem=58.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s]Capturing num tokens (num_tokens=704 avail_mem=58.15 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.53it/s]Capturing num tokens (num_tokens=704 avail_mem=58.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=640 avail_mem=58.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=576 avail_mem=58.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=512 avail_mem=58.13 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=480 avail_mem=58.15 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]Capturing num tokens (num_tokens=448 avail_mem=58.14 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.58it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.14 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=416 avail_mem=58.14 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=384 avail_mem=58.14 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=352 avail_mem=58.13 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=320 avail_mem=58.13 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=288 avail_mem=58.13 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.47it/s]Capturing num tokens (num_tokens=288 avail_mem=58.13 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=256 avail_mem=58.12 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=240 avail_mem=58.12 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.55it/s]Capturing num tokens (num_tokens=224 avail_mem=58.12 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=208 avail_mem=58.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.55it/s]Capturing num tokens (num_tokens=192 avail_mem=58.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.55it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.11 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=176 avail_mem=58.11 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=160 avail_mem=58.11 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=144 avail_mem=58.10 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=128 avail_mem=58.10 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=112 avail_mem=58.10 GB):  71%|███████   | 41/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=112 avail_mem=58.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=96 avail_mem=58.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s] Capturing num tokens (num_tokens=80 avail_mem=58.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=64 avail_mem=58.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=48 avail_mem=58.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=32 avail_mem=58.08 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.80it/s]

    Capturing num tokens (num_tokens=32 avail_mem=58.08 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=28 avail_mem=58.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=24 avail_mem=58.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=20 avail_mem=58.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=16 avail_mem=58.07 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=12 avail_mem=58.06 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.91it/s]Capturing num tokens (num_tokens=12 avail_mem=58.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.50it/s]Capturing num tokens (num_tokens=8 avail_mem=58.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.50it/s] Capturing num tokens (num_tokens=4 avail_mem=58.06 GB):  97%|█████████▋| 56/58 [00:01<00:00, 47.50it/s]Capturing num tokens (num_tokens=4 avail_mem=58.06 GB): 100%|██████████| 58/58 [00:01<00:00, 40.82it/s]


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


<strong style='color: #00008B;'>{'id': 'c9fce4e094984249b537a3c24cd89ede', 'object': 'chat.completion', 'created': 1780921623, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '29863984db75448cacba82560b7e09ba', 'object': 'chat.completion', 'created': 1780921623, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='96e918b414f847f3b9750bab788db11b', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1780921624, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '95051248300743d4b58da6c043df7f25', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2719079479575157, 'response_sent_to_client_ts': 1780921624.9222927}}</strong>


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
