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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:29,  1.81it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]

    Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=2304):  14%|█▍        | 8/58 [00:04<00:14,  3.49it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=768):  28%|██▊       | 16/58 [00:04<00:04,  9.57it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:01, 17.65it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 21.81it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 41.63it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.83it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.52it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.01it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.01it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.01it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.01it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.01it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.86it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.03it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.03it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.71it/s]


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


<strong style='color: #00008B;'>{'id': '1213f8a78adb4ff895c2b495950f5009', 'object': 'chat.completion', 'created': 1779468727, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'dff71688616a4f34a6d47fc2265a8780', 'object': 'chat.completion', 'created': 1779468727, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='5c4412c806954eacb6aecb8b86fec08a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1779468728, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'e6eca86af5db4579af415e1372cff160', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22644315008074045, 'response_sent_to_client_ts': 1779468728.7916312}}</strong>


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
