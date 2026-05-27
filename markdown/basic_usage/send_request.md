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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:16,  3.01it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]

    Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:06,  6.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:02, 12.29it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 18.19it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 25.85it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 35.10it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 44.23it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 44.23it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 44.23it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 44.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.58 GB):   3%|▎         | 2/58 [00:00<00:03, 14.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.58 GB):   7%|▋         | 4/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.58 GB):   7%|▋         | 4/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.57 GB):   7%|▋         | 4/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.56 GB):   7%|▋         | 4/58 [00:00<00:03, 16.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.56 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.56 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=71.53 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.53 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.52 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.51 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.51 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.51 GB):  21%|██        | 12/58 [00:00<00:03, 11.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.51 GB):  21%|██        | 12/58 [00:00<00:03, 11.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.51 GB):  21%|██        | 12/58 [00:01<00:03, 11.65it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.51 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.50 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.50 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.50 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.49 GB):  24%|██▍       | 14/58 [00:01<00:03, 12.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.49 GB):  31%|███       | 18/58 [00:01<00:02, 17.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.49 GB):  31%|███       | 18/58 [00:01<00:02, 17.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.49 GB):  31%|███       | 18/58 [00:01<00:02, 17.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.47 GB):  31%|███       | 18/58 [00:01<00:02, 17.39it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.48 GB):  31%|███       | 18/58 [00:01<00:02, 17.39it/s] Capturing num tokens (num_tokens=960 avail_mem=71.48 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=896 avail_mem=71.48 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=832 avail_mem=71.48 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=768 avail_mem=71.47 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=704 avail_mem=71.47 GB):  38%|███▊      | 22/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=704 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=640 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=576 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=512 avail_mem=71.45 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.25it/s]

    Capturing num tokens (num_tokens=480 avail_mem=71.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.25it/s]Capturing num tokens (num_tokens=480 avail_mem=71.47 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.22it/s]Capturing num tokens (num_tokens=448 avail_mem=71.47 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.22it/s]Capturing num tokens (num_tokens=416 avail_mem=71.46 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.22it/s]Capturing num tokens (num_tokens=384 avail_mem=71.46 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.22it/s]Capturing num tokens (num_tokens=352 avail_mem=71.46 GB):  52%|█████▏    | 30/58 [00:01<00:01, 27.22it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.46 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.05it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.25it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.25it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.25it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.25it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 29.25it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:02<00:00, 29.25it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:02<00:00, 36.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.53it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.53it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.53it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.53it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.53it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:02<00:00, 24.81it/s]


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


<strong style='color: #00008B;'>{'id': '039c86589da34012b7c45f638f7dcd60', 'object': 'chat.completion', 'created': 1779915723, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'f59c999763dc4e09b8d66909b44a4677', 'object': 'chat.completion', 'created': 1779915724, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='d8204769edc44276b71be0621f1db442', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1779915724, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'af087990e80b4abf8e5f3174327c2038', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.25190022960305214, 'response_sent_to_client_ts': 1779915725.0531223}}</strong>


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
