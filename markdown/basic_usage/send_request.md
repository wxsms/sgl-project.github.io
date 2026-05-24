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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:04<00:19,  2.57it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]

    Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03, 11.15it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]

    Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 18.06it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 25.72it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:04<00:00, 34.89it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 44.22it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 44.22it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 44.22it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 44.22it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 44.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.55 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.55 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.54 GB):   3%|▎         | 2/58 [00:00<00:03, 14.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.54 GB):   7%|▋         | 4/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.54 GB):   7%|▋         | 4/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.54 GB):   7%|▋         | 4/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.53 GB):   7%|▋         | 4/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.53 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.52 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.52 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.63it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=71.52 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.52 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.51 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.51 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.51 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.51 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.51 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.50 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.71it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=71.50 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.50 GB):  22%|██▏       | 13/58 [00:00<00:01, 23.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.50 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.49 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.49 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.49 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.47 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=960 avail_mem=71.48 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.30it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=71.48 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=832 avail_mem=71.48 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=768 avail_mem=71.47 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.30it/s]Capturing num tokens (num_tokens=768 avail_mem=71.47 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=704 avail_mem=71.47 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=640 avail_mem=71.47 GB):  43%|████▎     | 25/58 [00:00<00:01, 31.68it/s]Capturing num tokens (num_tokens=576 avail_mem=71.47 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.68it/s]Capturing num tokens (num_tokens=512 avail_mem=71.45 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.68it/s]Capturing num tokens (num_tokens=512 avail_mem=71.45 GB):  50%|█████     | 29/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=480 avail_mem=71.47 GB):  50%|█████     | 29/58 [00:01<00:00, 33.34it/s]

    Capturing num tokens (num_tokens=448 avail_mem=71.47 GB):  50%|█████     | 29/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=416 avail_mem=71.46 GB):  50%|█████     | 29/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=384 avail_mem=71.46 GB):  50%|█████     | 29/58 [00:01<00:00, 33.34it/s]Capturing num tokens (num_tokens=384 avail_mem=71.46 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=352 avail_mem=71.46 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=320 avail_mem=71.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=288 avail_mem=71.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=256 avail_mem=71.45 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.74it/s]Capturing num tokens (num_tokens=256 avail_mem=71.45 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=240 avail_mem=71.44 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.87it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.44 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=208 avail_mem=71.43 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=192 avail_mem=71.43 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.87it/s]Capturing num tokens (num_tokens=192 avail_mem=71.43 GB):  71%|███████   | 41/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=176 avail_mem=71.43 GB):  71%|███████   | 41/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=160 avail_mem=71.43 GB):  71%|███████   | 41/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=144 avail_mem=71.42 GB):  71%|███████   | 41/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=128 avail_mem=71.42 GB):  71%|███████   | 41/58 [00:01<00:00, 36.66it/s]Capturing num tokens (num_tokens=128 avail_mem=71.42 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.22it/s]Capturing num tokens (num_tokens=112 avail_mem=71.42 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.22it/s]

    Capturing num tokens (num_tokens=96 avail_mem=71.42 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.22it/s] Capturing num tokens (num_tokens=80 avail_mem=71.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.22it/s]Capturing num tokens (num_tokens=64 avail_mem=71.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.22it/s]Capturing num tokens (num_tokens=64 avail_mem=71.41 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=48 avail_mem=71.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=32 avail_mem=71.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=28 avail_mem=71.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=24 avail_mem=71.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.41it/s]Capturing num tokens (num_tokens=24 avail_mem=71.40 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=20 avail_mem=71.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.54it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=12 avail_mem=71.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.54it/s]Capturing num tokens (num_tokens=8 avail_mem=71.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 37.54it/s] Capturing num tokens (num_tokens=8 avail_mem=71.38 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=4 avail_mem=71.38 GB):  98%|█████████▊| 57/58 [00:01<00:00, 37.97it/s]Capturing num tokens (num_tokens=4 avail_mem=71.38 GB): 100%|██████████| 58/58 [00:01<00:00, 31.83it/s]


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


<strong style='color: #00008B;'>{'id': 'db59b283c5964f64a4af0de0e3bf1889', 'object': 'chat.completion', 'created': 1779608822, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'e9a080473f7e4ec18ea4650e19b6ed9c', 'object': 'chat.completion', 'created': 1779608822, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='1ce3174979ee4c249b2154124f7283e3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1779608822, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '80d28c1fc0a04bddab48387f5ea54852', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2307542897760868, 'response_sent_to_client_ts': 1779608823.4932065}}</strong>


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
