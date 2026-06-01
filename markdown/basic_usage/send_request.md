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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:52,  1.04it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:52,  1.04it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:52,  1.04it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:05<00:52,  1.04it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.52it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:05<00:08,  5.48it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03, 10.51it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 15.72it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 22.57it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 30.23it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 37.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.79 GB):   3%|▎         | 2/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.78 GB):   3%|▎         | 2/58 [00:00<00:03, 14.63it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.78 GB):   7%|▋         | 4/58 [00:00<00:03, 16.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.78 GB):   7%|▋         | 4/58 [00:00<00:03, 16.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.78 GB):   7%|▋         | 4/58 [00:00<00:03, 16.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.77 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=57.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.34it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.76 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.75 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.75 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.75 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.93it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=57.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=960 avail_mem=57.72 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.35it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=57.72 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=832 avail_mem=57.72 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=768 avail_mem=57.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=768 avail_mem=57.71 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.55it/s]Capturing num tokens (num_tokens=704 avail_mem=57.71 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.55it/s]Capturing num tokens (num_tokens=640 avail_mem=57.71 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.55it/s]Capturing num tokens (num_tokens=576 avail_mem=57.71 GB):  43%|████▎     | 25/58 [00:01<00:01, 28.55it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.71 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.43it/s]Capturing num tokens (num_tokens=512 avail_mem=56.36 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.43it/s]Capturing num tokens (num_tokens=480 avail_mem=56.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.43it/s]Capturing num tokens (num_tokens=448 avail_mem=56.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.43it/s]Capturing num tokens (num_tokens=416 avail_mem=56.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.43it/s]Capturing num tokens (num_tokens=416 avail_mem=56.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=384 avail_mem=56.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=352 avail_mem=56.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=320 avail_mem=56.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.93it/s]

    Capturing num tokens (num_tokens=288 avail_mem=56.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=288 avail_mem=56.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=256 avail_mem=56.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=240 avail_mem=56.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=224 avail_mem=56.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=208 avail_mem=56.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=208 avail_mem=56.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.66it/s]Capturing num tokens (num_tokens=192 avail_mem=56.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.66it/s]Capturing num tokens (num_tokens=176 avail_mem=56.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.66it/s]

    Capturing num tokens (num_tokens=160 avail_mem=56.31 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.66it/s]Capturing num tokens (num_tokens=144 avail_mem=56.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 29.66it/s]Capturing num tokens (num_tokens=144 avail_mem=56.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=128 avail_mem=56.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=112 avail_mem=56.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.37it/s]Capturing num tokens (num_tokens=96 avail_mem=56.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.37it/s] Capturing num tokens (num_tokens=96 avail_mem=56.29 GB):  81%|████████  | 47/58 [00:01<00:00, 28.63it/s]Capturing num tokens (num_tokens=80 avail_mem=56.29 GB):  81%|████████  | 47/58 [00:01<00:00, 28.63it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.29 GB):  81%|████████  | 47/58 [00:01<00:00, 28.63it/s]Capturing num tokens (num_tokens=48 avail_mem=56.28 GB):  81%|████████  | 47/58 [00:01<00:00, 28.63it/s]Capturing num tokens (num_tokens=32 avail_mem=58.91 GB):  81%|████████  | 47/58 [00:01<00:00, 28.63it/s]Capturing num tokens (num_tokens=32 avail_mem=58.91 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.63it/s]Capturing num tokens (num_tokens=28 avail_mem=58.90 GB):  88%|████████▊ | 51/58 [00:01<00:00, 24.63it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.90 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.63it/s]Capturing num tokens (num_tokens=20 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.63it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.63it/s]Capturing num tokens (num_tokens=16 avail_mem=58.89 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.68it/s]Capturing num tokens (num_tokens=12 avail_mem=58.89 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.68it/s]Capturing num tokens (num_tokens=8 avail_mem=58.89 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.68it/s] Capturing num tokens (num_tokens=4 avail_mem=58.88 GB):  95%|█████████▍| 55/58 [00:02<00:00, 27.68it/s]Capturing num tokens (num_tokens=4 avail_mem=58.88 GB): 100%|██████████| 58/58 [00:02<00:00, 26.89it/s]


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


<strong style='color: #00008B;'>{'id': '311901568721409ca74afbaef3f0401b', 'object': 'chat.completion', 'created': 1780304532, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'f16f8a9cff884580ab08f01cf367082f', 'object': 'chat.completion', 'created': 1780304532, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='20c94806e76b4508ab47caa13ee35623', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1780304533, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'f72b65585a4140feb1564a4ad3c7a903', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2585073560476303, 'response_sent_to_client_ts': 1780304533.7312899}}</strong>


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
