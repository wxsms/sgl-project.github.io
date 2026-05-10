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


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-10 09:48:23] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


    2026-05-10 09:48:27,054 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 09:48:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.40it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.79it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.04it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 30.24it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 30.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.08 GB):   9%|▊         | 5/58 [00:00<00:02, 18.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.07 GB):   9%|▊         | 5/58 [00:00<00:02, 18.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.06 GB):   9%|▊         | 5/58 [00:00<00:02, 18.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.32it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.32it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.05 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.05 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.05 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.04 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.04 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.04 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.60it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.04 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.03 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.03 GB):  28%|██▊       | 16/58 [00:00<00:02, 16.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.03 GB):  28%|██▊       | 16/58 [00:00<00:02, 16.96it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.03 GB):  28%|██▊       | 16/58 [00:01<00:02, 16.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.03 GB):  31%|███       | 18/58 [00:01<00:02, 14.00it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.02 GB):  31%|███       | 18/58 [00:01<00:02, 14.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.02 GB):  31%|███       | 18/58 [00:01<00:02, 14.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.00 GB):  31%|███       | 18/58 [00:01<00:02, 14.00it/s]Capturing num tokens (num_tokens=960 avail_mem=58.02 GB):  31%|███       | 18/58 [00:01<00:02, 14.00it/s] Capturing num tokens (num_tokens=960 avail_mem=58.02 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=896 avail_mem=58.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=832 avail_mem=58.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]

    Capturing num tokens (num_tokens=768 avail_mem=58.01 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=704 avail_mem=58.00 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=704 avail_mem=58.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.89it/s]Capturing num tokens (num_tokens=640 avail_mem=58.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.89it/s]Capturing num tokens (num_tokens=576 avail_mem=58.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.89it/s]Capturing num tokens (num_tokens=512 avail_mem=57.99 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.89it/s]Capturing num tokens (num_tokens=480 avail_mem=58.00 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.89it/s]Capturing num tokens (num_tokens=480 avail_mem=58.00 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.46it/s]Capturing num tokens (num_tokens=448 avail_mem=58.00 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.46it/s]Capturing num tokens (num_tokens=416 avail_mem=58.00 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.46it/s]

    Capturing num tokens (num_tokens=384 avail_mem=57.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.46it/s]Capturing num tokens (num_tokens=352 avail_mem=57.99 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.46it/s]Capturing num tokens (num_tokens=352 avail_mem=57.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 28.46it/s]Capturing num tokens (num_tokens=320 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 28.46it/s]Capturing num tokens (num_tokens=288 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 28.46it/s]Capturing num tokens (num_tokens=256 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 28.46it/s]Capturing num tokens (num_tokens=240 avail_mem=57.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 28.46it/s]Capturing num tokens (num_tokens=240 avail_mem=57.98 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.76it/s]Capturing num tokens (num_tokens=224 avail_mem=57.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.76it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.76it/s]Capturing num tokens (num_tokens=192 avail_mem=57.97 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.76it/s]Capturing num tokens (num_tokens=176 avail_mem=57.96 GB):  66%|██████▌   | 38/58 [00:01<00:00, 28.76it/s]Capturing num tokens (num_tokens=176 avail_mem=57.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=160 avail_mem=57.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=144 avail_mem=57.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=128 avail_mem=57.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=112 avail_mem=57.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s]Capturing num tokens (num_tokens=96 avail_mem=57.95 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.66it/s] Capturing num tokens (num_tokens=96 avail_mem=57.95 GB):  81%|████████  | 47/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=80 avail_mem=57.95 GB):  81%|████████  | 47/58 [00:01<00:00, 34.27it/s]

    Capturing num tokens (num_tokens=64 avail_mem=57.94 GB):  81%|████████  | 47/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=48 avail_mem=57.94 GB):  81%|████████  | 47/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=32 avail_mem=57.94 GB):  81%|████████  | 47/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=28 avail_mem=57.93 GB):  81%|████████  | 47/58 [00:02<00:00, 34.27it/s]Capturing num tokens (num_tokens=28 avail_mem=57.93 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s]Capturing num tokens (num_tokens=24 avail_mem=57.93 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s]Capturing num tokens (num_tokens=20 avail_mem=57.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s]Capturing num tokens (num_tokens=16 avail_mem=57.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s]Capturing num tokens (num_tokens=12 avail_mem=57.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s]Capturing num tokens (num_tokens=8 avail_mem=57.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 37.51it/s] Capturing num tokens (num_tokens=8 avail_mem=57.92 GB):  98%|█████████▊| 57/58 [00:02<00:00, 40.07it/s]Capturing num tokens (num_tokens=4 avail_mem=57.91 GB):  98%|█████████▊| 57/58 [00:02<00:00, 40.07it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.91 GB): 100%|██████████| 58/58 [00:02<00:00, 26.98it/s]


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


<strong style='color: #00008B;'>{'id': 'ddf270f1e1854553b9bfc907fb82e5f9', 'object': 'chat.completion', 'created': 1778406522, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '037e65826b4f422f8a67b15088b2fdf3', 'object': 'chat.completion', 'created': 1778406522, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='e1b476bd83b146e7a15b27fd38c00fc4', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1778406522, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'e84bc6d4cf0b464a89e392995949679b', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22904851008206606, 'response_sent_to_client_ts': 1778406523.5064964}}</strong>


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
