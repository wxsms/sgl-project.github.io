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


    [2026-05-07 21:30:05] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]


    2026-05-07 21:30:08,550 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 21:30:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:57,  4.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.79it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.91it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.48it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.48it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.48it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.48it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.48it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.55it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=352):  43%|████▎     | 25/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:00, 25.81it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:00, 25.81it/s]

    Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=144):  59%|█████▊    | 34/58 [00:05<00:00, 25.81it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:05<00:00, 38.26it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 49.28it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 49.28it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 49.28it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 49.28it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 49.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.41 GB):   2%|▏         | 1/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.38 GB):   2%|▏         | 1/58 [00:00<00:07,  7.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.38 GB):   3%|▎         | 2/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.38 GB):   3%|▎         | 2/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.37 GB):   3%|▎         | 2/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.37 GB):   3%|▎         | 2/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.37 GB):   9%|▊         | 5/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.37 GB):   9%|▊         | 5/58 [00:00<00:03, 15.08it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:03, 15.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 16.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.46it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.46it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:01<00:01, 30.46it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:01<00:01, 30.46it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 30.46it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 30.46it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:01<00:00, 35.43it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.26it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.41it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.76it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 33.14it/s]


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


<strong style='color: #00008B;'>{'id': 'e12d7376d85c4f098c3afe69f6288db1', 'object': 'chat.completion', 'created': 1778189423, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'fdccae216a0e4460bae1904d29794abe', 'object': 'chat.completion', 'created': 1778189423, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='71673a164ff649c1a047ed872411ffac', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1778189423, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'c6c98cd2d04045c7ab70a5bac68e1f7e', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2082052957266569, 'response_sent_to_client_ts': 1778189424.0948253}}</strong>


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
