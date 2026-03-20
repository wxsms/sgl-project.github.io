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

    [2026-03-20 13:06:22] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-20 13:06:22] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-20 13:06:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 13:06:27] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 13:06:27] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 13:06:27] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-20 13:06:30] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
    [2026-03-20 13:06:30] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.


    [2026-03-20 13:06:30] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:175: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [2026-03-20 13:06:31] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 13:06:36] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 13:06:36] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 13:06:36] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-20 13:06:36] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 13:06:36] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 13:06:36] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 13:06:38] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 13:06:39] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  7.57it/s]Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  7.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:12,  2.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:12,  2.33s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:58,  1.05s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:58,  1.05s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:58,  1.05s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.91it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.91it/s]

    Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.91it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.91it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:07,  6.76it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:02<00:04, 11.16it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:02<00:04, 11.16it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:02<00:04, 11.16it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04, 11.16it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04, 11.16it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:04, 11.16it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 17.14it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 24.72it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:00, 33.48it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 41.49it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 49.38it/s]

    Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 56.14it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 56.14it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 56.14it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 56.14it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 56.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=132.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=132.32 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=132.31 GB):   2%|▏         | 1/58 [00:00<00:08,  6.88it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=132.31 GB):   3%|▎         | 2/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.31 GB):   3%|▎         | 2/58 [00:00<00:07,  7.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=132.31 GB):   5%|▌         | 3/58 [00:00<00:07,  7.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=132.33 GB):   5%|▌         | 3/58 [00:00<00:07,  7.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=132.33 GB):   7%|▋         | 4/58 [00:00<00:06,  8.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=132.35 GB):   7%|▋         | 4/58 [00:00<00:06,  8.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=132.35 GB):   9%|▊         | 5/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=132.16 GB):   9%|▊         | 5/58 [00:00<00:06,  8.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=132.20 GB):   9%|▊         | 5/58 [00:00<00:06,  8.81it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=132.20 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=132.22 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.23 GB):  12%|█▏        | 7/58 [00:00<00:05, 10.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=132.23 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=3840 avail_mem=132.27 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.85it/s]Capturing num tokens (num_tokens=3584 avail_mem=132.26 GB):  16%|█▌        | 9/58 [00:00<00:04, 11.85it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=132.26 GB):  19%|█▉        | 11/58 [00:01<00:03, 13.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=132.27 GB):  19%|█▉        | 11/58 [00:01<00:03, 13.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.28 GB):  19%|█▉        | 11/58 [00:01<00:03, 13.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=132.28 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=132.28 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=132.29 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.23it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=132.31 GB):  22%|██▏       | 13/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=132.31 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=132.42 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=132.34 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=132.34 GB):  28%|██▊       | 16/58 [00:01<00:02, 18.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=132.34 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=132.33 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=132.39 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.55it/s]

    Capturing num tokens (num_tokens=960 avail_mem=132.40 GB):  33%|███▎      | 19/58 [00:01<00:01, 20.55it/s] Capturing num tokens (num_tokens=960 avail_mem=132.40 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.98it/s]Capturing num tokens (num_tokens=896 avail_mem=132.39 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.98it/s]Capturing num tokens (num_tokens=832 avail_mem=132.37 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.98it/s]Capturing num tokens (num_tokens=768 avail_mem=132.37 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.98it/s]Capturing num tokens (num_tokens=704 avail_mem=132.36 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.98it/s]Capturing num tokens (num_tokens=704 avail_mem=132.36 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=640 avail_mem=132.35 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=576 avail_mem=132.33 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.19it/s]

    Capturing num tokens (num_tokens=512 avail_mem=132.35 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=480 avail_mem=132.36 GB):  45%|████▍     | 26/58 [00:01<00:01, 26.19it/s]Capturing num tokens (num_tokens=480 avail_mem=132.36 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=448 avail_mem=132.35 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=416 avail_mem=132.35 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=384 avail_mem=132.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=352 avail_mem=132.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=352 avail_mem=132.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.10it/s]Capturing num tokens (num_tokens=320 avail_mem=132.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.10it/s]

    Capturing num tokens (num_tokens=288 avail_mem=132.28 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.10it/s]Capturing num tokens (num_tokens=256 avail_mem=132.27 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.10it/s]Capturing num tokens (num_tokens=240 avail_mem=132.26 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.10it/s]Capturing num tokens (num_tokens=240 avail_mem=132.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.07it/s]Capturing num tokens (num_tokens=224 avail_mem=132.26 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.07it/s]Capturing num tokens (num_tokens=208 avail_mem=132.25 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.07it/s]Capturing num tokens (num_tokens=192 avail_mem=132.24 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.07it/s]Capturing num tokens (num_tokens=176 avail_mem=132.23 GB):  66%|██████▌   | 38/58 [00:02<00:00, 31.07it/s]Capturing num tokens (num_tokens=176 avail_mem=132.23 GB):  72%|███████▏  | 42/58 [00:02<00:00, 32.48it/s]Capturing num tokens (num_tokens=160 avail_mem=132.22 GB):  72%|███████▏  | 42/58 [00:02<00:00, 32.48it/s]

    Capturing num tokens (num_tokens=144 avail_mem=132.21 GB):  72%|███████▏  | 42/58 [00:02<00:00, 32.48it/s]Capturing num tokens (num_tokens=128 avail_mem=132.20 GB):  72%|███████▏  | 42/58 [00:02<00:00, 32.48it/s]Capturing num tokens (num_tokens=112 avail_mem=132.19 GB):  72%|███████▏  | 42/58 [00:02<00:00, 32.48it/s]Capturing num tokens (num_tokens=112 avail_mem=132.19 GB):  79%|███████▉  | 46/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=96 avail_mem=132.18 GB):  79%|███████▉  | 46/58 [00:02<00:00, 33.25it/s] Capturing num tokens (num_tokens=80 avail_mem=132.17 GB):  79%|███████▉  | 46/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=64 avail_mem=132.19 GB):  79%|███████▉  | 46/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=48 avail_mem=132.17 GB):  79%|███████▉  | 46/58 [00:02<00:00, 33.25it/s]Capturing num tokens (num_tokens=48 avail_mem=132.17 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.97it/s]Capturing num tokens (num_tokens=32 avail_mem=132.16 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.97it/s]

    Capturing num tokens (num_tokens=28 avail_mem=132.15 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.97it/s]Capturing num tokens (num_tokens=24 avail_mem=132.14 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.97it/s]Capturing num tokens (num_tokens=20 avail_mem=132.13 GB):  86%|████████▌ | 50/58 [00:02<00:00, 33.97it/s]Capturing num tokens (num_tokens=20 avail_mem=132.13 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.76it/s]Capturing num tokens (num_tokens=16 avail_mem=132.12 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.76it/s]Capturing num tokens (num_tokens=12 avail_mem=132.11 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.76it/s]Capturing num tokens (num_tokens=8 avail_mem=132.10 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.76it/s] Capturing num tokens (num_tokens=4 avail_mem=132.10 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.76it/s]Capturing num tokens (num_tokens=4 avail_mem=132.10 GB): 100%|██████████| 58/58 [00:02<00:00, 35.83it/s]Capturing num tokens (num_tokens=4 avail_mem=132.10 GB): 100%|██████████| 58/58 [00:02<00:00, 23.40it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
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


<strong style='color: #00008B;'>{'id': '7774c9f6d48442d697a5310f67a60c14', 'object': 'chat.completion', 'created': 1774012023, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '94ee95e2b7294e6185901a99cd9d35fa', 'object': 'chat.completion', 'created': 1774012023, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='a81a02af92d641de90abc2f475ceb027', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1774012024, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=37, total_tokens=76, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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

    ,

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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '2317bd608faf43d79f734fd004ebbed1', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.24587354436516762, 'response_sent_to_client_ts': 1774012024.6828697}}</strong>


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
