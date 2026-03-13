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

    [2026-03-13 08:42:26] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-13 08:42:26] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-13 08:42:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-13 08:42:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-13 08:42:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-13 08:42:31] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-13 08:42:33] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.
    [2026-03-13 08:42:33] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-13 08:42:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-13 08:42:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-13 08:42:38] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-13 08:42:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-13 08:42:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-13 08:42:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-13 08:42:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-13 08:42:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-13 08:42:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.08it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:10,  2.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:10,  2.29s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:56,  1.02s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:56,  1.02s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:56,  1.02s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:12,  4.01it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:07,  6.89it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:04, 10.13it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:04, 10.13it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:04, 10.13it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:04, 10.13it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:02<00:03, 13.39it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:02<00:03, 13.39it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 13.39it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 13.39it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 13.39it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 18.34it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 18.34it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 18.34it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 18.34it/s] 

    Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 18.34it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 22.13it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 22.13it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 22.13it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 22.13it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 22.13it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]

    Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 26.11it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 30.99it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 34.59it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 37.33it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]

    Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 39.87it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:03<00:00, 43.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.62 GB):   2%|▏         | 1/58 [00:00<00:08,  7.08it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.59 GB):   2%|▏         | 1/58 [00:00<00:08,  7.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.58 GB):   2%|▏         | 1/58 [00:00<00:08,  7.08it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.58 GB):   2%|▏         | 1/58 [00:00<00:08,  7.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.66it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=57.58 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.54 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.54 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.54 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.54 GB):  16%|█▌        | 9/58 [00:00<00:05,  8.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.53 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.53 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.53 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.86it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.71 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.71 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.71 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.70 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.32it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=56.70 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.52 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.52 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.52 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.75 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.26it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=56.75 GB):  29%|██▉       | 17/58 [00:01<00:03, 11.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.75 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.52 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.78 GB):  33%|███▎      | 19/58 [00:01<00:03, 11.21it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=56.78 GB):  36%|███▌      | 21/58 [00:01<00:03, 11.32it/s]Capturing num tokens (num_tokens=960 avail_mem=56.80 GB):  36%|███▌      | 21/58 [00:01<00:03, 11.32it/s] Capturing num tokens (num_tokens=896 avail_mem=55.72 GB):  36%|███▌      | 21/58 [00:02<00:03, 11.32it/s]Capturing num tokens (num_tokens=896 avail_mem=55.72 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.58it/s]Capturing num tokens (num_tokens=832 avail_mem=43.17 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.58it/s]

    Capturing num tokens (num_tokens=768 avail_mem=42.51 GB):  40%|███▉      | 23/58 [00:02<00:03, 11.58it/s]Capturing num tokens (num_tokens=768 avail_mem=42.51 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=704 avail_mem=42.50 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.59it/s]Capturing num tokens (num_tokens=640 avail_mem=42.50 GB):  43%|████▎     | 25/58 [00:02<00:02, 11.59it/s]

    Capturing num tokens (num_tokens=640 avail_mem=42.50 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.74it/s]Capturing num tokens (num_tokens=576 avail_mem=43.16 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.74it/s]Capturing num tokens (num_tokens=512 avail_mem=42.55 GB):  47%|████▋     | 27/58 [00:02<00:02, 11.74it/s]Capturing num tokens (num_tokens=512 avail_mem=42.55 GB):  50%|█████     | 29/58 [00:02<00:02, 11.68it/s]Capturing num tokens (num_tokens=480 avail_mem=42.56 GB):  50%|█████     | 29/58 [00:02<00:02, 11.68it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.17 GB):  50%|█████     | 29/58 [00:02<00:02, 11.68it/s]Capturing num tokens (num_tokens=448 avail_mem=43.17 GB):  53%|█████▎    | 31/58 [00:02<00:02, 11.98it/s]Capturing num tokens (num_tokens=416 avail_mem=42.62 GB):  53%|█████▎    | 31/58 [00:02<00:02, 11.98it/s]Capturing num tokens (num_tokens=384 avail_mem=42.62 GB):  53%|█████▎    | 31/58 [00:02<00:02, 11.98it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.62 GB):  57%|█████▋    | 33/58 [00:02<00:02, 12.50it/s]Capturing num tokens (num_tokens=352 avail_mem=43.16 GB):  57%|█████▋    | 33/58 [00:02<00:02, 12.50it/s]Capturing num tokens (num_tokens=320 avail_mem=42.66 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.50it/s]Capturing num tokens (num_tokens=320 avail_mem=42.66 GB):  60%|██████    | 35/58 [00:03<00:01, 12.57it/s]Capturing num tokens (num_tokens=288 avail_mem=43.16 GB):  60%|██████    | 35/58 [00:03<00:01, 12.57it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.15 GB):  60%|██████    | 35/58 [00:03<00:01, 12.57it/s]Capturing num tokens (num_tokens=256 avail_mem=43.15 GB):  64%|██████▍   | 37/58 [00:03<00:01, 13.24it/s]Capturing num tokens (num_tokens=240 avail_mem=42.72 GB):  64%|██████▍   | 37/58 [00:03<00:01, 13.24it/s]Capturing num tokens (num_tokens=224 avail_mem=43.15 GB):  64%|██████▍   | 37/58 [00:03<00:01, 13.24it/s]Capturing num tokens (num_tokens=224 avail_mem=43.15 GB):  67%|██████▋   | 39/58 [00:03<00:01, 14.66it/s]Capturing num tokens (num_tokens=208 avail_mem=42.73 GB):  67%|██████▋   | 39/58 [00:03<00:01, 14.66it/s]

    Capturing num tokens (num_tokens=192 avail_mem=42.73 GB):  67%|██████▋   | 39/58 [00:03<00:01, 14.66it/s]Capturing num tokens (num_tokens=192 avail_mem=42.73 GB):  71%|███████   | 41/58 [00:03<00:01, 14.66it/s]Capturing num tokens (num_tokens=176 avail_mem=43.15 GB):  71%|███████   | 41/58 [00:03<00:01, 14.66it/s]Capturing num tokens (num_tokens=160 avail_mem=42.76 GB):  71%|███████   | 41/58 [00:03<00:01, 14.66it/s]Capturing num tokens (num_tokens=160 avail_mem=42.76 GB):  74%|███████▍  | 43/58 [00:03<00:01, 14.96it/s]Capturing num tokens (num_tokens=144 avail_mem=42.75 GB):  74%|███████▍  | 43/58 [00:03<00:01, 14.96it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.14 GB):  74%|███████▍  | 43/58 [00:03<00:01, 14.96it/s]Capturing num tokens (num_tokens=128 avail_mem=43.14 GB):  78%|███████▊  | 45/58 [00:03<00:00, 15.41it/s]Capturing num tokens (num_tokens=112 avail_mem=42.77 GB):  78%|███████▊  | 45/58 [00:03<00:00, 15.41it/s]Capturing num tokens (num_tokens=96 avail_mem=43.13 GB):  78%|███████▊  | 45/58 [00:03<00:00, 15.41it/s] Capturing num tokens (num_tokens=80 avail_mem=42.80 GB):  78%|███████▊  | 45/58 [00:03<00:00, 15.41it/s]

    Capturing num tokens (num_tokens=80 avail_mem=42.80 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.93it/s]Capturing num tokens (num_tokens=64 avail_mem=42.79 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.93it/s]Capturing num tokens (num_tokens=48 avail_mem=43.12 GB):  83%|████████▎ | 48/58 [00:03<00:00, 15.93it/s]Capturing num tokens (num_tokens=48 avail_mem=43.12 GB):  86%|████████▌ | 50/58 [00:04<00:00, 16.37it/s]Capturing num tokens (num_tokens=32 avail_mem=42.81 GB):  86%|████████▌ | 50/58 [00:04<00:00, 16.37it/s]Capturing num tokens (num_tokens=28 avail_mem=43.11 GB):  86%|████████▌ | 50/58 [00:04<00:00, 16.37it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.11 GB):  90%|████████▉ | 52/58 [00:04<00:00, 16.57it/s]Capturing num tokens (num_tokens=24 avail_mem=42.83 GB):  90%|████████▉ | 52/58 [00:04<00:00, 16.57it/s]Capturing num tokens (num_tokens=20 avail_mem=43.10 GB):  90%|████████▉ | 52/58 [00:04<00:00, 16.57it/s]Capturing num tokens (num_tokens=20 avail_mem=43.10 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.13it/s]Capturing num tokens (num_tokens=16 avail_mem=42.85 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.13it/s]Capturing num tokens (num_tokens=12 avail_mem=43.09 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.13it/s]

    Capturing num tokens (num_tokens=8 avail_mem=42.87 GB):  93%|█████████▎| 54/58 [00:04<00:00, 17.13it/s] Capturing num tokens (num_tokens=8 avail_mem=42.87 GB):  98%|█████████▊| 57/58 [00:04<00:00, 19.30it/s]Capturing num tokens (num_tokens=4 avail_mem=43.09 GB):  98%|█████████▊| 57/58 [00:04<00:00, 19.30it/s]Capturing num tokens (num_tokens=4 avail_mem=43.09 GB): 100%|██████████| 58/58 [00:04<00:00, 13.04it/s]


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


<strong style='color: #00008B;'>{'id': '2b36351ad7f14e3590ef72f954fb5106', 'object': 'chat.completion', 'created': 1773391378, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'f6d02f136a364a6a9b87acd7bd1cd6f0', 'object': 'chat.completion', 'created': 1773391378, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='4b680276e1a64f029af9a64b1cb6cc4e', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1773391379, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'f8b9cccf05da41ec80b1de7a3f82a99c', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.19656437262892723, 'response_sent_to_client_ts': 1773391379.8366113}}</strong>


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
