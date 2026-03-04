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

    [2026-03-04 09:26:09] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 09:26:09] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 09:26:09] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-04 09:26:14] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 09:26:14] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 09:26:14] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-04 09:26:16] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.
    [2026-03-04 09:26:16] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 09:26:16] INFO utils.py:452: Successfully reserved port 35904 on host '0.0.0.0'


    [2026-03-04 09:26:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 09:26:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 09:26:21] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-04 09:26:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 09:26:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 09:26:21] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-04 09:26:26] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-04 09:26:26] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-04 09:26:26] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.60 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.60 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.42it/s]Capturing batches (bs=2 avail_mem=59.54 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.42it/s]

    Capturing batches (bs=1 avail_mem=59.54 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.42it/s]Capturing batches (bs=1 avail_mem=59.54 GB): 100%|██████████| 3/3 [00:00<00:00, 13.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:21,  2.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:21,  2.49s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.20it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.77it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.77it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.77it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.99it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.99it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:03, 12.36it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:03, 12.36it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:03, 12.36it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:03, 12.36it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:03, 12.36it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:02, 16.88it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:02, 16.88it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:02, 16.88it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:02, 16.88it/s]

    Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:02, 16.88it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 21.05it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 21.05it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 21.05it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 21.05it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 21.05it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 24.96it/s]

    Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 24.96it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 28.15it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]

    Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 32.95it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 36.48it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 39.77it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 39.77it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 39.77it/s]

    Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 39.77it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 39.77it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 39.77it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 39.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.53 GB):   3%|▎         | 2/58 [00:00<00:03, 17.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.52 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.51 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.51 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.63it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=56.47 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.47 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.47 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.47 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.99it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=58.13 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.13 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.23it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.46 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.46 GB):  24%|██▍       | 14/58 [00:00<00:02, 16.23it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=57.46 GB):  28%|██▊       | 16/58 [00:00<00:02, 15.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.64 GB):  28%|██▊       | 16/58 [00:00<00:02, 15.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.63 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.63 GB):  31%|███       | 18/58 [00:01<00:02, 13.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.63 GB):  31%|███       | 18/58 [00:01<00:02, 13.84it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.45 GB):  31%|███       | 18/58 [00:01<00:02, 13.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.45 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.43 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s]Capturing num tokens (num_tokens=960 avail_mem=56.68 GB):  34%|███▍      | 20/58 [00:01<00:02, 14.47it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=56.68 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.06it/s]Capturing num tokens (num_tokens=896 avail_mem=56.67 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.06it/s]Capturing num tokens (num_tokens=832 avail_mem=56.67 GB):  38%|███▊      | 22/58 [00:01<00:02, 14.06it/s]Capturing num tokens (num_tokens=832 avail_mem=56.67 GB):  41%|████▏     | 24/58 [00:01<00:02, 12.73it/s]Capturing num tokens (num_tokens=768 avail_mem=57.44 GB):  41%|████▏     | 24/58 [00:01<00:02, 12.73it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.73 GB):  41%|████▏     | 24/58 [00:01<00:02, 12.73it/s]Capturing num tokens (num_tokens=704 avail_mem=56.73 GB):  45%|████▍     | 26/58 [00:01<00:02, 12.13it/s]Capturing num tokens (num_tokens=640 avail_mem=56.72 GB):  45%|████▍     | 26/58 [00:01<00:02, 12.13it/s]Capturing num tokens (num_tokens=576 avail_mem=56.72 GB):  45%|████▍     | 26/58 [00:01<00:02, 12.13it/s]

    Capturing num tokens (num_tokens=576 avail_mem=56.72 GB):  48%|████▊     | 28/58 [00:01<00:02, 12.44it/s]Capturing num tokens (num_tokens=512 avail_mem=57.42 GB):  48%|████▊     | 28/58 [00:01<00:02, 12.44it/s]Capturing num tokens (num_tokens=480 avail_mem=56.51 GB):  48%|████▊     | 28/58 [00:01<00:02, 12.44it/s]Capturing num tokens (num_tokens=480 avail_mem=56.51 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.57it/s]Capturing num tokens (num_tokens=448 avail_mem=55.80 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.57it/s]

    Capturing num tokens (num_tokens=416 avail_mem=56.51 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.57it/s]Capturing num tokens (num_tokens=416 avail_mem=56.51 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.01it/s]Capturing num tokens (num_tokens=384 avail_mem=56.46 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.01it/s]Capturing num tokens (num_tokens=352 avail_mem=55.85 GB):  55%|█████▌    | 32/58 [00:02<00:01, 13.01it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.85 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.78it/s]Capturing num tokens (num_tokens=320 avail_mem=55.85 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.78it/s]Capturing num tokens (num_tokens=288 avail_mem=56.45 GB):  59%|█████▊    | 34/58 [00:02<00:01, 12.78it/s]Capturing num tokens (num_tokens=288 avail_mem=56.45 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.07it/s]Capturing num tokens (num_tokens=256 avail_mem=55.90 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.07it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.90 GB):  62%|██████▏   | 36/58 [00:02<00:01, 13.07it/s]Capturing num tokens (num_tokens=240 avail_mem=55.90 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.16it/s]Capturing num tokens (num_tokens=224 avail_mem=56.02 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.16it/s]Capturing num tokens (num_tokens=208 avail_mem=56.02 GB):  66%|██████▌   | 38/58 [00:02<00:01, 13.16it/s]Capturing num tokens (num_tokens=208 avail_mem=56.02 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.09it/s]Capturing num tokens (num_tokens=192 avail_mem=55.52 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.09it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.67 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.09it/s]Capturing num tokens (num_tokens=176 avail_mem=55.67 GB):  72%|███████▏  | 42/58 [00:02<00:01, 14.57it/s]Capturing num tokens (num_tokens=160 avail_mem=56.01 GB):  72%|███████▏  | 42/58 [00:02<00:01, 14.57it/s]Capturing num tokens (num_tokens=144 avail_mem=55.57 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.57it/s]Capturing num tokens (num_tokens=144 avail_mem=55.57 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.37it/s]Capturing num tokens (num_tokens=128 avail_mem=56.01 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.37it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.59 GB):  76%|███████▌  | 44/58 [00:03<00:00, 14.37it/s]Capturing num tokens (num_tokens=112 avail_mem=55.59 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.61it/s]Capturing num tokens (num_tokens=96 avail_mem=55.59 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.61it/s] Capturing num tokens (num_tokens=80 avail_mem=55.99 GB):  79%|███████▉  | 46/58 [00:03<00:00, 14.61it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.99 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.88it/s]Capturing num tokens (num_tokens=64 avail_mem=55.58 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.88it/s]Capturing num tokens (num_tokens=48 avail_mem=55.96 GB):  83%|████████▎ | 48/58 [00:03<00:00, 12.88it/s]Capturing num tokens (num_tokens=48 avail_mem=55.96 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.64it/s]Capturing num tokens (num_tokens=32 avail_mem=55.57 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.64it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.98 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.64it/s]Capturing num tokens (num_tokens=28 avail_mem=55.98 GB):  90%|████████▉ | 52/58 [00:03<00:00, 14.29it/s]Capturing num tokens (num_tokens=24 avail_mem=55.92 GB):  90%|████████▉ | 52/58 [00:03<00:00, 14.29it/s]Capturing num tokens (num_tokens=20 avail_mem=55.59 GB):  90%|████████▉ | 52/58 [00:03<00:00, 14.29it/s]Capturing num tokens (num_tokens=20 avail_mem=55.59 GB):  93%|█████████▎| 54/58 [00:03<00:00, 14.76it/s]Capturing num tokens (num_tokens=16 avail_mem=55.92 GB):  93%|█████████▎| 54/58 [00:03<00:00, 14.76it/s]

    Capturing num tokens (num_tokens=12 avail_mem=55.61 GB):  93%|█████████▎| 54/58 [00:03<00:00, 14.76it/s]Capturing num tokens (num_tokens=12 avail_mem=55.61 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.56it/s]Capturing num tokens (num_tokens=8 avail_mem=55.91 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.56it/s] Capturing num tokens (num_tokens=4 avail_mem=55.63 GB):  97%|█████████▋| 56/58 [00:03<00:00, 15.56it/s]Capturing num tokens (num_tokens=4 avail_mem=55.63 GB): 100%|██████████| 58/58 [00:04<00:00, 16.17it/s]Capturing num tokens (num_tokens=4 avail_mem=55.63 GB): 100%|██████████| 58/58 [00:04<00:00, 14.49it/s]



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


<strong style='color: #00008B;'>{'id': 'ee933c023e5e45cab77271d61f0ef453', 'object': 'chat.completion', 'created': 1772616402, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '0808a57238c74060bd88d6b1ec6a3bb9', 'object': 'chat.completion', 'created': 1772616402, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='3243236de6e04b0facf447aa555e6cca', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1772616403, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '274b1e9a585348dc9d497926989a7172', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.047644536942243576, 'response_sent_to_client_ts': 1772616403.300297}}</strong>


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
