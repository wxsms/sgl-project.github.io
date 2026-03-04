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

    [2026-03-04 15:15:41] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 15:15:41] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 15:15:41] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-04 15:15:46] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 15:15:46] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 15:15:46] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-04 15:15:48] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.
    [2026-03-04 15:15:48] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 15:15:49] INFO utils.py:452: Successfully reserved port 36099 on host '0.0.0.0'


    [2026-03-04 15:15:54] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 15:15:54] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 15:15:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-04 15:15:54] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 15:15:54] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 15:15:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-04 15:15:59] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-04 15:15:59] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-04 15:15:59] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.08it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=133.90 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=133.90 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.04it/s]Capturing batches (bs=2 avail_mem=133.84 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.04it/s]Capturing batches (bs=1 avail_mem=133.83 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.04it/s]Capturing batches (bs=1 avail_mem=133.83 GB): 100%|██████████| 3/3 [00:00<00:00, 10.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.24it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.24it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:03<00:07,  5.92it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 12.86it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 19.90it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:03<00:00, 27.12it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:03<00:00, 35.16it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 42.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=133.49 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.46 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.46 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=133.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=133.45 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=133.44 GB):   3%|▎         | 2/58 [00:00<00:03, 18.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=133.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=133.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=133.44 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=133.43 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=133.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=133.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=133.43 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=133.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=133.42 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=133.42 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=3072 avail_mem=133.41 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=133.41 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=133.40 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=2304 avail_mem=133.40 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=133.39 GB):  21%|██        | 12/58 [00:00<00:01, 29.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=133.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=133.39 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=133.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=133.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=133.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s]

    Capturing num tokens (num_tokens=960 avail_mem=133.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.54it/s] Capturing num tokens (num_tokens=960 avail_mem=133.37 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=896 avail_mem=133.37 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=832 avail_mem=133.36 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=768 avail_mem=133.36 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=704 avail_mem=133.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=640 avail_mem=133.35 GB):  38%|███▊      | 22/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=640 avail_mem=133.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]Capturing num tokens (num_tokens=576 avail_mem=133.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]Capturing num tokens (num_tokens=512 avail_mem=133.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]Capturing num tokens (num_tokens=480 avail_mem=133.35 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]

    Capturing num tokens (num_tokens=448 avail_mem=133.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]Capturing num tokens (num_tokens=416 avail_mem=133.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 37.56it/s]Capturing num tokens (num_tokens=416 avail_mem=133.34 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=384 avail_mem=133.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=352 avail_mem=133.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=320 avail_mem=133.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 38.45it/s]Capturing num tokens (num_tokens=288 avail_mem=133.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 38.45it/s]Capturing num tokens (num_tokens=288 avail_mem=133.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=256 avail_mem=133.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=240 avail_mem=133.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=224 avail_mem=133.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]

    Capturing num tokens (num_tokens=208 avail_mem=133.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=192 avail_mem=133.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=192 avail_mem=133.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=176 avail_mem=133.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=160 avail_mem=133.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=144 avail_mem=133.28 GB):  71%|███████   | 41/58 [00:01<00:00, 38.56it/s]

    Capturing num tokens (num_tokens=128 avail_mem=133.28 GB):  71%|███████   | 41/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=128 avail_mem=133.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.96it/s]Capturing num tokens (num_tokens=112 avail_mem=133.27 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.96it/s]

    Capturing num tokens (num_tokens=96 avail_mem=137.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.96it/s] Capturing num tokens (num_tokens=80 avail_mem=137.14 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.96it/s]Capturing num tokens (num_tokens=64 avail_mem=137.13 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.96it/s]Capturing num tokens (num_tokens=64 avail_mem=137.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 20.15it/s]Capturing num tokens (num_tokens=48 avail_mem=137.13 GB):  84%|████████▍ | 49/58 [00:01<00:00, 20.15it/s]Capturing num tokens (num_tokens=32 avail_mem=137.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 20.15it/s]Capturing num tokens (num_tokens=28 avail_mem=137.12 GB):  84%|████████▍ | 49/58 [00:01<00:00, 20.15it/s]Capturing num tokens (num_tokens=24 avail_mem=137.11 GB):  84%|████████▍ | 49/58 [00:01<00:00, 20.15it/s]Capturing num tokens (num_tokens=24 avail_mem=137.11 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=20 avail_mem=137.11 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s]

    Capturing num tokens (num_tokens=16 avail_mem=137.10 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=12 avail_mem=137.10 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=8 avail_mem=137.09 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s] Capturing num tokens (num_tokens=4 avail_mem=137.09 GB):  91%|█████████▏| 53/58 [00:01<00:00, 23.48it/s]Capturing num tokens (num_tokens=4 avail_mem=137.09 GB): 100%|██████████| 58/58 [00:01<00:00, 27.49it/s]Capturing num tokens (num_tokens=4 avail_mem=137.09 GB): 100%|██████████| 58/58 [00:01<00:00, 29.40it/s]


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


<strong style='color: #00008B;'>{'id': 'c1de412fe2bc44258ddec0efcda3e573', 'object': 'chat.completion', 'created': 1772637373, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '79019ee99a8248aca7fa49bdd29fadde', 'object': 'chat.completion', 'created': 1772637373, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='76cc6c6055054397a4d9c13589f38b9e', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1772637374, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=37, total_tokens=76, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'be6daa5d895e403eb6334fb3a116b933', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.04696060693822801, 'response_sent_to_client_ts': 1772637374.3138049}}</strong>


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
