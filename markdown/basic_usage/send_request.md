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

    [2026-03-12 09:58:28] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-12 09:58:28] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-12 09:58:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-12 09:58:33] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-12 09:58:33] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-12 09:58:33] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-12 09:58:35] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.
    [2026-03-12 09:58:35] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-12 09:58:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-12 09:58:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-12 09:58:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-12 09:58:44] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-12 09:58:44] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-12 09:58:44] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-12 09:58:44] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-12 09:58:44] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-12 09:58:44] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.17it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.16it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:40,  1.35it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:13,  3.88it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:13,  3.88it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:13,  3.88it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:13,  3.88it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.07it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04, 10.68it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 15.04it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 15.04it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 15.04it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 15.04it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 15.04it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.87it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.87it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.87it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.87it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.87it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 10.70it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 10.70it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 10.70it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 10.70it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 10.70it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 13.51it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 13.51it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 13.51it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 13.51it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 13.51it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 20.59it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 24.93it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 24.93it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 24.93it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 24.93it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 24.93it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 27.66it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:04<00:00, 31.26it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:04<00:00, 31.26it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:04<00:00, 31.26it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:04<00:00, 31.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=65.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=65.41 GB):   2%|▏         | 1/58 [00:00<00:06,  8.47it/s]Capturing num tokens (num_tokens=7680 avail_mem=65.39 GB):   2%|▏         | 1/58 [00:00<00:06,  8.47it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=65.39 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=65.40 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=65.40 GB):   5%|▌         | 3/58 [00:00<00:06,  8.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=65.24 GB):   5%|▌         | 3/58 [00:00<00:06,  8.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=65.44 GB):   5%|▌         | 3/58 [00:00<00:06,  8.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=65.44 GB):   9%|▊         | 5/58 [00:00<00:05, 10.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=65.29 GB):   9%|▊         | 5/58 [00:00<00:05, 10.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=65.32 GB):   9%|▊         | 5/58 [00:00<00:05, 10.32it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=65.32 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=65.36 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=65.47 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=65.47 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=65.50 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=65.48 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.76it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=65.46 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=65.46 GB):  21%|██        | 12/58 [00:00<00:02, 16.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=65.46 GB):  21%|██        | 12/58 [00:00<00:02, 16.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=65.47 GB):  21%|██        | 12/58 [00:00<00:02, 16.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=65.47 GB):  21%|██        | 12/58 [00:00<00:02, 16.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=65.47 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=65.46 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=65.45 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=65.44 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=65.44 GB):  31%|███       | 18/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=65.43 GB):  31%|███       | 18/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=1280 avail_mem=65.41 GB):  31%|███       | 18/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=65.39 GB):  31%|███       | 18/58 [00:01<00:01, 20.10it/s]Capturing num tokens (num_tokens=1024 avail_mem=65.39 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.50it/s]Capturing num tokens (num_tokens=960 avail_mem=65.40 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.50it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=65.38 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.50it/s]Capturing num tokens (num_tokens=832 avail_mem=65.37 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.50it/s]Capturing num tokens (num_tokens=768 avail_mem=65.38 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.50it/s]Capturing num tokens (num_tokens=768 avail_mem=65.38 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.56it/s]Capturing num tokens (num_tokens=704 avail_mem=65.37 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.56it/s]Capturing num tokens (num_tokens=640 avail_mem=65.38 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.56it/s]Capturing num tokens (num_tokens=576 avail_mem=65.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.56it/s]Capturing num tokens (num_tokens=512 avail_mem=65.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.56it/s]Capturing num tokens (num_tokens=512 avail_mem=65.35 GB):  50%|█████     | 29/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=480 avail_mem=65.36 GB):  50%|█████     | 29/58 [00:01<00:01, 28.66it/s]

    Capturing num tokens (num_tokens=448 avail_mem=65.35 GB):  50%|█████     | 29/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=416 avail_mem=65.32 GB):  50%|█████     | 29/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=384 avail_mem=65.34 GB):  50%|█████     | 29/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=384 avail_mem=65.34 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=352 avail_mem=65.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=320 avail_mem=65.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=288 avail_mem=65.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=256 avail_mem=65.30 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.99it/s]Capturing num tokens (num_tokens=256 avail_mem=65.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.49it/s]Capturing num tokens (num_tokens=240 avail_mem=65.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.49it/s]

    Capturing num tokens (num_tokens=224 avail_mem=65.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.49it/s]Capturing num tokens (num_tokens=208 avail_mem=65.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.49it/s]Capturing num tokens (num_tokens=192 avail_mem=65.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.49it/s]Capturing num tokens (num_tokens=192 avail_mem=65.26 GB):  71%|███████   | 41/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=176 avail_mem=65.25 GB):  71%|███████   | 41/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=160 avail_mem=65.24 GB):  71%|███████   | 41/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=144 avail_mem=65.23 GB):  71%|███████   | 41/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=128 avail_mem=65.20 GB):  71%|███████   | 41/58 [00:01<00:00, 33.83it/s]Capturing num tokens (num_tokens=128 avail_mem=65.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=112 avail_mem=65.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.96it/s]

    Capturing num tokens (num_tokens=96 avail_mem=65.21 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.96it/s] Capturing num tokens (num_tokens=80 avail_mem=65.20 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.96it/s]Capturing num tokens (num_tokens=64 avail_mem=65.18 GB):  78%|███████▊  | 45/58 [00:02<00:00, 34.96it/s]Capturing num tokens (num_tokens=64 avail_mem=65.18 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.26it/s]Capturing num tokens (num_tokens=48 avail_mem=65.18 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.26it/s]Capturing num tokens (num_tokens=32 avail_mem=65.17 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.26it/s]Capturing num tokens (num_tokens=28 avail_mem=65.15 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.26it/s]Capturing num tokens (num_tokens=24 avail_mem=65.15 GB):  84%|████████▍ | 49/58 [00:02<00:00, 35.26it/s]

    Capturing num tokens (num_tokens=24 avail_mem=65.15 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s]Capturing num tokens (num_tokens=20 avail_mem=65.14 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s]Capturing num tokens (num_tokens=16 avail_mem=65.14 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s]Capturing num tokens (num_tokens=12 avail_mem=65.13 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s]Capturing num tokens (num_tokens=8 avail_mem=65.13 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s] Capturing num tokens (num_tokens=4 avail_mem=65.13 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.91it/s]Capturing num tokens (num_tokens=4 avail_mem=65.13 GB): 100%|██████████| 58/58 [00:02<00:00, 38.66it/s]Capturing num tokens (num_tokens=4 avail_mem=65.13 GB): 100%|██████████| 58/58 [00:02<00:00, 25.73it/s]


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


<strong style='color: #00008B;'>{'id': '933bb16eebcf4c06b552f00886f7c7ef', 'object': 'chat.completion', 'created': 1773309539, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': 'f14ec7ae945f443e8698518b83ef4bfd', 'object': 'chat.completion', 'created': 1773309539, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='4f2ed41640da444d8e03b77e0e4afae5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1773309540, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'cd05e234dc2b45689b39abca72ab5f6c', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.19241427397355437, 'response_sent_to_client_ts': 1773309540.832565}}</strong>


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
