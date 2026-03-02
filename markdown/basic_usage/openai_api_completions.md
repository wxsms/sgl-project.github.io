# OpenAI APIs - Completions

SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

This tutorial covers the following popular APIs:

- `chat/completions`
- `completions`

Check out other tutorials to learn about [vision APIs](openai_api_vision.ipynb) for vision-language models and [embedding APIs](openai_api_embeddings.ipynb) for embedding models.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    [2026-03-02 21:04:59] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-02 21:04:59] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-02 21:04:59] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-02 21:05:03] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-02 21:05:03] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-02 21:05:03] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-02 21:05:05] INFO server_args.py:1970: Attention backend not specified. Use fa3 backend by default.
    [2026-03-02 21:05:05] INFO server_args.py:3042: Set soft_watchdog_timeout since in CI


    [2026-03-02 21:05:10] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-02 21:05:10] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-02 21:05:10] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-02 21:05:10] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-02 21:05:10] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-02 21:05:10] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-02 21:05:15] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-02 21:05:15] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-02 21:05:15] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.07it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=76.42 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=76.42 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.21it/s]Capturing batches (bs=2 avail_mem=76.37 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.21it/s]

    Capturing batches (bs=1 avail_mem=76.37 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.21it/s]Capturing batches (bs=1 avail_mem=76.37 GB): 100%|██████████| 3/3 [00:00<00:00, 12.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:13,  2.34s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]Compiling num tokens (num_tokens=704):  29%|██▉       | 17/58 [00:02<00:03, 12.98it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:02<00:01, 22.26it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:02<00:00, 32.13it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:02<00:00, 43.29it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:02<00:00, 43.29it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:02<00:00, 43.29it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:02<00:00, 43.29it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:02<00:00, 43.29it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=12):  78%|███████▊  | 45/58 [00:03<00:00, 43.29it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 55.70it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 55.70it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 55.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 18.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.69 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.69 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.68 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.68 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.68 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.68 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.67 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.67 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.67 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.66 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.66 GB):  21%|██        | 12/58 [00:00<00:01, 30.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]Capturing num tokens (num_tokens=960 avail_mem=71.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s] Capturing num tokens (num_tokens=896 avail_mem=71.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 36.96it/s]

    Capturing num tokens (num_tokens=896 avail_mem=71.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=832 avail_mem=71.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=768 avail_mem=71.64 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=704 avail_mem=71.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=640 avail_mem=71.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=576 avail_mem=71.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=512 avail_mem=71.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 42.54it/s]Capturing num tokens (num_tokens=512 avail_mem=71.62 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]Capturing num tokens (num_tokens=480 avail_mem=71.63 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]Capturing num tokens (num_tokens=448 avail_mem=71.63 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]Capturing num tokens (num_tokens=416 avail_mem=71.63 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]Capturing num tokens (num_tokens=384 avail_mem=71.63 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.62 GB):  50%|█████     | 29/58 [00:00<00:00, 45.50it/s]Capturing num tokens (num_tokens=352 avail_mem=71.62 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=320 avail_mem=71.62 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=288 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=256 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=240 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=224 avail_mem=71.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=208 avail_mem=71.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=208 avail_mem=71.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=192 avail_mem=71.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=176 avail_mem=71.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=160 avail_mem=71.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=144 avail_mem=71.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 46.85it/s]Capturing num tokens (num_tokens=112 avail_mem=71.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=96 avail_mem=71.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s] Capturing num tokens (num_tokens=80 avail_mem=71.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=64 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=48 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=32 avail_mem=71.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=32 avail_mem=71.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=28 avail_mem=71.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=24 avail_mem=71.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=16 avail_mem=71.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=12 avail_mem=71.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=12 avail_mem=71.55 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=8 avail_mem=71.55 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s] Capturing num tokens (num_tokens=4 avail_mem=71.55 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=4 avail_mem=71.55 GB): 100%|██████████| 58/58 [00:01<00:00, 42.02it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33029


## Chat Completions

### Usage

The server fully implements the OpenAI API.
It will automatically apply the chat template specified in the Hugging Face tokenizer, if one is available.
You can also specify a custom chat template with `--chat-template` when launching the server.


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

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: ChatCompletion(id='beb8a7b91bc046d2b72a64ffe332c792', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington, D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1772485528, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=39, prompt_tokens=37, total_tokens=76, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


### Model Thinking/Reasoning Support

Some models support internal reasoning or thinking processes that can be exposed in the API response. SGLang provides unified support for various reasoning models through the `chat_template_kwargs` parameter and compatible reasoning parsers.

#### Supported Models and Configuration

| Model Family | Chat Template Parameter | Reasoning Parser | Notes |
|--------------|------------------------|------------------|--------|
| DeepSeek-R1 (R1, R1-0528, R1-Distill) | `enable_thinking` | `--reasoning-parser deepseek-r1` | Standard reasoning models |
| DeepSeek-V3.1 | `thinking` | `--reasoning-parser deepseek-v3` | Hybrid model (thinking/non-thinking modes) |
| Qwen3 (standard) | `enable_thinking` | `--reasoning-parser qwen3` | Hybrid model (thinking/non-thinking modes) |
| Qwen3-Thinking | N/A (always enabled) | `--reasoning-parser qwen3-thinking` | Always generates reasoning |
| Kimi | N/A (always enabled) | `--reasoning-parser kimi` | Kimi thinking models |
| Gpt-Oss | N/A (always enabled) | `--reasoning-parser gpt-oss` | Gpt-Oss thinking models |

#### Basic Usage

To enable reasoning output, you need to:
1. Launch the server with the appropriate reasoning parser
2. Set the model-specific parameter in `chat_template_kwargs`
3. Optionally use `separate_reasoning: False` to not get reasoning content separately (default to `True`)

**Note for Qwen3-Thinking models:** These models always generate thinking content and do not support the `enable_thinking` parameter. Use `--reasoning-parser qwen3-thinking` or `--reasoning-parser qwen3` to parse the thinking content.


#### Example: Qwen3 Models

```python
# Launch server:
# python3 -m sglang.launch_server --model Qwen/Qwen3-4B --reasoning-parser qwen3

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:30000/v1",
)

model = "Qwen/Qwen3-4B"
messages = [{"role": "user", "content": "How many r's are in 'strawberry'?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True
    }
)

print("Reasoning:", response.choices[0].message.reasoning_content)
print("-"*100)
print("Answer:", response.choices[0].message.content)
```

**ExampleOutput:**
```
Reasoning: Okay, so the user is asking how many 'r's are in the word 'strawberry'. Let me think. First, I need to make sure I have the word spelled correctly. Strawberry... S-T-R-A-W-B-E-R-R-Y. Wait, is that right? Let me break it down.

Starting with 'strawberry', let's write out the letters one by one. S, T, R, A, W, B, E, R, R, Y. Hmm, wait, that's 10 letters. Let me check again. S (1), T (2), R (3), A (4), W (5), B (6), E (7), R (8), R (9), Y (10). So the letters are S-T-R-A-W-B-E-R-R-Y. 
...
Therefore, the answer should be three R's in 'strawberry'. But I need to make sure I'm not counting any other letters as R. Let me check again. S, T, R, A, W, B, E, R, R, Y. No other R's. So three in total. Yeah, that seems right.

----------------------------------------------------------------------------------------------------
Answer: The word "strawberry" contains **three** letters 'r'. Here's the breakdown:

1. **S-T-R-A-W-B-E-R-R-Y**  
   - The **third letter** is 'R'.  
   - The **eighth and ninth letters** are also 'R's.  

Thus, the total count is **3**.  

**Answer:** 3.
```

**Note:** Setting `"enable_thinking": False` (or omitting it) will result in `reasoning_content` being `None`. Qwen3-Thinking models always generate reasoning content and don't support the `enable_thinking` parameter.


#### Logit Bias Support

SGLang supports the `logit_bias` parameter for both chat completions and completions APIs. This parameter allows you to modify the likelihood of specific tokens being generated by adding bias values to their logits. The bias values can range from -100 to 100, where:

- **Positive values** (0 to 100) increase the likelihood of the token being selected
- **Negative values** (-100 to 0) decrease the likelihood of the token being selected
- **-100** effectively prevents the token from being generated

The `logit_bias` parameter accepts a dictionary where keys are token IDs (as strings) and values are the bias amounts (as floats).


#### Getting Token IDs

To use `logit_bias` effectively, you need to know the token IDs for the words you want to bias. Here's how to get token IDs:

```python
# Get tokenizer to find token IDs
import tiktoken

# For OpenAI models, use the appropriate encoding
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or your model

# Get token IDs for specific words
word = "sunny"
token_ids = tokenizer.encode(word)
print(f"Token IDs for '{word}': {token_ids}")

# For SGLang models, you can access the tokenizer through the client
# and get token IDs for bias
```

**Important:** The `logit_bias` parameter uses token IDs as string keys, not the actual words.


#### Example: DeepSeek-V3 Models

DeepSeek-V3 models support thinking mode through the `thinking` parameter:

```python
# Launch server:
# python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.1 --tp 8  --reasoning-parser deepseek-v3

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:30000/v1",
)

model = "deepseek-ai/DeepSeek-V3.1"
messages = [{"role": "user", "content": "How many r's are in 'strawberry'?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"thinking": True},
        "separate_reasoning": True
    }
)

print("Reasoning:", response.choices[0].message.reasoning_content)
print("-"*100)
print("Answer:", response.choices[0].message.content)
```

**Example Output:**
```
Reasoning: First, the question is: "How many r's are in 'strawberry'?"

I need to count the number of times the letter 'r' appears in the word "strawberry".

Let me write out the word: S-T-R-A-W-B-E-R-R-Y.

Now, I'll go through each letter and count the 'r's.
...
So, I have three 'r's in "strawberry".

I should double-check. The word is spelled S-T-R-A-W-B-E-R-R-Y. The letters are at positions: 3, 8, and 9 are 'r's. Yes, that's correct.

Therefore, the answer should be 3.
----------------------------------------------------------------------------------------------------
Answer: The word "strawberry" contains **3** instances of the letter "r". Here's a breakdown for clarity:

- The word is spelled: S-T-R-A-W-B-E-R-R-Y
- The "r" appears at the 3rd, 8th, and 9th positions.
```

**Note:** DeepSeek-V3 models use the `thinking` parameter (not `enable_thinking`) to control reasoning output.



```python
# Example with logit_bias parameter
# Note: You need to get the actual token IDs from your tokenizer
# For demonstration, we'll use some example token IDs
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "Complete this sentence: The weather today is"}
    ],
    temperature=0.7,
    max_tokens=20,
    logit_bias={
        "12345": 50,  # Increase likelihood of token ID 12345
        "67890": -50,  # Decrease likelihood of token ID 67890
        "11111": 25,  # Slightly increase likelihood of token ID 11111
    },
)

print_highlight(f"Response with logit bias: {response.choices[0].message.content}")
```


<strong style='color: #00008B;'>Response with logit bias:  privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy</strong>


### Parameters

The chat completions API accepts OpenAI Chat Completions API's parameters. Refer to [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) for more details.

SGLang extends the standard API with the `extra_body` parameter, allowing for additional customization. One key option within `extra_body` is `chat_template_kwargs`, which can be used to pass arguments to the chat template processor.


```python
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a knowledgeable historian who provides concise responses.",
        },
        {"role": "user", "content": "Tell me about ancient Rome"},
        {
            "role": "assistant",
            "content": "Ancient Rome was a civilization centered in Italy.",
        },
        {"role": "user", "content": "What were their major achievements?"},
    ],
    temperature=0.3,  # Lower temperature for more focused responses
    max_tokens=128,  # Reasonable length for a concise response
    top_p=0.95,  # Slightly higher for better fluency
    presence_penalty=0.2,  # Mild penalty to avoid repetition
    frequency_penalty=0.2,  # Mild penalty for more natural language
    n=1,  # Single response is usually more stable
    seed=42,  # Keep for reproducibility
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Ancient Rome was a major civilization that had a profound impact on world history. Some of their major achievements include:<br><br>1. The construction of the Colosseum, one of the largest amphitheaters in the world, which served as a public entertainment venue.<br><br>2. The development of the Roman Republic, which lasted from 509 to 476 BCE, and the Roman Empire, which lasted from 27 BCE to 476 CE.<br><br>3. The establishment of the Roman Empire, which stretched from Italy to Spain and included parts of modern-day France, Germany, and Italy.<br><br>4. The construction of</strong>


Streaming mode is also supported.

#### Logit Bias Support

The completions API also supports the `logit_bias` parameter with the same functionality as described in the chat completions section above.



```python
stream = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

    Yes, I am a test of Qwen. I am created by Alibaba Cloud and am designed to provide helpful responses to users. Please let me know if you have any questions or if there's anything I can assist you with.

#### Returning Routed Experts (MoE Models)

For MoE models, set `return_routed_experts: true` in `extra_body` to return expert routing data. Requires `--enable-return-routed-experts` server flag. The `routed_experts` field will be returned in the `sgl_ext` object on each choice, containing base64-encoded int32 expert IDs as a flattened array with logical shape `[num_tokens, num_layers, top_k]`.


```python
# Example with logit_bias parameter for completions API
# Note: You need to get the actual token IDs from your tokenizer
# For demonstration, we'll use some example token IDs
response = client.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    prompt="The best programming language for AI is",
    temperature=0.7,
    max_tokens=20,
    logit_bias={
        "12345": 75,  # Strongly favor token ID 12345
        "67890": -100,  # Completely avoid token ID 67890
        "11111": -25,  # Slightly discourage token ID 11111
    },
)

print_highlight(f"Response with logit bias: {response.choices[0].text}")
```


<strong style='color: #00008B;'>Response with logit bias:  privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy privacy</strong>


## Completions

### Usage
Completions API is similar to Chat Completions API, but without the `messages` parameter or chat templates.


```python
response = client.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    prompt="List 3 countries and their capitals.",
    temperature=0,
    max_tokens=64,
    n=1,
    stop=None,
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: Completion(id='5e5f063daaa34309a7b2eab1dbb0edc6', choices=[CompletionChoice(finish_reason='length', index=0, logprobs=None, text=' 1. United States - Washington D.C.\n2. Canada - Ottawa\n3. France - Paris\n4. Germany - Berlin\n5. Japan - Tokyo\n6. Italy - Rome\n7. Spain - Madrid\n8. United Kingdom - London\n9. Australia - Canberra\n10. New Zealand', matched_stop=None)], created=1772485529, model='qwen/qwen2.5-0.5b-instruct', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=64, prompt_tokens=8, total_tokens=72, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


### Parameters

The completions API accepts OpenAI Completions API's parameters.  Refer to [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create) for more details.

Here is an example of a detailed completions request:


```python
response = client.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    prompt="Write a short story about a space explorer.",
    temperature=0.7,  # Moderate temperature for creative writing
    max_tokens=150,  # Longer response for a story
    top_p=0.9,  # Balanced diversity in word choice
    stop=["\n\n", "THE END"],  # Multiple stop sequences
    presence_penalty=0.3,  # Encourage novel elements
    frequency_penalty=0.3,  # Reduce repetitive phrases
    n=1,  # Generate one completion
    seed=123,  # For reproducible results
)

print_highlight(f"Response: {response}")
```


<strong style='color: #00008B;'>Response: Completion(id='193a4c89366d4776816479c93ccae8b6', choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text=' Once upon a time, in the year 2050, an astronaut named Jack found himself in the vastness of space. He was on a mission to explore new planets and discover new species of plants and animals. As he journeyed through the cosmos, Jack encountered a variety of alien creatures and landscapes.', matched_stop='\n\n')], created=1772485529, model='qwen/qwen2.5-0.5b-instruct', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=63, prompt_tokens=9, total_tokens=72, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


#### Returning Routed Experts (MoE Models)

For MoE models, set `return_routed_experts: true` in `extra_body` to return expert routing data. Requires `--enable-return-routed-experts` server flag. The `routed_experts` field will be returned in the `sgl_ext` object on each choice, containing base64-encoded int32 expert IDs as a flattened array with logical shape `[num_tokens, num_layers, top_k]`.

## Structured Outputs (JSON, Regex, EBNF)

For OpenAI compatible structured outputs API, refer to [Structured Outputs](../advanced_features/structured_outputs.ipynb) for more details.


## Using LoRA Adapters

SGLang supports LoRA (Low-Rank Adaptation) adapters with OpenAI-compatible APIs. You can specify which adapter to use directly in the `model` parameter using the `base-model:adapter-name` syntax.

**Server Setup:**
```bash
python -m sglang.launch_server \
    --model-path qwen/qwen2.5-0.5b-instruct \
    --enable-lora \
    --lora-paths adapter_a=/path/to/adapter_a adapter_b=/path/to/adapter_b
```

For more details on LoRA serving configuration, see the [LoRA documentation](../advanced_features/lora.ipynb).

**API Call:**

(Recommended) Use the `model:adapter` syntax to specify which adapter to use:
```python
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct:adapter_a",  # ← base-model:adapter-name
    messages=[{"role": "user", "content": "Convert to SQL: show all users"}],
    max_tokens=50,
)
```

**Backward Compatible: Using `extra_body`**

The old `extra_body` method is still supported for backward compatibility:
```python
# Backward compatible method
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Convert to SQL: show all users"}],
    extra_body={"lora_path": "adapter_a"},  # ← old method
    max_tokens=50,
)
```
**Note:** When both `model:adapter` and `extra_body["lora_path"]` are specified, the `model:adapter` syntax takes precedence.


```python
terminate_process(server_process)
```
