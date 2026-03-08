# SGLang Native APIs

Apart from the OpenAI compatible APIs, the SGLang Runtime also provides its native server APIs. We introduce the following APIs:

- `/generate` (text generation model)
- `/get_model_info`
- `/get_server_info`
- `/health`
- `/health_generate`
- `/flush_cache`
- `/update_weights`
- `/encode`(embedding model)
- `/v1/rerank`(cross encoder rerank model)
- `/v1/score`(decoder-only scoring)
- `/classify`(reward model)
- `/start_expert_distribution_record`
- `/stop_expert_distribution_record`
- `/dump_expert_distribution_record`
- `/tokenize`
- `/detokenize`
- A full list of these APIs can be found at [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)

We mainly use `requests` to test these APIs in the following examples. You can also use `curl`.


## Launch A Server


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    [2026-03-08 15:25:16] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-08 15:25:16] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-08 15:25:16] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-08 15:25:20] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:25:20] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:25:20] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:25:22] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:25:22] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:25:26] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:25:26] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:25:26] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:25:26] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:25:26] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:25:26] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:25:31] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:25:31] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:25:31] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.76it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.76it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=8.94 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=8.94 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.42it/s]Capturing batches (bs=2 avail_mem=8.88 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.42it/s]Capturing batches (bs=1 avail_mem=8.88 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.42it/s]Capturing batches (bs=1 avail_mem=8.88 GB): 100%|██████████| 3/3 [00:00<00:00, 10.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:03,  2.16s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:03,  2.16s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:55,  1.00it/s]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:55,  1.00it/s]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:32,  1.70it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:32,  1.70it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:32,  1.70it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:16,  3.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:16,  3.28it/s]

    Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:16,  3.28it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:10,  4.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:10,  4.91it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:10,  4.91it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:07,  6.73it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:07,  6.73it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:07,  6.73it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.78it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.78it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.78it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.78it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:05,  8.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 13.89it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 13.89it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 13.89it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 13.89it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 13.89it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 18.86it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 24.68it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 24.68it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 28.22it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 32.56it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 35.59it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s]

    Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 38.98it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 40.80it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s]

    Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 42.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=8.58 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=8.58 GB):   2%|▏         | 1/58 [00:00<00:08,  6.81it/s]Capturing num tokens (num_tokens=7680 avail_mem=8.55 GB):   2%|▏         | 1/58 [00:00<00:08,  6.81it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=8.55 GB):   3%|▎         | 2/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=8.54 GB):   3%|▎         | 2/58 [00:00<00:07,  7.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=8.54 GB):   5%|▌         | 3/58 [00:00<00:07,  7.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=8.54 GB):   5%|▌         | 3/58 [00:00<00:07,  7.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=8.54 GB):   7%|▋         | 4/58 [00:00<00:06,  7.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=8.54 GB):   7%|▋         | 4/58 [00:00<00:06,  7.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=8.54 GB):   9%|▊         | 5/58 [00:00<00:06,  8.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=8.54 GB):   9%|▊         | 5/58 [00:00<00:06,  8.30it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=8.54 GB):  10%|█         | 6/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=8.54 GB):  10%|█         | 6/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=8.54 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=8.53 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.79it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=8.53 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=8.53 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=8.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.27it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=8.22 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=7.82 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=7.41 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=7.41 GB):  21%|██        | 12/58 [00:01<00:04,  9.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=7.01 GB):  21%|██        | 12/58 [00:01<00:04,  9.77it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=6.63 GB):  21%|██        | 12/58 [00:01<00:04,  9.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=6.63 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=6.22 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=5.86 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.10it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=5.86 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=5.84 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=5.83 GB):  28%|██▊       | 16/58 [00:01<00:03, 10.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=5.83 GB):  31%|███       | 18/58 [00:01<00:03, 10.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=5.72 GB):  31%|███       | 18/58 [00:01<00:03, 10.77it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=3.66 GB):  31%|███       | 18/58 [00:02<00:03, 10.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=3.66 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=3.64 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.06it/s]Capturing num tokens (num_tokens=960 avail_mem=3.65 GB):  34%|███▍      | 20/58 [00:02<00:03, 11.06it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=3.65 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.51it/s]Capturing num tokens (num_tokens=896 avail_mem=3.65 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.51it/s]Capturing num tokens (num_tokens=832 avail_mem=3.64 GB):  38%|███▊      | 22/58 [00:02<00:03, 11.51it/s]Capturing num tokens (num_tokens=832 avail_mem=3.64 GB):  41%|████▏     | 24/58 [00:02<00:03, 11.30it/s]Capturing num tokens (num_tokens=768 avail_mem=1.08 GB):  41%|████▏     | 24/58 [00:02<00:03, 11.30it/s]

    Capturing num tokens (num_tokens=704 avail_mem=1.08 GB):  41%|████▏     | 24/58 [00:02<00:03, 11.30it/s]Capturing num tokens (num_tokens=704 avail_mem=1.08 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.74it/s]Capturing num tokens (num_tokens=640 avail_mem=1.08 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.74it/s]Capturing num tokens (num_tokens=576 avail_mem=1.07 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.74it/s]

    Capturing num tokens (num_tokens=576 avail_mem=1.07 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.12it/s]Capturing num tokens (num_tokens=512 avail_mem=1.06 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.12it/s]Capturing num tokens (num_tokens=480 avail_mem=1.08 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.12it/s]Capturing num tokens (num_tokens=480 avail_mem=1.08 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.31it/s]Capturing num tokens (num_tokens=448 avail_mem=1.08 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.31it/s]

    Capturing num tokens (num_tokens=416 avail_mem=1.07 GB):  52%|█████▏    | 30/58 [00:02<00:02, 12.31it/s]Capturing num tokens (num_tokens=416 avail_mem=1.07 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.53it/s]Capturing num tokens (num_tokens=384 avail_mem=1.07 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.53it/s]Capturing num tokens (num_tokens=352 avail_mem=1.07 GB):  55%|█████▌    | 32/58 [00:03<00:02, 12.53it/s]

    Capturing num tokens (num_tokens=352 avail_mem=1.07 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.66it/s]Capturing num tokens (num_tokens=320 avail_mem=1.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.66it/s]Capturing num tokens (num_tokens=288 avail_mem=1.06 GB):  59%|█████▊    | 34/58 [00:03<00:01, 12.66it/s]Capturing num tokens (num_tokens=288 avail_mem=1.06 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.80it/s]Capturing num tokens (num_tokens=256 avail_mem=1.06 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.80it/s]

    Capturing num tokens (num_tokens=240 avail_mem=1.06 GB):  62%|██████▏   | 36/58 [00:03<00:01, 12.80it/s]Capturing num tokens (num_tokens=240 avail_mem=1.06 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.88it/s]Capturing num tokens (num_tokens=224 avail_mem=1.05 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.88it/s]Capturing num tokens (num_tokens=208 avail_mem=1.05 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.88it/s]

    Capturing num tokens (num_tokens=208 avail_mem=1.05 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.97it/s]Capturing num tokens (num_tokens=192 avail_mem=1.05 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.97it/s]Capturing num tokens (num_tokens=176 avail_mem=1.04 GB):  69%|██████▉   | 40/58 [00:03<00:01, 12.97it/s]Capturing num tokens (num_tokens=176 avail_mem=1.04 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.04it/s]Capturing num tokens (num_tokens=160 avail_mem=1.04 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.04it/s]

    Capturing num tokens (num_tokens=144 avail_mem=1.04 GB):  72%|███████▏  | 42/58 [00:03<00:01, 13.04it/s]Capturing num tokens (num_tokens=144 avail_mem=1.04 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.02it/s]Capturing num tokens (num_tokens=128 avail_mem=1.03 GB):  76%|███████▌  | 44/58 [00:03<00:01, 13.02it/s]Capturing num tokens (num_tokens=112 avail_mem=1.03 GB):  76%|███████▌  | 44/58 [00:04<00:01, 13.02it/s]

    Capturing num tokens (num_tokens=112 avail_mem=1.03 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.02it/s]Capturing num tokens (num_tokens=96 avail_mem=1.02 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.02it/s] Capturing num tokens (num_tokens=80 avail_mem=1.02 GB):  79%|███████▉  | 46/58 [00:04<00:00, 13.02it/s]Capturing num tokens (num_tokens=80 avail_mem=1.02 GB):  83%|████████▎ | 48/58 [00:04<00:00, 13.07it/s]Capturing num tokens (num_tokens=64 avail_mem=1.02 GB):  83%|████████▎ | 48/58 [00:04<00:00, 13.07it/s]

    Capturing num tokens (num_tokens=48 avail_mem=1.02 GB):  83%|████████▎ | 48/58 [00:04<00:00, 13.07it/s]Capturing num tokens (num_tokens=48 avail_mem=1.02 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.11it/s]Capturing num tokens (num_tokens=32 avail_mem=1.01 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.11it/s]Capturing num tokens (num_tokens=28 avail_mem=1.01 GB):  86%|████████▌ | 50/58 [00:04<00:00, 13.11it/s]

    Capturing num tokens (num_tokens=28 avail_mem=1.01 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.15it/s]Capturing num tokens (num_tokens=24 avail_mem=1.01 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.15it/s]Capturing num tokens (num_tokens=20 avail_mem=1.00 GB):  90%|████████▉ | 52/58 [00:04<00:00, 13.15it/s]Capturing num tokens (num_tokens=20 avail_mem=1.00 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.14it/s]Capturing num tokens (num_tokens=16 avail_mem=1.00 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.14it/s]

    Capturing num tokens (num_tokens=12 avail_mem=1.00 GB):  93%|█████████▎| 54/58 [00:04<00:00, 13.14it/s]Capturing num tokens (num_tokens=12 avail_mem=1.00 GB):  97%|█████████▋| 56/58 [00:04<00:00, 13.11it/s]Capturing num tokens (num_tokens=8 avail_mem=0.99 GB):  97%|█████████▋| 56/58 [00:04<00:00, 13.11it/s] Capturing num tokens (num_tokens=4 avail_mem=0.99 GB):  97%|█████████▋| 56/58 [00:04<00:00, 13.11it/s]

    Capturing num tokens (num_tokens=4 avail_mem=0.99 GB): 100%|██████████| 58/58 [00:05<00:00, 13.14it/s]Capturing num tokens (num_tokens=4 avail_mem=0.99 GB): 100%|██████████| 58/58 [00:05<00:00, 11.59it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Generate (text generation model)
Generate completions. This is similar to the `/v1/completions` in OpenAI API. Detailed parameters can be found in the [sampling parameters](sampling_params.md).


```python
import requests

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': " The capital of France is Paris.\n\nGiven that Charlie is fond of chess, and Alex is not fond of chess, if Charlie's score is 8-4 and Alex's score is 8-2, what is the total score of chess they both play against each other? A) Less than 12\n\nB) 12 or more\n\nC) 4 points less than 12\n\nD) 4 points more than 12\n\nE) More than 12\nTo determine the total score of chess that Charlie and Alex both play against each other, we need to calculate the individual scores for both", 'output_ids': [576, 6722, 315, 9625, 374, 12095, 382, 22043, 429, 24927, 374, 21208, 315, 32719, 11, 323, 8515, 374, 537, 21208, 315, 32719, 11, 421, 24927, 594, 5456, 374, 220, 23, 12, 19, 323, 8515, 594, 5456, 374, 220, 23, 12, 17, 11, 1128, 374, 279, 2790, 5456, 315, 32719, 807, 2176, 1486, 2348, 1817, 1008, 30, 362, 8, 9668, 1091, 220, 16, 17, 271, 33, 8, 220, 16, 17, 476, 803, 271, 34, 8, 220, 19, 3501, 2686, 1091, 220, 16, 17, 271, 35, 8, 220, 19, 3501, 803, 1091, 220, 16, 17, 271, 36, 8, 4398, 1091, 220, 16, 17, 198, 1249, 8253, 279, 2790, 5456, 315, 32719, 429, 24927, 323, 8515, 2176, 1486, 2348, 1817, 1008, 11, 582, 1184, 311, 11047, 279, 3842, 12205, 369, 2176], 'meta_info': {'id': 'd0b21d3bf4ee40f4bbe3175fd93fc474', 'finish_reason': {'type': 'length', 'length': 128}, 'prompt_tokens': 7, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 128, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.55209841998294, 'response_sent_to_client_ts': 1772983549.656364}}</strong>


## Get Model Info

Get the information of the model.

- `model_path`: The path/name of the model.
- `is_generation`: Whether the model is used as generation model or embedding model.
- `tokenizer_path`: The path/name of the tokenizer.
- `preferred_sampling_params`: The default sampling params specified via `--preferred-sampling-params`. `None` is returned in this example as we did not explicitly configure it in server args.
- `weight_version`: This field contains the version of the model weights. This is often used to track changes or updates to the model’s trained parameters.
- `has_image_understanding`: Whether the model has image-understanding capability.
- `has_audio_understanding`: Whether the model has audio-understanding capability.
- `model_type`: The model type from the HuggingFace config (e.g., "qwen2", "llama").
- `architectures`: The model architectures from the HuggingFace config (e.g., ["Qwen2ForCausalLM"]).


```python
url = f"http://localhost:{port}/get_model_info"

response = requests.get(url)
response_json = response.json()
print_highlight(response_json)
assert response_json["model_path"] == "qwen/qwen2.5-0.5b-instruct"
assert response_json["is_generation"] is True
assert response_json["tokenizer_path"] == "qwen/qwen2.5-0.5b-instruct"
assert response_json["preferred_sampling_params"] is None
assert response_json.keys() == {
    "model_path",
    "is_generation",
    "tokenizer_path",
    "preferred_sampling_params",
    "weight_version",
    "has_image_understanding",
    "has_audio_understanding",
    "model_type",
    "architectures",
}
```

    [2026-03-08 15:25:49] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.



<strong style='color: #00008B;'>{'model_path': 'qwen/qwen2.5-0.5b-instruct', 'tokenizer_path': 'qwen/qwen2.5-0.5b-instruct', 'is_generation': True, 'preferred_sampling_params': None, 'weight_version': 'default', 'has_image_understanding': False, 'has_audio_understanding': False, 'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM']}</strong>


## Get Server Info
Gets the server information including CLI arguments, token limits, and memory pool sizes.
- Note: `get_server_info` merges the following deprecated endpoints:
  - `get_server_args`
  - `get_memory_pool_size`
  - `get_max_total_num_tokens`


```python
url = f"http://localhost:{port}/get_server_info"

response = requests.get(url)
print_highlight(response.text)
```

    [2026-03-08 15:25:49] Endpoint '/get_server_info' is deprecated and will be removed in a future version. Please use '/server_info' instead.



<strong style='color: #00008B;'>{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","host":"0.0.0.0","port":30144,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":null,"pp_async_batch_depth":0,"stream_interval":1,"stream_output":false,"enable_streaming_session":false,"random_seed":277746025,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"collect_tokens_histogram":false,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"attention_backend":"fa3","decode_attention_backend":null,"prefill_attention_backend":null,"sampling_backend":"flashinfer","grammar_backend":"xgrammar","mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"flashinfer_cutlass","nsa_prefill_backend":null,"nsa_decode_backend":null,"disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_ngram_min_match_window_size":1,"speculative_ngram_max_match_window_size":12,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_branch_length":18,"speculative_ngram_capacity":10000000,"enable_multi_layer_eagle":false,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","disable_hicache_numa_detect":false,"hicache_storage_backend":null,"hicache_storage_prefetch_policy":"best_effort","hicache_storage_backend_extra_config":null,"hierarchical_sparse_attention_extra_config":null,"enable_lmcache":false,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"enable_double_sparsity":false,"ds_channel_config_path":null,"ds_heavy_channel_num":32,"ds_heavy_token_num":256,"ds_heavy_channel_type":"qk","ds_sparse_decode_threshold":4096,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","multi_item_scoring_delimiter":null,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":false,"disable_cuda_graph_padding":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_nan_detection":false,"enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"disable_chunked_prefix_cache":false,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"enable_nsa_prefill_context_parallel":false,"nsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"mm_max_concurrent_calls":32,"mm_per_request_timeout":10.0,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"status":"ready","max_total_num_tokens":20480,"max_req_input_len":20474,"internal_states":[{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","host":"0.0.0.0","port":30144,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":128,"pp_async_batch_depth":0,"stream_interval":1,"stream_output":false,"enable_streaming_session":false,"random_seed":277746025,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"collect_tokens_histogram":false,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"attention_backend":"fa3","decode_attention_backend":"fa3","prefill_attention_backend":"fa3","sampling_backend":"flashinfer","grammar_backend":"xgrammar","mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"flashinfer_cutlass","nsa_prefill_backend":null,"nsa_decode_backend":null,"disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_ngram_min_match_window_size":1,"speculative_ngram_max_match_window_size":12,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_branch_length":18,"speculative_ngram_capacity":10000000,"enable_multi_layer_eagle":false,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","disable_hicache_numa_detect":false,"hicache_storage_backend":null,"hicache_storage_prefetch_policy":"best_effort","hicache_storage_backend_extra_config":null,"hierarchical_sparse_attention_extra_config":null,"enable_lmcache":false,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"enable_double_sparsity":false,"ds_channel_config_path":null,"ds_heavy_channel_num":32,"ds_heavy_token_num":256,"ds_heavy_channel_type":"qk","ds_sparse_decode_threshold":4096,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","multi_item_scoring_delimiter":null,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":false,"disable_cuda_graph_padding":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_nan_detection":false,"enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"disable_chunked_prefix_cache":true,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"enable_nsa_prefill_context_parallel":false,"nsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"mm_max_concurrent_calls":32,"mm_per_request_timeout":10.0,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"use_mla_backend":false,"last_gen_throughput":284.6802401625865,"memory_usage":{"weight":1.58,"kvcache":0.23,"token_capacity":20480,"graph":0.06},"effective_max_running_requests_per_dp":128}],"version":"0.0.0.dev1+g07359efce"}</strong>


## Health Check
- `/health`: Check the health of the server.
- `/health_generate`: Check the health of the server by generating one token.


```python
url = f"http://localhost:{port}/health_generate"

response = requests.get(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'></strong>



```python
url = f"http://localhost:{port}/health"

response = requests.get(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'></strong>


## Flush Cache

Flush the radix cache. It will be automatically triggered when the model weights are updated by the `/update_weights` API.


```python
url = f"http://localhost:{port}/flush_cache"

response = requests.post(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'>Cache flushed.<br>Please check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)<br></strong>


## Update Weights From Disk

Update model weights from disk without restarting the server. Only applicable for models with the same architecture and parameter size.

SGLang support `update_weights_from_disk` API for continuous evaluation during training (save checkpoint to disk and update weights from disk).



```python
# successful update with same architecture and size

url = f"http://localhost:{port}/update_weights_from_disk"
data = {"model_path": "qwen/qwen2.5-0.5b-instruct"}

response = requests.post(url, json=data)
print_highlight(response.text)
assert response.json()["success"] is True
assert response.json()["message"] == "Succeeded to update model weights."
```

    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.82it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.82it/s]
    
    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:324: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>{"success":true,"message":"Succeeded to update model weights.","num_paused_requests":0}</strong>



```python
# failed update with different parameter size or wrong name

url = f"http://localhost:{port}/update_weights_from_disk"
data = {"model_path": "qwen/qwen2.5-0.5b-instruct-wrong"}

response = requests.post(url, json=data)
response_json = response.json()
print_highlight(response_json)
assert response_json["success"] is False
assert response_json["message"] == (
    "Failed to get weights iterator: "
    "qwen/qwen2.5-0.5b-instruct-wrong"
    " (repository not found)."
)
```

    [2026-03-08 15:25:52] Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).



<strong style='color: #00008B;'>{'success': False, 'message': 'Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).', 'num_paused_requests': 0}</strong>



```python
terminate_process(server_process)
```

## Encode (embedding model)

Encode text into embeddings. Note that this API is only available for [embedding models](openai_api_embeddings.ipynb) and will raise an error for generation models.
Therefore, we launch a new server to server an embedding model.


```python
embedding_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --host 0.0.0.0 --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=embedding_process)
```

    [2026-03-08 15:25:55] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:25:55] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:25:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:25:57] INFO model_config.py:1231: Downcasting torch.float32 to torch.float16.
    [2026-03-08 15:25:57] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:25:57] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:26:01] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:01] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:01] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:26:01] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:01] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:01] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:26:06] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:06] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:06] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.45it/s]


    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.08it/s]
    Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.13it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:38,  1.43it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:38,  1.43it/s]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:03<00:38,  1.43it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:18,  2.82it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:18,  2.82it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:18,  2.82it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.30it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:08,  5.97it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:08,  5.97it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:08,  5.97it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:08,  5.97it/s]

    Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:08,  5.97it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:03<00:04, 10.65it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]

    Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:01, 20.44it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:03<00:01, 27.73it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:03<00:00, 40.40it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 53.21it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 53.21it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 53.21it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:04<00:00, 59.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.63 GB):   2%|▏         | 1/58 [00:00<00:07,  7.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.60 GB):   2%|▏         | 1/58 [00:00<00:07,  7.75it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=73.60 GB):   3%|▎         | 2/58 [00:00<00:07,  7.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.60 GB):   3%|▎         | 2/58 [00:00<00:07,  7.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.60 GB):   3%|▎         | 2/58 [00:00<00:07,  7.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.60 GB):   7%|▋         | 4/58 [00:00<00:05, 10.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.60 GB):   7%|▋         | 4/58 [00:00<00:05, 10.08it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=73.60 GB):   7%|▋         | 4/58 [00:00<00:05, 10.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.60 GB):  10%|█         | 6/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.59 GB):  10%|█         | 6/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.59 GB):  10%|█         | 6/58 [00:00<00:04, 11.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.89it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.89it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=73.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.89it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.57 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.57 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.57 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.56 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.14it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.56 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.07 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.97 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.90 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.89 GB):  26%|██▌       | 15/58 [00:01<00:02, 16.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=72.89 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.89 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.89 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=960 avail_mem=72.86 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s] Capturing num tokens (num_tokens=896 avail_mem=72.85 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=832 avail_mem=72.86 GB):  33%|███▎      | 19/58 [00:01<00:01, 22.12it/s]Capturing num tokens (num_tokens=832 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]Capturing num tokens (num_tokens=768 avail_mem=72.87 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]Capturing num tokens (num_tokens=704 avail_mem=72.87 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]Capturing num tokens (num_tokens=640 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]Capturing num tokens (num_tokens=576 avail_mem=72.86 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.85 GB):  41%|████▏     | 24/58 [00:01<00:01, 28.35it/s]Capturing num tokens (num_tokens=512 avail_mem=72.85 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=480 avail_mem=72.85 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=448 avail_mem=72.85 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=416 avail_mem=72.84 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=384 avail_mem=72.84 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=352 avail_mem=72.84 GB):  50%|█████     | 29/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=352 avail_mem=72.84 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=320 avail_mem=72.83 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=288 avail_mem=72.84 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=256 avail_mem=72.84 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=240 avail_mem=72.83 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=224 avail_mem=72.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=208 avail_mem=72.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=192 avail_mem=72.83 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=176 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=160 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.11it/s]Capturing num tokens (num_tokens=144 avail_mem=72.82 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=128 avail_mem=72.81 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=112 avail_mem=72.81 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=96 avail_mem=72.80 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=72.80 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  76%|███████▌  | 44/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=64 avail_mem=72.80 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=48 avail_mem=72.79 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=32 avail_mem=72.79 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=24 avail_mem=72.78 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  84%|████████▍ | 49/58 [00:01<00:00, 41.76it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=12 avail_mem=72.77 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.57it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.77 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.57it/s] Capturing num tokens (num_tokens=4 avail_mem=72.76 GB):  93%|█████████▎| 54/58 [00:02<00:00, 42.57it/s]Capturing num tokens (num_tokens=4 avail_mem=72.76 GB): 100%|██████████| 58/58 [00:02<00:00, 28.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# successful encode for embedding model

url = f"http://localhost:{port}/encode"
data = {"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "text": "Once upon a time"}

response = requests.post(url, json=data)
response_json = response.json()
print_highlight(f"Text embedding (first 10): {response_json['embedding'][:10]}")
```


<strong style='color: #00008B;'>Text embedding (first 10): [-0.00023102760314941406, -0.04986572265625, -0.0032711029052734375, 0.011077880859375, -0.0140533447265625, 0.0159912109375, -0.01441192626953125, 0.0059051513671875, -0.0228424072265625, 0.0272979736328125]</strong>



```python
terminate_process(embedding_process)
```

## v1/rerank (cross encoder rerank model)
Rerank a list of documents given a query using a cross-encoder model. Note that this API is only available for cross encoder model like [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) with `attention-backend` `triton` and `torch_native`.



```python
reranker_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path BAAI/bge-reranker-v2-m3 \
    --host 0.0.0.0 --disable-radix-cache --chunked-prefill-size -1 --attention-backend triton --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=reranker_process)
```

    [2026-03-08 15:26:24] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:24] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:24] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:26:26] INFO model_config.py:1231: Downcasting torch.float32 to torch.float16.
    [2026-03-08 15:26:26] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:26:27] No HuggingFace chat template found


    [2026-03-08 15:26:30] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:30] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:30] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:26:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:31] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:26:36] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:36] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:36] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.00s/it]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.01s/it]
    


    [2026-03-08 15:26:37] Disable piecewise CUDA graph because the model is not a language model


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# compute rerank scores for query and documents

url = f"http://localhost:{port}/v1/rerank"
data = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
    ],
}

response = requests.post(url, json=data)
response_json = response.json()
for item in response_json:
    print_highlight(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
```


<strong style='color: #00008B;'>Score: 5.26 - Document: 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'</strong>



<strong style='color: #00008B;'>Score: -8.19 - Document: 'hi'</strong>



```python
terminate_process(reranker_process)
```

## v1/score (decoder-only scoring)

Compute token probabilities for specified tokens given a query and items. This is useful for classification tasks, scoring responses, or computing log-probabilities.

Parameters:
- `query`: Query text
- `items`: Item text(s) to score
- `label_token_ids`: Token IDs to compute probabilities for
- `apply_softmax`: Whether to apply softmax to get normalized probabilities (default: False)
- `item_first`: Whether items come first in concatenation order (default: False)
- `model`: Model name

The response contains `scores` - a list of probability lists, one per item, each in the order of `label_token_ids`.


```python
score_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=score_process)
```

    [2026-03-08 15:26:48] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:48] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:26:49] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:26:49] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:26:54] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:54] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:54] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:26:54] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:26:54] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:26:54] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:26:59] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:59] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:26:59] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.73it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.73it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=58.48 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=58.48 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.49it/s]Capturing batches (bs=2 avail_mem=58.42 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.49it/s]Capturing batches (bs=1 avail_mem=58.42 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.49it/s]Capturing batches (bs=1 avail_mem=58.42 GB): 100%|██████████| 3/3 [00:00<00:00, 10.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:04,  2.19s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<00:54,  1.03it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:21,  2.48it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:21,  2.48it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:21,  2.48it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:12,  4.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:12,  4.20it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:12,  4.20it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:12,  4.20it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=1536):  16%|█▌        | 9/58 [00:02<00:06,  7.17it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]

    Compiling num tokens (num_tokens=512):  33%|███▎      | 19/58 [00:02<00:01, 20.48it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:02<00:00, 33.78it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:02<00:00, 33.78it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:02<00:00, 33.78it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:02<00:00, 33.78it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:02<00:00, 33.78it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:00, 33.78it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:00, 33.78it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=112):  60%|██████    | 35/58 [00:03<00:00, 36.23it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 58.99it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 58.99it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 58.99it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 58.99it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.68 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.66 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=960 avail_mem=76.65 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s] Capturing num tokens (num_tokens=896 avail_mem=76.65 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=832 avail_mem=76.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=704 avail_mem=76.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=704 avail_mem=76.64 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=640 avail_mem=76.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=576 avail_mem=76.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=512 avail_mem=76.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=480 avail_mem=76.64 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=448 avail_mem=76.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=416 avail_mem=76.63 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.16it/s]Capturing num tokens (num_tokens=416 avail_mem=76.63 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=384 avail_mem=76.63 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=352 avail_mem=76.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=320 avail_mem=76.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=256 avail_mem=76.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=240 avail_mem=76.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.78it/s]Capturing num tokens (num_tokens=240 avail_mem=76.61 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.20it/s]Capturing num tokens (num_tokens=224 avail_mem=76.61 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.20it/s]Capturing num tokens (num_tokens=208 avail_mem=76.61 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.20it/s]Capturing num tokens (num_tokens=192 avail_mem=76.60 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.20it/s]Capturing num tokens (num_tokens=176 avail_mem=76.60 GB):  66%|██████▌   | 38/58 [00:00<00:00, 48.20it/s]Capturing num tokens (num_tokens=160 avail_mem=76.60 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=144 avail_mem=76.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 48.20it/s]Capturing num tokens (num_tokens=144 avail_mem=76.59 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]Capturing num tokens (num_tokens=128 avail_mem=76.59 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]Capturing num tokens (num_tokens=112 avail_mem=76.59 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s] Capturing num tokens (num_tokens=80 avail_mem=76.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]Capturing num tokens (num_tokens=64 avail_mem=76.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]Capturing num tokens (num_tokens=48 avail_mem=76.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 49.60it/s]Capturing num tokens (num_tokens=48 avail_mem=76.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=32 avail_mem=76.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=28 avail_mem=76.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=24 avail_mem=76.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=20 avail_mem=76.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=16 avail_mem=76.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=12 avail_mem=76.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 50.22it/s]Capturing num tokens (num_tokens=12 avail_mem=76.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s]Capturing num tokens (num_tokens=8 avail_mem=76.55 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.55 GB):  97%|█████████▋| 56/58 [00:01<00:00, 50.71it/s]Capturing num tokens (num_tokens=4 avail_mem=76.55 GB): 100%|██████████| 58/58 [00:01<00:00, 44.25it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# Score the probability of different completions given a query
query = "The capital of France is"
items = ["Paris", "London", "Berlin"]

url = f"http://localhost:{port}/v1/score"
data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "query": query,
    "items": items,
    "label_token_ids": [9454, 2753],  # e.g. "Yes" and "No" token ids
    "apply_softmax": True,  # Normalize probabilities to sum to 1
}

response = requests.post(url, json=data)
response_json = response.json()

# Display scores for each item
for item, scores in zip(items, response_json["scores"]):
    print_highlight(f"Item '{item}': probabilities = {[f'{s:.4f}' for s in scores]}")
```


<strong style='color: #00008B;'>Item 'Paris': probabilities = ['0.0237', '0.9763']</strong>



<strong style='color: #00008B;'>Item 'London': probabilities = ['0.0284', '0.9716']</strong>



<strong style='color: #00008B;'>Item 'Berlin': probabilities = ['0.0637', '0.9363']</strong>



```python
terminate_process(score_process)
```

## Classify (reward model)

SGLang Runtime also supports reward models. Here we use a reward model to classify the quality of pairwise generations.


```python
# Note that SGLang now treats embedding models and reward models as the same type of models.
# This will be updated in the future.

reward_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --host 0.0.0.0 --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=reward_process)
```

    [2026-03-08 15:27:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:27:17] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:27:17] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:27:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:21] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:27:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:21] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:27:26] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:27:26] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:27:26] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.13it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.03it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.62it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.43it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.36it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:00,  3.17s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:00,  3.17s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:32,  1.64s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:32,  1.64s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:55,  1.01s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:55,  1.01s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:37,  1.45it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:37,  1.45it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:27,  1.96it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:27,  1.96it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:20,  2.53it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:20,  2.53it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.18it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.18it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  3.88it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  3.88it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:10,  4.64it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:08,  5.48it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:08,  5.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:06,  7.01it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:06,  7.01it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:06,  7.01it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.45it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.45it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  8.45it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:04, 10.17it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:04, 10.17it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:04, 10.17it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:03, 12.09it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:03, 12.09it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:03, 12.09it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:03, 12.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 15.79it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 15.79it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 15.79it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 15.79it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 15.79it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:01, 21.62it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:00, 30.66it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:00, 30.66it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:00, 30.66it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:00, 30.66it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:00, 30.66it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:06<00:00, 30.66it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:00, 38.31it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 46.81it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 52.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=58.00 GB):   2%|▏         | 1/58 [00:00<00:20,  2.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.97 GB):   2%|▏         | 1/58 [00:00<00:20,  2.83it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.97 GB):   3%|▎         | 2/58 [00:00<00:18,  3.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.98 GB):   3%|▎         | 2/58 [00:00<00:18,  3.00it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.98 GB):   5%|▌         | 3/58 [00:00<00:17,  3.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.98 GB):   5%|▌         | 3/58 [00:00<00:17,  3.23it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.98 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.98 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.98 GB):   9%|▊         | 5/58 [00:01<00:14,  3.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.99 GB):   9%|▊         | 5/58 [00:01<00:14,  3.68it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.99 GB):  10%|█         | 6/58 [00:01<00:12,  4.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.99 GB):  10%|█         | 6/58 [00:01<00:12,  4.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.99 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.37it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.99 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.37it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.99 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.00 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.00 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.00 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.27it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.00 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.00 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.00 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.00 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.19it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.00 GB):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.00 GB):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.00 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.00 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.35it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.00 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.00 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.00 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.00 GB):  26%|██▌       | 15/58 [00:02<00:04,  8.72it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=58.00 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.00 GB):  29%|██▉       | 17/58 [00:02<00:04, 10.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.00 GB):  29%|██▉       | 17/58 [00:03<00:04, 10.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.00 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.12it/s]Capturing num tokens (num_tokens=960 avail_mem=57.99 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.12it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=57.99 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.52it/s]Capturing num tokens (num_tokens=896 avail_mem=57.99 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.52it/s]Capturing num tokens (num_tokens=832 avail_mem=57.99 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.52it/s]Capturing num tokens (num_tokens=768 avail_mem=57.98 GB):  38%|███▊      | 22/58 [00:03<00:02, 15.52it/s]Capturing num tokens (num_tokens=768 avail_mem=57.98 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Capturing num tokens (num_tokens=704 avail_mem=57.98 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Capturing num tokens (num_tokens=640 avail_mem=57.97 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Capturing num tokens (num_tokens=576 avail_mem=57.97 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.97 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.01it/s]Capturing num tokens (num_tokens=512 avail_mem=57.97 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.01it/s]Capturing num tokens (num_tokens=480 avail_mem=57.96 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.01it/s]Capturing num tokens (num_tokens=448 avail_mem=57.96 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.01it/s]Capturing num tokens (num_tokens=416 avail_mem=57.95 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.01it/s]Capturing num tokens (num_tokens=416 avail_mem=57.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.61it/s]Capturing num tokens (num_tokens=384 avail_mem=57.95 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.61it/s]Capturing num tokens (num_tokens=352 avail_mem=57.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.61it/s]

    Capturing num tokens (num_tokens=320 avail_mem=57.94 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.61it/s]Capturing num tokens (num_tokens=320 avail_mem=57.94 GB):  60%|██████    | 35/58 [00:03<00:00, 23.19it/s]Capturing num tokens (num_tokens=288 avail_mem=57.93 GB):  60%|██████    | 35/58 [00:03<00:00, 23.19it/s]Capturing num tokens (num_tokens=256 avail_mem=57.93 GB):  60%|██████    | 35/58 [00:03<00:00, 23.19it/s]Capturing num tokens (num_tokens=240 avail_mem=57.93 GB):  60%|██████    | 35/58 [00:03<00:00, 23.19it/s]

    Capturing num tokens (num_tokens=240 avail_mem=57.93 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=224 avail_mem=57.92 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=208 avail_mem=57.92 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=192 avail_mem=57.91 GB):  66%|██████▌   | 38/58 [00:03<00:00, 20.57it/s]Capturing num tokens (num_tokens=192 avail_mem=57.91 GB):  71%|███████   | 41/58 [00:04<00:00, 22.10it/s]Capturing num tokens (num_tokens=176 avail_mem=57.91 GB):  71%|███████   | 41/58 [00:04<00:00, 22.10it/s]Capturing num tokens (num_tokens=160 avail_mem=57.90 GB):  71%|███████   | 41/58 [00:04<00:00, 22.10it/s]Capturing num tokens (num_tokens=144 avail_mem=57.90 GB):  71%|███████   | 41/58 [00:04<00:00, 22.10it/s]

    Capturing num tokens (num_tokens=144 avail_mem=57.90 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.04it/s]Capturing num tokens (num_tokens=128 avail_mem=60.56 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.04it/s]Capturing num tokens (num_tokens=112 avail_mem=60.57 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.04it/s]Capturing num tokens (num_tokens=96 avail_mem=60.57 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.04it/s] Capturing num tokens (num_tokens=80 avail_mem=60.56 GB):  76%|███████▌  | 44/58 [00:04<00:00, 21.04it/s]Capturing num tokens (num_tokens=80 avail_mem=60.56 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.14it/s]Capturing num tokens (num_tokens=64 avail_mem=60.56 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.14it/s]Capturing num tokens (num_tokens=48 avail_mem=60.55 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.14it/s]Capturing num tokens (num_tokens=32 avail_mem=60.55 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.14it/s]Capturing num tokens (num_tokens=28 avail_mem=60.55 GB):  83%|████████▎ | 48/58 [00:04<00:00, 25.14it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.55 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.66it/s]Capturing num tokens (num_tokens=24 avail_mem=60.54 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.66it/s]Capturing num tokens (num_tokens=20 avail_mem=60.53 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.66it/s]Capturing num tokens (num_tokens=16 avail_mem=60.53 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.66it/s]Capturing num tokens (num_tokens=12 avail_mem=60.53 GB):  90%|████████▉ | 52/58 [00:04<00:00, 28.66it/s]Capturing num tokens (num_tokens=12 avail_mem=60.53 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.47it/s]Capturing num tokens (num_tokens=8 avail_mem=60.52 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.47it/s] Capturing num tokens (num_tokens=4 avail_mem=60.52 GB):  97%|█████████▋| 56/58 [00:04<00:00, 31.47it/s]Capturing num tokens (num_tokens=4 avail_mem=60.52 GB): 100%|██████████| 58/58 [00:04<00:00, 12.79it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
from transformers import AutoTokenizer

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)

RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]

tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
prompts = tokenizer.apply_chat_template(CONVS, tokenize=False, return_dict=False)

url = f"http://localhost:{port}/classify"
data = {"model": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", "text": prompts}

responses = requests.post(url, json=data).json()
for response in responses:
    print_highlight(f"reward: {response['embedding'][0]}")
```


<strong style='color: #00008B;'>reward: -24.125</strong>



<strong style='color: #00008B;'>reward: 1.1015625</strong>



```python
terminate_process(reward_process)
```

## Capture expert selection distribution in MoE models

SGLang Runtime supports recording the number of times an expert is selected in a MoE model run for each expert in the model. This is useful when analyzing the throughput of the model and plan for optimization.

*Note: We only print out the first 10 lines of the csv below for better readability. Please adjust accordingly if you want to analyze the results more deeply.*


```python
expert_record_server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B --host 0.0.0.0 --expert-distribution-recorder-mode stat --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=expert_record_server_process)
```

    [2026-03-08 15:27:51] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:51] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:51] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:27:53] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:27:53] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:27:57] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:57] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:57] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:27:57] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:27:57] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:27:57] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-08 15:28:02] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:28:02] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:28:02] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/8 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  12% Completed | 1/8 [00:00<00:04,  1.45it/s]


    Loading safetensors checkpoint shards:  25% Completed | 2/8 [00:01<00:04,  1.31it/s]


    Loading safetensors checkpoint shards:  38% Completed | 3/8 [00:02<00:03,  1.28it/s]


    Loading safetensors checkpoint shards:  50% Completed | 4/8 [00:03<00:03,  1.25it/s]


    Loading safetensors checkpoint shards:  62% Completed | 5/8 [00:03<00:02,  1.25it/s]


    Loading safetensors checkpoint shards:  75% Completed | 6/8 [00:04<00:01,  1.25it/s]


    Loading safetensors checkpoint shards:  88% Completed | 7/8 [00:05<00:00,  1.26it/s]


    Loading safetensors checkpoint shards: 100% Completed | 8/8 [00:05<00:00,  1.64it/s]
    Loading safetensors checkpoint shards: 100% Completed | 8/8 [00:05<00:00,  1.40it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=12.99 GB):   0%|          | 0/3 [00:00<?, ?it/s][2026-03-08 15:28:09] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
    [2026-03-08 15:28:09] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton


    Capturing batches (bs=4 avail_mem=12.99 GB):  33%|███▎      | 1/3 [00:01<00:02,  1.19s/it]Capturing batches (bs=2 avail_mem=11.68 GB):  33%|███▎      | 1/3 [00:01<00:02,  1.19s/it]

    Capturing batches (bs=2 avail_mem=11.68 GB):  67%|██████▋   | 2/3 [00:02<00:00,  1.03it/s]Capturing batches (bs=1 avail_mem=11.68 GB):  67%|██████▋   | 2/3 [00:02<00:00,  1.03it/s]

    Capturing batches (bs=1 avail_mem=11.68 GB): 100%|██████████| 3/3 [00:02<00:00,  1.50it/s]Capturing batches (bs=1 avail_mem=11.68 GB): 100%|██████████| 3/3 [00:02<00:00,  1.30it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
response = requests.post(f"http://localhost:{port}/start_expert_distribution_record")
print_highlight(response)

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())

response = requests.post(f"http://localhost:{port}/stop_expert_distribution_record")
print_highlight(response)

response = requests.post(f"http://localhost:{port}/dump_expert_distribution_record")
print_highlight(response)
```


<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'>{'text': ' The capital of France is Paris.', 'output_ids': [576, 6722, 315, 9625, 374, 12095, 13, 151643], 'meta_info': {'id': '262d8c296fd54bbc83b43229267764f6', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 7, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 8, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 0.11096358485519886, 'response_sent_to_client_ts': 1772983698.4745748}}</strong>



<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'><Response [200]></strong>



```python
terminate_process(expert_record_server_process)
```

## Tokenize/Detokenize Example (Round Trip)

This example demonstrates how to use the /tokenize and /detokenize endpoints together. We first tokenize a string, then detokenize the resulting IDs to reconstruct the original text. This workflow is useful when you need to handle tokenization externally but still leverage the server for detokenization.


```python
tokenizer_free_server_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct
""")

wait_for_server(f"http://localhost:{port}", process=tokenizer_free_server_process)
```

    [2026-03-08 15:28:22] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:28:22] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:28:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-08 15:28:24] INFO server_args.py:2110: Attention backend not specified. Use fa3 backend by default.
    [2026-03-08 15:28:24] INFO server_args.py:3217: Set soft_watchdog_timeout since in CI


    [2026-03-08 15:28:24] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=31756, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.836, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_output=False, enable_streaming_session=False, random_seed=767244647, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='flashinfer_cutlass', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_match_window_size=1, speculative_ngram_max_match_window_size=12, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_branch_length=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', disable_hicache_numa_detect=False, hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, hierarchical_sparse_attention_extra_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4], disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, mm_max_concurrent_calls=32, mm_per_request_timeout=10.0, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)


    [2026-03-08 15:28:25] Watchdog TokenizerManager initialized.
    [2026-03-08 15:28:25] Using default HuggingFace chat template with detected content format: string


    [2026-03-08 15:28:28] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:28:28] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:28:28] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-08 15:28:28] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-08 15:28:28] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-08 15:28:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-08 15:28:30] Watchdog DetokenizerManager initialized.


    [2026-03-08 15:28:30] Mamba selective_state_update backend initialized: triton
    [2026-03-08 15:28:30] Init torch distributed begin.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [2026-03-08 15:28:31] Init torch distributed ends. elapsed=0.23 s, mem usage=0.09 GB


    [2026-03-08 15:28:32] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:28:32] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-08 15:28:32] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-03-08 15:28:33] Load weight begin. avail mem=78.50 GB


    [2026-03-08 15:28:33] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /root/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.
    [2026-03-08 15:28:33] No model.safetensors.index.json found in remote.
    Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.43it/s]
    Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.43it/s]
    
    [2026-03-08 15:28:33] Load weight end. elapsed=0.44 s, type=Qwen2ForCausalLM, avail mem=77.53 GB, mem usage=0.98 GB.
    [2026-03-08 15:28:33] Using KV cache dtype: torch.bfloat16
    [2026-03-08 15:28:33] KV Cache is allocated. #tokens: 20480, K size: 0.12 GB, V size: 0.12 GB
    [2026-03-08 15:28:33] Memory pool end. avail mem=77.20 GB


    [2026-03-08 15:28:33] Capture cuda graph begin. This can take up to several minutes. avail mem=77.10 GB
    [2026-03-08 15:28:33] Capture cuda graph bs [1, 2, 4]


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=77.10 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=77.10 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.91it/s]Capturing batches (bs=2 avail_mem=77.04 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.91it/s]Capturing batches (bs=1 avail_mem=77.04 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.91it/s]Capturing batches (bs=1 avail_mem=77.04 GB): 100%|██████████| 3/3 [00:00<00:00, 12.49it/s]


    [2026-03-08 15:28:34] Capture cuda graph end. Time elapsed: 0.76 s. mem usage=0.06 GB. avail mem=77.04 GB.
    [2026-03-08 15:28:34] Capture piecewise CUDA graph begin. avail mem=77.04 GB
    [2026-03-08 15:28:34] Capture cuda graph num tokens [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
    [2026-03-08 15:28:34] install_torch_compiled
      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    [2026-03-08 15:28:35] Initializing SGLangBackend
    [2026-03-08 15:28:35] SGLangBackend __call__


    [2026-03-08 15:28:36] Compiling a graph for dynamic shape takes 0.19 s
    [2026-03-08 15:28:36] Computation graph saved to /root/.cache/sglang/torch_compile_cache/rank_0_0/backbone/computation_graph_1772983716.2359486.py


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:02,  2.15s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:02,  2.15s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:02,  2.15s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:02,  2.15s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:23,  2.30it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:06,  6.96it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]

    Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:02<00:02, 16.53it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:02<00:01, 26.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:02<00:00, 39.26it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:02<00:00, 50.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:02<00:00, 20.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.08it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.69 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.68 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.67 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.66 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.66 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=960 avail_mem=76.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s] Capturing num tokens (num_tokens=896 avail_mem=76.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=768 avail_mem=76.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.33it/s]Capturing num tokens (num_tokens=768 avail_mem=76.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=704 avail_mem=76.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=640 avail_mem=76.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=576 avail_mem=76.63 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=512 avail_mem=76.62 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=480 avail_mem=76.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=448 avail_mem=76.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.00it/s]Capturing num tokens (num_tokens=448 avail_mem=76.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=416 avail_mem=76.63 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=384 avail_mem=76.63 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=352 avail_mem=76.63 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=288 avail_mem=76.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=256 avail_mem=76.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=256 avail_mem=76.62 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=240 avail_mem=76.62 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=224 avail_mem=76.61 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=208 avail_mem=76.61 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=192 avail_mem=76.61 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=176 avail_mem=76.60 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.00it/s]Capturing num tokens (num_tokens=160 avail_mem=76.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 48.00it/s]Capturing num tokens (num_tokens=160 avail_mem=76.60 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]Capturing num tokens (num_tokens=144 avail_mem=76.60 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]Capturing num tokens (num_tokens=128 avail_mem=76.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]

    Capturing num tokens (num_tokens=112 avail_mem=76.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]Capturing num tokens (num_tokens=96 avail_mem=76.59 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s] Capturing num tokens (num_tokens=80 avail_mem=76.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]Capturing num tokens (num_tokens=64 avail_mem=76.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.26it/s]Capturing num tokens (num_tokens=64 avail_mem=76.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=48 avail_mem=76.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=32 avail_mem=76.58 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=28 avail_mem=76.57 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=24 avail_mem=76.57 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=20 avail_mem=76.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=16 avail_mem=76.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 49.87it/s]Capturing num tokens (num_tokens=16 avail_mem=76.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.45it/s]Capturing num tokens (num_tokens=12 avail_mem=76.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.45it/s]

    Capturing num tokens (num_tokens=8 avail_mem=76.55 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.45it/s] Capturing num tokens (num_tokens=4 avail_mem=76.55 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.45it/s]Capturing num tokens (num_tokens=4 avail_mem=76.55 GB): 100%|██████████| 58/58 [00:01<00:00, 43.89it/s]
    [2026-03-08 15:28:38] Capture piecewise CUDA graph end. Time elapsed: 4.49 s. mem usage=0.49 GB. avail mem=76.55 GB.


    [2026-03-08 15:28:39] max_total_num_tokens=20480, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=32768, available_gpu_mem=76.55 GB


    [2026-03-08 15:28:39] Successfully reserved port 31756 on host '127.0.0.1'
    [2026-03-08 15:28:39] INFO:     Started server process [1495959]
    [2026-03-08 15:28:39] INFO:     Waiting for application startup.
    [2026-03-08 15:28:39] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
    [2026-03-08 15:28:39] INFO:     Application startup complete.
    [2026-03-08 15:28:39] INFO:     Uvicorn running on socket ('127.0.0.1', 31756) (Press CTRL+C to quit)


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)
    [2026-03-08 15:28:39] INFO:     127.0.0.1:46056 - "GET /v1/models HTTP/1.1" 200 OK


    [2026-03-08 15:28:40] INFO:     127.0.0.1:46062 - "GET /model_info HTTP/1.1" 200 OK


    [2026-03-08 15:28:41] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, input throughput (token/s): 0.00, cuda graph: True
    [2026-03-08 15:28:41] INFO:     127.0.0.1:46074 - "POST /generate HTTP/1.1" 200 OK
    [2026-03-08 15:28:41] The server is fired up and ready to roll!



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
import requests
from sglang.utils import print_highlight

base_url = f"http://localhost:{port}"
tokenize_url = f"{base_url}/tokenize"
detokenize_url = f"{base_url}/detokenize"

model_name = "qwen/qwen2.5-0.5b-instruct"
input_text = "SGLang provides efficient tokenization endpoints."
print_highlight(f"Original Input Text:\n'{input_text}'")

# --- tokenize the input text ---
tokenize_payload = {
    "model": model_name,
    "prompt": input_text,
    "add_special_tokens": False,
}
try:
    tokenize_response = requests.post(tokenize_url, json=tokenize_payload)
    tokenize_response.raise_for_status()
    tokenization_result = tokenize_response.json()
    token_ids = tokenization_result.get("tokens")

    if not token_ids:
        raise ValueError("Tokenization returned empty tokens.")

    print_highlight(f"\nTokenized Output (IDs):\n{token_ids}")
    print_highlight(f"Token Count: {tokenization_result.get('count')}")
    print_highlight(f"Max Model Length: {tokenization_result.get('max_model_len')}")

    # --- detokenize the obtained token IDs ---
    detokenize_payload = {
        "model": model_name,
        "tokens": token_ids,
        "skip_special_tokens": True,
    }

    detokenize_response = requests.post(detokenize_url, json=detokenize_payload)
    detokenize_response.raise_for_status()
    detokenization_result = detokenize_response.json()
    reconstructed_text = detokenization_result.get("text")

    print_highlight(f"\nDetokenized Output (Text):\n'{reconstructed_text}'")

    if input_text == reconstructed_text:
        print_highlight(
            "\nRound Trip Successful: Original and reconstructed text match."
        )
    else:
        print_highlight(
            "\nRound Trip Mismatch: Original and reconstructed text differ."
        )

except requests.exceptions.RequestException as e:
    print_highlight(f"\nHTTP Request Error: {e}")
except Exception as e:
    print_highlight(f"\nAn error occurred: {e}")
```


<strong style='color: #00008B;'>Original Input Text:<br>'SGLang provides efficient tokenization endpoints.'</strong>


    [2026-03-08 15:28:44] INFO:     127.0.0.1:46076 - "POST /tokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Tokenized Output (IDs):<br>[50, 3825, 524, 5707, 11050, 3950, 2022, 36342, 13]</strong>



<strong style='color: #00008B;'>Token Count: 9</strong>



<strong style='color: #00008B;'>Max Model Length: 131072</strong>


    [2026-03-08 15:28:44] INFO:     127.0.0.1:46092 - "POST /detokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Detokenized Output (Text):<br>'SGLang provides efficient tokenization endpoints.'</strong>



<strong style='color: #00008B;'><br>Round Trip Successful: Original and reconstructed text match.</strong>



```python
terminate_process(tokenizer_free_server_process)
```
