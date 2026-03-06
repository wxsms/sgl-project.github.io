# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    [2026-03-06 08:06:28] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-06 08:06:28] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-06 08:06:28] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-06 08:06:32] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:06:32] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:06:32] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-06 08:06:34] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.
    [2026-03-06 08:06:34] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-06 08:06:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:06:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:06:39] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-06 08:06:39] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:06:39] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:06:39] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-06 08:06:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-06 08:06:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-06 08:06:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.65it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.51it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.54it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.87 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.87 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.13it/s]Capturing batches (bs=2 avail_mem=62.81 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.13it/s]Capturing batches (bs=1 avail_mem=62.81 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.13it/s]Capturing batches (bs=1 avail_mem=62.81 GB): 100%|██████████| 3/3 [00:00<00:00, 10.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:17,  1.39s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:17,  1.39s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:46,  1.18it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:46,  1.18it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:31,  1.71it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:31,  1.71it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  2.97it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  2.97it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:13,  3.70it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:13,  3.70it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.47it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.47it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.34it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.34it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.97it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.37it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.37it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.37it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.82it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.82it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.69it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.69it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.69it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.69it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.56it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.56it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 19.64it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.79it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:00, 34.50it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 44.62it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 50.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=34.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=34.24 GB):   2%|▏         | 1/58 [00:00<00:17,  3.31it/s]Capturing num tokens (num_tokens=7680 avail_mem=34.21 GB):   2%|▏         | 1/58 [00:00<00:17,  3.31it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=34.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=34.21 GB):   3%|▎         | 2/58 [00:00<00:15,  3.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=34.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=34.21 GB):   5%|▌         | 3/58 [00:00<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=34.21 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=34.22 GB):   7%|▋         | 4/58 [00:01<00:13,  4.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=34.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=34.22 GB):   9%|▊         | 5/58 [00:01<00:12,  4.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=34.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=34.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.65it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=34.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=34.23 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.02it/s]Capturing num tokens (num_tokens=4608 avail_mem=34.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=34.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.50it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=34.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=34.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=34.23 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=34.23 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.48it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=34.23 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.23 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=34.23 GB):  21%|██        | 12/58 [00:02<00:06,  7.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=34.23 GB):  21%|██        | 12/58 [00:02<00:06,  7.63it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=34.23 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=34.23 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.23 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=34.23 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=34.23 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.61it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=34.23 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=34.23 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.23 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=34.23 GB):  29%|██▉       | 17/58 [00:02<00:03, 11.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=34.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=34.22 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.94it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=34.23 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.94it/s]Capturing num tokens (num_tokens=960 avail_mem=34.22 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.94it/s] Capturing num tokens (num_tokens=960 avail_mem=34.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.28it/s]Capturing num tokens (num_tokens=896 avail_mem=34.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.28it/s]Capturing num tokens (num_tokens=832 avail_mem=34.22 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.28it/s]Capturing num tokens (num_tokens=768 avail_mem=34.21 GB):  38%|███▊      | 22/58 [00:02<00:02, 16.28it/s]Capturing num tokens (num_tokens=768 avail_mem=34.21 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.35it/s]Capturing num tokens (num_tokens=704 avail_mem=34.21 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.35it/s]

    Capturing num tokens (num_tokens=640 avail_mem=34.21 GB):  43%|████▎     | 25/58 [00:02<00:01, 19.35it/s]Capturing num tokens (num_tokens=576 avail_mem=34.20 GB):  43%|████▎     | 25/58 [00:03<00:01, 19.35it/s]Capturing num tokens (num_tokens=576 avail_mem=34.20 GB):  48%|████▊     | 28/58 [00:03<00:01, 22.06it/s]Capturing num tokens (num_tokens=512 avail_mem=34.19 GB):  48%|████▊     | 28/58 [00:03<00:01, 22.06it/s]Capturing num tokens (num_tokens=480 avail_mem=34.19 GB):  48%|████▊     | 28/58 [00:03<00:01, 22.06it/s]Capturing num tokens (num_tokens=448 avail_mem=34.19 GB):  48%|████▊     | 28/58 [00:03<00:01, 22.06it/s]Capturing num tokens (num_tokens=416 avail_mem=34.18 GB):  48%|████▊     | 28/58 [00:03<00:01, 22.06it/s]Capturing num tokens (num_tokens=416 avail_mem=34.18 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Capturing num tokens (num_tokens=384 avail_mem=34.18 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]

    Capturing num tokens (num_tokens=352 avail_mem=34.18 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Capturing num tokens (num_tokens=320 avail_mem=34.17 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Capturing num tokens (num_tokens=288 avail_mem=34.17 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.56it/s]Capturing num tokens (num_tokens=288 avail_mem=34.17 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.79it/s]Capturing num tokens (num_tokens=256 avail_mem=34.17 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.79it/s]Capturing num tokens (num_tokens=240 avail_mem=34.16 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.79it/s]Capturing num tokens (num_tokens=224 avail_mem=34.16 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.79it/s]Capturing num tokens (num_tokens=208 avail_mem=34.16 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.79it/s]Capturing num tokens (num_tokens=208 avail_mem=34.16 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=192 avail_mem=34.15 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.48it/s]

    Capturing num tokens (num_tokens=176 avail_mem=34.15 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=160 avail_mem=34.15 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=144 avail_mem=34.14 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.48it/s]Capturing num tokens (num_tokens=144 avail_mem=34.14 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s]Capturing num tokens (num_tokens=128 avail_mem=34.15 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s]Capturing num tokens (num_tokens=112 avail_mem=34.15 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s]Capturing num tokens (num_tokens=96 avail_mem=34.14 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s] Capturing num tokens (num_tokens=80 avail_mem=34.14 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s]Capturing num tokens (num_tokens=64 avail_mem=34.14 GB):  76%|███████▌  | 44/58 [00:03<00:00, 33.64it/s]

    Capturing num tokens (num_tokens=64 avail_mem=34.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=48 avail_mem=34.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=32 avail_mem=34.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=28 avail_mem=34.13 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=24 avail_mem=34.12 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=20 avail_mem=34.12 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.72it/s]Capturing num tokens (num_tokens=20 avail_mem=34.12 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.28it/s]Capturing num tokens (num_tokens=16 avail_mem=34.11 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.28it/s]Capturing num tokens (num_tokens=12 avail_mem=34.11 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.28it/s]Capturing num tokens (num_tokens=8 avail_mem=34.11 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.28it/s] Capturing num tokens (num_tokens=4 avail_mem=34.10 GB):  93%|█████████▎| 54/58 [00:03<00:00, 37.28it/s]

    Capturing num tokens (num_tokens=4 avail_mem=34.10 GB): 100%|██████████| 58/58 [00:03<00:00, 15.08it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34422


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-06 08:07:04] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. Japan - Tokyo<br>2. France - Paris<br>3. India - New Delhi</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Sure! Here are three countries along with their respective capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Mexico** - Mexico City</strong>



<strong style='color: #00008B;'>Of course! Here’s another list of three countries and their capitals:<br><br>1. **India** - New Delhi<br>2. **Germany** - Berlin<br>3. **Brazil** - Brasília</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2. Let's solve it.<br><br>\[ 2 * 2 = 4 \]<br><br>So, 2 * 2 equals 4. You didn't actually need a calculator for this; it's a simple multiplication!</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>### Tips for Staying Healthy:<br>1. **Balanced Diet:**<br>   - **Fruits and Vegetables:** Consume a variety of fruits and vegetables to meet your daily nutritional needs.<br>   - **Whole Grains:** Include whole grains like oats, brown rice, and quinoa to provide necessary carbohydrates and fiber.<br>   - **Lean Proteins:** Incorporate lean proteins such as chicken, fish, beans, and legumes to support muscle health and recovery.<br>   - **Healthy Fats:** Include foods rich in healthy fats like nuts, seeds, avocados, and olive oil to support overall health.<br>   - **Limit Processed Foods:** Reduce consumption of processed foods, sugary drinks, and foods high in saturated and trans fats.<br><br>2. **Regular Exercise:**<br>   - **Aerobic Activity:** Engage in at least 150 minutes of moderate aerobic activity (like brisk walking, cycling, or swimming) or 75 minutes of vigorous aerobic activity (like running or jumping rope) each week.<br>   - **Muscle-Strengthening:** Include muscle-strengthening exercises on two or more days a week, such as weightlifting, bodyweight exercises, or resistance band workouts.<br>   - **Flexibility and Balance:** Incorporate flexibility and balance exercises like yoga or Pilates to enhance overall fitness and reduce the risk of injuries.<br><br>Together, these two elements can significantly contribute to maintaining a healthy and active lifestyle.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Phoenix Feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "stag",<br>    "bogart": "The Dementor"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

    100%|██████████| 3/3 [00:00<00:00, 37.27it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is东京 (Tokyo).</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    [2026-03-06 08:07:15] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:07:15] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:07:15] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-06 08:07:16] INFO server_args.py:2048: Attention backend not specified. Use fa3 backend by default.
    [2026-03-06 08:07:16] INFO server_args.py:3155: Set soft_watchdog_timeout since in CI


    [2026-03-06 08:07:19] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-06 08:07:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:07:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:07:21] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-06 08:07:21] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-06 08:07:21] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-06 08:07:21] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-06 08:07:27] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-06 08:07:27] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-06 08:07:27] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.71it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:01,  1.51it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:01<00:00,  2.01it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.82it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.55it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.64it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=60.77 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=60.77 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.87it/s]Capturing batches (bs=2 avail_mem=59.77 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.87it/s]Capturing batches (bs=1 avail_mem=59.77 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.87it/s]Capturing batches (bs=1 avail_mem=59.77 GB): 100%|██████████| 3/3 [00:00<00:00,  5.08it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33410



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-06 08:07:38] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>The image shows a man on a street, dressed in an Mountains Equipment Company (MEC) yellow shirt, stepping away from the back of a yellow taxi while ironing a shirt on a crossbar. The truck bed appears to be used as a temporary ironing station, with an iron and holding straps for the shirt visible. The street scene suggests an urban setting, possibly a busy city, indicated by the presence of taxis and other vehicles in the background. Additional details such as flags, stores, and the American flag hanging by the taxi contribute to the urban setting.</strong>



```python
terminate_process(server_process)
```
