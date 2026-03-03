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

    [2026-03-03 00:23:05] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 00:23:05] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 00:23:05] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-03 00:23:10] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:10] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:10] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 00:23:12] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 00:23:12] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 00:23:16] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:16] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:16] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 00:23:16] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:16] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:16] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 00:23:20] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 00:23:20] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 00:23:20] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.69it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.56it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.55it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.52it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=62.87 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=62.87 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.19it/s]Capturing batches (bs=2 avail_mem=62.81 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.19it/s]Capturing batches (bs=1 avail_mem=62.81 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.19it/s]Capturing batches (bs=1 avail_mem=62.81 GB): 100%|██████████| 3/3 [00:00<00:00, 10.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:25,  2.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:25,  2.56s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:43,  1.26it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:29,  1.81it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:21,  2.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:21,  2.43it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.08it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.08it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.82it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.82it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.59it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.59it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:08,  5.46it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:08,  5.46it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:08,  5.46it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.07it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.07it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.07it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.36it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.36it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.36it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.90it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.90it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.90it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.78it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 15.01it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 15.01it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 15.01it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 15.01it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 15.01it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:01, 20.49it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:00, 29.31it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:00, 29.31it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:00, 29.31it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 36.75it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 47.01it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 55.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.83it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.97 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=60.97 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.94 GB):   2%|▏         | 1/58 [00:00<00:16,  3.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=60.94 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.94 GB):   3%|▎         | 2/58 [00:00<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=60.94 GB):   5%|▌         | 3/58 [00:00<00:14,  3.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.97 GB):   5%|▌         | 3/58 [00:00<00:14,  3.73it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.97 GB):   7%|▋         | 4/58 [00:01<00:13,  3.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.97 GB):   9%|▊         | 5/58 [00:01<00:12,  4.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.98 GB):   9%|▊         | 5/58 [00:01<00:12,  4.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.98 GB):  10%|█         | 6/58 [00:01<00:11,  4.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.55 GB):  10%|█         | 6/58 [00:01<00:11,  4.50it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=59.55 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.56 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.86it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.56 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.52 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.60it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.52 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.51 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.09it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.51 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.50 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.50 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.50 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.50 GB):  21%|██        | 12/58 [00:02<00:06,  6.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.50 GB):  21%|██        | 12/58 [00:02<00:06,  6.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.50 GB):  21%|██        | 12/58 [00:02<00:06,  6.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.50 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.50 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.34it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=59.50 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.50 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.50 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.50 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.81it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=59.50 GB):  31%|███       | 18/58 [00:02<00:03, 10.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.50 GB):  31%|███       | 18/58 [00:02<00:03, 10.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.50 GB):  31%|███       | 18/58 [00:02<00:03, 10.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.50 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.50 GB):  34%|███▍      | 20/58 [00:02<00:03, 12.32it/s]Capturing num tokens (num_tokens=960 avail_mem=59.50 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.32it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=59.50 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.65it/s]Capturing num tokens (num_tokens=896 avail_mem=59.49 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.65it/s]Capturing num tokens (num_tokens=832 avail_mem=59.49 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.65it/s]Capturing num tokens (num_tokens=832 avail_mem=59.49 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=768 avail_mem=59.48 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.07it/s]Capturing num tokens (num_tokens=704 avail_mem=59.48 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.07it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.48 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.23it/s]Capturing num tokens (num_tokens=640 avail_mem=59.48 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.23it/s]Capturing num tokens (num_tokens=576 avail_mem=59.47 GB):  45%|████▍     | 26/58 [00:03<00:01, 16.23it/s]Capturing num tokens (num_tokens=576 avail_mem=59.47 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.97it/s]Capturing num tokens (num_tokens=512 avail_mem=59.47 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.97it/s]Capturing num tokens (num_tokens=480 avail_mem=59.46 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.97it/s]Capturing num tokens (num_tokens=448 avail_mem=59.46 GB):  48%|████▊     | 28/58 [00:03<00:01, 16.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.46 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.86it/s]Capturing num tokens (num_tokens=416 avail_mem=59.46 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.86it/s]Capturing num tokens (num_tokens=384 avail_mem=59.45 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.86it/s]Capturing num tokens (num_tokens=352 avail_mem=59.45 GB):  53%|█████▎    | 31/58 [00:03<00:01, 18.86it/s]Capturing num tokens (num_tokens=352 avail_mem=59.45 GB):  59%|█████▊    | 34/58 [00:03<00:01, 20.41it/s]Capturing num tokens (num_tokens=320 avail_mem=59.45 GB):  59%|█████▊    | 34/58 [00:03<00:01, 20.41it/s]Capturing num tokens (num_tokens=288 avail_mem=59.44 GB):  59%|█████▊    | 34/58 [00:03<00:01, 20.41it/s]Capturing num tokens (num_tokens=256 avail_mem=59.44 GB):  59%|█████▊    | 34/58 [00:03<00:01, 20.41it/s]

    Capturing num tokens (num_tokens=256 avail_mem=59.44 GB):  64%|██████▍   | 37/58 [00:03<00:00, 22.25it/s]Capturing num tokens (num_tokens=240 avail_mem=59.43 GB):  64%|██████▍   | 37/58 [00:03<00:00, 22.25it/s]Capturing num tokens (num_tokens=224 avail_mem=59.43 GB):  64%|██████▍   | 37/58 [00:03<00:00, 22.25it/s]Capturing num tokens (num_tokens=208 avail_mem=59.43 GB):  64%|██████▍   | 37/58 [00:03<00:00, 22.25it/s]Capturing num tokens (num_tokens=192 avail_mem=59.42 GB):  64%|██████▍   | 37/58 [00:03<00:00, 22.25it/s]Capturing num tokens (num_tokens=192 avail_mem=59.42 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=176 avail_mem=59.42 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=160 avail_mem=59.42 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=144 avail_mem=59.41 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=128 avail_mem=59.42 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]

    Capturing num tokens (num_tokens=112 avail_mem=59.42 GB):  71%|███████   | 41/58 [00:03<00:00, 26.66it/s]Capturing num tokens (num_tokens=112 avail_mem=59.42 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s]Capturing num tokens (num_tokens=96 avail_mem=59.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s] Capturing num tokens (num_tokens=80 avail_mem=59.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s]Capturing num tokens (num_tokens=64 avail_mem=59.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s]Capturing num tokens (num_tokens=48 avail_mem=59.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s]Capturing num tokens (num_tokens=32 avail_mem=59.40 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.02it/s]Capturing num tokens (num_tokens=32 avail_mem=59.40 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]Capturing num tokens (num_tokens=28 avail_mem=59.40 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]Capturing num tokens (num_tokens=24 avail_mem=59.40 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]Capturing num tokens (num_tokens=20 avail_mem=59.39 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]

    Capturing num tokens (num_tokens=16 avail_mem=59.39 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]Capturing num tokens (num_tokens=12 avail_mem=59.38 GB):  88%|████████▊ | 51/58 [00:04<00:00, 34.27it/s]Capturing num tokens (num_tokens=12 avail_mem=59.38 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.42it/s]Capturing num tokens (num_tokens=8 avail_mem=59.38 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.42it/s] Capturing num tokens (num_tokens=4 avail_mem=59.37 GB):  97%|█████████▋| 56/58 [00:04<00:00, 36.42it/s]Capturing num tokens (num_tokens=4 avail_mem=59.37 GB): 100%|██████████| 58/58 [00:04<00:00, 13.51it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33617


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-03 00:23:40] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries and their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Brazil - Brasília</strong>


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


<strong style='color: #00008B;'>Sure! Here is a list of three countries along with their respective capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Brazil** - Brasília</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries along with their respective capitals:<br><br>1. **Germany** - Berlin<br>2. **India** - New Delhi<br>3. **Mexico** - Mexico City</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>You don't necessarily need a calculator for this problem, as it's a straightforward multiplication. The answer is 4.<br><br>However, if you want me to simulate using a calculator, here's how it would look:<br><br>1. Type `2 * 2` into your calculator.<br>2. Press the '=' or 'Enter' button.<br><br>The calculator will display: 4.<br><br>So, 2 * 2 = 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Eating a variety of foods in appropriate portions to ensure you get all necessary nutrients. Include whole grains, lean proteins, healthy fats, and plenty of fruits and vegetables. Stay hydrated and limit sugary and processed foods.<br>2. **Regular Exercise**: Engage in physical activities regularly to improve cardiovascular health, boost immunity, and enhance overall well-being. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, plus muscle-strengthening exercises on two or more days.<br><br>By following these tips, you can significantly improve your health and quality of life.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holm oak",<br>        "core": "Phoenix feather",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Argus Filch"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 40.46it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


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

    [2026-03-03 00:23:49] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:49] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:49] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 00:23:51] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 00:23:51] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 00:23:53] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-03 00:23:55] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:55] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:55] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 00:23:55] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 00:23:55] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 00:23:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 00:24:01] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 00:24:01] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 00:24:01] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.57it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.42it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.29it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.31it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.68it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.52it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.44 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.44 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.08it/s]Capturing batches (bs=2 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.08it/s]Capturing batches (bs=1 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.08it/s]Capturing batches (bs=1 avail_mem=61.42 GB): 100%|██████████| 3/3 [00:00<00:00,  5.67it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:31670



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-03 00:24:13] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image shows a person engaging in an unconventional activity on a New York City street. The individual is standing on the hood of a yellow cab, using crutches to support their position. They are ironing a shirt on an ironing board, which is placed on the cab's rear window. In the background, there is another yellow cab, and the setting appears to be a busy urban street with buildings and flags visible. This scene captures an amusing juxtaposition of everyday activities and transportation in an urban environment.</strong>



```python
terminate_process(server_process)
```
