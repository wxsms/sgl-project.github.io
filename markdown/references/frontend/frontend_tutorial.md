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

    [2026-03-05 04:34:45] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-05 04:34:45] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-05 04:34:45] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-05 04:34:49] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:34:49] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:34:49] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-05 04:34:51] INFO server_args.py:2038: Attention backend not specified. Use fa3 backend by default.
    [2026-03-05 04:34:51] INFO server_args.py:3129: Set soft_watchdog_timeout since in CI


    [2026-03-05 04:34:52] INFO utils.py:452: Successfully reserved port 30734 on host '0.0.0.0'


    [2026-03-05 04:34:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:34:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:34:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-05 04:34:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:34:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:34:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-05 04:35:00] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-05 04:35:00] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-05 04:35:00] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.60it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.50it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.52it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.53it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.53it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=32.83 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=32.83 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]Capturing batches (bs=2 avail_mem=32.78 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]

    Capturing batches (bs=1 avail_mem=32.78 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.35it/s]Capturing batches (bs=1 avail_mem=32.78 GB): 100%|██████████| 3/3 [00:00<00:00, 12.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.64s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:14,  1.33s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:14,  1.33s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:44,  1.22it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:44,  1.22it/s]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:22,  2.38it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:17,  3.02it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:17,  3.02it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.73it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.73it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.50it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.50it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.35it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.35it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.35it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.98it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.34it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.34it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.34it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  9.73it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  9.73it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  9.73it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.62it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.62it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.62it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 14.72it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 14.72it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 14.72it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 14.72it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 14.72it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 19.00it/s]

    Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 19.00it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 21.41it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 21.41it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 21.41it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 21.41it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 21.41it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 25.94it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 25.94it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 25.94it/s]

    Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 25.94it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 25.57it/s]

    Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 28.27it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 28.53it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 41.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 46.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=27.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=27.55 GB):   2%|▏         | 1/58 [00:00<00:19,  2.91it/s]Capturing num tokens (num_tokens=7680 avail_mem=30.18 GB):   2%|▏         | 1/58 [00:00<00:19,  2.91it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=30.18 GB):   3%|▎         | 2/58 [00:00<00:19,  2.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=30.18 GB):   3%|▎         | 2/58 [00:00<00:19,  2.87it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=30.18 GB):   5%|▌         | 3/58 [00:00<00:16,  3.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=30.18 GB):   5%|▌         | 3/58 [00:00<00:16,  3.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=30.18 GB):   7%|▋         | 4/58 [00:01<00:16,  3.22it/s]Capturing num tokens (num_tokens=6144 avail_mem=30.18 GB):   7%|▋         | 4/58 [00:01<00:16,  3.22it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=30.18 GB):   9%|▊         | 5/58 [00:01<00:14,  3.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=30.19 GB):   9%|▊         | 5/58 [00:01<00:14,  3.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=30.19 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=30.19 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=30.19 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=30.19 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=30.19 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=30.20 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.11it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=30.20 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=30.20 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=30.20 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.35it/s]Capturing num tokens (num_tokens=3584 avail_mem=30.20 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.35it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=30.20 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=30.20 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=30.20 GB):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=30.20 GB):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=30.20 GB):  21%|██        | 12/58 [00:02<00:06,  6.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=30.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=30.20 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=30.19 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.17it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=30.19 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=30.16 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=30.16 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=30.16 GB):  31%|███       | 18/58 [00:03<00:03, 10.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=30.16 GB):  31%|███       | 18/58 [00:03<00:03, 10.56it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=30.16 GB):  31%|███       | 18/58 [00:03<00:03, 10.56it/s]Capturing num tokens (num_tokens=1280 avail_mem=30.16 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=30.16 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.58it/s]Capturing num tokens (num_tokens=960 avail_mem=30.16 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.58it/s] Capturing num tokens (num_tokens=896 avail_mem=30.15 GB):  34%|███▍      | 20/58 [00:03<00:03, 12.58it/s]Capturing num tokens (num_tokens=896 avail_mem=30.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.99it/s]Capturing num tokens (num_tokens=832 avail_mem=30.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.99it/s]

    Capturing num tokens (num_tokens=768 avail_mem=30.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.99it/s]Capturing num tokens (num_tokens=704 avail_mem=30.14 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.99it/s]Capturing num tokens (num_tokens=704 avail_mem=30.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Capturing num tokens (num_tokens=640 avail_mem=30.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Capturing num tokens (num_tokens=576 avail_mem=30.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Capturing num tokens (num_tokens=512 avail_mem=30.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Capturing num tokens (num_tokens=480 avail_mem=30.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.90it/s]Capturing num tokens (num_tokens=480 avail_mem=30.13 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=448 avail_mem=30.12 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.75it/s]

    Capturing num tokens (num_tokens=416 avail_mem=30.12 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=384 avail_mem=30.12 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=352 avail_mem=30.11 GB):  52%|█████▏    | 30/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=352 avail_mem=30.11 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.95it/s]Capturing num tokens (num_tokens=320 avail_mem=30.11 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.95it/s]Capturing num tokens (num_tokens=288 avail_mem=30.10 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.95it/s]Capturing num tokens (num_tokens=256 avail_mem=30.10 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.95it/s]Capturing num tokens (num_tokens=240 avail_mem=30.10 GB):  59%|█████▊    | 34/58 [00:03<00:00, 25.95it/s]Capturing num tokens (num_tokens=240 avail_mem=30.10 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.10it/s]Capturing num tokens (num_tokens=224 avail_mem=30.09 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.10it/s]

    Capturing num tokens (num_tokens=208 avail_mem=30.09 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.10it/s]Capturing num tokens (num_tokens=192 avail_mem=30.08 GB):  66%|██████▌   | 38/58 [00:03<00:00, 29.10it/s]Capturing num tokens (num_tokens=192 avail_mem=30.08 GB):  71%|███████   | 41/58 [00:03<00:00, 28.42it/s]Capturing num tokens (num_tokens=176 avail_mem=30.08 GB):  71%|███████   | 41/58 [00:03<00:00, 28.42it/s]Capturing num tokens (num_tokens=160 avail_mem=30.08 GB):  71%|███████   | 41/58 [00:03<00:00, 28.42it/s]Capturing num tokens (num_tokens=144 avail_mem=30.07 GB):  71%|███████   | 41/58 [00:03<00:00, 28.42it/s]Capturing num tokens (num_tokens=128 avail_mem=30.08 GB):  71%|███████   | 41/58 [00:03<00:00, 28.42it/s]Capturing num tokens (num_tokens=128 avail_mem=30.08 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.19it/s]Capturing num tokens (num_tokens=112 avail_mem=30.08 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.19it/s]

    Capturing num tokens (num_tokens=96 avail_mem=30.08 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.19it/s] Capturing num tokens (num_tokens=80 avail_mem=30.07 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.19it/s]Capturing num tokens (num_tokens=64 avail_mem=30.07 GB):  78%|███████▊  | 45/58 [00:04<00:00, 31.19it/s]Capturing num tokens (num_tokens=64 avail_mem=30.07 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.59it/s]Capturing num tokens (num_tokens=48 avail_mem=30.07 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.59it/s]Capturing num tokens (num_tokens=32 avail_mem=30.06 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.59it/s]Capturing num tokens (num_tokens=28 avail_mem=30.06 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.59it/s]Capturing num tokens (num_tokens=24 avail_mem=30.06 GB):  84%|████████▍ | 49/58 [00:04<00:00, 33.59it/s]Capturing num tokens (num_tokens=24 avail_mem=30.06 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.87it/s]Capturing num tokens (num_tokens=20 avail_mem=30.05 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.87it/s]

    Capturing num tokens (num_tokens=16 avail_mem=30.05 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.87it/s]Capturing num tokens (num_tokens=12 avail_mem=30.04 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.87it/s]Capturing num tokens (num_tokens=8 avail_mem=30.04 GB):  91%|█████████▏| 53/58 [00:04<00:00, 34.87it/s] Capturing num tokens (num_tokens=8 avail_mem=30.04 GB):  98%|█████████▊| 57/58 [00:04<00:00, 36.33it/s]Capturing num tokens (num_tokens=4 avail_mem=30.04 GB):  98%|█████████▊| 57/58 [00:04<00:00, 36.33it/s]Capturing num tokens (num_tokens=4 avail_mem=30.04 GB): 100%|██████████| 58/58 [00:04<00:00, 13.49it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:30734


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-05 04:35:21] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Brazil - Brasília<br>3. Japan - Tokyo</strong>


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


<strong style='color: #00008B;'>Sure! Here is a list of three countries and their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Italy** - Rome</strong>



<strong style='color: #00008B;'>Certainly! Here’s another list of three countries and their capitals:<br><br>1. **Japan** - Tokyo<br>2. **Australia** - Canberra<br>3. **Nigeria** - Abuja</strong>


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



<strong style='color: #00008B;'>2 * 2, which equals 4. You don't actually need a calculator for this, but let me show you:<br><br>1) 2 * 2 means 2 added to itself 2 times.<br>2) So, 2 + 2 = 4.<br><br>Therefore, the answer is 4.</strong>


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


<strong style='color: #00008B;'>### Tip 1: Balanced Diet<br>Eating a balanced diet is crucial for maintaining good health. It ensures that you receive all the necessary nutrients your body needs to function optimally. A balanced diet includes a variety of foods from different food groups such as fruits, vegetables, whole grains, lean proteins, and healthy fats. Fruits and vegetables should be abundant as they are rich in vitamins, minerals, and fiber, which help in promoting digestive health and boosting the immune system. Whole grains provide essential nutrients and help maintain stable blood sugar levels. Lean proteins, like fish, poultry, and legumes, are important for muscle and tissue repair and support. Healthy fats, found in foods like nuts, avocados, and olive oil, are vital for brain function and hormone production. Additionally, staying hydrated by drinking plenty of water also plays a key role in overall health. By incorporating these elements into your daily diet, you can enhance your energy levels, improve cognitive function, and reduce the risk of chronic diseases.<br><br>### Tip 2: Regular Exercise<br>Regular exercise is crucial for maintaining good health and preventing various diseases. Engaging in physical activities like walking, jogging, swimming, or cycling helps to enhance cardiovascular health, boost immunity, and improve overall well-being. Regular exercise also helps maintain a healthy weight by burning calories, reducing the risk of obesity and related conditions such as type 2 diabetes, hypertension, and high cholesterol. Moreover, physical activity has been shown to reduce stress, improve mood, and enhance cognitive function. Consistent exercise routines can also strengthen muscles and bones, increasing flexibility and aiding in better posture. To reap these benefits, aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, as recommended by the World Health Organization. Adding strength training exercises at least twice a week further enhances these health benefits.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Fur from astral2",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Scruffy baby elf"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 40.02it/s]

    



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

    [2026-03-05 04:35:32] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:35:32] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:35:32] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-05 04:35:34] INFO server_args.py:2038: Attention backend not specified. Use fa3 backend by default.
    [2026-03-05 04:35:34] INFO server_args.py:3129: Set soft_watchdog_timeout since in CI


    [2026-03-05 04:35:34] INFO utils.py:452: Successfully reserved port 35704 on host '0.0.0.0'


    [2026-03-05 04:35:36] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-05 04:35:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:35:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:35:38] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-05 04:35:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-05 04:35:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-05 04:35:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-05 04:35:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-05 04:35:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-05 04:35:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.54it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.38it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.29it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.33it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.71it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.53it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=61.44 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=61.44 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.71it/s]Capturing batches (bs=2 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.71it/s]Capturing batches (bs=1 avail_mem=61.42 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.71it/s]Capturing batches (bs=1 avail_mem=61.42 GB): 100%|██████████| 3/3 [00:00<00:00,  4.71it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:35704



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-05 04:35:56] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image depicts a man engaged in the activity of ironing clothes while standing next to the open back of a yellow taxi. The taxi has a pole with an ironing board attached to it, which is ingeniously used for the man to iron garments. The scene takes place on a street, with other taxis and a yellow sports car visible in the background, and a cityscape at the back. The setting appears to be in a metropolitan area, possibly during daytime.</strong>



```python
terminate_process(server_process)
```
