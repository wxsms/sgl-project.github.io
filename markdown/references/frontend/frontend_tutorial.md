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

    [2026-03-11 09:46:51] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-11 09:46:51] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-11 09:46:51] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-11 09:46:56] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:46:56] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:46:56] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-11 09:46:58] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.
    [2026-03-11 09:46:58] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-11 09:47:04] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:47:04] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:47:04] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-11 09:47:04] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:47:04] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:47:04] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-11 09:47:08] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-11 09:47:08] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-11 09:47:08] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.91it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.80it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.77it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.77it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.78it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:01,  3.18s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.54s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.54s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.07it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.54it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.54it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:25,  2.07it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:25,  2.07it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:20,  2.59it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:20,  2.59it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.15it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.15it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:13,  3.71it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:13,  3.71it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.32it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.32it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:09,  5.02it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:09,  5.02it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:08,  5.62it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:08,  5.62it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  6.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  6.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:07,  6.37it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:05,  8.57it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:05,  8.57it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:05,  8.57it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:03, 10.60it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:03, 10.60it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:03, 10.60it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:03, 10.60it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:02, 13.97it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:02, 13.97it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:02, 13.97it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:01, 18.87it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 25.12it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 33.61it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 33.61it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:06<00:00, 33.61it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 41.55it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]

    Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:06<00:00, 42.43it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 48.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=119.67 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.63 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=119.63 GB):   3%|▎         | 2/58 [00:00<00:15,  3.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.63 GB):   3%|▎         | 2/58 [00:00<00:15,  3.70it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=119.63 GB):   5%|▌         | 3/58 [00:00<00:15,  3.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.64 GB):   5%|▌         | 3/58 [00:00<00:15,  3.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=119.64 GB):   7%|▋         | 4/58 [00:01<00:17,  3.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.64 GB):   7%|▋         | 4/58 [00:01<00:17,  3.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.10it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.10it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=119.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.64 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.21it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=119.64 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.64 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.64 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.64 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.37it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=119.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.64 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.64 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.64 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.64 GB):  21%|██        | 12/58 [00:02<00:06,  7.65it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=119.64 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.64 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.61 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.61 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.61 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.38it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.61 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.61 GB):  31%|███       | 18/58 [00:02<00:03, 11.09it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.60 GB):  31%|███       | 18/58 [00:02<00:03, 11.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=122.31 GB):  31%|███       | 18/58 [00:02<00:03, 11.09it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=122.31 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=122.31 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.07it/s]Capturing num tokens (num_tokens=960 avail_mem=122.31 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.07it/s] Capturing num tokens (num_tokens=896 avail_mem=122.30 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.07it/s]Capturing num tokens (num_tokens=896 avail_mem=122.30 GB):  40%|███▉      | 23/58 [00:03<00:02, 14.43it/s]Capturing num tokens (num_tokens=832 avail_mem=122.30 GB):  40%|███▉      | 23/58 [00:03<00:02, 14.43it/s]Capturing num tokens (num_tokens=768 avail_mem=122.29 GB):  40%|███▉      | 23/58 [00:03<00:02, 14.43it/s]Capturing num tokens (num_tokens=704 avail_mem=122.28 GB):  40%|███▉      | 23/58 [00:03<00:02, 14.43it/s]

    Capturing num tokens (num_tokens=704 avail_mem=122.28 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.50it/s]Capturing num tokens (num_tokens=640 avail_mem=122.28 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.50it/s]Capturing num tokens (num_tokens=576 avail_mem=122.27 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.50it/s]Capturing num tokens (num_tokens=512 avail_mem=122.27 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.50it/s]Capturing num tokens (num_tokens=480 avail_mem=122.26 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.50it/s]Capturing num tokens (num_tokens=480 avail_mem=122.26 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.31it/s]Capturing num tokens (num_tokens=448 avail_mem=122.26 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.31it/s]Capturing num tokens (num_tokens=416 avail_mem=122.25 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.31it/s]Capturing num tokens (num_tokens=384 avail_mem=122.25 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.31it/s]

    Capturing num tokens (num_tokens=352 avail_mem=122.24 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.31it/s]Capturing num tokens (num_tokens=352 avail_mem=122.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.43it/s]Capturing num tokens (num_tokens=320 avail_mem=122.24 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.43it/s]Capturing num tokens (num_tokens=288 avail_mem=122.23 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.43it/s]Capturing num tokens (num_tokens=256 avail_mem=122.23 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.43it/s]Capturing num tokens (num_tokens=240 avail_mem=122.22 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.43it/s]Capturing num tokens (num_tokens=240 avail_mem=122.22 GB):  66%|██████▌   | 38/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=224 avail_mem=122.21 GB):  66%|██████▌   | 38/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=208 avail_mem=122.21 GB):  66%|██████▌   | 38/58 [00:03<00:00, 27.22it/s]

    Capturing num tokens (num_tokens=192 avail_mem=122.20 GB):  66%|██████▌   | 38/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=176 avail_mem=122.20 GB):  66%|██████▌   | 38/58 [00:03<00:00, 27.22it/s]Capturing num tokens (num_tokens=176 avail_mem=122.20 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.44it/s]Capturing num tokens (num_tokens=160 avail_mem=122.19 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.44it/s]Capturing num tokens (num_tokens=144 avail_mem=122.19 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.44it/s]Capturing num tokens (num_tokens=128 avail_mem=122.19 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.44it/s]Capturing num tokens (num_tokens=112 avail_mem=122.19 GB):  72%|███████▏  | 42/58 [00:03<00:00, 29.44it/s]Capturing num tokens (num_tokens=112 avail_mem=122.19 GB):  79%|███████▉  | 46/58 [00:03<00:00, 31.02it/s]Capturing num tokens (num_tokens=96 avail_mem=122.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 31.02it/s] Capturing num tokens (num_tokens=80 avail_mem=122.18 GB):  79%|███████▉  | 46/58 [00:03<00:00, 31.02it/s]

    Capturing num tokens (num_tokens=64 avail_mem=122.17 GB):  79%|███████▉  | 46/58 [00:03<00:00, 31.02it/s]Capturing num tokens (num_tokens=48 avail_mem=122.17 GB):  79%|███████▉  | 46/58 [00:03<00:00, 31.02it/s]Capturing num tokens (num_tokens=48 avail_mem=122.17 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=32 avail_mem=122.16 GB):  86%|████████▌ | 50/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=28 avail_mem=122.16 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.18it/s]Capturing num tokens (num_tokens=24 avail_mem=122.15 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.18it/s]Capturing num tokens (num_tokens=20 avail_mem=122.14 GB):  86%|████████▌ | 50/58 [00:04<00:00, 32.18it/s]Capturing num tokens (num_tokens=20 avail_mem=122.14 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.98it/s]Capturing num tokens (num_tokens=16 avail_mem=122.14 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.98it/s]

    Capturing num tokens (num_tokens=12 avail_mem=122.13 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.98it/s]Capturing num tokens (num_tokens=8 avail_mem=122.13 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.98it/s] Capturing num tokens (num_tokens=4 avail_mem=122.12 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.98it/s]Capturing num tokens (num_tokens=4 avail_mem=122.12 GB): 100%|██████████| 58/58 [00:04<00:00, 33.61it/s]Capturing num tokens (num_tokens=4 avail_mem=122.12 GB): 100%|██████████| 58/58 [00:04<00:00, 13.75it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:36998


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-11 09:47:29] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries and their capitals:<br><br>1. **France** - Paris<br>2. **Brazil** - Brasília<br>3. **Japan** - Tokyo</strong>


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


<strong style='color: #00008B;'>Sure! Here is a list of three countries and their respective capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Italy** - Rome</strong>



<strong style='color: #00008B;'>Certainly! Here is another list of three countries and their capitals:<br><br>1. **Spain** - Madrid<br>2. **India** - New Delhi<br>3. **Canada** - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>You don't need a calculator for this one! The answer is 4.<br><br>Still, here is a quick confirmation using a simple calculation:<br><br>2 * 2 = 4</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet:**<br>   - Eat a variety of nutrient-rich foods including fruits, vegetables, whole grains, lean proteins, and healthy fats.<br>   - Limit the intake of processed foods, sugars, and saturated fats.<br>   - Regular meal planning and mindful eating can help maintain a balanced diet.<br><br>2. **Regular Exercise:**<br>   - Engage in physical activities that you enjoy, such as walking, jogging, cycling, or swimming.<br>   - Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, along with muscle-strengthening activities on two or more days.<br>   - Incorporate exercise into your daily routine to reap multiple health benefits.<br><br>Both are essential for maintaining good health and improving quality of life.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "A spider"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 27.98it/s]

    100%|██████████| 3/3 [00:00<00:00, 27.77it/s]

    



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

    [2026-03-11 09:47:40] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:47:40] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:47:40] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-11 09:47:42] INFO server_args.py:2140: Attention backend not specified. Use fa3 backend by default.
    [2026-03-11 09:47:42] INFO server_args.py:3279: Set soft_watchdog_timeout since in CI


    [2026-03-11 09:47:45] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-11 09:47:48] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:47:48] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:47:48] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-11 09:47:48] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-11 09:47:48] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-11 09:47:48] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-11 09:47:54] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-11 09:47:54] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-11 09:47:54] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.34it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.08it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.49it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:03<00:00,  1.35it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.30it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.30it/s]
    


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:37223



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-11 09:48:07] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>The image shows a man on the back of a yellow taxi parked on a street. He appears to be ironing a pair of blue pants on an ironing board, which is attached to the rear of the taxi. The ironing board is positioned using a few sticks, and a few buildings and other urban elements are visible in the background. The overall scene appears to be an urban street setting, possibly in a city.</strong>



```python
terminate_process(server_process)
```
