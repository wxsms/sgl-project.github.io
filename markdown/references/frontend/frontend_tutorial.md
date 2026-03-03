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

    [2026-03-03 08:17:42] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-03 08:17:42] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-03 08:17:42] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-03 08:17:47] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:17:47] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:17:47] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 08:17:49] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 08:17:49] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 08:17:55] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:17:55] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:17:55] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 08:17:55] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:17:55] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:17:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 08:18:00] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:18:00] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:18:00] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.30it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.19it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.18it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.22it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.21it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=120.74 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=120.74 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.69it/s]Capturing batches (bs=2 avail_mem=120.68 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.69it/s]Capturing batches (bs=1 avail_mem=120.68 GB):  33%|███▎      | 1/3 [00:00<00:00,  4.69it/s]Capturing batches (bs=1 avail_mem=120.68 GB): 100%|██████████| 3/3 [00:00<00:00, 11.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:03,  3.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:03,  3.21s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:26,  1.55s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.08it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.59it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.59it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:24,  2.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:24,  2.18it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.83it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.83it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.56it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.38it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.38it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.28it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:07,  6.16it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:05,  7.97it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:05,  7.97it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:05,  7.97it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:04,  9.64it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:03, 11.29it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:03, 11.29it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:02, 14.47it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:02, 14.47it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:02, 14.47it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:01, 19.46it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]

    Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 25.83it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:00, 35.75it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 45.24it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 50.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=121.60 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.57 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=121.57 GB):   3%|▎         | 2/58 [00:00<00:15,  3.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.56 GB):   3%|▎         | 2/58 [00:00<00:15,  3.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=121.56 GB):   5%|▌         | 3/58 [00:00<00:14,  3.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=121.57 GB):   5%|▌         | 3/58 [00:00<00:14,  3.91it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=121.57 GB):   7%|▋         | 4/58 [00:00<00:12,  4.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.57 GB):   7%|▋         | 4/58 [00:00<00:12,  4.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=121.57 GB):   9%|▊         | 5/58 [00:01<00:11,  4.44it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=121.57 GB):   9%|▊         | 5/58 [00:01<00:11,  4.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=121.57 GB):  10%|█         | 6/58 [00:01<00:11,  4.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=121.57 GB):  10%|█         | 6/58 [00:01<00:11,  4.68it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=121.57 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.57 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=121.57 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=121.30 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.21it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=121.30 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.60 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.60 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.60 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.30it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.60 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.59 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.59 GB):  21%|██        | 12/58 [00:02<00:06,  7.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.59 GB):  21%|██        | 12/58 [00:02<00:06,  7.32it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.59 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.59 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.16 GB):  22%|██▏       | 13/58 [00:02<00:05,  7.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.16 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.16 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.30it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=120.16 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.16 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.16 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.15 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.83it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.15 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.15 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.68it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.15 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.68it/s]Capturing num tokens (num_tokens=960 avail_mem=120.15 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.68it/s] Capturing num tokens (num_tokens=960 avail_mem=120.15 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=896 avail_mem=120.14 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=832 avail_mem=120.14 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=768 avail_mem=120.13 GB):  38%|███▊      | 22/58 [00:02<00:02, 15.84it/s]Capturing num tokens (num_tokens=768 avail_mem=120.13 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.51it/s]Capturing num tokens (num_tokens=704 avail_mem=120.12 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.51it/s]

    Capturing num tokens (num_tokens=640 avail_mem=120.08 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.51it/s]Capturing num tokens (num_tokens=576 avail_mem=120.08 GB):  43%|████▎     | 25/58 [00:03<00:01, 18.51it/s]Capturing num tokens (num_tokens=576 avail_mem=120.08 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.50it/s]Capturing num tokens (num_tokens=512 avail_mem=120.07 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.50it/s]Capturing num tokens (num_tokens=480 avail_mem=120.07 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.50it/s]Capturing num tokens (num_tokens=448 avail_mem=120.06 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.50it/s]Capturing num tokens (num_tokens=448 avail_mem=120.06 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=416 avail_mem=120.03 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.75it/s]

    Capturing num tokens (num_tokens=384 avail_mem=120.02 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=352 avail_mem=120.01 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=320 avail_mem=120.00 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.75it/s]Capturing num tokens (num_tokens=320 avail_mem=120.00 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=288 avail_mem=120.00 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=256 avail_mem=119.99 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=240 avail_mem=119.99 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]Capturing num tokens (num_tokens=224 avail_mem=119.98 GB):  60%|██████    | 35/58 [00:03<00:00, 25.72it/s]

    Capturing num tokens (num_tokens=224 avail_mem=119.98 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.53it/s]Capturing num tokens (num_tokens=208 avail_mem=119.98 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.53it/s]Capturing num tokens (num_tokens=192 avail_mem=119.97 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.53it/s]Capturing num tokens (num_tokens=176 avail_mem=119.97 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.53it/s]Capturing num tokens (num_tokens=160 avail_mem=119.96 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.53it/s]Capturing num tokens (num_tokens=160 avail_mem=119.96 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.75it/s]Capturing num tokens (num_tokens=144 avail_mem=119.95 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.75it/s]Capturing num tokens (num_tokens=128 avail_mem=119.96 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.75it/s]Capturing num tokens (num_tokens=112 avail_mem=119.96 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.75it/s]Capturing num tokens (num_tokens=96 avail_mem=119.95 GB):  74%|███████▍  | 43/58 [00:03<00:00, 30.75it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=119.95 GB):  81%|████████  | 47/58 [00:03<00:00, 31.91it/s]Capturing num tokens (num_tokens=80 avail_mem=119.94 GB):  81%|████████  | 47/58 [00:03<00:00, 31.91it/s]Capturing num tokens (num_tokens=64 avail_mem=119.94 GB):  81%|████████  | 47/58 [00:03<00:00, 31.91it/s]Capturing num tokens (num_tokens=48 avail_mem=119.93 GB):  81%|████████  | 47/58 [00:03<00:00, 31.91it/s]Capturing num tokens (num_tokens=32 avail_mem=119.93 GB):  81%|████████  | 47/58 [00:03<00:00, 31.91it/s]Capturing num tokens (num_tokens=32 avail_mem=119.93 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.12it/s]Capturing num tokens (num_tokens=28 avail_mem=119.92 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.12it/s]Capturing num tokens (num_tokens=24 avail_mem=119.92 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.12it/s]Capturing num tokens (num_tokens=20 avail_mem=119.91 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.12it/s]Capturing num tokens (num_tokens=16 avail_mem=119.90 GB):  88%|████████▊ | 51/58 [00:03<00:00, 33.12it/s]

    Capturing num tokens (num_tokens=16 avail_mem=119.90 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.92it/s]Capturing num tokens (num_tokens=12 avail_mem=119.90 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.92it/s]Capturing num tokens (num_tokens=8 avail_mem=119.89 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.92it/s] Capturing num tokens (num_tokens=4 avail_mem=119.89 GB):  95%|█████████▍| 55/58 [00:03<00:00, 33.92it/s]Capturing num tokens (num_tokens=4 avail_mem=119.89 GB): 100%|██████████| 58/58 [00:03<00:00, 14.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:35591


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-03 08:18:21] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Brazil - Brasília</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries and their respective capitals:<br><br>1. Japan - Tokyo<br>2. France - Paris<br>3. Spain - Madrid</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries and their capitals:<br><br>1. Italy - Rome<br>2. Canada - Ottawa<br>3. Australia - Canberra</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>Let's solve it:<br><br>2 * 2 = 4<br><br>Therefore, 2 * 2 equals 4.<br><br>You didn't actually need a calculator for this, but I'm happy to verify it for you!</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Consuming a variety of nutrient-rich foods such as fruits, vegetables, lean proteins, whole grains, and healthy fats is essential. Staying hydrated by drinking plenty of water also supports overall health.<br>2. **Regular Exercise**: Engaging in at least 150 to 300 minutes of moderate-intensity aerobic activity or 75 to 150 minutes of vigorous-intensity aerobic activity each week, along with muscle-strengthening activities on two or more days a week, is crucial. This helps maintain a healthy weight, strengthen the cardiovascular system, improve mental health, and enhance immune function.<br><br>Both of these habits are foundational for living a healthy and fulfilling life.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Walnut",<br>        "core": "Callisto hair",<br>        "length": 11.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Lord Voldemort"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 47.43it/s]

    



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

    [2026-03-03 08:18:30] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:18:30] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:18:30] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /root/actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:34: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-03 08:18:32] INFO server_args.py:1967: Attention backend not specified. Use fa3 backend by default.
    [2026-03-03 08:18:32] INFO server_args.py:3039: Set soft_watchdog_timeout since in CI


    [2026-03-03 08:18:35] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-03 08:18:37] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:18:37] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:18:37] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-03 08:18:37] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-03 08:18:37] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-03 08:18:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-03 08:18:44] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:18:44] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-03 08:18:44] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:03,  1.33it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.10it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.54it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.45it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.36it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.36it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=104.58 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=104.58 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.63it/s]Capturing batches (bs=2 avail_mem=104.55 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.63it/s]Capturing batches (bs=1 avail_mem=104.54 GB):  33%|███▎      | 1/3 [00:00<00:01,  1.63it/s]Capturing batches (bs=1 avail_mem=104.54 GB): 100%|██████████| 3/3 [00:00<00:00,  4.45it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34339



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-03 08:18:57] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image shows a person standing on the bed of a yellow SUV van parked in a city street. The person is using crutches to keep their balance as they iron clothes that are hanging on a clothesline fixed to the front bumper of the vehicle. The clothesline is strung between two poles connected to the van, and it holds a blue shirt, among other items. The scene appears to be in an urban environment, with other vehicles, including a yellow taxi, visible in the background. The setting suggests a unique, possibly humorous, approach to laundry for someone possibly unable to stand due to an injury or mobility issue.</strong>



```python
terminate_process(server_process)
```
