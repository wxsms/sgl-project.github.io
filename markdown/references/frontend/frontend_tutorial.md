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

    [2026-03-04 17:25:07] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-04 17:25:07] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-04 17:25:07] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-04 17:25:11] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:25:11] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:25:11] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-04 17:25:13] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.
    [2026-03-04 17:25:13] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 17:25:14] INFO utils.py:452: Successfully reserved port 38596 on host '0.0.0.0'


    [2026-03-04 17:25:18] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:25:18] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:25:18] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-04 17:25:18] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:25:18] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:25:18] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-04 17:25:23] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-04 17:25:23] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-04 17:25:23] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.59it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.34it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.32it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.29it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.32it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=38.65 GB):   0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=38.65 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.46it/s]Capturing batches (bs=2 avail_mem=38.59 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.46it/s]

    Capturing batches (bs=1 avail_mem=38.59 GB):  33%|███▎      | 1/3 [00:00<00:00,  5.46it/s]Capturing batches (bs=1 avail_mem=38.59 GB): 100%|██████████| 3/3 [00:00<00:00, 13.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:16,  1.37s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:16,  1.37s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:51,  1.06it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:51,  1.06it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:39,  1.38it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:39,  1.38it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.87it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:21,  2.47it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:21,  2.47it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.16it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.16it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  3.93it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  3.93it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.28it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.39it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:10,  4.52it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:10,  4.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  4.74it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  4.74it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:08,  5.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:08,  5.42it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:08,  5.42it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  7.65it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04, 10.01it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04, 10.01it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04, 10.01it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04, 10.01it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 13.68it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 13.68it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 13.68it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:01, 17.81it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 17.89it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 17.89it/s]

    Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 17.89it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:01, 17.94it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:01, 17.94it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:01, 17.94it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:01, 17.94it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:01, 19.67it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:01, 19.67it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:01, 19.67it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:01, 19.67it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:06<00:01, 20.17it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:06<00:01, 20.17it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:01, 20.17it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:01, 20.17it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 21.25it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:00, 30.59it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 42.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.11 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.08 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.08 GB):   3%|▎         | 2/58 [00:00<00:26,  2.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.08 GB):   3%|▎         | 2/58 [00:00<00:26,  2.14it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.08 GB):   5%|▌         | 3/58 [00:01<00:20,  2.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.92 GB):   5%|▌         | 3/58 [00:01<00:20,  2.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.92 GB):   7%|▋         | 4/58 [00:01<00:19,  2.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.35 GB):   7%|▋         | 4/58 [00:01<00:19,  2.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.35 GB):   9%|▊         | 5/58 [00:02<00:20,  2.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.35 GB):   9%|▊         | 5/58 [00:02<00:20,  2.59it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.35 GB):  10%|█         | 6/58 [00:02<00:18,  2.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.35 GB):  10%|█         | 6/58 [00:02<00:18,  2.74it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.35 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.63 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.63 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.63 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.75it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.63 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.18it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  17%|█▋        | 10/58 [00:02<00:09,  5.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  19%|█▉        | 11/58 [00:03<00:07,  5.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.64 GB):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.63 GB):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:03<00:05,  8.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.64 GB):  24%|██▍       | 14/58 [00:03<00:05,  8.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:03<00:05,  8.12it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.64 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.64 GB):  31%|███       | 18/58 [00:03<00:03, 11.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.63 GB):  31%|███       | 18/58 [00:03<00:03, 11.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.63 GB):  31%|███       | 18/58 [00:03<00:03, 11.39it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.63 GB):  31%|███       | 18/58 [00:03<00:03, 11.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.63 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.50it/s]Capturing num tokens (num_tokens=960 avail_mem=61.63 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.50it/s] Capturing num tokens (num_tokens=896 avail_mem=61.63 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.50it/s]Capturing num tokens (num_tokens=832 avail_mem=61.62 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.50it/s]Capturing num tokens (num_tokens=832 avail_mem=61.62 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=768 avail_mem=61.62 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=704 avail_mem=61.61 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]

    Capturing num tokens (num_tokens=640 avail_mem=61.61 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.63it/s]Capturing num tokens (num_tokens=576 avail_mem=61.60 GB):  41%|████▏     | 24/58 [00:04<00:01, 17.63it/s]Capturing num tokens (num_tokens=576 avail_mem=61.60 GB):  48%|████▊     | 28/58 [00:04<00:01, 21.41it/s]Capturing num tokens (num_tokens=512 avail_mem=61.60 GB):  48%|████▊     | 28/58 [00:04<00:01, 21.41it/s]Capturing num tokens (num_tokens=480 avail_mem=61.60 GB):  48%|████▊     | 28/58 [00:04<00:01, 21.41it/s]Capturing num tokens (num_tokens=448 avail_mem=61.59 GB):  48%|████▊     | 28/58 [00:04<00:01, 21.41it/s]Capturing num tokens (num_tokens=416 avail_mem=61.59 GB):  48%|████▊     | 28/58 [00:04<00:01, 21.41it/s]Capturing num tokens (num_tokens=416 avail_mem=61.59 GB):  55%|█████▌    | 32/58 [00:04<00:01, 24.55it/s]Capturing num tokens (num_tokens=384 avail_mem=61.59 GB):  55%|█████▌    | 32/58 [00:04<00:01, 24.55it/s]

    Capturing num tokens (num_tokens=352 avail_mem=61.59 GB):  55%|█████▌    | 32/58 [00:04<00:01, 24.55it/s]Capturing num tokens (num_tokens=320 avail_mem=61.58 GB):  55%|█████▌    | 32/58 [00:04<00:01, 24.55it/s]Capturing num tokens (num_tokens=320 avail_mem=61.58 GB):  60%|██████    | 35/58 [00:04<00:00, 25.56it/s]Capturing num tokens (num_tokens=288 avail_mem=61.58 GB):  60%|██████    | 35/58 [00:04<00:00, 25.56it/s]Capturing num tokens (num_tokens=256 avail_mem=61.57 GB):  60%|██████    | 35/58 [00:04<00:00, 25.56it/s]

    Capturing num tokens (num_tokens=240 avail_mem=61.57 GB):  60%|██████    | 35/58 [00:04<00:00, 25.56it/s]Capturing num tokens (num_tokens=240 avail_mem=61.57 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.20it/s]Capturing num tokens (num_tokens=224 avail_mem=61.57 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.20it/s]Capturing num tokens (num_tokens=208 avail_mem=61.56 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.20it/s]Capturing num tokens (num_tokens=192 avail_mem=61.56 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.20it/s]Capturing num tokens (num_tokens=192 avail_mem=61.56 GB):  71%|███████   | 41/58 [00:04<00:00, 22.96it/s]Capturing num tokens (num_tokens=176 avail_mem=61.55 GB):  71%|███████   | 41/58 [00:04<00:00, 22.96it/s]Capturing num tokens (num_tokens=160 avail_mem=61.55 GB):  71%|███████   | 41/58 [00:04<00:00, 22.96it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.55 GB):  71%|███████   | 41/58 [00:04<00:00, 22.96it/s]Capturing num tokens (num_tokens=128 avail_mem=61.56 GB):  71%|███████   | 41/58 [00:04<00:00, 22.96it/s]Capturing num tokens (num_tokens=128 avail_mem=61.56 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.11it/s]Capturing num tokens (num_tokens=112 avail_mem=61.55 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.11it/s]Capturing num tokens (num_tokens=96 avail_mem=61.55 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.11it/s] Capturing num tokens (num_tokens=80 avail_mem=61.54 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.11it/s]Capturing num tokens (num_tokens=64 avail_mem=61.54 GB):  78%|███████▊  | 45/58 [00:04<00:00, 25.11it/s]

    Capturing num tokens (num_tokens=64 avail_mem=61.54 GB):  84%|████████▍ | 49/58 [00:04<00:00, 27.03it/s]Capturing num tokens (num_tokens=48 avail_mem=61.54 GB):  84%|████████▍ | 49/58 [00:04<00:00, 27.03it/s]Capturing num tokens (num_tokens=32 avail_mem=61.53 GB):  84%|████████▍ | 49/58 [00:04<00:00, 27.03it/s]Capturing num tokens (num_tokens=28 avail_mem=61.53 GB):  84%|████████▍ | 49/58 [00:04<00:00, 27.03it/s]Capturing num tokens (num_tokens=24 avail_mem=61.53 GB):  84%|████████▍ | 49/58 [00:04<00:00, 27.03it/s]Capturing num tokens (num_tokens=24 avail_mem=61.53 GB):  91%|█████████▏| 53/58 [00:04<00:00, 28.37it/s]Capturing num tokens (num_tokens=20 avail_mem=61.53 GB):  91%|█████████▏| 53/58 [00:04<00:00, 28.37it/s]Capturing num tokens (num_tokens=16 avail_mem=61.52 GB):  91%|█████████▏| 53/58 [00:04<00:00, 28.37it/s]Capturing num tokens (num_tokens=12 avail_mem=61.52 GB):  91%|█████████▏| 53/58 [00:05<00:00, 28.37it/s]

    Capturing num tokens (num_tokens=12 avail_mem=61.52 GB):  97%|█████████▋| 56/58 [00:05<00:00, 27.23it/s]Capturing num tokens (num_tokens=8 avail_mem=61.51 GB):  97%|█████████▋| 56/58 [00:05<00:00, 27.23it/s] Capturing num tokens (num_tokens=4 avail_mem=61.51 GB):  97%|█████████▋| 56/58 [00:05<00:00, 27.23it/s]Capturing num tokens (num_tokens=4 avail_mem=61.51 GB): 100%|██████████| 58/58 [00:05<00:00, 11.33it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:38596


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-04 17:25:45] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Canada - Ottawa</strong>


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their respective capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **India** - New Delhi</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries and their capitals:<br><br>1. **Japan** - Tokyo<br>2. **Brazil** - Brasília<br>3. **Canada** - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>Let's solve this without needing a calculator.<br>2 * 2 equals 4.<br><br>So, the answer to 2 * 2 is 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: A balanced diet is essential for maintaining overall health. It involves consuming a variety of nutrient-rich foods, including:<br>   - Fruits and vegetables<br>   - Whole grains<br>   - Lean proteins (fish, poultry, beans, legumes)<br>   - Healthy fats (avocados, nuts, olive oil)<br>   - Staying hydrated by drinking plenty of water<br><br>2. **Regular Exercise**: Regular physical activity is vital for several reasons:<br>   - Maintaining a healthy weight<br>   - Improving cardiovascular health<br>   - Strengthening muscles and bones<br>   - Enhancing flexibility<br>   - Boosting immune function<br>   - Reducing stress, anxiety, and depression<br>   - Improving sleep and cognitive function<br>   - Increasing energy levels<br><br>By incorporating these lifestyle changes, you can significantly improve your overall health and well-being.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Phoenix feather",<br>        "length": 10.4<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Hubert Pringle"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 39.88it/s]

    



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

    [2026-03-04 17:25:55] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:25:55] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:25:55] INFO utils.py:164: NumExpr defaulting to 16 threads.


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:41: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-04 17:25:57] INFO server_args.py:1975: Attention backend not specified. Use fa3 backend by default.
    [2026-03-04 17:25:57] INFO server_args.py:3066: Set soft_watchdog_timeout since in CI


    [2026-03-04 17:25:57] INFO utils.py:452: Successfully reserved port 39302 on host '0.0.0.0'


    [2026-03-04 17:25:59] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-04 17:26:01] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:26:01] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:26:01] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-04 17:26:01] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-04 17:26:01] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-04 17:26:01] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-04 17:26:07] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-04 17:26:07] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-04 17:26:07] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.69it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:01,  1.56it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.36it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.34it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.73it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.58it/s]
    


      0%|          | 0/3 [00:00<?, ?it/s]Capturing batches (bs=4 avail_mem=59.31 GB):   0%|          | 0/3 [00:00<?, ?it/s]

    Capturing batches (bs=4 avail_mem=59.31 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.05it/s]Capturing batches (bs=2 avail_mem=58.99 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.05it/s]Capturing batches (bs=1 avail_mem=58.99 GB):  33%|███▎      | 1/3 [00:00<00:00,  2.05it/s]Capturing batches (bs=1 avail_mem=58.99 GB): 100%|██████████| 3/3 [00:00<00:00,  5.37it/s]



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:39302



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-04 17:26:19] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image shows a person standing on the back of a taxi van, ironing a shirt on an ironing board mounted on a portable frame. The taxi van is parked on a city street with other yellow taxis and a street scene visible in the background, including buildings, a pedestrian crossing, and some greenery. The person appears to be balancing carefully while ironing, which is an unusual and challenging task given the location. This scene likely depicts an effort to multitask, potentially as part of a mock scenario or a stunt.</strong>



```python
terminate_process(server_process)
```
