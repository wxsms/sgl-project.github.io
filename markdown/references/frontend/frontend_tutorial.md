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

    [2026-03-16 17:29:44] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-16 17:29:44] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-16 17:29:44] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-16 17:29:49] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:29:49] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:29:49] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-16 17:29:51] INFO server_args.py:2155: Attention backend not specified. Use fa3 backend by default.
    [2026-03-16 17:29:51] INFO server_args.py:3311: Set soft_watchdog_timeout since in CI


    [2026-03-16 17:29:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:29:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:29:56] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-16 17:29:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:29:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:29:56] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-16 17:30:00] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-16 17:30:00] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-16 17:30:00] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.46it/s]


    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.29it/s]


    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.27it/s]


    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.24it/s]
    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.27it/s]
    


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_dynamo/variables/functions.py:1692: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS="+dynamo" for a DEBUG stack trace.
      torch._dynamo.utils.warn_once(msg)


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:25,  2.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:25,  2.55s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:13,  1.31s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:13,  1.31s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:48,  1.14it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:48,  1.14it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:34,  1.55it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:26,  2.03it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:26,  2.03it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:20,  2.54it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:16,  3.12it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:16,  3.12it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:13,  3.72it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:13,  3.72it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:11,  4.41it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:11,  4.41it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:09,  5.15it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:09,  5.15it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.05it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.05it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:07,  6.05it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  7.94it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  7.94it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  7.94it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.04it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.04it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.04it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.86it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.86it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.86it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 11.08it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 11.08it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 11.08it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 12.72it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 12.72it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:02, 15.73it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 20.25it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 20.25it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 24.83it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.80it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:00, 29.80it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:00, 29.80it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:00, 29.80it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:00, 29.80it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 35.79it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:06<00:00, 40.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=59.23 GB):   2%|▏         | 1/58 [00:00<00:20,  2.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.19 GB):   2%|▏         | 1/58 [00:00<00:20,  2.75it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=59.19 GB):   3%|▎         | 2/58 [00:00<00:19,  2.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.17 GB):   3%|▎         | 2/58 [00:00<00:19,  2.94it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=59.17 GB):   5%|▌         | 3/58 [00:00<00:17,  3.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.16 GB):   5%|▌         | 3/58 [00:00<00:17,  3.14it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.16 GB):   7%|▋         | 4/58 [00:01<00:15,  3.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.16 GB):   7%|▋         | 4/58 [00:01<00:15,  3.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.16 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.15 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=59.15 GB):  10%|█         | 6/58 [00:01<00:13,  3.84it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.14 GB):  10%|█         | 6/58 [00:01<00:13,  3.84it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.14 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.14 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.26it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=59.14 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.14 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.87it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.14 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.15 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.45it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.15 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.15 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.68it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.15 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.68it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=59.15 GB):  21%|██        | 12/58 [00:02<00:06,  7.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.14 GB):  21%|██        | 12/58 [00:02<00:06,  7.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.15 GB):  21%|██        | 12/58 [00:02<00:06,  7.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.15 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.15 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=59.15 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.14 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.15 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.15 GB):  31%|███       | 18/58 [00:02<00:03, 11.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.14 GB):  31%|███       | 18/58 [00:02<00:03, 11.82it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=59.14 GB):  31%|███       | 18/58 [00:03<00:03, 11.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.14 GB):  31%|███       | 18/58 [00:03<00:03, 11.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.14 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.96it/s]Capturing num tokens (num_tokens=960 avail_mem=59.14 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.96it/s] Capturing num tokens (num_tokens=896 avail_mem=59.13 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.96it/s]Capturing num tokens (num_tokens=832 avail_mem=59.13 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.96it/s]Capturing num tokens (num_tokens=832 avail_mem=59.13 GB):  41%|████▏     | 24/58 [00:03<00:01, 18.13it/s]Capturing num tokens (num_tokens=768 avail_mem=59.13 GB):  41%|████▏     | 24/58 [00:03<00:01, 18.13it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.12 GB):  41%|████▏     | 24/58 [00:03<00:01, 18.13it/s]Capturing num tokens (num_tokens=640 avail_mem=59.12 GB):  41%|████▏     | 24/58 [00:03<00:01, 18.13it/s]Capturing num tokens (num_tokens=576 avail_mem=59.12 GB):  41%|████▏     | 24/58 [00:03<00:01, 18.13it/s]Capturing num tokens (num_tokens=576 avail_mem=59.12 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.97it/s]Capturing num tokens (num_tokens=512 avail_mem=59.11 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.97it/s]Capturing num tokens (num_tokens=480 avail_mem=59.11 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.97it/s]Capturing num tokens (num_tokens=448 avail_mem=59.11 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.97it/s]

    Capturing num tokens (num_tokens=448 avail_mem=59.11 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.99it/s]Capturing num tokens (num_tokens=416 avail_mem=59.10 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.99it/s]Capturing num tokens (num_tokens=384 avail_mem=59.10 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.99it/s]Capturing num tokens (num_tokens=352 avail_mem=59.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.99it/s]Capturing num tokens (num_tokens=320 avail_mem=59.09 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.99it/s]Capturing num tokens (num_tokens=320 avail_mem=59.09 GB):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Capturing num tokens (num_tokens=288 avail_mem=59.09 GB):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Capturing num tokens (num_tokens=256 avail_mem=59.08 GB):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Capturing num tokens (num_tokens=240 avail_mem=59.08 GB):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Capturing num tokens (num_tokens=224 avail_mem=59.07 GB):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]

    Capturing num tokens (num_tokens=224 avail_mem=59.07 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=208 avail_mem=59.07 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=192 avail_mem=59.07 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=176 avail_mem=59.06 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=160 avail_mem=59.06 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=144 avail_mem=59.05 GB):  67%|██████▋   | 39/58 [00:03<00:00, 28.89it/s]Capturing num tokens (num_tokens=144 avail_mem=59.05 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s]Capturing num tokens (num_tokens=128 avail_mem=59.06 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s]Capturing num tokens (num_tokens=112 avail_mem=59.06 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s]Capturing num tokens (num_tokens=96 avail_mem=59.06 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s] Capturing num tokens (num_tokens=80 avail_mem=59.05 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s]

    Capturing num tokens (num_tokens=64 avail_mem=59.05 GB):  76%|███████▌  | 44/58 [00:03<00:00, 32.34it/s]Capturing num tokens (num_tokens=64 avail_mem=59.05 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.01it/s]Capturing num tokens (num_tokens=48 avail_mem=59.05 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.01it/s]Capturing num tokens (num_tokens=32 avail_mem=59.04 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.01it/s]Capturing num tokens (num_tokens=28 avail_mem=59.04 GB):  84%|████████▍ | 49/58 [00:03<00:00, 35.01it/s]Capturing num tokens (num_tokens=24 avail_mem=59.04 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.01it/s]Capturing num tokens (num_tokens=20 avail_mem=59.03 GB):  84%|████████▍ | 49/58 [00:04<00:00, 35.01it/s]Capturing num tokens (num_tokens=20 avail_mem=59.03 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.31it/s]Capturing num tokens (num_tokens=16 avail_mem=59.03 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.31it/s]Capturing num tokens (num_tokens=12 avail_mem=59.02 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.31it/s]Capturing num tokens (num_tokens=8 avail_mem=58.99 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.31it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.99 GB):  93%|█████████▎| 54/58 [00:04<00:00, 37.31it/s]Capturing num tokens (num_tokens=4 avail_mem=58.99 GB): 100%|██████████| 58/58 [00:04<00:00, 13.99it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32276


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-16 17:30:21] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries with their respective capitals:<br><br>1. **France** - Paris<br>2. **Spain** - Madrid<br>3. **Italy** - Rome</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries along with their respective capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Mexico** - Mexico City</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries along with their respective capitals:<br><br>1. **India** - New Delhi<br>2. **Brazil** - Brasília<br>3. **Canada** - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>Let's calculate:<br><br>\[ 2 * 2 = 4 \]<br><br>Therefore, 2 * 2 equals 4. There was no need to use a calculator for this simple multiplication.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Eating a balanced diet is crucial for maintaining good health. It involves consuming a variety of nutrient-rich foods in appropriate proportions to ensure your body gets essential vitamins, minerals, proteins, carbohydrates, and fats. A balanced diet typically includes fruits, vegetables (a variety of colors to ensure a broad spectrum of nutrients), whole grains, lean proteins, and healthy fats. It also means limiting the intake of processed foods, sugary drinks, and foods high in saturated and trans fats. Consistently meeting the nutritional needs of your body can help in weight management, boost your immune system, prevent chronic diseases, and improve overall well-being.<br><br>2. **Regular Exercise**: Engaging in regular physical activity is essential for maintaining good health and well-being. It helps you stay fit, manage weight, reduce the risk of chronic diseases, and improve your mood. Whether you prefer aerobic exercises, strength training, or other forms of physical activity, consistency is key. Aim for at least 150 minutes of moderately intense aerobic exercise or 75 minutes of vigorous aerobic exercise per week, and incorporate strength training exercises at least two days a week. Even small amounts of physical activity can have a positive impact, so start with activities that you enjoy and can manage.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Oak",<br>        "core": "Phoenix Feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "stag",<br>    "bogart": "The Dementor"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 34.94it/s]

    



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

    [2026-03-16 17:30:31] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:30:31] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:30:31] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-16 17:30:33] INFO server_args.py:2155: Attention backend not specified. Use fa3 backend by default.
    [2026-03-16 17:30:33] INFO server_args.py:3311: Set soft_watchdog_timeout since in CI


    [2026-03-16 17:30:35] Ignore import error when loading sglang.srt.multimodal.processors.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-03-16 17:30:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:30:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:30:38] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-16 17:30:38] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-16 17:30:38] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-16 17:30:38] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-03-16 17:30:43] Ignore import error when loading sglang.srt.models.glm_ocr: No module named 'transformers.models.glm_ocr'
    [2026-03-16 17:30:43] Ignore import error when loading sglang.srt.models.glm_ocr_nextn: No module named 'transformers.models.glm_ocr'
    [2026-03-16 17:30:43] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]


    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.50it/s]


    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.29it/s]


    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:01<00:01,  1.72it/s]


    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:02<00:00,  1.59it/s]


    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.38it/s]
    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.44it/s]
    


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:35997



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-16 17:30:54] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image shows a man standing on the back of a moving car in what appears to be a busy urban street. He is using a portable ironing board to iron a shirt. There is another yellow taxi visible in the background on the left side of the image. The setting suggests this could be happening in a city, possibly in New York given the style of the taxi and the building architecture.</strong>



```python
terminate_process(server_process)
```
