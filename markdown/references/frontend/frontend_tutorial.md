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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    2026-04-08 06:50:56.754 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:50:56] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:50:56.754 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:50:56] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:50:56.754 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:50:56] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:50:56.754 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:50:56] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:50:56.755 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:50:56] Persistent cache disabled, using in-memory JIT cache


    [2026-04-08 06:50:58] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:50:58] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:50:58] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:50:58] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  2.24it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.80it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.66it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.61it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.67it/s]


    2026-04-08 06:51:01,620 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:51:01] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.08s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.50s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.80it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.47it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.47it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:12,  4.16it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:12,  4.16it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  4.97it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  4.97it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  4.97it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:07,  6.59it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:07,  6.59it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:07,  6.59it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.15it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.15it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.15it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:04,  9.74it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:04,  9.74it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04,  9.74it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 11.61it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 11.61it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.61it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.61it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 14.95it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 14.95it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 14.95it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 14.95it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 14.95it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.06it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]

    Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.60it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 39.17it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 42.35it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 46.42it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 46.42it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 46.42it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 46.42it/s]

    Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:06<00:00, 46.42it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:06<00:00, 46.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=101.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=101.33 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]Capturing num tokens (num_tokens=7680 avail_mem=101.30 GB):   2%|▏         | 1/58 [00:00<00:16,  3.55it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=101.30 GB):   3%|▎         | 2/58 [00:00<00:15,  3.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=101.30 GB):   3%|▎         | 2/58 [00:00<00:15,  3.66it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=101.30 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=101.30 GB):   5%|▌         | 3/58 [00:00<00:14,  3.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=101.30 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=101.30 GB):   7%|▋         | 4/58 [00:01<00:13,  4.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=101.30 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=101.31 GB):   9%|▊         | 5/58 [00:01<00:12,  4.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=101.31 GB):  10%|█         | 6/58 [00:01<00:10,  4.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=101.31 GB):  10%|█         | 6/58 [00:01<00:10,  4.75it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=101.31 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=101.31 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=101.31 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=101.32 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=101.32 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=101.32 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.75it/s]Capturing num tokens (num_tokens=3840 avail_mem=101.32 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=101.32 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.27it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=101.32 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=101.32 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=101.32 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=101.32 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=101.32 GB):  21%|██        | 12/58 [00:02<00:06,  7.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=101.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=101.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=101.32 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.68it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=101.32 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=101.32 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=101.32 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=101.32 GB):  31%|███       | 18/58 [00:02<00:04,  8.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=101.31 GB):  31%|███       | 18/58 [00:02<00:04,  8.66it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=101.31 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=101.32 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=101.32 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=101.32 GB):  36%|███▌      | 21/58 [00:03<00:04,  8.84it/s]Capturing num tokens (num_tokens=960 avail_mem=101.31 GB):  36%|███▌      | 21/58 [00:03<00:04,  8.84it/s] Capturing num tokens (num_tokens=896 avail_mem=101.31 GB):  36%|███▌      | 21/58 [00:03<00:04,  8.84it/s]Capturing num tokens (num_tokens=832 avail_mem=101.30 GB):  36%|███▌      | 21/58 [00:03<00:04,  8.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=101.30 GB):  41%|████▏     | 24/58 [00:03<00:02, 12.50it/s]Capturing num tokens (num_tokens=768 avail_mem=101.30 GB):  41%|████▏     | 24/58 [00:03<00:02, 12.50it/s]Capturing num tokens (num_tokens=704 avail_mem=101.29 GB):  41%|████▏     | 24/58 [00:03<00:02, 12.50it/s]Capturing num tokens (num_tokens=640 avail_mem=101.29 GB):  41%|████▏     | 24/58 [00:03<00:02, 12.50it/s]Capturing num tokens (num_tokens=640 avail_mem=101.29 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=576 avail_mem=101.29 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=512 avail_mem=101.28 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.93it/s]Capturing num tokens (num_tokens=480 avail_mem=101.28 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.93it/s]

    Capturing num tokens (num_tokens=480 avail_mem=101.28 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.48it/s]Capturing num tokens (num_tokens=448 avail_mem=101.28 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.48it/s]Capturing num tokens (num_tokens=416 avail_mem=101.28 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.48it/s]Capturing num tokens (num_tokens=384 avail_mem=101.27 GB):  52%|█████▏    | 30/58 [00:03<00:01, 18.48it/s]Capturing num tokens (num_tokens=384 avail_mem=101.27 GB):  57%|█████▋    | 33/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=352 avail_mem=101.26 GB):  57%|█████▋    | 33/58 [00:03<00:01, 17.21it/s]

    Capturing num tokens (num_tokens=320 avail_mem=101.26 GB):  57%|█████▋    | 33/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=288 avail_mem=101.26 GB):  57%|█████▋    | 33/58 [00:03<00:01, 17.21it/s]Capturing num tokens (num_tokens=288 avail_mem=101.26 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.75it/s]Capturing num tokens (num_tokens=256 avail_mem=101.25 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.75it/s]Capturing num tokens (num_tokens=240 avail_mem=101.25 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.75it/s]

    Capturing num tokens (num_tokens=224 avail_mem=101.24 GB):  62%|██████▏   | 36/58 [00:03<00:01, 16.75it/s]Capturing num tokens (num_tokens=208 avail_mem=101.24 GB):  62%|██████▏   | 36/58 [00:04<00:01, 16.75it/s]Capturing num tokens (num_tokens=208 avail_mem=101.24 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.62it/s]Capturing num tokens (num_tokens=192 avail_mem=101.24 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.62it/s]Capturing num tokens (num_tokens=176 avail_mem=101.23 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.62it/s]Capturing num tokens (num_tokens=160 avail_mem=101.23 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.62it/s]Capturing num tokens (num_tokens=144 avail_mem=101.22 GB):  69%|██████▉   | 40/58 [00:04<00:00, 20.62it/s]Capturing num tokens (num_tokens=144 avail_mem=101.22 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.78it/s]Capturing num tokens (num_tokens=128 avail_mem=101.23 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.78it/s]

    Capturing num tokens (num_tokens=112 avail_mem=101.23 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.78it/s]Capturing num tokens (num_tokens=96 avail_mem=101.23 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.78it/s] Capturing num tokens (num_tokens=80 avail_mem=101.19 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.78it/s]Capturing num tokens (num_tokens=80 avail_mem=101.19 GB):  83%|████████▎ | 48/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=64 avail_mem=101.19 GB):  83%|████████▎ | 48/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=48 avail_mem=101.19 GB):  83%|████████▎ | 48/58 [00:04<00:00, 26.61it/s]

    Capturing num tokens (num_tokens=32 avail_mem=101.18 GB):  83%|████████▎ | 48/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=32 avail_mem=101.18 GB):  88%|████████▊ | 51/58 [00:04<00:00, 24.65it/s]Capturing num tokens (num_tokens=28 avail_mem=101.18 GB):  88%|████████▊ | 51/58 [00:04<00:00, 24.65it/s]Capturing num tokens (num_tokens=24 avail_mem=101.18 GB):  88%|████████▊ | 51/58 [00:04<00:00, 24.65it/s]Capturing num tokens (num_tokens=20 avail_mem=101.17 GB):  88%|████████▊ | 51/58 [00:04<00:00, 24.65it/s]Capturing num tokens (num_tokens=16 avail_mem=101.17 GB):  88%|████████▊ | 51/58 [00:04<00:00, 24.65it/s]Capturing num tokens (num_tokens=16 avail_mem=101.17 GB):  95%|█████████▍| 55/58 [00:04<00:00, 26.96it/s]Capturing num tokens (num_tokens=12 avail_mem=101.16 GB):  95%|█████████▍| 55/58 [00:04<00:00, 26.96it/s]Capturing num tokens (num_tokens=8 avail_mem=101.16 GB):  95%|█████████▍| 55/58 [00:04<00:00, 26.96it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=101.16 GB):  95%|█████████▍| 55/58 [00:04<00:00, 26.96it/s]Capturing num tokens (num_tokens=4 avail_mem=101.16 GB): 100%|██████████| 58/58 [00:04<00:00, 12.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:30210


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-08 06:51:20] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Japan** - Tokyo</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Japan** - Tokyo</strong>



<strong style='color: #00008B;'>Sure! Here is another list of three countries and their capitals:<br><br>1. **Italy** - Rome<br>2. **Spain** - Madrid<br>3. **Canada** - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2<br><br>The solution is straightforward multiplication:<br>2 * 2 = 4<br><br>Therefore, the answer is 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: A balanced diet is essential for maintaining overall health. It includes a variety of foods from all food groups, such as fruits, vegetables, whole grains, lean proteins, and healthy fats. Consistent consumption of fruits and vegetables provides essential vitamins, minerals, and fiber. Whole grains offer sustained energy and benefits for digestive health. Lean proteins are vital for muscle repair and growth, while healthy fats support brain health. Additionally, staying hydrated and avoiding processed foods and excessive sugars are important for preventing health issues such as obesity, heart disease, and diabetes.<br>2. **Regular Exercise**: Regular exercise is crucial for maintaining physical and mental well-being. It helps strengthen muscles and bones, improve cardiovascular and respiratory health, and boost energy levels. Engage in activities you enjoy, such as walking, jogging, cycling, or joining a fitness class. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, and incorporate strength training at least two days per week. Warm up and cool down during your workouts, stay hydrated, and listen to your body to avoid injuries and overexertion.<br><br>By combining these two tips, you can lead a healthier and more active lifestyle.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry JamesPotty",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Phoenix feather",<br>        "length": 12.25<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Quirrell"<br>}</strong>


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

     33%|███▎      | 1/3 [00:00<00:00,  9.37it/s]

     67%|██████▋   | 2/3 [00:00<00:00, 14.79it/s]

    100%|██████████| 3/3 [00:00<00:00, 23.37it/s]

    



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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    2026-04-08 06:51:37.265 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:37.265 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:37.265 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:37.265 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:37] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:37.265 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:37] Persistent cache disabled, using in-memory JIT cache


    [2026-04-08 06:51:38] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-08 06:51:41] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:41] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:41] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:41] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    2026-04-08 06:51:47.292 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:47] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:47.292 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:47] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:47.292 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:47] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:47.292 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:47] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:51:47.293 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:51:47] Persistent cache disabled, using in-memory JIT cache


    [2026-04-08 06:51:48] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [2026-04-08 06:51:49] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:49] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:49] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 06:51:49] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:01,  2.10it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:00<00:01,  2.65it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:01<00:00,  2.25it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:01<00:00,  1.89it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:02<00:00,  1.67it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:02<00:00,  1.86it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-08 06:51:58,207 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:51:58] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:31697



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-08 06:52:01] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:799: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person wearing a yellow shirt standing on the rear area of a taxi, using an iron to iron clothing on an ironing board set up on the roof of the taxi. The ironing board is elevated, which is an unconventional method of ironing and likely poses safety issues. The setting appears to be a busy urban street with other taxis and pedestrians visible in the background. This activity is not typical and seems to be for show or a demonstration purpose.</strong>



```python
terminate_process(server_process)
```
