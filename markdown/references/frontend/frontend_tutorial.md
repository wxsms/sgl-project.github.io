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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-10 03:31:01] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:03,  1.15s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:02<00:01,  1.02it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.08it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.16it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.10it/s]


    2026-05-10 03:31:08,505 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 03:31:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:06,  5.39s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:27,  2.63s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:27,  2.63s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:34,  1.73s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:34,  1.73s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:09,  1.28s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:09,  1.28s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:53,  1.00s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:53,  1.00s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:43,  1.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:43,  1.20it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:35,  1.42it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:09<00:30,  1.66it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:09<00:30,  1.66it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.91it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.91it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:21,  2.19it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:21,  2.19it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:10<00:19,  2.46it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:10<00:19,  2.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:10<00:16,  2.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:10<00:16,  2.78it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.08it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.08it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.38it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.38it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:11<00:11,  3.86it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:11<00:11,  3.86it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:11<00:09,  4.33it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:11<00:09,  4.33it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:11<00:08,  4.92it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:11<00:08,  4.92it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.62it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.62it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  6.21it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  6.21it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:05,  6.92it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:05,  6.92it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:11<00:05,  6.92it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:11<00:04,  8.98it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:11<00:04,  8.98it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:11<00:04,  8.98it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:11<00:03, 10.94it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:11<00:03, 10.94it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:11<00:03, 10.94it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:12<00:03, 10.94it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:12<00:02, 14.05it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:12<00:02, 14.05it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:12<00:02, 14.05it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:12<00:02, 14.05it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:12<00:01, 17.05it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:12<00:01, 17.05it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:12<00:01, 17.05it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:12<00:01, 17.05it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:12<00:01, 19.53it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:12<00:01, 19.53it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:12<00:01, 19.53it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:12<00:01, 19.53it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:12<00:01, 21.95it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:12<00:01, 21.95it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:12<00:01, 21.95it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:12<00:01, 21.95it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:12<00:01, 21.95it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:12<00:00, 25.76it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:12<00:00, 25.76it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:12<00:00, 25.76it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:12<00:00, 25.76it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:12<00:00, 25.76it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:12<00:00, 28.34it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:12<00:00, 36.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.52 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   2%|▏         | 1/58 [00:00<00:53,  1.07it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   3%|▎         | 2/58 [00:01<00:49,  1.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   3%|▎         | 2/58 [00:01<00:49,  1.14it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:45,  1.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.83 GB):   5%|▌         | 3/58 [00:02<00:45,  1.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.83 GB):   7%|▋         | 4/58 [00:03<00:41,  1.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   7%|▋         | 4/58 [00:03<00:41,  1.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:37,  1.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.92 GB):   9%|▊         | 5/58 [00:03<00:37,  1.41it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.92 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.42 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.42 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.56 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.67it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.56 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.84it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.66 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.84it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.66 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.72 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.01it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.72 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.24 GB):  17%|█▋        | 10/58 [00:06<00:21,  2.19it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.24 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.27 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.39it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.27 GB):  21%|██        | 12/58 [00:06<00:17,  2.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  21%|██        | 12/58 [00:06<00:17,  2.62it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.66 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.65 GB):  22%|██▏       | 13/58 [00:06<00:15,  2.89it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.65 GB):  24%|██▍       | 14/58 [00:07<00:13,  3.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  24%|██▍       | 14/58 [00:07<00:13,  3.21it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.65 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.64 GB):  26%|██▌       | 15/58 [00:07<00:12,  3.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.64 GB):  28%|██▊       | 16/58 [00:07<00:10,  3.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  28%|██▊       | 16/58 [00:07<00:10,  3.98it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.24 GB):  29%|██▉       | 17/58 [00:07<00:09,  4.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.24 GB):  31%|███       | 18/58 [00:07<00:08,  4.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.43 GB):  31%|███       | 18/58 [00:07<00:08,  4.70it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.43 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.58 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.58 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s]Capturing num tokens (num_tokens=960 avail_mem=26.55 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.73it/s]Capturing num tokens (num_tokens=896 avail_mem=26.54 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.73it/s]Capturing num tokens (num_tokens=832 avail_mem=26.46 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.73it/s]Capturing num tokens (num_tokens=832 avail_mem=26.46 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.84it/s]Capturing num tokens (num_tokens=768 avail_mem=26.46 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.84it/s]

    Capturing num tokens (num_tokens=704 avail_mem=26.45 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.84it/s]Capturing num tokens (num_tokens=704 avail_mem=26.45 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.14it/s]Capturing num tokens (num_tokens=640 avail_mem=26.50 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.14it/s]Capturing num tokens (num_tokens=576 avail_mem=26.49 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.14it/s]

    Capturing num tokens (num_tokens=576 avail_mem=26.49 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.27it/s]Capturing num tokens (num_tokens=512 avail_mem=26.47 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.27it/s]Capturing num tokens (num_tokens=480 avail_mem=26.46 GB):  48%|████▊     | 28/58 [00:08<00:02, 11.27it/s]Capturing num tokens (num_tokens=480 avail_mem=26.46 GB):  52%|█████▏    | 30/58 [00:08<00:02, 12.53it/s]Capturing num tokens (num_tokens=448 avail_mem=26.45 GB):  52%|█████▏    | 30/58 [00:08<00:02, 12.53it/s]Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  52%|█████▏    | 30/58 [00:08<00:02, 12.53it/s]

    Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.31it/s]Capturing num tokens (num_tokens=384 avail_mem=26.40 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.31it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.31it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.22it/s]Capturing num tokens (num_tokens=320 avail_mem=26.41 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.22it/s]Capturing num tokens (num_tokens=288 avail_mem=26.41 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.22it/s]

    Capturing num tokens (num_tokens=288 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.50it/s]Capturing num tokens (num_tokens=256 avail_mem=26.40 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.50it/s]Capturing num tokens (num_tokens=240 avail_mem=26.39 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.50it/s]Capturing num tokens (num_tokens=240 avail_mem=26.39 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.50it/s]Capturing num tokens (num_tokens=224 avail_mem=26.36 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.50it/s]Capturing num tokens (num_tokens=208 avail_mem=26.35 GB):  66%|██████▌   | 38/58 [00:09<00:01, 16.50it/s]

    Capturing num tokens (num_tokens=208 avail_mem=26.35 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.59it/s]Capturing num tokens (num_tokens=192 avail_mem=26.36 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.59it/s]Capturing num tokens (num_tokens=176 avail_mem=26.35 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.59it/s]Capturing num tokens (num_tokens=160 avail_mem=26.33 GB):  69%|██████▉   | 40/58 [00:09<00:01, 16.59it/s]Capturing num tokens (num_tokens=160 avail_mem=26.33 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.97it/s]Capturing num tokens (num_tokens=144 avail_mem=26.32 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.97it/s]Capturing num tokens (num_tokens=128 avail_mem=26.31 GB):  74%|███████▍  | 43/58 [00:09<00:00, 17.97it/s]

    Capturing num tokens (num_tokens=128 avail_mem=26.31 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.26it/s]Capturing num tokens (num_tokens=112 avail_mem=26.31 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.26it/s]Capturing num tokens (num_tokens=96 avail_mem=26.29 GB):  78%|███████▊  | 45/58 [00:09<00:00, 18.26it/s] Capturing num tokens (num_tokens=96 avail_mem=26.29 GB):  81%|████████  | 47/58 [00:09<00:00, 18.36it/s]Capturing num tokens (num_tokens=80 avail_mem=26.28 GB):  81%|████████  | 47/58 [00:09<00:00, 18.36it/s]Capturing num tokens (num_tokens=64 avail_mem=26.27 GB):  81%|████████  | 47/58 [00:09<00:00, 18.36it/s]Capturing num tokens (num_tokens=48 avail_mem=26.26 GB):  81%|████████  | 47/58 [00:09<00:00, 18.36it/s]

    Capturing num tokens (num_tokens=48 avail_mem=26.26 GB):  86%|████████▌ | 50/58 [00:09<00:00, 19.45it/s]Capturing num tokens (num_tokens=32 avail_mem=26.24 GB):  86%|████████▌ | 50/58 [00:09<00:00, 19.45it/s]Capturing num tokens (num_tokens=28 avail_mem=26.23 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.45it/s]Capturing num tokens (num_tokens=24 avail_mem=26.23 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.45it/s]Capturing num tokens (num_tokens=24 avail_mem=26.23 GB):  91%|█████████▏| 53/58 [00:10<00:00, 21.09it/s]Capturing num tokens (num_tokens=20 avail_mem=26.22 GB):  91%|█████████▏| 53/58 [00:10<00:00, 21.09it/s]Capturing num tokens (num_tokens=16 avail_mem=26.21 GB):  91%|█████████▏| 53/58 [00:10<00:00, 21.09it/s]Capturing num tokens (num_tokens=12 avail_mem=26.21 GB):  91%|█████████▏| 53/58 [00:10<00:00, 21.09it/s]

    Capturing num tokens (num_tokens=8 avail_mem=26.20 GB):  91%|█████████▏| 53/58 [00:10<00:00, 21.09it/s] Capturing num tokens (num_tokens=8 avail_mem=26.20 GB):  98%|█████████▊| 57/58 [00:10<00:00, 25.46it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB):  98%|█████████▊| 57/58 [00:10<00:00, 25.46it/s]Capturing num tokens (num_tokens=4 avail_mem=26.20 GB): 100%|██████████| 58/58 [00:10<00:00,  5.68it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:39389


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-10 03:31:52] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Spain** - Madrid<br>3. **Australia** - Canberra</strong>


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


<strong style='color: #00008B;'>Certainly! Here's a list of three countries along with their capitals:<br><br>1. **France - Paris**<br>2. **Germany - Berlin**<br>3. **Japan - Tokyo**</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries and their capitals:<br><br>1. **Italy - Rome**<br>2. **Spain - Madrid**<br>3. **Canada - Ottawa**</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>Let's calculate this:<br><br>2 * 2 = 4<br><br>So, the answer is 4. No calculator was necessary for this simple multiplication.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Focus on eating a variety of nutritious foods from all food groups: fruits, vegetables, whole grains, lean proteins, and dairy. Include plenty of whole foods and limit processed and sugary items. Staying hydrated is also essential.<br>2. **Regular Exercise**: Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, plus strength training exercises at least two days a week. Find activities you enjoy to maintain consistency and commitment.<br><br>Both of these habits are foundational for maintaining good health and can significantly impact your overall well-being.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holm oak",<br>        "core": "Phoenix feather",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Basilisk"<br>}</strong>


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

     33%|███▎      | 1/3 [00:00<00:00,  6.95it/s]

     67%|██████▋   | 2/3 [00:00<00:00, 11.01it/s]

    100%|██████████| 3/3 [00:00<00:00, 17.32it/s]

    



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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-10 03:32:21] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-05-10 03:32:22] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-10 03:32:25] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-05-10 03:32:30] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:02<00:09,  2.27s/it]

    Multi-thread loading shards:  40% Completed | 2/5 [00:03<00:04,  1.62s/it]

    Multi-thread loading shards:  60% Completed | 3/5 [00:04<00:02,  1.34s/it]

    Multi-thread loading shards:  80% Completed | 4/5 [00:05<00:01,  1.19s/it]

    Multi-thread loading shards: 100% Completed | 5/5 [00:05<00:00,  1.14it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:05<00:00,  1.14s/it]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-10 03:32:43,218 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 03:32:43] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:39960



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-10 03:32:46] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:830: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person in a yellow shirt standing on the side of a yellow car in an urban setting. The individual is holding an iron and working on what appears to be a shirt spread out on a portable ironing board. The car is a taxi, and there are other yellow taxis visible in the background, suggesting that this is an urban environment, likely a city street. The person appears to be demonstrating or performing an action, possibly related to cleaning or ironing clothes.</strong>



```python
terminate_process(server_process)
```
