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


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.65it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.56it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.51it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.55it/s]


    2026-04-29 05:59:24,905 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 05:59:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:56,  5.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:56,  5.21s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.27s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.27s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.15it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.79it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.41it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.26it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.26it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:09,  4.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:09,  4.97it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:08,  5.59it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:08,  5.59it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:06<00:07,  6.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:06<00:07,  6.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:06<00:07,  6.31it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  7.79it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  7.79it/s]

    Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  7.79it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.08it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.08it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.08it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:03, 10.40it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:03, 10.40it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:03, 10.40it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:03, 12.19it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:03, 12.19it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:03, 12.19it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:03, 12.19it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:07<00:02, 15.51it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:07<00:02, 15.51it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:07<00:02, 15.51it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:07<00:02, 15.51it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:07<00:02, 15.51it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:07<00:01, 21.26it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:07<00:00, 30.56it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:07<00:00, 42.07it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:07<00:00, 42.07it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:07<00:00, 42.07it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:07<00:00, 42.07it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:07<00:00, 42.07it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:08<00:00, 42.07it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:08<00:00, 42.07it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:08<00:00, 42.07it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:08<00:00, 42.07it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 50.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.78 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.73 GB):   2%|▏         | 1/58 [00:00<00:17,  3.25it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.73 GB):   3%|▎         | 2/58 [00:00<00:16,  3.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:16,  3.33it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.07 GB):   5%|▌         | 3/58 [00:00<00:15,  3.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.07 GB):   5%|▌         | 3/58 [00:00<00:15,  3.66it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.07 GB):   7%|▋         | 4/58 [00:01<00:13,  3.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   7%|▋         | 4/58 [00:01<00:13,  3.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:01<00:12,  4.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:01<00:12,  4.29it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.07 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):  10%|█         | 6/58 [00:01<00:11,  4.69it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.07 GB):  12%|█▏        | 7/58 [00:01<00:10,  5.05it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.07 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.07 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.56it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.07 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.07 GB):  16%|█▌        | 9/58 [00:01<00:08,  6.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.07 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.07 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.58it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.07 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.07 GB):  19%|█▉        | 11/58 [00:02<00:06,  7.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.07 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.06 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=61.06 GB):  21%|██        | 12/58 [00:02<00:05,  7.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.06 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.88it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.06 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.88it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.06 GB):  24%|██▍       | 14/58 [00:02<00:04,  8.88it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.06 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.43 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.32 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=34.32 GB):  31%|███       | 18/58 [00:02<00:03, 11.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=34.32 GB):  31%|███       | 18/58 [00:02<00:03, 11.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=34.32 GB):  31%|███       | 18/58 [00:02<00:03, 11.49it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=34.30 GB):  31%|███       | 18/58 [00:02<00:03, 11.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=34.30 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.61it/s]Capturing num tokens (num_tokens=960 avail_mem=34.30 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.61it/s] Capturing num tokens (num_tokens=896 avail_mem=34.30 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.61it/s]Capturing num tokens (num_tokens=832 avail_mem=34.29 GB):  36%|███▌      | 21/58 [00:02<00:02, 14.61it/s]Capturing num tokens (num_tokens=832 avail_mem=34.29 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.78it/s]Capturing num tokens (num_tokens=768 avail_mem=34.29 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.78it/s]Capturing num tokens (num_tokens=704 avail_mem=34.28 GB):  41%|████▏     | 24/58 [00:02<00:01, 17.78it/s]

    Capturing num tokens (num_tokens=640 avail_mem=34.28 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.78it/s]Capturing num tokens (num_tokens=576 avail_mem=34.28 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.78it/s]Capturing num tokens (num_tokens=576 avail_mem=34.28 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.71it/s]Capturing num tokens (num_tokens=512 avail_mem=34.27 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.71it/s]Capturing num tokens (num_tokens=480 avail_mem=34.27 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.71it/s]Capturing num tokens (num_tokens=448 avail_mem=34.27 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.71it/s]Capturing num tokens (num_tokens=416 avail_mem=34.26 GB):  48%|████▊     | 28/58 [00:03<00:01, 21.71it/s]Capturing num tokens (num_tokens=416 avail_mem=34.26 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.13it/s]Capturing num tokens (num_tokens=384 avail_mem=34.26 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.13it/s]

    Capturing num tokens (num_tokens=352 avail_mem=34.25 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.13it/s]Capturing num tokens (num_tokens=320 avail_mem=34.25 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.13it/s]Capturing num tokens (num_tokens=288 avail_mem=34.26 GB):  55%|█████▌    | 32/58 [00:03<00:01, 25.13it/s]Capturing num tokens (num_tokens=288 avail_mem=34.26 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=256 avail_mem=34.25 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=240 avail_mem=34.25 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=224 avail_mem=34.24 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=208 avail_mem=34.24 GB):  62%|██████▏   | 36/58 [00:03<00:00, 28.52it/s]Capturing num tokens (num_tokens=208 avail_mem=34.24 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]Capturing num tokens (num_tokens=192 avail_mem=34.24 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]

    Capturing num tokens (num_tokens=176 avail_mem=34.23 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]Capturing num tokens (num_tokens=160 avail_mem=34.23 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]Capturing num tokens (num_tokens=144 avail_mem=34.23 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]Capturing num tokens (num_tokens=128 avail_mem=34.23 GB):  69%|██████▉   | 40/58 [00:03<00:00, 31.51it/s]Capturing num tokens (num_tokens=128 avail_mem=34.23 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s]Capturing num tokens (num_tokens=112 avail_mem=34.23 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s]Capturing num tokens (num_tokens=96 avail_mem=34.22 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s] Capturing num tokens (num_tokens=80 avail_mem=34.21 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s]Capturing num tokens (num_tokens=64 avail_mem=34.21 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s]Capturing num tokens (num_tokens=48 avail_mem=34.21 GB):  78%|███████▊  | 45/58 [00:03<00:00, 34.49it/s]

    Capturing num tokens (num_tokens=48 avail_mem=34.21 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=32 avail_mem=34.20 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=28 avail_mem=34.20 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=24 avail_mem=34.20 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=20 avail_mem=34.20 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=16 avail_mem=34.19 GB):  86%|████████▌ | 50/58 [00:03<00:00, 36.65it/s]Capturing num tokens (num_tokens=16 avail_mem=34.19 GB):  95%|█████████▍| 55/58 [00:03<00:00, 38.34it/s]Capturing num tokens (num_tokens=12 avail_mem=34.19 GB):  95%|█████████▍| 55/58 [00:03<00:00, 38.34it/s]Capturing num tokens (num_tokens=8 avail_mem=34.18 GB):  95%|█████████▍| 55/58 [00:03<00:00, 38.34it/s] Capturing num tokens (num_tokens=4 avail_mem=34.18 GB):  95%|█████████▍| 55/58 [00:03<00:00, 38.34it/s]Capturing num tokens (num_tokens=4 avail_mem=34.18 GB): 100%|██████████| 58/58 [00:03<00:00, 15.18it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:31114


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-29 05:59:44] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Mexico - Mexico City</strong>


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


<strong style='color: #00008B;'>Sure! Here are three countries with their respective capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Australia** - Canberra</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries and their capitals:<br><br>1. **Canada** - Ottawa<br>2. **Mexico** - Mexico City<br>3. **Germany** - Berlin</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>Let's calculate it.<br><br>\[ 2 * 2 = 4 \]<br><br>So, 2 * 2 equals 4. I didn't strictly need a calculator for this simple multiplication, but if you wanted to use one, you could input "2 * 2" into it and you would get 4 as the result.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Consuming a variety of nutrients including carbohydrates, proteins, fats, vitamins, and minerals from foods like fruits, vegetables, whole grains, lean proteins, and healthy fats. Aim for consistency and balance in your diet to support your body's needs and maintain a healthy weight.<br>2. **Regular Exercise**: Engaging in physical activities such as aerobic exercises (150 minutes of moderate activity or 75 minutes of vigorous activity per week) and strength training (two or more days a week) to improve cardiovascular health, strength, endurance, and overall well-being. Choose activities you enjoy to ensure long-term sustainability.<br><br>Both these tips are crucial for maintaining optimal health and overall wellness.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "fir",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "A treacle broom"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 30.52it/s]

    



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


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-29 05:59:59] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-29 06:00:03] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.42it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.39it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.37it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.36it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.72it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.55it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-29 06:00:13,865 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 06:00:13] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34419



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-29 06:00:17] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:811: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person standing on the rear roof of a yellow taxi cab, using an iron to iron clothes laid out on a makeshift ironing board, which is set up on the roof. The kabob pole in the image's title is a reference to this activity, as it suggests that the ironing service is being carried out using a "kabob pole," which is a term for the ironing board serving as a stable place to stand while ironing on-the-go. This appears to be an outdoor setting, likely in an urban environment, given the visible taxi and surrounding buildings.</strong>



```python
terminate_process(server_process)
```
