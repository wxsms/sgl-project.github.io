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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-23 22:31:58] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 22:31:59] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-23 22:32:00] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 22:32:06] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.47it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.33it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.28it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.29it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.30it/s]


    2026-04-23 22:32:13,986 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 22:32:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:42,  2.85s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:42,  2.85s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:34,  1.68s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:34,  1.68s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.07s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.07s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:40,  1.32it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.75it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:23,  2.20it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.81it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:14,  3.56it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:11,  4.42it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:11,  4.42it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.75it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.75it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:09,  4.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:09,  4.75it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  5.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  5.00it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:08,  5.46it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:08,  5.46it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:07,  5.95it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:07,  5.95it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:06,  6.18it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:06,  6.18it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:06,  6.49it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:06,  6.49it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:05,  6.97it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:05,  6.97it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:05,  6.97it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:06<00:04,  8.71it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:06<00:04,  8.71it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:06<00:04,  8.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:03, 10.16it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:03, 10.16it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:03, 10.16it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:02, 11.69it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:02, 11.69it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:02, 11.69it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:06<00:02, 13.01it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:06<00:02, 13.01it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:07<00:02, 13.01it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:07<00:02, 13.01it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:07<00:01, 16.03it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:07<00:01, 16.03it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:07<00:01, 16.03it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:07<00:01, 16.03it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:07<00:01, 18.55it/s]

    Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:07<00:01, 18.55it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:07<00:01, 18.80it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:07<00:01, 18.80it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:07<00:01, 18.80it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:07<00:01, 18.80it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:01, 20.57it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:01, 20.57it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:01, 20.57it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:01, 20.57it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:01, 20.57it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:07<00:00, 25.20it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 27.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 27.12it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 27.12it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 27.12it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 27.12it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:07<00:00, 28.04it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 31.37it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 31.37it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 31.37it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 31.37it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 31.37it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 32.76it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 32.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=40.14 GB):   2%|▏         | 1/58 [00:00<00:33,  1.70it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.20 GB):   2%|▏         | 1/58 [00:00<00:33,  1.70it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.20 GB):   3%|▎         | 2/58 [00:01<00:30,  1.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.61 GB):   3%|▎         | 2/58 [00:01<00:30,  1.84it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=40.61 GB):   5%|▌         | 3/58 [00:01<00:27,  2.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.29 GB):   5%|▌         | 3/58 [00:01<00:27,  2.02it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.29 GB):   7%|▋         | 4/58 [00:01<00:24,  2.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.62 GB):   7%|▋         | 4/58 [00:01<00:24,  2.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=40.62 GB):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.63 GB):   9%|▊         | 5/58 [00:02<00:22,  2.39it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.63 GB):  10%|█         | 6/58 [00:02<00:19,  2.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.43 GB):  10%|█         | 6/58 [00:02<00:19,  2.72it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=40.43 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.63 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.01it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=40.63 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.63 GB):  14%|█▍        | 8/58 [00:03<00:15,  3.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.63 GB):  16%|█▌        | 9/58 [00:03<00:12,  3.77it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.61 GB):  16%|█▌        | 9/58 [00:03<00:12,  3.77it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.61 GB):  17%|█▋        | 10/58 [00:03<00:11,  4.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.62 GB):  17%|█▋        | 10/58 [00:03<00:11,  4.08it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.62 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.59 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=40.59 GB):  21%|██        | 12/58 [00:03<00:09,  4.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.60 GB):  21%|██        | 12/58 [00:03<00:09,  4.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.60 GB):  22%|██▏       | 13/58 [00:03<00:08,  5.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.60 GB):  22%|██▏       | 13/58 [00:03<00:08,  5.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=40.60 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.59 GB):  24%|██▍       | 14/58 [00:04<00:07,  5.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.59 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.55 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.56it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=40.55 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.57 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.57 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.57 GB):  31%|███       | 18/58 [00:04<00:04,  8.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.55 GB):  31%|███       | 18/58 [00:04<00:04,  8.71it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=40.56 GB):  31%|███       | 18/58 [00:04<00:04,  8.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.56 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.55 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.18it/s]Capturing num tokens (num_tokens=960 avail_mem=40.55 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.18it/s] Capturing num tokens (num_tokens=960 avail_mem=40.55 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.06it/s]Capturing num tokens (num_tokens=896 avail_mem=40.54 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.06it/s]

    Capturing num tokens (num_tokens=832 avail_mem=40.53 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.06it/s]Capturing num tokens (num_tokens=832 avail_mem=40.53 GB):  41%|████▏     | 24/58 [00:04<00:02, 13.80it/s]Capturing num tokens (num_tokens=768 avail_mem=40.52 GB):  41%|████▏     | 24/58 [00:04<00:02, 13.80it/s]Capturing num tokens (num_tokens=704 avail_mem=40.51 GB):  41%|████▏     | 24/58 [00:04<00:02, 13.80it/s]Capturing num tokens (num_tokens=640 avail_mem=40.51 GB):  41%|████▏     | 24/58 [00:04<00:02, 13.80it/s]Capturing num tokens (num_tokens=640 avail_mem=40.51 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.10it/s]Capturing num tokens (num_tokens=576 avail_mem=40.51 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.10it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.50 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.10it/s]Capturing num tokens (num_tokens=480 avail_mem=40.50 GB):  47%|████▋     | 27/58 [00:05<00:01, 16.10it/s]Capturing num tokens (num_tokens=480 avail_mem=40.50 GB):  52%|█████▏    | 30/58 [00:05<00:01, 18.20it/s]Capturing num tokens (num_tokens=448 avail_mem=40.49 GB):  52%|█████▏    | 30/58 [00:05<00:01, 18.20it/s]Capturing num tokens (num_tokens=416 avail_mem=40.48 GB):  52%|█████▏    | 30/58 [00:05<00:01, 18.20it/s]Capturing num tokens (num_tokens=384 avail_mem=40.48 GB):  52%|█████▏    | 30/58 [00:05<00:01, 18.20it/s]

    Capturing num tokens (num_tokens=384 avail_mem=40.48 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.75it/s]Capturing num tokens (num_tokens=352 avail_mem=40.47 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.75it/s]Capturing num tokens (num_tokens=320 avail_mem=40.46 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.75it/s]Capturing num tokens (num_tokens=288 avail_mem=40.45 GB):  57%|█████▋    | 33/58 [00:05<00:01, 19.75it/s]Capturing num tokens (num_tokens=288 avail_mem=40.45 GB):  62%|██████▏   | 36/58 [00:05<00:01, 21.53it/s]Capturing num tokens (num_tokens=256 avail_mem=40.44 GB):  62%|██████▏   | 36/58 [00:05<00:01, 21.53it/s]Capturing num tokens (num_tokens=240 avail_mem=40.43 GB):  62%|██████▏   | 36/58 [00:05<00:01, 21.53it/s]Capturing num tokens (num_tokens=224 avail_mem=40.43 GB):  62%|██████▏   | 36/58 [00:05<00:01, 21.53it/s]

    Capturing num tokens (num_tokens=224 avail_mem=40.43 GB):  67%|██████▋   | 39/58 [00:05<00:00, 23.26it/s]Capturing num tokens (num_tokens=208 avail_mem=40.42 GB):  67%|██████▋   | 39/58 [00:05<00:00, 23.26it/s]Capturing num tokens (num_tokens=192 avail_mem=40.41 GB):  67%|██████▋   | 39/58 [00:05<00:00, 23.26it/s]Capturing num tokens (num_tokens=176 avail_mem=40.40 GB):  67%|██████▋   | 39/58 [00:05<00:00, 23.26it/s]Capturing num tokens (num_tokens=160 avail_mem=40.40 GB):  67%|██████▋   | 39/58 [00:05<00:00, 23.26it/s]Capturing num tokens (num_tokens=160 avail_mem=40.40 GB):  74%|███████▍  | 43/58 [00:05<00:00, 26.58it/s]Capturing num tokens (num_tokens=144 avail_mem=40.39 GB):  74%|███████▍  | 43/58 [00:05<00:00, 26.58it/s]Capturing num tokens (num_tokens=128 avail_mem=40.40 GB):  74%|███████▍  | 43/58 [00:05<00:00, 26.58it/s]Capturing num tokens (num_tokens=112 avail_mem=40.40 GB):  74%|███████▍  | 43/58 [00:05<00:00, 26.58it/s]Capturing num tokens (num_tokens=96 avail_mem=40.40 GB):  74%|███████▍  | 43/58 [00:05<00:00, 26.58it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=40.40 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=80 avail_mem=40.39 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=64 avail_mem=40.39 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=48 avail_mem=40.39 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=32 avail_mem=40.38 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=28 avail_mem=40.38 GB):  81%|████████  | 47/58 [00:05<00:00, 30.08it/s]Capturing num tokens (num_tokens=28 avail_mem=40.38 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s]Capturing num tokens (num_tokens=24 avail_mem=40.38 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s]Capturing num tokens (num_tokens=20 avail_mem=40.37 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s]Capturing num tokens (num_tokens=16 avail_mem=40.37 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s]Capturing num tokens (num_tokens=12 avail_mem=40.36 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s]

    Capturing num tokens (num_tokens=8 avail_mem=40.36 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.52it/s] Capturing num tokens (num_tokens=8 avail_mem=40.36 GB):  98%|█████████▊| 57/58 [00:05<00:00, 35.88it/s]Capturing num tokens (num_tokens=4 avail_mem=40.36 GB):  98%|█████████▊| 57/58 [00:05<00:00, 35.88it/s]Capturing num tokens (num_tokens=4 avail_mem=40.36 GB): 100%|██████████| 58/58 [00:05<00:00,  9.81it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:35554


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-23 22:32:35] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Australia - Canberra<br>3. Japan - Tokyo</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries and their respective capitals:<br><br>1. France - Paris<br>2. Brazil - Brasília<br>3. Japan - Tokyo</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries and their respective capitals:<br><br>1. Germany - Berlin<br>2. Italy - Rome<br>3. Canada - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>This calculation simply requires basic multiplication. Without needing a calculator, we can determine that:<br><br>2 * 2 = 4<br><br>So the answer is 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Eating a variety of nutrients from different food groups is essential for overall health. Include plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods and sugary drinks, and stay well-hydrated with water.<br><br>2. **Regular Exercise**: Engage in a mix of aerobic activities (like jogging, swimming, or cycling), strength training (like weightlifting or bodyweight exercises), and flexibility exercises (like yoga or stretching). Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, along with strength training exercises on two or more days a week.<br><br>By combining these two practices, you can significantly improve your physical and mental health, enhance your quality of life, and reduce the risk of various health issues.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holme Pine",<br>        "core": "Phoenix feather",<br>        "length": 11.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "The person who j"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 33.02it/s]

    



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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-23 22:32:45] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 22:32:47] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-23 22:32:47] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-23 22:32:51] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 22:32:54] `torch_dtype` is deprecated! Use `dtype` instead!


    The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-23 22:32:55] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:03,  1.27it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.22it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.20it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:03<00:00,  1.20it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.52it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.37it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-23 22:33:10,089 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 22:33:10] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34911



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-23 22:33:13] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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



<strong style='color: #00008B;'>The image shows a man on a yellow cab wearing a bright yellow shirt and使用的衣補排，该排相对于在车尾部打开。车本身停在城市的街道上，可以看到其它交通标志和建筑物。这看起来像是一个街景，可能是在纽约市，因为出租车的颜色和风格是纽约出租车的特点。</strong>



```python
terminate_process(server_process)
```
