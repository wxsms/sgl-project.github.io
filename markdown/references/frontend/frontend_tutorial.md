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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 16:06:09] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-18 16:06:10] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 16:06:17] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.85it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.41it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.35it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.35it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.38it/s]


    2026-04-18 16:06:24,660 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 16:06:24] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:55,  3.07s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:23,  1.49s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:49,  1.11it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:33,  1.63it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:18,  2.87it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.62it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.41it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.41it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.29it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.99it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.99it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.99it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.08it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.08it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:04, 10.08it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 12.03it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:02, 15.32it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:02, 15.32it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:02, 15.32it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:02, 15.32it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:02, 15.32it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:05<00:01, 20.49it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 26.99it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 34.45it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 37.43it/s]

    Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 39.29it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 41.51it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 46.11it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 46.11it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 46.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=117.03 GB):   2%|▏         | 1/58 [00:00<00:24,  2.28it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.94 GB):   2%|▏         | 1/58 [00:00<00:24,  2.28it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=116.94 GB):   3%|▎         | 2/58 [00:00<00:23,  2.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.94 GB):   3%|▎         | 2/58 [00:00<00:23,  2.40it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=116.94 GB):   5%|▌         | 3/58 [00:01<00:21,  2.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.94 GB):   5%|▌         | 3/58 [00:01<00:21,  2.50it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.94 GB):   7%|▋         | 4/58 [00:01<00:20,  2.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.95 GB):   7%|▋         | 4/58 [00:01<00:20,  2.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=116.95 GB):   9%|▊         | 5/58 [00:01<00:19,  2.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.95 GB):   9%|▊         | 5/58 [00:01<00:19,  2.75it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=116.95 GB):  10%|█         | 6/58 [00:02<00:17,  3.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.95 GB):  10%|█         | 6/58 [00:02<00:17,  3.04it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=116.95 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.70 GB):  12%|█▏        | 7/58 [00:02<00:16,  3.07it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=119.70 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.71 GB):  14%|█▍        | 8/58 [00:02<00:16,  3.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.71 GB):  16%|█▌        | 9/58 [00:02<00:13,  3.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.71 GB):  16%|█▌        | 9/58 [00:03<00:13,  3.55it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=119.71 GB):  17%|█▋        | 10/58 [00:03<00:11,  4.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.71 GB):  17%|█▋        | 10/58 [00:03<00:11,  4.05it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.71 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.71 GB):  19%|█▉        | 11/58 [00:03<00:10,  4.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=119.71 GB):  21%|██        | 12/58 [00:03<00:08,  5.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.71 GB):  21%|██        | 12/58 [00:03<00:08,  5.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.71 GB):  22%|██▏       | 13/58 [00:03<00:07,  5.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.71 GB):  22%|██▏       | 13/58 [00:03<00:07,  5.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=119.71 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.71 GB):  24%|██▍       | 14/58 [00:03<00:06,  6.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.71 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.71 GB):  26%|██▌       | 15/58 [00:03<00:06,  7.03it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=119.71 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.71 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.71 GB):  28%|██▊       | 16/58 [00:04<00:05,  7.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.71 GB):  31%|███       | 18/58 [00:04<00:04,  9.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.70 GB):  31%|███       | 18/58 [00:04<00:04,  9.11it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=119.71 GB):  31%|███       | 18/58 [00:04<00:04,  9.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.71 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.71 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.69it/s]Capturing num tokens (num_tokens=960 avail_mem=119.70 GB):  34%|███▍      | 20/58 [00:04<00:03, 10.69it/s] Capturing num tokens (num_tokens=960 avail_mem=119.70 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.70it/s]Capturing num tokens (num_tokens=896 avail_mem=119.70 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.70it/s]

    Capturing num tokens (num_tokens=832 avail_mem=119.70 GB):  38%|███▊      | 22/58 [00:04<00:02, 12.70it/s]Capturing num tokens (num_tokens=832 avail_mem=119.70 GB):  41%|████▏     | 24/58 [00:04<00:02, 14.38it/s]Capturing num tokens (num_tokens=768 avail_mem=119.69 GB):  41%|████▏     | 24/58 [00:04<00:02, 14.38it/s]Capturing num tokens (num_tokens=704 avail_mem=119.69 GB):  41%|████▏     | 24/58 [00:04<00:02, 14.38it/s]Capturing num tokens (num_tokens=640 avail_mem=119.69 GB):  41%|████▏     | 24/58 [00:04<00:02, 14.38it/s]Capturing num tokens (num_tokens=640 avail_mem=119.69 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.86it/s]Capturing num tokens (num_tokens=576 avail_mem=119.68 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.86it/s]

    Capturing num tokens (num_tokens=512 avail_mem=119.68 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.86it/s]Capturing num tokens (num_tokens=480 avail_mem=119.67 GB):  47%|████▋     | 27/58 [00:04<00:01, 16.86it/s]Capturing num tokens (num_tokens=480 avail_mem=119.67 GB):  52%|█████▏    | 30/58 [00:04<00:01, 18.95it/s]Capturing num tokens (num_tokens=448 avail_mem=119.67 GB):  52%|█████▏    | 30/58 [00:04<00:01, 18.95it/s]Capturing num tokens (num_tokens=416 avail_mem=119.67 GB):  52%|█████▏    | 30/58 [00:04<00:01, 18.95it/s]Capturing num tokens (num_tokens=384 avail_mem=119.66 GB):  52%|█████▏    | 30/58 [00:04<00:01, 18.95it/s]Capturing num tokens (num_tokens=384 avail_mem=119.66 GB):  57%|█████▋    | 33/58 [00:04<00:01, 20.94it/s]Capturing num tokens (num_tokens=352 avail_mem=119.66 GB):  57%|█████▋    | 33/58 [00:04<00:01, 20.94it/s]

    Capturing num tokens (num_tokens=320 avail_mem=119.65 GB):  57%|█████▋    | 33/58 [00:04<00:01, 20.94it/s]Capturing num tokens (num_tokens=288 avail_mem=119.65 GB):  57%|█████▋    | 33/58 [00:04<00:01, 20.94it/s]Capturing num tokens (num_tokens=288 avail_mem=119.65 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.84it/s]Capturing num tokens (num_tokens=256 avail_mem=119.65 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.84it/s]Capturing num tokens (num_tokens=240 avail_mem=119.64 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.84it/s]Capturing num tokens (num_tokens=224 avail_mem=119.64 GB):  62%|██████▏   | 36/58 [00:04<00:00, 22.84it/s]Capturing num tokens (num_tokens=208 avail_mem=119.63 GB):  62%|██████▏   | 36/58 [00:05<00:00, 22.84it/s]

    Capturing num tokens (num_tokens=208 avail_mem=119.63 GB):  69%|██████▉   | 40/58 [00:05<00:00, 25.24it/s]Capturing num tokens (num_tokens=192 avail_mem=119.63 GB):  69%|██████▉   | 40/58 [00:05<00:00, 25.24it/s]Capturing num tokens (num_tokens=176 avail_mem=119.63 GB):  69%|██████▉   | 40/58 [00:05<00:00, 25.24it/s]Capturing num tokens (num_tokens=160 avail_mem=119.63 GB):  69%|██████▉   | 40/58 [00:05<00:00, 25.24it/s]Capturing num tokens (num_tokens=144 avail_mem=119.62 GB):  69%|██████▉   | 40/58 [00:05<00:00, 25.24it/s]Capturing num tokens (num_tokens=144 avail_mem=119.62 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.86it/s]Capturing num tokens (num_tokens=128 avail_mem=119.63 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.86it/s]Capturing num tokens (num_tokens=112 avail_mem=119.63 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.86it/s]Capturing num tokens (num_tokens=96 avail_mem=119.62 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.86it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=119.62 GB):  76%|███████▌  | 44/58 [00:05<00:00, 26.86it/s]Capturing num tokens (num_tokens=80 avail_mem=119.62 GB):  83%|████████▎ | 48/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=64 avail_mem=119.62 GB):  83%|████████▎ | 48/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=48 avail_mem=119.61 GB):  83%|████████▎ | 48/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=32 avail_mem=119.61 GB):  83%|████████▎ | 48/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=28 avail_mem=119.61 GB):  83%|████████▎ | 48/58 [00:05<00:00, 28.08it/s]Capturing num tokens (num_tokens=28 avail_mem=119.61 GB):  90%|████████▉ | 52/58 [00:05<00:00, 28.97it/s]Capturing num tokens (num_tokens=24 avail_mem=119.60 GB):  90%|████████▉ | 52/58 [00:05<00:00, 28.97it/s]Capturing num tokens (num_tokens=20 avail_mem=119.60 GB):  90%|████████▉ | 52/58 [00:05<00:00, 28.97it/s]

    Capturing num tokens (num_tokens=16 avail_mem=119.60 GB):  90%|████████▉ | 52/58 [00:05<00:00, 28.97it/s]Capturing num tokens (num_tokens=12 avail_mem=119.59 GB):  90%|████████▉ | 52/58 [00:05<00:00, 28.97it/s]Capturing num tokens (num_tokens=12 avail_mem=119.59 GB):  97%|█████████▋| 56/58 [00:05<00:00, 29.72it/s]Capturing num tokens (num_tokens=8 avail_mem=119.59 GB):  97%|█████████▋| 56/58 [00:05<00:00, 29.72it/s] Capturing num tokens (num_tokens=4 avail_mem=119.58 GB):  97%|█████████▋| 56/58 [00:05<00:00, 29.72it/s]Capturing num tokens (num_tokens=4 avail_mem=119.58 GB): 100%|██████████| 58/58 [00:05<00:00, 10.30it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33814


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-18 16:06:49] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Mexico** - Mexico City</strong>


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Brazil - Brasília</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries and their capitals:<br><br>1. Italy - Rome<br>2. Canada - Ottawa<br>3. India - New Delhi</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>This equals: 4.<br><br>No calculator is necessary for this simple multiplication.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet:** Eating a variety of nutritious foods from all the major food groups is essential. Focus on whole grains, fruits and vegetables, lean proteins, and healthy fats. Limit processed and high-sugar foods, and drink plenty of water. This helps support your immune system, maintain energy levels, and prevent chronic diseases.<br><br>2. **Regular Exercise:** Engaging in physical activities regularly can improve your cardiovascular health, strengthen muscles and bones, and help manage your weight. It also boosts mood, helps manage stress, improves sleep quality, enhances cognitive function, and increases energy levels. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.<br><br>Both these tips are crucial for maintaining overall health and wellbeing.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holm oak",<br>        "core": "Phoenix feather",<br>        "length": 11.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Dementors"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 26.36it/s]

    100%|██████████| 3/3 [00:00<00:00, 26.08it/s]

    



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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 16:07:02] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-18 16:07:03] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-18 16:07:06] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 16:07:10] `torch_dtype` is deprecated! Use `dtype` instead!


    The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-04-18 16:07:11] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.58it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.46it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.42it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.40it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.74it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.59it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-18 16:07:23,006 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 16:07:23] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32274



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-18 16:07:26] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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



<strong style='color: #00008B;'>The image shows a person standing on the back of a taxi, using an iron to press a pair of jeans laid out on the vehicle's trunk. The scene takes place on a city street, as suggested by the presence of yellow taxis, American flags, and a building with a wall and windows in the background. The action seems humorous or out of the ordinary, given the unusual setting of ironing clothes on a moving vehicle. The individual is dressed casually in a yellow shirt.</strong>



```python
terminate_process(server_process)
```
