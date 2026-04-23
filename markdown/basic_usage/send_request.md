# Sending Requests
This notebook provides a quick-start guide to use SGLang in chat completions after installation. Once your server is running, API documentation is available at `http://localhost:30000/docs` (Swagger UI), `http://localhost:30000/redoc` (ReDoc), or `http://localhost:30000/openapi.json` (OpenAPI spec, useful for AI agents). Replace `30000` with your port if using a different one.

- For Vision Language Models, see [OpenAI APIs - Vision](openai_api_vision.ipynb).
- For Embedding Models, see [OpenAI APIs - Embedding](openai_api_embeddings.ipynb) and [Encode (embedding model)](native_api.html#Encode-(embedding-model)).
- For Reward Models, see [Classify (reward model)](native_api.html#Classify-(reward-model)).

## Launch A Server


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
 --host 0.0.0.0 --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-23 04:49:36] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 04:49:37] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-23 04:49:38] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 04:49:45] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.31it/s]


    2026-04-23 04:49:50,489 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 04:49:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:06,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.05it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:26,  2.05it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:26,  2.05it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.35it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:11,  4.35it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.13it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.13it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.13it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.13it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:03<00:04, 10.94it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 16.91it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 23.81it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 26.77it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]

    Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 34.74it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 43.21it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 43.21it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 43.21it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 43.21it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 43.21it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 43.21it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 43.21it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 46.70it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 46.70it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 46.70it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 46.70it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 46.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 13.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   2%|▏         | 1/58 [00:00<00:06,  8.95it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.93 GB):   2%|▏         | 1/58 [00:00<00:06,  8.95it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.93 GB):   3%|▎         | 2/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.25 GB):   3%|▎         | 2/58 [00:00<00:14,  3.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.25 GB):   5%|▌         | 3/58 [00:00<00:10,  5.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.95 GB):   5%|▌         | 3/58 [00:00<00:10,  5.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.22 GB):   5%|▌         | 3/58 [00:00<00:10,  5.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.22 GB):   9%|▊         | 5/58 [00:00<00:06,  7.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.25 GB):   9%|▊         | 5/58 [00:00<00:06,  7.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.00 GB):   9%|▊         | 5/58 [00:00<00:06,  7.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.00 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.24 GB):  12%|█▏        | 7/58 [00:00<00:05,  9.06it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.28 GB):  12%|█▏        | 7/58 [00:01<00:05,  9.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.28 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.24 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.23 GB):  16%|█▌        | 9/58 [00:01<00:04, 11.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.23 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.77it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.06 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.23 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.77it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=58.23 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.54it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.23 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.11 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.21 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.21 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.21 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.09 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.20 GB):  28%|██▊       | 16/58 [00:01<00:02, 17.48it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.20 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.19 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.19 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.76it/s]Capturing num tokens (num_tokens=960 avail_mem=58.16 GB):  33%|███▎      | 19/58 [00:01<00:01, 19.76it/s] Capturing num tokens (num_tokens=960 avail_mem=58.16 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=896 avail_mem=58.17 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=832 avail_mem=58.16 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=768 avail_mem=58.11 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.16it/s]

    Capturing num tokens (num_tokens=704 avail_mem=58.15 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.16it/s]Capturing num tokens (num_tokens=704 avail_mem=58.15 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.69it/s]Capturing num tokens (num_tokens=640 avail_mem=58.14 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.69it/s]Capturing num tokens (num_tokens=576 avail_mem=58.14 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.12 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.69it/s]Capturing num tokens (num_tokens=512 avail_mem=58.12 GB):  50%|█████     | 29/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=480 avail_mem=58.14 GB):  50%|█████     | 29/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=448 avail_mem=58.13 GB):  50%|█████     | 29/58 [00:01<00:01, 25.13it/s]

    Capturing num tokens (num_tokens=416 avail_mem=58.13 GB):  50%|█████     | 29/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=416 avail_mem=58.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.25it/s]Capturing num tokens (num_tokens=384 avail_mem=58.10 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.25it/s]Capturing num tokens (num_tokens=352 avail_mem=58.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.25it/s]Capturing num tokens (num_tokens=320 avail_mem=58.08 GB):  55%|█████▌    | 32/58 [00:02<00:00, 26.25it/s]Capturing num tokens (num_tokens=320 avail_mem=58.08 GB):  60%|██████    | 35/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=288 avail_mem=58.06 GB):  60%|██████    | 35/58 [00:02<00:00, 24.67it/s]

    Capturing num tokens (num_tokens=256 avail_mem=58.07 GB):  60%|██████    | 35/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=240 avail_mem=58.06 GB):  60%|██████    | 35/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=240 avail_mem=58.06 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=224 avail_mem=58.05 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=208 avail_mem=58.04 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.27it/s]Capturing num tokens (num_tokens=192 avail_mem=58.02 GB):  66%|██████▌   | 38/58 [00:02<00:00, 21.27it/s]

    Capturing num tokens (num_tokens=192 avail_mem=58.02 GB):  71%|███████   | 41/58 [00:02<00:00, 22.65it/s]Capturing num tokens (num_tokens=176 avail_mem=57.94 GB):  71%|███████   | 41/58 [00:02<00:00, 22.65it/s]Capturing num tokens (num_tokens=160 avail_mem=57.44 GB):  71%|███████   | 41/58 [00:02<00:00, 22.65it/s]Capturing num tokens (num_tokens=144 avail_mem=57.38 GB):  71%|███████   | 41/58 [00:02<00:00, 22.65it/s]Capturing num tokens (num_tokens=128 avail_mem=57.37 GB):  71%|███████   | 41/58 [00:02<00:00, 22.65it/s]Capturing num tokens (num_tokens=128 avail_mem=57.37 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.19it/s]Capturing num tokens (num_tokens=112 avail_mem=57.37 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.19it/s]Capturing num tokens (num_tokens=96 avail_mem=57.36 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.19it/s] Capturing num tokens (num_tokens=80 avail_mem=56.28 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.19it/s]

    Capturing num tokens (num_tokens=64 avail_mem=56.18 GB):  78%|███████▊  | 45/58 [00:02<00:00, 25.19it/s]Capturing num tokens (num_tokens=64 avail_mem=56.18 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.93it/s]Capturing num tokens (num_tokens=48 avail_mem=42.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.93it/s]Capturing num tokens (num_tokens=32 avail_mem=41.64 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.93it/s]Capturing num tokens (num_tokens=28 avail_mem=41.63 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.93it/s]Capturing num tokens (num_tokens=24 avail_mem=41.60 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.93it/s]Capturing num tokens (num_tokens=24 avail_mem=41.60 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.99it/s]Capturing num tokens (num_tokens=20 avail_mem=41.61 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.99it/s]Capturing num tokens (num_tokens=16 avail_mem=41.61 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.99it/s]

    Capturing num tokens (num_tokens=12 avail_mem=41.60 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.99it/s]Capturing num tokens (num_tokens=12 avail_mem=41.60 GB):  97%|█████████▋| 56/58 [00:02<00:00, 24.08it/s]Capturing num tokens (num_tokens=8 avail_mem=41.59 GB):  97%|█████████▋| 56/58 [00:02<00:00, 24.08it/s] Capturing num tokens (num_tokens=4 avail_mem=41.59 GB):  97%|█████████▋| 56/58 [00:02<00:00, 24.08it/s]Capturing num tokens (num_tokens=4 avail_mem=41.59 GB): 100%|██████████| 58/58 [00:03<00:00, 19.22it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Using cURL



```python
import subprocess, json

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "qwen/qwen2.5-0.5b-instruct", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
"""

response = json.loads(subprocess.check_output(curl_command, shell=True))
print_highlight(response)
```


<strong style='color: #00008B;'>{'id': 'ce3a4e6c8bda4d8aa9b1b5407d0b2733', 'object': 'chat.completion', 'created': 1776919805, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


## Using Python Requests


```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'id': '6e10ccfd4f634e8e8a347788a9c3dd31', 'object': 'chat.completion', 'created': 1776919805, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


## Using OpenAI Python Client


```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print_highlight(response)
```


<strong style='color: #00008B;'>ChatCompletion(id='8dc1f64c2f7941d397b3ad0679331201', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1776919805, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


### Streaming


```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# Use stream=True for streaming responses
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
    stream=True,
)

# Handle the streaming output
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

    Sure

    ,

     here

     are

     three

     countries

     and

     their

     capitals

    :
    


    1

    .

     **

    United

     States

    **

     -

     Washington

     D

    .C

    .


    2

    .

     **

    Canada

    **

     -

     Ottawa

    


    3

    .

     **

    Australia

    **

     -

     Canberra

## Using Native Generation APIs

You can also use the native `/generate` endpoint with requests, which provides more flexibility. An API reference is available at [Sampling Parameters](sampling_params.md).


```python
import requests

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'e0282ac6dcca46c39e9e4213da0adc5e', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.21450444497168064, 'response_sent_to_client_ts': 1776919806.3964577}}</strong>


### Streaming


```python
import requests, json

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"]
        print(output[prev:], end="", flush=True)
        prev = len(output)
```

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     second

     largest

     city

     in

     the

     world

    .

     It

     is

     located

     in

     the

     south

     of

     France

    ,

     on

     the

     banks

     of

     the


```python
terminate_process(server_process)
```
