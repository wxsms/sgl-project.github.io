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
    [2026-04-24 00:55:31] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 00:55:32] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-24 00:55:33] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 00:55:41] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.98it/s]


    2026-04-24 00:55:47,088 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 00:55:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:05,  1.18s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:25,  2.12it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:25,  2.12it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.46it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.46it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.46it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.46it/s]

    Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:11,  4.46it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s]

    Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:03<00:03, 13.71it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:00, 29.59it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 38.66it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 46.19it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 51.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.31 GB):   2%|▏         | 1/58 [00:00<00:06,  8.77it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.28 GB):   2%|▏         | 1/58 [00:00<00:06,  8.77it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=116.28 GB):   3%|▎         | 2/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.28 GB):   3%|▎         | 2/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.28 GB):   3%|▎         | 2/58 [00:00<00:06,  9.25it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.28 GB):   7%|▋         | 4/58 [00:00<00:05, 10.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.28 GB):   7%|▋         | 4/58 [00:00<00:05, 10.38it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=116.28 GB):   7%|▋         | 4/58 [00:00<00:05, 10.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.28 GB):  10%|█         | 6/58 [00:00<00:04, 11.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.28 GB):  10%|█         | 6/58 [00:00<00:04, 11.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):  10%|█         | 6/58 [00:00<00:04, 11.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.99it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=116.27 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.99it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.27 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.26 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.63it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.26 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.26 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.08it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=116.25 GB):  22%|██▏       | 13/58 [00:00<00:02, 17.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.25 GB):  22%|██▏       | 13/58 [00:01<00:02, 17.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.25 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.25 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.24 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.24 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.23 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.21 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.48it/s]

    Capturing num tokens (num_tokens=960 avail_mem=116.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.48it/s] Capturing num tokens (num_tokens=896 avail_mem=116.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.48it/s]Capturing num tokens (num_tokens=832 avail_mem=116.22 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.48it/s]Capturing num tokens (num_tokens=832 avail_mem=116.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.78it/s]Capturing num tokens (num_tokens=768 avail_mem=116.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.78it/s]Capturing num tokens (num_tokens=704 avail_mem=116.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.78it/s]Capturing num tokens (num_tokens=640 avail_mem=116.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.78it/s]Capturing num tokens (num_tokens=576 avail_mem=116.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.78it/s]Capturing num tokens (num_tokens=576 avail_mem=116.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=512 avail_mem=116.20 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.31it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=448 avail_mem=116.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=416 avail_mem=116.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 29.31it/s]Capturing num tokens (num_tokens=416 avail_mem=116.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=384 avail_mem=116.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=352 avail_mem=116.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=320 avail_mem=116.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=288 avail_mem=116.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=288 avail_mem=116.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=256 avail_mem=116.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.11it/s]

    Capturing num tokens (num_tokens=240 avail_mem=116.19 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=224 avail_mem=116.19 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=208 avail_mem=116.19 GB):  62%|██████▏   | 36/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=208 avail_mem=116.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=192 avail_mem=116.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=176 avail_mem=116.18 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=160 avail_mem=116.18 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.16it/s]

    Capturing num tokens (num_tokens=144 avail_mem=116.17 GB):  69%|██████▉   | 40/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=144 avail_mem=116.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.99it/s]Capturing num tokens (num_tokens=128 avail_mem=116.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.99it/s]Capturing num tokens (num_tokens=112 avail_mem=116.17 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.99it/s]Capturing num tokens (num_tokens=96 avail_mem=116.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.99it/s] Capturing num tokens (num_tokens=80 avail_mem=116.16 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.99it/s]Capturing num tokens (num_tokens=80 avail_mem=116.16 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.33it/s]Capturing num tokens (num_tokens=64 avail_mem=116.16 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.33it/s]Capturing num tokens (num_tokens=48 avail_mem=116.16 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.33it/s]Capturing num tokens (num_tokens=32 avail_mem=116.15 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.33it/s]

    Capturing num tokens (num_tokens=28 avail_mem=116.15 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.33it/s]Capturing num tokens (num_tokens=28 avail_mem=116.15 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.41it/s]Capturing num tokens (num_tokens=24 avail_mem=116.14 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.41it/s]Capturing num tokens (num_tokens=20 avail_mem=116.14 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.41it/s]Capturing num tokens (num_tokens=16 avail_mem=116.14 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.41it/s]Capturing num tokens (num_tokens=12 avail_mem=116.13 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.41it/s]Capturing num tokens (num_tokens=12 avail_mem=116.13 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.38it/s]Capturing num tokens (num_tokens=8 avail_mem=116.13 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.38it/s] Capturing num tokens (num_tokens=4 avail_mem=116.13 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.38it/s]Capturing num tokens (num_tokens=4 avail_mem=116.13 GB): 100%|██████████| 58/58 [00:02<00:00, 25.33it/s]


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


<strong style='color: #00008B;'>{'id': 'f8b03d167b474367af22b207f3f3f5e8', 'object': 'chat.completion', 'created': 1776992161, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '96f948c40fe14fd9918bd718a93d9d98', 'object': 'chat.completion', 'created': 1776992161, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='83f4630055904b5595eaf35f04a9f057', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1776992162, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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

     respective

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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '5138aefc20b64f798ee445bd40011c65', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2654476137831807, 'response_sent_to_client_ts': 1776992163.036066}}</strong>


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
