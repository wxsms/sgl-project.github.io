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
    [2026-04-23 21:58:49] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 21:58:50] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-23 21:58:51] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-23 21:58:59] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]


    2026-04-23 21:59:04,071 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 21:59:04] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:08,  1.23s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:08,  1.23s/it]

    Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:03<01:08,  1.23s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:27,  1.94it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:15,  3.25it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:15,  3.25it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:03<00:15,  3.25it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.76it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.76it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:03<00:10,  4.76it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.53it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:04,  9.29it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:04,  9.29it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:04,  9.29it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:04,  9.29it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 12.01it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 14.73it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 14.73it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 14.73it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 14.73it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:02, 17.21it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:02, 17.21it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 17.21it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 17.21it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 19.75it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 19.75it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 24.70it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 24.70it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 24.70it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 24.70it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 25.78it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 25.60it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 25.60it/s]

    Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 25.60it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 25.60it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 25.60it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 27.78it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 27.78it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 27.78it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 27.78it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 27.89it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 27.89it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 27.89it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 27.91it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 27.91it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 27.91it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 27.91it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]

    Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 28.07it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 33.70it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 33.70it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 33.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=98.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=98.06 GB):   2%|▏         | 1/58 [00:00<00:11,  4.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=98.03 GB):   2%|▏         | 1/58 [00:00<00:11,  4.83it/s]Capturing num tokens (num_tokens=7680 avail_mem=98.03 GB):   3%|▎         | 2/58 [00:00<00:08,  6.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=98.03 GB):   3%|▎         | 2/58 [00:00<00:08,  6.39it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=98.03 GB):   5%|▌         | 3/58 [00:00<00:09,  5.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=98.03 GB):   5%|▌         | 3/58 [00:00<00:09,  5.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=98.03 GB):   7%|▋         | 4/58 [00:00<00:09,  5.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=98.03 GB):   7%|▋         | 4/58 [00:00<00:09,  5.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=98.03 GB):   9%|▊         | 5/58 [00:00<00:09,  5.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=98.02 GB):   9%|▊         | 5/58 [00:00<00:09,  5.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=98.02 GB):  10%|█         | 6/58 [00:01<00:09,  5.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=98.03 GB):  10%|█         | 6/58 [00:01<00:09,  5.57it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=98.03 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=98.02 GB):  12%|█▏        | 7/58 [00:01<00:09,  5.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=98.02 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=98.02 GB):  14%|█▍        | 8/58 [00:01<00:08,  5.78it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=98.02 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=98.02 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=98.02 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=98.01 GB):  17%|█▋        | 10/58 [00:01<00:07,  6.11it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=98.01 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=98.01 GB):  19%|█▉        | 11/58 [00:01<00:07,  6.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=98.01 GB):  21%|██        | 12/58 [00:01<00:06,  6.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=98.01 GB):  21%|██        | 12/58 [00:01<00:06,  6.81it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=98.01 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=98.01 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=98.01 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=98.00 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.43it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=98.00 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=98.00 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=98.00 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=97.99 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=97.99 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=97.98 GB):  28%|██▊       | 16/58 [00:02<00:04, 10.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=97.98 GB):  34%|███▍      | 20/58 [00:02<00:02, 16.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=97.96 GB):  34%|███▍      | 20/58 [00:02<00:02, 16.40it/s]Capturing num tokens (num_tokens=960 avail_mem=97.98 GB):  34%|███▍      | 20/58 [00:02<00:02, 16.40it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=97.98 GB):  34%|███▍      | 20/58 [00:02<00:02, 16.40it/s]Capturing num tokens (num_tokens=896 avail_mem=97.98 GB):  40%|███▉      | 23/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=832 avail_mem=97.97 GB):  40%|███▉      | 23/58 [00:02<00:01, 19.23it/s]Capturing num tokens (num_tokens=768 avail_mem=97.97 GB):  40%|███▉      | 23/58 [00:02<00:01, 19.23it/s]

    Capturing num tokens (num_tokens=768 avail_mem=97.97 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.25it/s]Capturing num tokens (num_tokens=704 avail_mem=116.21 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.25it/s]Capturing num tokens (num_tokens=640 avail_mem=116.21 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.25it/s]Capturing num tokens (num_tokens=576 avail_mem=116.21 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.25it/s]Capturing num tokens (num_tokens=512 avail_mem=116.20 GB):  43%|████▎     | 25/58 [00:02<00:02, 15.25it/s]Capturing num tokens (num_tokens=512 avail_mem=116.20 GB):  50%|█████     | 29/58 [00:02<00:01, 20.11it/s]Capturing num tokens (num_tokens=480 avail_mem=116.21 GB):  50%|█████     | 29/58 [00:02<00:01, 20.11it/s]Capturing num tokens (num_tokens=448 avail_mem=116.21 GB):  50%|█████     | 29/58 [00:02<00:01, 20.11it/s]Capturing num tokens (num_tokens=416 avail_mem=116.21 GB):  50%|█████     | 29/58 [00:02<00:01, 20.11it/s]Capturing num tokens (num_tokens=384 avail_mem=116.21 GB):  50%|█████     | 29/58 [00:02<00:01, 20.11it/s]

    Capturing num tokens (num_tokens=384 avail_mem=116.21 GB):  57%|█████▋    | 33/58 [00:03<00:01, 24.12it/s]Capturing num tokens (num_tokens=352 avail_mem=116.20 GB):  57%|█████▋    | 33/58 [00:03<00:01, 24.12it/s]Capturing num tokens (num_tokens=320 avail_mem=116.20 GB):  57%|█████▋    | 33/58 [00:03<00:01, 24.12it/s]Capturing num tokens (num_tokens=288 avail_mem=116.20 GB):  57%|█████▋    | 33/58 [00:03<00:01, 24.12it/s]Capturing num tokens (num_tokens=256 avail_mem=116.19 GB):  57%|█████▋    | 33/58 [00:03<00:01, 24.12it/s]Capturing num tokens (num_tokens=256 avail_mem=116.19 GB):  64%|██████▍   | 37/58 [00:03<00:00, 27.09it/s]Capturing num tokens (num_tokens=240 avail_mem=116.19 GB):  64%|██████▍   | 37/58 [00:03<00:00, 27.09it/s]Capturing num tokens (num_tokens=224 avail_mem=116.19 GB):  64%|██████▍   | 37/58 [00:03<00:00, 27.09it/s]Capturing num tokens (num_tokens=208 avail_mem=116.18 GB):  64%|██████▍   | 37/58 [00:03<00:00, 27.09it/s]Capturing num tokens (num_tokens=192 avail_mem=116.18 GB):  64%|██████▍   | 37/58 [00:03<00:00, 27.09it/s]

    Capturing num tokens (num_tokens=192 avail_mem=116.18 GB):  71%|███████   | 41/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=176 avail_mem=116.18 GB):  71%|███████   | 41/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=160 avail_mem=116.18 GB):  71%|███████   | 41/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=144 avail_mem=116.17 GB):  71%|███████   | 41/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=128 avail_mem=116.17 GB):  71%|███████   | 41/58 [00:03<00:00, 29.48it/s]Capturing num tokens (num_tokens=128 avail_mem=116.17 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=112 avail_mem=116.17 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=96 avail_mem=116.16 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.06it/s] Capturing num tokens (num_tokens=80 avail_mem=116.16 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.06it/s]Capturing num tokens (num_tokens=64 avail_mem=116.16 GB):  78%|███████▊  | 45/58 [00:03<00:00, 31.06it/s]

    Capturing num tokens (num_tokens=64 avail_mem=116.16 GB):  84%|████████▍ | 49/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=48 avail_mem=116.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=32 avail_mem=116.15 GB):  84%|████████▍ | 49/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=28 avail_mem=116.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=24 avail_mem=116.14 GB):  84%|████████▍ | 49/58 [00:03<00:00, 32.18it/s]Capturing num tokens (num_tokens=24 avail_mem=116.14 GB):  91%|█████████▏| 53/58 [00:03<00:00, 33.30it/s]Capturing num tokens (num_tokens=20 avail_mem=116.14 GB):  91%|█████████▏| 53/58 [00:03<00:00, 33.30it/s]Capturing num tokens (num_tokens=16 avail_mem=116.14 GB):  91%|█████████▏| 53/58 [00:03<00:00, 33.30it/s]Capturing num tokens (num_tokens=12 avail_mem=116.13 GB):  91%|█████████▏| 53/58 [00:03<00:00, 33.30it/s]Capturing num tokens (num_tokens=8 avail_mem=116.13 GB):  91%|█████████▏| 53/58 [00:03<00:00, 33.30it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=116.13 GB):  98%|█████████▊| 57/58 [00:03<00:00, 34.01it/s]Capturing num tokens (num_tokens=4 avail_mem=116.13 GB):  98%|█████████▊| 57/58 [00:03<00:00, 34.01it/s]Capturing num tokens (num_tokens=4 avail_mem=116.13 GB): 100%|██████████| 58/58 [00:03<00:00, 15.54it/s]


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


<strong style='color: #00008B;'>{'id': 'f58481c4626f4b65b95ae2644d1e48b0', 'object': 'chat.completion', 'created': 1776981560, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '6af91599d3184d5abef13a8d48f4b84b', 'object': 'chat.completion', 'created': 1776981560, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='faf52e53be4c4c828beed3442fe33ece', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their respective capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1776981561, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=38, prompt_tokens=37, total_tokens=75, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '986ab17678bb4253a7784d1e1f5622a2', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.25817520823329687, 'response_sent_to_client_ts': 1776981561.8971984}}</strong>


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
