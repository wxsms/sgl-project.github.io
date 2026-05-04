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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.07it/s]


    2026-05-04 16:16:02,563 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 16:16:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:06,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.64it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.64it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.64it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.64it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.62it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:10,  4.62it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.02it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.83it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.83it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.83it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.83it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.83it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.71it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]

    Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.84it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 33.31it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 36.47it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 39.01it/s]

    Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 39.01it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 39.01it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 39.01it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:06<00:00, 42.81it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:06<00:00, 42.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.10 GB):   2%|▏         | 1/58 [00:00<00:07,  7.59it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.07 GB):   2%|▏         | 1/58 [00:00<00:07,  7.59it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=56.07 GB):   3%|▎         | 2/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.06 GB):   3%|▎         | 2/58 [00:00<00:07,  7.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.06 GB):   5%|▌         | 3/58 [00:00<00:07,  7.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.06 GB):   5%|▌         | 3/58 [00:00<00:07,  7.73it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.06 GB):   7%|▋         | 4/58 [00:00<00:06,  7.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.06 GB):   7%|▋         | 4/58 [00:00<00:06,  7.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.06 GB):   9%|▊         | 5/58 [00:00<00:06,  8.14it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.05 GB):   9%|▊         | 5/58 [00:00<00:06,  8.14it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=56.05 GB):  10%|█         | 6/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.04 GB):  10%|█         | 6/58 [00:00<00:06,  8.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.04 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.04 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.73it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.04 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.08it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.04 GB):  14%|█▍        | 8/58 [00:00<00:05,  9.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.04 GB):  14%|█▍        | 8/58 [00:01<00:05,  9.08it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=56.04 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.03 GB):  17%|█▋        | 10/58 [00:01<00:05,  9.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.03 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.03 GB):  19%|█▉        | 11/58 [00:01<00:05,  9.23it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.03 GB):  21%|██        | 12/58 [00:01<00:05,  8.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.02 GB):  21%|██        | 12/58 [00:01<00:05,  8.85it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.02 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.02 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.02 GB):  22%|██▏       | 13/58 [00:01<00:06,  7.08it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=56.02 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.02 GB):  26%|██▌       | 15/58 [00:01<00:05,  7.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.02 GB):  28%|██▊       | 16/58 [00:01<00:05,  7.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.01 GB):  28%|██▊       | 16/58 [00:01<00:05,  7.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=56.01 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.01 GB):  29%|██▉       | 17/58 [00:02<00:05,  7.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.01 GB):  31%|███       | 18/58 [00:02<00:05,  7.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.01 GB):  31%|███       | 18/58 [00:02<00:05,  7.74it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=56.01 GB):  33%|███▎      | 19/58 [00:02<00:04,  7.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.00 GB):  33%|███▎      | 19/58 [00:02<00:04,  7.90it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.00 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.98 GB):  34%|███▍      | 20/58 [00:02<00:04,  8.22it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.98 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=960 avail_mem=56.00 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s] Capturing num tokens (num_tokens=896 avail_mem=56.00 GB):  36%|███▌      | 21/58 [00:02<00:04,  8.63it/s]Capturing num tokens (num_tokens=896 avail_mem=56.00 GB):  40%|███▉      | 23/58 [00:02<00:03,  9.45it/s]Capturing num tokens (num_tokens=832 avail_mem=55.99 GB):  40%|███▉      | 23/58 [00:02<00:03,  9.45it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.99 GB):  40%|███▉      | 23/58 [00:02<00:03,  9.45it/s]Capturing num tokens (num_tokens=768 avail_mem=55.99 GB):  43%|████▎     | 25/58 [00:02<00:03,  9.94it/s]Capturing num tokens (num_tokens=704 avail_mem=55.98 GB):  43%|████▎     | 25/58 [00:02<00:03,  9.94it/s]Capturing num tokens (num_tokens=640 avail_mem=55.98 GB):  43%|████▎     | 25/58 [00:03<00:03,  9.94it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.98 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.35it/s]Capturing num tokens (num_tokens=576 avail_mem=55.98 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.35it/s]Capturing num tokens (num_tokens=512 avail_mem=55.96 GB):  47%|████▋     | 27/58 [00:03<00:02, 10.35it/s]Capturing num tokens (num_tokens=512 avail_mem=55.96 GB):  50%|█████     | 29/58 [00:03<00:02, 10.50it/s]Capturing num tokens (num_tokens=480 avail_mem=55.98 GB):  50%|█████     | 29/58 [00:03<00:02, 10.50it/s]

    Capturing num tokens (num_tokens=448 avail_mem=55.98 GB):  50%|█████     | 29/58 [00:03<00:02, 10.50it/s]Capturing num tokens (num_tokens=448 avail_mem=55.98 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.71it/s]Capturing num tokens (num_tokens=416 avail_mem=55.98 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.71it/s]Capturing num tokens (num_tokens=384 avail_mem=55.97 GB):  53%|█████▎    | 31/58 [00:03<00:02, 10.71it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.97 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.30it/s]Capturing num tokens (num_tokens=352 avail_mem=55.97 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.30it/s]Capturing num tokens (num_tokens=320 avail_mem=55.96 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.30it/s]Capturing num tokens (num_tokens=288 avail_mem=55.96 GB):  57%|█████▋    | 33/58 [00:03<00:02, 12.30it/s]Capturing num tokens (num_tokens=288 avail_mem=55.96 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.48it/s]Capturing num tokens (num_tokens=256 avail_mem=55.96 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.48it/s]

    Capturing num tokens (num_tokens=240 avail_mem=55.95 GB):  62%|██████▏   | 36/58 [00:03<00:01, 14.48it/s]Capturing num tokens (num_tokens=240 avail_mem=55.95 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=224 avail_mem=55.64 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.12it/s]

    Capturing num tokens (num_tokens=208 avail_mem=55.91 GB):  66%|██████▌   | 38/58 [00:03<00:01, 14.12it/s]Capturing num tokens (num_tokens=208 avail_mem=55.91 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.10it/s]Capturing num tokens (num_tokens=192 avail_mem=55.91 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.10it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.68 GB):  69%|██████▉   | 40/58 [00:04<00:01, 12.10it/s]Capturing num tokens (num_tokens=176 avail_mem=55.68 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.09it/s]Capturing num tokens (num_tokens=160 avail_mem=55.90 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.09it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.90 GB):  72%|███████▏  | 42/58 [00:04<00:01, 11.09it/s]Capturing num tokens (num_tokens=144 avail_mem=55.90 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.43it/s]Capturing num tokens (num_tokens=128 avail_mem=55.89 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.43it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.88 GB):  76%|███████▌  | 44/58 [00:04<00:01, 10.43it/s]Capturing num tokens (num_tokens=112 avail_mem=55.88 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.17it/s]Capturing num tokens (num_tokens=96 avail_mem=55.70 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.17it/s] Capturing num tokens (num_tokens=80 avail_mem=55.87 GB):  79%|███████▉  | 46/58 [00:04<00:01, 10.17it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.87 GB):  83%|████████▎ | 48/58 [00:04<00:00, 10.30it/s]Capturing num tokens (num_tokens=64 avail_mem=55.86 GB):  83%|████████▎ | 48/58 [00:04<00:00, 10.30it/s]Capturing num tokens (num_tokens=48 avail_mem=55.86 GB):  83%|████████▎ | 48/58 [00:05<00:00, 10.30it/s]Capturing num tokens (num_tokens=48 avail_mem=55.86 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.69it/s]Capturing num tokens (num_tokens=32 avail_mem=55.85 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.69it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.84 GB):  86%|████████▌ | 50/58 [00:05<00:00, 10.69it/s]Capturing num tokens (num_tokens=28 avail_mem=55.84 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.02it/s]Capturing num tokens (num_tokens=24 avail_mem=55.84 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.02it/s]Capturing num tokens (num_tokens=20 avail_mem=55.83 GB):  90%|████████▉ | 52/58 [00:05<00:00, 11.02it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.83 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.00it/s]Capturing num tokens (num_tokens=16 avail_mem=55.83 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.00it/s]Capturing num tokens (num_tokens=12 avail_mem=55.82 GB):  93%|█████████▎| 54/58 [00:05<00:00, 11.00it/s]Capturing num tokens (num_tokens=12 avail_mem=55.82 GB):  97%|█████████▋| 56/58 [00:05<00:00, 10.96it/s]Capturing num tokens (num_tokens=8 avail_mem=55.81 GB):  97%|█████████▋| 56/58 [00:05<00:00, 10.96it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=55.81 GB):  97%|█████████▋| 56/58 [00:05<00:00, 10.96it/s]Capturing num tokens (num_tokens=4 avail_mem=55.81 GB): 100%|██████████| 58/58 [00:05<00:00, 11.12it/s]Capturing num tokens (num_tokens=4 avail_mem=55.81 GB): 100%|██████████| 58/58 [00:05<00:00,  9.97it/s]


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


<strong style='color: #00008B;'>{'id': '4451a07d2bcf40f39b2b9c05e25cf015', 'object': 'chat.completion', 'created': 1777911382, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '0c35cce3527949bb8165f468a50e2e07', 'object': 'chat.completion', 'created': 1777911383, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='79e6af38db6d47189201afe7887745ed', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1777911383, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'f3cf74b120a54798b6dcdd08e8569971', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.2181732626631856, 'response_sent_to_client_ts': 1777911384.1438107}}</strong>


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
