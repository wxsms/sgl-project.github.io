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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-04-09 05:43:02] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 05:43:02] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 05:43:02] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 05:43:02] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.63it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]


    2026-04-09 05:43:03,180 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 05:43:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:01,  1.10s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:24,  2.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:24,  2.21it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:24,  2.21it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:13,  3.79it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:13,  3.79it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:13,  3.79it/s]

    Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:13,  3.79it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:07,  6.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]

    Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:04,  9.63it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:02, 14.22it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:02, 14.22it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:02, 14.22it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:02, 14.22it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:02, 14.22it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 18.68it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 24.15it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:00, 28.13it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 32.87it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 36.54it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 39.18it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 43.38it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:04<00:00, 47.75it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:04<00:00, 47.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.29 GB):   2%|▏         | 1/58 [00:00<00:08,  6.39it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.23 GB):   2%|▏         | 1/58 [00:00<00:08,  6.39it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.23 GB):   3%|▎         | 2/58 [00:00<00:08,  6.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.22 GB):   3%|▎         | 2/58 [00:00<00:08,  6.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.22 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.22 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.22 GB):   5%|▌         | 3/58 [00:00<00:07,  7.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.22 GB):   9%|▊         | 5/58 [00:00<00:05,  8.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.40 GB):   9%|▊         | 5/58 [00:00<00:05,  8.85it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.40 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.40 GB):  10%|█         | 6/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.40 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.78it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.22 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.80it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.22 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.44 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.44 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.21 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.21 GB):  21%|██        | 12/58 [00:01<00:05,  7.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.50 GB):  21%|██        | 12/58 [00:01<00:05,  7.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.50 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.50 GB):  22%|██▏       | 13/58 [00:01<00:05,  8.07it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=57.50 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.21 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.55 GB):  24%|██▍       | 14/58 [00:01<00:05,  8.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.55 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.54 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.46it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=58.20 GB):  28%|██▊       | 16/58 [00:01<00:04,  9.46it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.20 GB):  31%|███       | 18/58 [00:02<00:03, 10.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.20 GB):  31%|███       | 18/58 [00:02<00:03, 10.58it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=57.59 GB):  31%|███       | 18/58 [00:02<00:03, 10.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.59 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.18 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s]Capturing num tokens (num_tokens=960 avail_mem=58.19 GB):  34%|███▍      | 20/58 [00:02<00:04,  9.13it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=58.19 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.39it/s]Capturing num tokens (num_tokens=896 avail_mem=57.64 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.39it/s]Capturing num tokens (num_tokens=832 avail_mem=57.64 GB):  38%|███▊      | 22/58 [00:02<00:03, 10.39it/s]Capturing num tokens (num_tokens=832 avail_mem=57.64 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.35it/s]Capturing num tokens (num_tokens=768 avail_mem=58.18 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.35it/s]

    Capturing num tokens (num_tokens=704 avail_mem=57.69 GB):  41%|████▏     | 24/58 [00:02<00:02, 11.35it/s]Capturing num tokens (num_tokens=704 avail_mem=57.69 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.81it/s]Capturing num tokens (num_tokens=640 avail_mem=57.68 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.81it/s]Capturing num tokens (num_tokens=576 avail_mem=58.18 GB):  45%|████▍     | 26/58 [00:02<00:02, 11.81it/s]

    Capturing num tokens (num_tokens=576 avail_mem=58.18 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.96it/s]Capturing num tokens (num_tokens=512 avail_mem=57.73 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.96it/s]Capturing num tokens (num_tokens=480 avail_mem=58.19 GB):  48%|████▊     | 28/58 [00:02<00:02, 12.96it/s]Capturing num tokens (num_tokens=480 avail_mem=58.19 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.92it/s]Capturing num tokens (num_tokens=448 avail_mem=58.18 GB):  52%|█████▏    | 30/58 [00:02<00:02, 13.92it/s]Capturing num tokens (num_tokens=416 avail_mem=57.77 GB):  52%|█████▏    | 30/58 [00:03<00:02, 13.92it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.77 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.32it/s]Capturing num tokens (num_tokens=384 avail_mem=57.89 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.32it/s]Capturing num tokens (num_tokens=352 avail_mem=58.18 GB):  55%|█████▌    | 32/58 [00:03<00:01, 14.32it/s]Capturing num tokens (num_tokens=352 avail_mem=58.18 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.49it/s]Capturing num tokens (num_tokens=320 avail_mem=57.79 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.49it/s]

    Capturing num tokens (num_tokens=288 avail_mem=58.17 GB):  59%|█████▊    | 34/58 [00:03<00:01, 14.49it/s]Capturing num tokens (num_tokens=288 avail_mem=58.17 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.29it/s]Capturing num tokens (num_tokens=256 avail_mem=57.90 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.29it/s]Capturing num tokens (num_tokens=240 avail_mem=57.81 GB):  62%|██████▏   | 36/58 [00:03<00:01, 11.29it/s]Capturing num tokens (num_tokens=240 avail_mem=57.81 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.22it/s]Capturing num tokens (num_tokens=224 avail_mem=58.16 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.22it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.83 GB):  66%|██████▌   | 38/58 [00:03<00:01, 12.22it/s]Capturing num tokens (num_tokens=208 avail_mem=57.83 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.28it/s]Capturing num tokens (num_tokens=192 avail_mem=58.16 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.28it/s]Capturing num tokens (num_tokens=176 avail_mem=57.85 GB):  69%|██████▉   | 40/58 [00:03<00:01, 13.28it/s]Capturing num tokens (num_tokens=176 avail_mem=57.85 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.38it/s]Capturing num tokens (num_tokens=160 avail_mem=58.15 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.38it/s]Capturing num tokens (num_tokens=144 avail_mem=58.09 GB):  72%|███████▏  | 42/58 [00:03<00:01, 14.38it/s]

    Capturing num tokens (num_tokens=144 avail_mem=58.09 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.38it/s]Capturing num tokens (num_tokens=128 avail_mem=58.14 GB):  76%|███████▌  | 44/58 [00:03<00:00, 15.38it/s]Capturing num tokens (num_tokens=112 avail_mem=57.89 GB):  76%|███████▌  | 44/58 [00:04<00:00, 15.38it/s]Capturing num tokens (num_tokens=112 avail_mem=57.89 GB):  79%|███████▉  | 46/58 [00:04<00:00, 16.14it/s]Capturing num tokens (num_tokens=96 avail_mem=58.14 GB):  79%|███████▉  | 46/58 [00:04<00:00, 16.14it/s] Capturing num tokens (num_tokens=80 avail_mem=57.91 GB):  79%|███████▉  | 46/58 [00:04<00:00, 16.14it/s]Capturing num tokens (num_tokens=64 avail_mem=58.13 GB):  79%|███████▉  | 46/58 [00:04<00:00, 16.14it/s]

    Capturing num tokens (num_tokens=64 avail_mem=58.13 GB):  84%|████████▍ | 49/58 [00:04<00:00, 18.07it/s]Capturing num tokens (num_tokens=48 avail_mem=57.93 GB):  84%|████████▍ | 49/58 [00:04<00:00, 18.07it/s]Capturing num tokens (num_tokens=32 avail_mem=58.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 18.07it/s]Capturing num tokens (num_tokens=28 avail_mem=58.12 GB):  84%|████████▍ | 49/58 [00:04<00:00, 18.07it/s]Capturing num tokens (num_tokens=28 avail_mem=58.12 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.85it/s]Capturing num tokens (num_tokens=24 avail_mem=57.97 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.85it/s]Capturing num tokens (num_tokens=20 avail_mem=58.11 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.85it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.11 GB):  90%|████████▉ | 52/58 [00:04<00:00, 19.85it/s]Capturing num tokens (num_tokens=16 avail_mem=58.11 GB):  95%|█████████▍| 55/58 [00:04<00:00, 21.74it/s]Capturing num tokens (num_tokens=12 avail_mem=57.99 GB):  95%|█████████▍| 55/58 [00:04<00:00, 21.74it/s]Capturing num tokens (num_tokens=8 avail_mem=58.09 GB):  95%|█████████▍| 55/58 [00:04<00:00, 21.74it/s] Capturing num tokens (num_tokens=4 avail_mem=58.09 GB):  95%|█████████▍| 55/58 [00:04<00:00, 21.74it/s]Capturing num tokens (num_tokens=4 avail_mem=58.09 GB): 100%|██████████| 58/58 [00:04<00:00, 22.68it/s]Capturing num tokens (num_tokens=4 avail_mem=58.09 GB): 100%|██████████| 58/58 [00:04<00:00, 12.64it/s]


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


<strong style='color: #00008B;'>{'id': 'ece046e4b9fc4d73b5f507ea9c536ece', 'object': 'chat.completion', 'created': 1775713400, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '90e551d9d81b43fba4bf39a4b2b63417', 'object': 'chat.completion', 'created': 1775713400, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='f0016203401d4bb597f2a1e523bd05af', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1775713401, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '5c723eeefe78402987eaee2b432c9724', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.18646281701512635, 'response_sent_to_client_ts': 1775713401.4911842}}</strong>


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
