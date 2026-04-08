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


    [2026-04-08 21:54:45] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 21:54:45] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 21:54:45] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 21:54:45] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


    2026-04-08 21:54:46,390 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 21:54:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.26it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:02<00:03, 11.97it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:03<00:01, 19.51it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 27.55it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 34.31it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 40.68it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 50.54it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 50.54it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 50.54it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.26 GB):   2%|▏         | 1/58 [00:00<00:08,  6.78it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.23 GB):   2%|▏         | 1/58 [00:00<00:08,  6.78it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.23 GB):   3%|▎         | 2/58 [00:00<00:08,  6.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.23 GB):   3%|▎         | 2/58 [00:00<00:08,  6.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.23 GB):   5%|▌         | 3/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.23 GB):   5%|▌         | 3/58 [00:00<00:07,  7.10it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.23 GB):   7%|▋         | 4/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.23 GB):   7%|▋         | 4/58 [00:00<00:07,  7.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=38.23 GB):   9%|▊         | 5/58 [00:00<00:06,  7.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.22 GB):   9%|▊         | 5/58 [00:00<00:06,  7.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.22 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.23 GB):  10%|█         | 6/58 [00:00<00:06,  8.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.23 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.22 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.31it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=38.22 GB):  12%|█▏        | 7/58 [00:00<00:06,  8.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.22 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.21 GB):  16%|█▌        | 9/58 [00:01<00:05,  9.00it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=38.21 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.21 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.21 GB):  19%|█▉        | 11/58 [00:01<00:04,  9.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.21 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.21 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.03it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.20 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.20 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.20 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.19 GB):  26%|██▌       | 15/58 [00:01<00:04, 10.26it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.19 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.19 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.19 GB):  29%|██▉       | 17/58 [00:01<00:03, 10.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.19 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.18 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.84it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=38.16 GB):  33%|███▎      | 19/58 [00:02<00:03, 10.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.16 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.57it/s]Capturing num tokens (num_tokens=960 avail_mem=38.18 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.57it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=38.17 GB):  36%|███▌      | 21/58 [00:02<00:03,  9.57it/s]Capturing num tokens (num_tokens=896 avail_mem=38.17 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.51it/s]Capturing num tokens (num_tokens=832 avail_mem=38.17 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.51it/s]Capturing num tokens (num_tokens=768 avail_mem=38.17 GB):  40%|███▉      | 23/58 [00:02<00:03, 10.51it/s]

    Capturing num tokens (num_tokens=768 avail_mem=38.17 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.87it/s]Capturing num tokens (num_tokens=704 avail_mem=38.16 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.87it/s]Capturing num tokens (num_tokens=640 avail_mem=38.16 GB):  43%|████▎     | 25/58 [00:02<00:03, 10.87it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.16 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.30it/s]Capturing num tokens (num_tokens=576 avail_mem=38.16 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.30it/s]Capturing num tokens (num_tokens=512 avail_mem=38.15 GB):  47%|████▋     | 27/58 [00:02<00:03, 10.30it/s]Capturing num tokens (num_tokens=512 avail_mem=38.15 GB):  50%|█████     | 29/58 [00:02<00:02, 10.47it/s]Capturing num tokens (num_tokens=480 avail_mem=38.16 GB):  50%|█████     | 29/58 [00:02<00:02, 10.47it/s]

    Capturing num tokens (num_tokens=448 avail_mem=38.16 GB):  50%|█████     | 29/58 [00:03<00:02, 10.47it/s]Capturing num tokens (num_tokens=448 avail_mem=38.16 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.41it/s]Capturing num tokens (num_tokens=416 avail_mem=38.16 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.41it/s]Capturing num tokens (num_tokens=384 avail_mem=38.16 GB):  53%|█████▎    | 31/58 [00:03<00:02, 11.41it/s]Capturing num tokens (num_tokens=384 avail_mem=38.16 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.78it/s]Capturing num tokens (num_tokens=352 avail_mem=38.15 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.78it/s]

    Capturing num tokens (num_tokens=320 avail_mem=38.15 GB):  57%|█████▋    | 33/58 [00:03<00:01, 12.78it/s]Capturing num tokens (num_tokens=320 avail_mem=38.15 GB):  60%|██████    | 35/58 [00:03<00:01, 13.15it/s]Capturing num tokens (num_tokens=288 avail_mem=38.14 GB):  60%|██████    | 35/58 [00:03<00:01, 13.15it/s]Capturing num tokens (num_tokens=256 avail_mem=38.14 GB):  60%|██████    | 35/58 [00:03<00:01, 13.15it/s]Capturing num tokens (num_tokens=240 avail_mem=38.14 GB):  60%|██████    | 35/58 [00:03<00:01, 13.15it/s]

    Capturing num tokens (num_tokens=240 avail_mem=38.14 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.07it/s]Capturing num tokens (num_tokens=224 avail_mem=38.14 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.07it/s]Capturing num tokens (num_tokens=208 avail_mem=38.13 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.07it/s]Capturing num tokens (num_tokens=192 avail_mem=38.13 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.07it/s]Capturing num tokens (num_tokens=176 avail_mem=38.12 GB):  66%|██████▌   | 38/58 [00:03<00:01, 15.07it/s]Capturing num tokens (num_tokens=176 avail_mem=38.12 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.40it/s]Capturing num tokens (num_tokens=160 avail_mem=38.12 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.40it/s]Capturing num tokens (num_tokens=144 avail_mem=38.11 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.40it/s]Capturing num tokens (num_tokens=128 avail_mem=38.11 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.40it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.11 GB):  72%|███████▏  | 42/58 [00:03<00:00, 19.40it/s]Capturing num tokens (num_tokens=112 avail_mem=38.11 GB):  79%|███████▉  | 46/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=96 avail_mem=38.10 GB):  79%|███████▉  | 46/58 [00:03<00:00, 23.10it/s] Capturing num tokens (num_tokens=80 avail_mem=38.10 GB):  79%|███████▉  | 46/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=64 avail_mem=38.09 GB):  79%|███████▉  | 46/58 [00:03<00:00, 23.10it/s]Capturing num tokens (num_tokens=64 avail_mem=38.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 23.54it/s]Capturing num tokens (num_tokens=48 avail_mem=38.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 23.54it/s]Capturing num tokens (num_tokens=32 avail_mem=38.09 GB):  84%|████████▍ | 49/58 [00:03<00:00, 23.54it/s]

    Capturing num tokens (num_tokens=28 avail_mem=38.05 GB):  84%|████████▍ | 49/58 [00:04<00:00, 23.54it/s]Capturing num tokens (num_tokens=28 avail_mem=38.05 GB):  90%|████████▉ | 52/58 [00:04<00:00, 21.12it/s]Capturing num tokens (num_tokens=24 avail_mem=38.05 GB):  90%|████████▉ | 52/58 [00:04<00:00, 21.12it/s]Capturing num tokens (num_tokens=20 avail_mem=38.05 GB):  90%|████████▉ | 52/58 [00:04<00:00, 21.12it/s]Capturing num tokens (num_tokens=16 avail_mem=38.05 GB):  90%|████████▉ | 52/58 [00:04<00:00, 21.12it/s]Capturing num tokens (num_tokens=12 avail_mem=38.04 GB):  90%|████████▉ | 52/58 [00:04<00:00, 21.12it/s]Capturing num tokens (num_tokens=12 avail_mem=38.04 GB):  97%|█████████▋| 56/58 [00:04<00:00, 25.09it/s]Capturing num tokens (num_tokens=8 avail_mem=38.04 GB):  97%|█████████▋| 56/58 [00:04<00:00, 25.09it/s] Capturing num tokens (num_tokens=4 avail_mem=38.03 GB):  97%|█████████▋| 56/58 [00:04<00:00, 25.09it/s]Capturing num tokens (num_tokens=4 avail_mem=38.03 GB): 100%|██████████| 58/58 [00:04<00:00, 13.72it/s]


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


<strong style='color: #00008B;'>{'id': 'c85e1019404c48388146583596bc4951', 'object': 'chat.completion', 'created': 1775685301, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '3441692e3d824e718b7942a45a51f648', 'object': 'chat.completion', 'created': 1775685301, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='ec74fbf5f1ad4b45ba68cdda4ceeb64a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1775685302, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': '10bbed3c9d53482996e10b54c0abffe0', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.19339475687593222, 'response_sent_to_client_ts': 1775685302.9230545}}</strong>


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
