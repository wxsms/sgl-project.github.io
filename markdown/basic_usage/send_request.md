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


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-07 18:50:26] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


    2026-05-07 18:50:28,847 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 18:50:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:53,  4.10s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.58it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.69it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.68it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:04<00:02, 14.32it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 21.76it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:04<00:00, 30.04it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:04<00:00, 38.94it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:04<00:00, 38.94it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:04<00:00, 38.94it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:04<00:00, 38.94it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:04<00:00, 38.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.72 GB):   3%|▎         | 2/58 [00:00<00:03, 14.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:03, 14.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:03, 14.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.71 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.71 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.70 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.69 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.69 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.69 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.69 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.69 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.69 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.68 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.68 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 24.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.66 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.84it/s]Capturing num tokens (num_tokens=960 avail_mem=55.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.84it/s] Capturing num tokens (num_tokens=896 avail_mem=55.65 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.84it/s]Capturing num tokens (num_tokens=768 avail_mem=55.64 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.84it/s]Capturing num tokens (num_tokens=768 avail_mem=55.64 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.17it/s]Capturing num tokens (num_tokens=704 avail_mem=55.64 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.17it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.63 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.17it/s]Capturing num tokens (num_tokens=576 avail_mem=55.63 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.17it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.63 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.38it/s]Capturing num tokens (num_tokens=512 avail_mem=55.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.38it/s]Capturing num tokens (num_tokens=480 avail_mem=55.63 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.38it/s]

    Capturing num tokens (num_tokens=448 avail_mem=55.63 GB):  48%|████▊     | 28/58 [00:01<00:01, 17.38it/s]Capturing num tokens (num_tokens=448 avail_mem=55.63 GB):  53%|█████▎    | 31/58 [00:01<00:01, 14.14it/s]Capturing num tokens (num_tokens=416 avail_mem=55.63 GB):  53%|█████▎    | 31/58 [00:01<00:01, 14.14it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.63 GB):  53%|█████▎    | 31/58 [00:01<00:01, 14.14it/s]Capturing num tokens (num_tokens=384 avail_mem=55.63 GB):  57%|█████▋    | 33/58 [00:01<00:01, 12.74it/s]Capturing num tokens (num_tokens=352 avail_mem=55.62 GB):  57%|█████▋    | 33/58 [00:01<00:01, 12.74it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.62 GB):  57%|█████▋    | 33/58 [00:01<00:01, 12.74it/s]Capturing num tokens (num_tokens=320 avail_mem=55.62 GB):  60%|██████    | 35/58 [00:02<00:01, 11.87it/s]Capturing num tokens (num_tokens=288 avail_mem=55.61 GB):  60%|██████    | 35/58 [00:02<00:01, 11.87it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.61 GB):  60%|██████    | 35/58 [00:02<00:01, 11.87it/s]Capturing num tokens (num_tokens=256 avail_mem=55.61 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.27it/s]Capturing num tokens (num_tokens=240 avail_mem=55.61 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.27it/s]

    Capturing num tokens (num_tokens=224 avail_mem=55.60 GB):  64%|██████▍   | 37/58 [00:02<00:01, 11.27it/s]Capturing num tokens (num_tokens=224 avail_mem=55.60 GB):  67%|██████▋   | 39/58 [00:02<00:01, 10.66it/s]Capturing num tokens (num_tokens=208 avail_mem=55.60 GB):  67%|██████▋   | 39/58 [00:02<00:01, 10.66it/s]Capturing num tokens (num_tokens=192 avail_mem=55.60 GB):  67%|██████▋   | 39/58 [00:02<00:01, 10.66it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.60 GB):  71%|███████   | 41/58 [00:02<00:01, 10.54it/s]Capturing num tokens (num_tokens=176 avail_mem=55.60 GB):  71%|███████   | 41/58 [00:02<00:01, 10.54it/s]Capturing num tokens (num_tokens=160 avail_mem=55.59 GB):  71%|███████   | 41/58 [00:02<00:01, 10.54it/s]Capturing num tokens (num_tokens=160 avail_mem=55.59 GB):  74%|███████▍  | 43/58 [00:02<00:01, 10.38it/s]Capturing num tokens (num_tokens=144 avail_mem=55.59 GB):  74%|███████▍  | 43/58 [00:02<00:01, 10.38it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.59 GB):  74%|███████▍  | 43/58 [00:03<00:01, 10.38it/s]Capturing num tokens (num_tokens=128 avail_mem=55.59 GB):  78%|███████▊  | 45/58 [00:03<00:01, 10.27it/s]Capturing num tokens (num_tokens=112 avail_mem=55.59 GB):  78%|███████▊  | 45/58 [00:03<00:01, 10.27it/s]Capturing num tokens (num_tokens=96 avail_mem=55.58 GB):  78%|███████▊  | 45/58 [00:03<00:01, 10.27it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=55.58 GB):  81%|████████  | 47/58 [00:03<00:01, 10.19it/s]Capturing num tokens (num_tokens=80 avail_mem=55.58 GB):  81%|████████  | 47/58 [00:03<00:01, 10.19it/s]Capturing num tokens (num_tokens=64 avail_mem=55.57 GB):  81%|████████  | 47/58 [00:03<00:01, 10.19it/s]Capturing num tokens (num_tokens=64 avail_mem=55.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 10.17it/s]Capturing num tokens (num_tokens=48 avail_mem=55.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 10.17it/s]

    Capturing num tokens (num_tokens=32 avail_mem=55.57 GB):  84%|████████▍ | 49/58 [00:03<00:00, 10.17it/s]Capturing num tokens (num_tokens=32 avail_mem=55.57 GB):  88%|████████▊ | 51/58 [00:03<00:00, 10.10it/s]Capturing num tokens (num_tokens=28 avail_mem=55.56 GB):  88%|████████▊ | 51/58 [00:03<00:00, 10.10it/s]Capturing num tokens (num_tokens=24 avail_mem=55.56 GB):  88%|████████▊ | 51/58 [00:03<00:00, 10.10it/s]

    Capturing num tokens (num_tokens=24 avail_mem=55.56 GB):  91%|█████████▏| 53/58 [00:03<00:00, 10.08it/s]Capturing num tokens (num_tokens=20 avail_mem=55.56 GB):  91%|█████████▏| 53/58 [00:03<00:00, 10.08it/s]Capturing num tokens (num_tokens=16 avail_mem=55.56 GB):  91%|█████████▏| 53/58 [00:04<00:00, 10.08it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.56 GB):  95%|█████████▍| 55/58 [00:04<00:00,  9.84it/s]Capturing num tokens (num_tokens=12 avail_mem=55.55 GB):  95%|█████████▍| 55/58 [00:04<00:00,  9.84it/s]Capturing num tokens (num_tokens=12 avail_mem=55.55 GB):  97%|█████████▋| 56/58 [00:04<00:00,  9.82it/s]Capturing num tokens (num_tokens=8 avail_mem=55.55 GB):  97%|█████████▋| 56/58 [00:04<00:00,  9.82it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=55.55 GB):  98%|█████████▊| 57/58 [00:04<00:00,  9.82it/s]Capturing num tokens (num_tokens=4 avail_mem=55.55 GB):  98%|█████████▊| 57/58 [00:04<00:00,  9.82it/s]Capturing num tokens (num_tokens=4 avail_mem=55.55 GB): 100%|██████████| 58/58 [00:04<00:00, 13.15it/s]


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


<strong style='color: #00008B;'>{'id': '7aff0736dfbe4f5aa8db9d90f372d4ac', 'object': 'chat.completion', 'created': 1778179845, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>{'id': '688e30cea79144b8a5a9809d47b68ab2', 'object': 'chat.completion', 'created': 1778179845, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is Paris.', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 36, 'total_tokens': 44, 'completion_tokens': 8, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}</strong>


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


<strong style='color: #00008B;'>ChatCompletion(id='b72a0b8595414ddb9f74133651fe3803', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Sure, here are three countries and their capitals:\n\n1. **United States** - Washington D.C.\n2. **Canada** - Ottawa\n3. **Australia** - Canberra', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content=None), matched_stop=151645)], created=1778179846, model='qwen/qwen2.5-0.5b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=37, prompt_tokens=37, total_tokens=74, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})</strong>


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


<strong style='color: #00008B;'>{'text': ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks of the', 'output_ids': [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 3283, 304, 279, 1879, 13, 1084, 374, 7407, 304, 279, 9806, 315, 9625, 11, 389, 279, 13959, 315, 279], 'meta_info': {'id': 'e77e35ab3b9f4bf7884e84285725b30b', 'finish_reason': {'type': 'length', 'length': 32}, 'prompt_tokens': 5, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 32, 'cached_tokens': 2, 'cached_tokens_details': {'device': 2, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.22351694712415338, 'response_sent_to_client_ts': 1778179846.8148484}}</strong>


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
