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

    [2026-03-20 01:06:22] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-20 01:06:22] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-20 01:06:22] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 01:06:27] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:06:27] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:06:27] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-20 01:06:29] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:06:30] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.
    [2026-03-20 01:06:30] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:175: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [2026-03-20 01:06:31] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:06:37] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:06:37] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:06:37] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-20 01:06:37] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:06:37] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:06:37] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 01:06:39] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:06:40] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.47it/s]

    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.26it/s]

    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.17it/s]

    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.17it/s]Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:58,  3.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:58,  3.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:04<01:41,  1.82s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:04<01:41,  1.82s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:30,  1.73it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:30,  1.73it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:24,  2.05it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:24,  2.05it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:21,  2.36it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:21,  2.36it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:17,  2.73it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:17,  2.73it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:15,  3.12it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:15,  3.12it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:13,  3.47it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:13,  3.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:11,  3.86it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:11,  3.86it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:09,  4.66it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:09,  4.66it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:08,  5.13it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:08,  5.13it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:07,  5.67it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:07,  5.67it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:06,  6.23it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:06,  6.23it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:05,  6.85it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:05,  6.85it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:08<00:05,  6.85it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:04,  8.26it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:04,  8.26it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:04,  8.26it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:08<00:03,  9.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:08<00:03,  9.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:08<00:03,  9.91it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:02, 11.34it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:02, 11.34it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:02, 11.34it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:08<00:02, 12.90it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:08<00:02, 12.90it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:08<00:02, 12.90it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:08<00:02, 14.36it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:08<00:02, 14.36it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:08<00:02, 14.36it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:08<00:02, 14.36it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:08<00:01, 17.14it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:08<00:01, 17.14it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:08<00:01, 17.14it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:08<00:01, 17.14it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:08<00:01, 19.11it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:08<00:01, 19.11it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:08<00:01, 19.11it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:08<00:01, 19.11it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:08<00:01, 20.81it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:08<00:01, 20.81it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:08<00:01, 20.81it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:09<00:01, 20.81it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:09<00:01, 20.81it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:09<00:00, 23.70it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:09<00:00, 23.70it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:09<00:00, 23.70it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:09<00:00, 23.70it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:09<00:00, 25.00it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:09<00:00, 25.00it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:09<00:00, 25.00it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:09<00:00, 25.00it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:09<00:00, 25.00it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:09<00:00, 27.43it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:09<00:00, 27.43it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:09<00:00, 27.43it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:09<00:00, 27.43it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:09<00:00, 27.43it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:09<00:00, 29.22it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:09<00:00, 29.22it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:09<00:00, 29.22it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:09<00:00, 29.22it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:09<00:00, 29.22it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:09<00:00, 30.80it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:09<00:00, 30.80it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:09<00:00, 30.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=104.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=104.66 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=104.64 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=104.64 GB):   3%|▎         | 2/58 [00:01<00:36,  1.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=104.65 GB):   3%|▎         | 2/58 [00:01<00:36,  1.55it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=104.65 GB):   5%|▌         | 3/58 [00:01<00:35,  1.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=104.65 GB):   5%|▌         | 3/58 [00:01<00:35,  1.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=104.65 GB):   7%|▋         | 4/58 [00:04<01:12,  1.34s/it]Capturing num tokens (num_tokens=6144 avail_mem=111.69 GB):   7%|▋         | 4/58 [00:04<01:12,  1.34s/it]

    Capturing num tokens (num_tokens=6144 avail_mem=111.69 GB):   9%|▊         | 5/58 [00:04<00:54,  1.02s/it]Capturing num tokens (num_tokens=5632 avail_mem=105.08 GB):   9%|▊         | 5/58 [00:04<00:54,  1.02s/it]

    Capturing num tokens (num_tokens=5632 avail_mem=105.08 GB):  10%|█         | 6/58 [00:05<00:41,  1.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=103.65 GB):  10%|█         | 6/58 [00:05<00:41,  1.24it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=103.65 GB):  12%|█▏        | 7/58 [00:05<00:33,  1.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=103.65 GB):  12%|█▏        | 7/58 [00:05<00:33,  1.50it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=103.65 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=103.65 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=103.65 GB):  16%|█▌        | 9/58 [00:06<00:20,  2.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=103.66 GB):  16%|█▌        | 9/58 [00:06<00:20,  2.37it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=103.66 GB):  17%|█▋        | 10/58 [00:06<00:15,  3.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=103.65 GB):  17%|█▋        | 10/58 [00:06<00:15,  3.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=103.65 GB):  19%|█▉        | 11/58 [00:06<00:12,  3.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=111.70 GB):  19%|█▉        | 11/58 [00:06<00:12,  3.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=111.70 GB):  21%|██        | 12/58 [00:06<00:11,  4.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=103.85 GB):  21%|██        | 12/58 [00:06<00:11,  4.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=103.85 GB):  22%|██▏       | 13/58 [00:06<00:09,  4.64it/s]Capturing num tokens (num_tokens=2816 avail_mem=103.21 GB):  22%|██▏       | 13/58 [00:06<00:09,  4.64it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=103.21 GB):  24%|██▍       | 14/58 [00:06<00:08,  5.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=103.21 GB):  24%|██▍       | 14/58 [00:06<00:08,  5.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=103.21 GB):  26%|██▌       | 15/58 [00:06<00:06,  6.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=103.21 GB):  26%|██▌       | 15/58 [00:06<00:06,  6.16it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=103.21 GB):  28%|██▊       | 16/58 [00:06<00:06,  6.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=103.21 GB):  28%|██▊       | 16/58 [00:06<00:06,  6.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=103.22 GB):  28%|██▊       | 16/58 [00:07<00:06,  6.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=103.22 GB):  31%|███       | 18/58 [00:07<00:04,  8.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=103.21 GB):  31%|███       | 18/58 [00:07<00:04,  8.73it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=103.21 GB):  31%|███       | 18/58 [00:07<00:04,  8.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=103.21 GB):  34%|███▍      | 20/58 [00:07<00:03,  9.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=103.19 GB):  34%|███▍      | 20/58 [00:07<00:03,  9.86it/s]Capturing num tokens (num_tokens=960 avail_mem=103.19 GB):  34%|███▍      | 20/58 [00:07<00:03,  9.86it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=103.19 GB):  38%|███▊      | 22/58 [00:07<00:03, 10.08it/s]Capturing num tokens (num_tokens=896 avail_mem=103.18 GB):  38%|███▊      | 22/58 [00:07<00:03, 10.08it/s]Capturing num tokens (num_tokens=832 avail_mem=102.69 GB):  38%|███▊      | 22/58 [00:07<00:03, 10.08it/s]Capturing num tokens (num_tokens=832 avail_mem=102.69 GB):  41%|████▏     | 24/58 [00:07<00:02, 11.36it/s]Capturing num tokens (num_tokens=768 avail_mem=102.58 GB):  41%|████▏     | 24/58 [00:07<00:02, 11.36it/s]Capturing num tokens (num_tokens=704 avail_mem=102.52 GB):  41%|████▏     | 24/58 [00:07<00:02, 11.36it/s]

    Capturing num tokens (num_tokens=640 avail_mem=102.51 GB):  41%|████▏     | 24/58 [00:07<00:02, 11.36it/s]Capturing num tokens (num_tokens=640 avail_mem=102.51 GB):  47%|████▋     | 27/58 [00:07<00:02, 14.08it/s]Capturing num tokens (num_tokens=576 avail_mem=110.16 GB):  47%|████▋     | 27/58 [00:07<00:02, 14.08it/s]Capturing num tokens (num_tokens=512 avail_mem=110.15 GB):  47%|████▋     | 27/58 [00:07<00:02, 14.08it/s]

    Capturing num tokens (num_tokens=512 avail_mem=110.15 GB):  50%|█████     | 29/58 [00:07<00:02, 12.01it/s]Capturing num tokens (num_tokens=480 avail_mem=110.15 GB):  50%|█████     | 29/58 [00:07<00:02, 12.01it/s]Capturing num tokens (num_tokens=448 avail_mem=110.15 GB):  50%|█████     | 29/58 [00:08<00:02, 12.01it/s]

    Capturing num tokens (num_tokens=448 avail_mem=110.15 GB):  53%|█████▎    | 31/58 [00:08<00:02, 10.85it/s]Capturing num tokens (num_tokens=416 avail_mem=110.14 GB):  53%|█████▎    | 31/58 [00:08<00:02, 10.85it/s]Capturing num tokens (num_tokens=384 avail_mem=110.14 GB):  53%|█████▎    | 31/58 [00:08<00:02, 10.85it/s]

    Capturing num tokens (num_tokens=384 avail_mem=110.14 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.10it/s]Capturing num tokens (num_tokens=352 avail_mem=110.13 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.10it/s]Capturing num tokens (num_tokens=320 avail_mem=110.13 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.10it/s]

    Capturing num tokens (num_tokens=320 avail_mem=110.13 GB):  60%|██████    | 35/58 [00:08<00:02,  9.71it/s]Capturing num tokens (num_tokens=288 avail_mem=109.85 GB):  60%|██████    | 35/58 [00:08<00:02,  9.71it/s]Capturing num tokens (num_tokens=256 avail_mem=109.15 GB):  60%|██████    | 35/58 [00:08<00:02,  9.71it/s]

    Capturing num tokens (num_tokens=256 avail_mem=109.15 GB):  64%|██████▍   | 37/58 [00:08<00:02,  9.60it/s]Capturing num tokens (num_tokens=240 avail_mem=109.14 GB):  64%|██████▍   | 37/58 [00:08<00:02,  9.60it/s]Capturing num tokens (num_tokens=240 avail_mem=109.14 GB):  66%|██████▌   | 38/58 [00:08<00:02,  9.58it/s]Capturing num tokens (num_tokens=224 avail_mem=109.14 GB):  66%|██████▌   | 38/58 [00:08<00:02,  9.58it/s]

    Capturing num tokens (num_tokens=224 avail_mem=109.14 GB):  67%|██████▋   | 39/58 [00:09<00:01,  9.52it/s]Capturing num tokens (num_tokens=208 avail_mem=109.14 GB):  67%|██████▋   | 39/58 [00:09<00:01,  9.52it/s]Capturing num tokens (num_tokens=208 avail_mem=109.14 GB):  69%|██████▉   | 40/58 [00:09<00:01,  9.43it/s]Capturing num tokens (num_tokens=192 avail_mem=109.13 GB):  69%|██████▉   | 40/58 [00:09<00:01,  9.43it/s]

    Capturing num tokens (num_tokens=192 avail_mem=109.13 GB):  71%|███████   | 41/58 [00:09<00:01,  9.41it/s]Capturing num tokens (num_tokens=176 avail_mem=109.13 GB):  71%|███████   | 41/58 [00:09<00:01,  9.41it/s]Capturing num tokens (num_tokens=176 avail_mem=109.13 GB):  72%|███████▏  | 42/58 [00:09<00:01,  9.51it/s]Capturing num tokens (num_tokens=160 avail_mem=109.13 GB):  72%|███████▏  | 42/58 [00:09<00:01,  9.51it/s]

    Capturing num tokens (num_tokens=160 avail_mem=109.13 GB):  74%|███████▍  | 43/58 [00:09<00:01,  9.57it/s]Capturing num tokens (num_tokens=144 avail_mem=108.79 GB):  74%|███████▍  | 43/58 [00:09<00:01,  9.57it/s]Capturing num tokens (num_tokens=144 avail_mem=108.79 GB):  76%|███████▌  | 44/58 [00:09<00:01,  9.43it/s]Capturing num tokens (num_tokens=128 avail_mem=108.70 GB):  76%|███████▌  | 44/58 [00:09<00:01,  9.43it/s]

    Capturing num tokens (num_tokens=128 avail_mem=108.70 GB):  78%|███████▊  | 45/58 [00:09<00:01,  9.41it/s]Capturing num tokens (num_tokens=112 avail_mem=107.01 GB):  78%|███████▊  | 45/58 [00:09<00:01,  9.41it/s]Capturing num tokens (num_tokens=112 avail_mem=107.01 GB):  79%|███████▉  | 46/58 [00:09<00:01,  9.32it/s]Capturing num tokens (num_tokens=96 avail_mem=105.19 GB):  79%|███████▉  | 46/58 [00:09<00:01,  9.32it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=105.19 GB):  81%|████████  | 47/58 [00:09<00:01,  9.43it/s]Capturing num tokens (num_tokens=80 avail_mem=105.18 GB):  81%|████████  | 47/58 [00:09<00:01,  9.43it/s]Capturing num tokens (num_tokens=64 avail_mem=105.18 GB):  81%|████████  | 47/58 [00:10<00:01,  9.43it/s]

    Capturing num tokens (num_tokens=64 avail_mem=105.18 GB):  84%|████████▍ | 49/58 [00:10<00:00,  9.66it/s]Capturing num tokens (num_tokens=48 avail_mem=105.18 GB):  84%|████████▍ | 49/58 [00:10<00:00,  9.66it/s]Capturing num tokens (num_tokens=48 avail_mem=105.18 GB):  86%|████████▌ | 50/58 [00:10<00:00,  9.71it/s]Capturing num tokens (num_tokens=32 avail_mem=105.17 GB):  86%|████████▌ | 50/58 [00:10<00:00,  9.71it/s]

    Capturing num tokens (num_tokens=28 avail_mem=105.17 GB):  86%|████████▌ | 50/58 [00:10<00:00,  9.71it/s]Capturing num tokens (num_tokens=28 avail_mem=105.17 GB):  90%|████████▉ | 52/58 [00:10<00:00,  9.81it/s]Capturing num tokens (num_tokens=24 avail_mem=105.17 GB):  90%|████████▉ | 52/58 [00:10<00:00,  9.81it/s]

    Capturing num tokens (num_tokens=24 avail_mem=105.17 GB):  91%|█████████▏| 53/58 [00:10<00:00,  8.10it/s]Capturing num tokens (num_tokens=20 avail_mem=105.16 GB):  91%|█████████▏| 53/58 [00:10<00:00,  8.10it/s]Capturing num tokens (num_tokens=20 avail_mem=105.16 GB):  93%|█████████▎| 54/58 [00:10<00:00,  8.41it/s]Capturing num tokens (num_tokens=16 avail_mem=105.16 GB):  93%|█████████▎| 54/58 [00:10<00:00,  8.41it/s]

    Capturing num tokens (num_tokens=12 avail_mem=105.15 GB):  93%|█████████▎| 54/58 [00:10<00:00,  8.41it/s]Capturing num tokens (num_tokens=12 avail_mem=105.15 GB):  97%|█████████▋| 56/58 [00:10<00:00,  9.01it/s]Capturing num tokens (num_tokens=8 avail_mem=105.15 GB):  97%|█████████▋| 56/58 [00:10<00:00,  9.01it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=105.15 GB):  98%|█████████▊| 57/58 [00:11<00:00,  9.18it/s]Capturing num tokens (num_tokens=4 avail_mem=105.14 GB):  98%|█████████▊| 57/58 [00:11<00:00,  9.18it/s]Capturing num tokens (num_tokens=4 avail_mem=105.14 GB): 100%|██████████| 58/58 [00:11<00:00,  5.21it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:37890


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-20 01:07:48] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Australia** - Canberra</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries along with their respective capitals:<br><br>1. France - Paris<br>2. Germany - Berlin<br>3. Italy - Rome</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries along with their respective capitals:<br><br>1. Japan - Tokyo<br>2. Canada - Ottawa<br>3. Brazil - Brasília</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>When we multiply 2 by 2, we get:<br><br>2 * 2 = 4<br><br>So, the answer is 4. You didn't really need a calculator for this, but here it is:<br><br>2 * 2 = 4<br><br>If you wanted to confirm this using a calculator, you could input `2 * 2 =` and it would show you the result, which is 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet:** A balanced diet is crucial for maintaining overall health. It involves eating a variety of foods from all major food groups to ensure you get essential nutrients, vitamins, and minerals. Focus on:<br>   - Fruits and vegetables: Rich in dietary fiber, vitamins, and minerals.<br>   - Whole grains: Provide energy and are rich in fiber.<br>   - Lean proteins: Good for muscle health and repair (e.g., poultry, fish, beans, legumes).<br>   - Healthy fats: From sources like avocado, nuts, seeds, and olive oil.<br>   - Staying hydrated: Drinking plenty of water is essential.<br><br>2. **Regular Exercise:** Regular physical activity is vital for various health benefits. It helps maintain a healthy weight, improves cardiovascular health, enhances immune function, and boosts mood. Key points include:<br>   - Cardiovascular exercise: For heart health and improved circulation.<br>   - Strength training: For building and maintaining muscle mass.<br>   - Combination of activities: To work different muscle groups and avoid plateaus.<br>   - Gradual progression: Increase intensity and duration over time to stay safe and effective.<br><br>Together, these two practices form a strong foundation for a healthy lifestyle.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Hornbeam",<br>        "core": "Phoenix feather",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Dementor"<br>}</strong>


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

     33%|███▎      | 1/3 [00:00<00:00,  4.98it/s]

     67%|██████▋   | 2/3 [00:00<00:00,  7.89it/s]

    100%|██████████| 3/3 [00:00<00:00, 13.25it/s]

    



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

    [2026-03-20 01:08:05] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:08:05] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:08:05] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-20 01:08:07] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
    [2026-03-20 01:08:07] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.
    [2026-03-20 01:08:07] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:175: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [2026-03-20 01:08:07] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:08:13] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:08:13] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:08:13] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-20 01:08:13] INFO utils.py:148: Note: detected 192 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 01:08:13] INFO utils.py:151: Note: NumExpr detected 192 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 01:08:13] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 01:08:15] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 01:08:18] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:01<00:05,  1.46s/it]

    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.11it/s]

    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:03<00:02,  1.16s/it]

    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:04<00:01,  1.30s/it]

    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:06<00:00,  1.53s/it]Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:06<00:00,  1.38s/it]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34095



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-20 01:08:38] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image depicts a man standing on the back of a yellow taxi, using a portable ironing board and iron to iron clothes. The taxi is parked on a city street, and there are other yellow taxis visible in the background, suggesting this is a busy urban area. The man appears to be engaging in an unusual and somewhat amusing activity, as the setting typically doesn't facilitate such labeling or behavior.</strong>



```python
terminate_process(server_process)
```
