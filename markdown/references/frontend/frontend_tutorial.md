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

    [2026-03-20 00:16:52] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.


    [2026-03-20 00:16:52] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.


    [2026-03-20 00:16:52] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 00:16:56] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:16:56] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:16:56] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-20 00:16:58] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
    [2026-03-20 00:16:58] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.
    [2026-03-20 00:16:58] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:175: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [2026-03-20 00:16:59] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 00:17:03] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:17:03] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:17:03] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-20 00:17:03] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:17:03] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:17:03] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 00:17:05] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 00:17:06] Transformers version 5.3.0 is used for model type qwen2. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  1.64it/s]

    Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.49it/s]

    Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.50it/s]

    Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.47it/s]Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:12,  1.30s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:43,  1.25it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:43,  1.25it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:21,  2.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:21,  2.43it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:03<00:16,  3.09it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:13,  3.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:13,  3.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:03<00:10,  4.61it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:03<00:10,  4.61it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:08,  5.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:08,  5.48it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:08,  5.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  7.11it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  7.11it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.56it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.56it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.56it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.05it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.05it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04, 10.05it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.96it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.96it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:03, 11.96it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:03, 11.96it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 15.17it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 15.17it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 15.17it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:01, 20.59it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 25.03it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 25.03it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 25.03it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 25.03it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 23.84it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 23.84it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 23.84it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 23.84it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 23.78it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 24.28it/s]

    Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 25.66it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 25.66it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 25.66it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 25.66it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 25.90it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 25.90it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 25.90it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 25.90it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 25.90it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 27.08it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 27.08it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 27.08it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 27.08it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 27.26it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 27.26it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 27.26it/s]

    Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 27.26it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 27.26it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:06<00:00, 28.84it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:06<00:00, 28.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.22 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.19 GB):   2%|▏         | 1/58 [00:00<00:38,  1.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.19 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.19 GB):   3%|▎         | 2/58 [00:01<00:34,  1.61it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.19 GB):   5%|▌         | 3/58 [00:01<00:31,  1.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.19 GB):   5%|▌         | 3/58 [00:01<00:31,  1.74it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.19 GB):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.20 GB):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.20 GB):   9%|▊         | 5/58 [00:02<00:26,  2.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.20 GB):   9%|▊         | 5/58 [00:02<00:26,  2.02it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.20 GB):  10%|█         | 6/58 [00:03<00:23,  2.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.20 GB):  10%|█         | 6/58 [00:03<00:23,  2.25it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.20 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.21 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.50it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.21 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.86it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.21 GB):  14%|█▍        | 8/58 [00:03<00:17,  2.86it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.21 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.21 GB):  16%|█▌        | 9/58 [00:03<00:14,  3.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.21 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.21 GB):  17%|█▋        | 10/58 [00:03<00:12,  3.72it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.21 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.21 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.21 GB):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.20 GB):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=43.21 GB):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.21 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.21 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.45it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.21 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.91it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.17 GB):  26%|██▌       | 15/58 [00:04<00:06,  6.91it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.17 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.17 GB):  28%|██▊       | 16/58 [00:04<00:06,  6.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.17 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.18 GB):  29%|██▉       | 17/58 [00:04<00:06,  6.20it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=43.18 GB):  31%|███       | 18/58 [00:05<00:06,  6.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.17 GB):  31%|███       | 18/58 [00:05<00:06,  6.46it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.17 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.35 GB):  33%|███▎      | 19/58 [00:05<00:06,  6.41it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.35 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.35 GB):  34%|███▍      | 20/58 [00:05<00:05,  6.46it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.35 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.09it/s]Capturing num tokens (num_tokens=960 avail_mem=43.17 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.09it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=43.17 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.41it/s]Capturing num tokens (num_tokens=896 avail_mem=42.40 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.41it/s]Capturing num tokens (num_tokens=896 avail_mem=42.40 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.43it/s]Capturing num tokens (num_tokens=832 avail_mem=42.40 GB):  40%|███▉      | 23/58 [00:05<00:04,  7.43it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.40 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.66it/s]Capturing num tokens (num_tokens=768 avail_mem=43.17 GB):  41%|████▏     | 24/58 [00:05<00:04,  7.66it/s]Capturing num tokens (num_tokens=768 avail_mem=43.17 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.15it/s]Capturing num tokens (num_tokens=704 avail_mem=42.45 GB):  43%|████▎     | 25/58 [00:05<00:04,  8.15it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.45 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.11it/s]Capturing num tokens (num_tokens=640 avail_mem=42.45 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.11it/s]Capturing num tokens (num_tokens=640 avail_mem=42.45 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.09it/s]Capturing num tokens (num_tokens=576 avail_mem=42.44 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.09it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.44 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.51it/s]Capturing num tokens (num_tokens=512 avail_mem=43.15 GB):  48%|████▊     | 28/58 [00:06<00:03,  8.51it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.15 GB):  50%|█████     | 29/58 [00:06<00:04,  5.94it/s]Capturing num tokens (num_tokens=480 avail_mem=42.49 GB):  50%|█████     | 29/58 [00:06<00:04,  5.94it/s]Capturing num tokens (num_tokens=448 avail_mem=43.15 GB):  50%|█████     | 29/58 [00:06<00:04,  5.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.15 GB):  53%|█████▎    | 31/58 [00:06<00:03,  7.23it/s]Capturing num tokens (num_tokens=416 avail_mem=42.54 GB):  53%|█████▎    | 31/58 [00:06<00:03,  7.23it/s]Capturing num tokens (num_tokens=416 avail_mem=42.54 GB):  55%|█████▌    | 32/58 [00:06<00:03,  7.60it/s]Capturing num tokens (num_tokens=384 avail_mem=42.54 GB):  55%|█████▌    | 32/58 [00:06<00:03,  7.60it/s]Capturing num tokens (num_tokens=352 avail_mem=43.14 GB):  55%|█████▌    | 32/58 [00:07<00:03,  7.60it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.14 GB):  59%|█████▊    | 34/58 [00:07<00:02,  8.81it/s]Capturing num tokens (num_tokens=320 avail_mem=42.59 GB):  59%|█████▊    | 34/58 [00:07<00:02,  8.81it/s]Capturing num tokens (num_tokens=288 avail_mem=42.59 GB):  59%|█████▊    | 34/58 [00:07<00:02,  8.81it/s]

    Capturing num tokens (num_tokens=288 avail_mem=42.59 GB):  62%|██████▏   | 36/58 [00:07<00:02,  7.55it/s]Capturing num tokens (num_tokens=256 avail_mem=43.13 GB):  62%|██████▏   | 36/58 [00:07<00:02,  7.55it/s]Capturing num tokens (num_tokens=240 avail_mem=42.64 GB):  62%|██████▏   | 36/58 [00:07<00:02,  7.55it/s]Capturing num tokens (num_tokens=240 avail_mem=42.64 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.29it/s]Capturing num tokens (num_tokens=224 avail_mem=42.63 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.29it/s]

    Capturing num tokens (num_tokens=208 avail_mem=43.12 GB):  66%|██████▌   | 38/58 [00:07<00:02,  8.29it/s]Capturing num tokens (num_tokens=208 avail_mem=43.12 GB):  69%|██████▉   | 40/58 [00:07<00:02,  8.87it/s]Capturing num tokens (num_tokens=192 avail_mem=42.68 GB):  69%|██████▉   | 40/58 [00:07<00:02,  8.87it/s]Capturing num tokens (num_tokens=176 avail_mem=43.12 GB):  69%|██████▉   | 40/58 [00:07<00:02,  8.87it/s]

    Capturing num tokens (num_tokens=176 avail_mem=43.12 GB):  72%|███████▏  | 42/58 [00:08<00:01,  9.51it/s]Capturing num tokens (num_tokens=160 avail_mem=42.71 GB):  72%|███████▏  | 42/58 [00:08<00:01,  9.51it/s]Capturing num tokens (num_tokens=160 avail_mem=42.71 GB):  74%|███████▍  | 43/58 [00:08<00:01,  9.59it/s]Capturing num tokens (num_tokens=144 avail_mem=42.73 GB):  74%|███████▍  | 43/58 [00:08<00:01,  9.59it/s]Capturing num tokens (num_tokens=128 avail_mem=43.12 GB):  74%|███████▍  | 43/58 [00:08<00:01,  9.59it/s]

    Capturing num tokens (num_tokens=128 avail_mem=43.12 GB):  78%|███████▊  | 45/58 [00:08<00:01, 10.16it/s]Capturing num tokens (num_tokens=112 avail_mem=42.74 GB):  78%|███████▊  | 45/58 [00:08<00:01, 10.16it/s]Capturing num tokens (num_tokens=96 avail_mem=43.11 GB):  78%|███████▊  | 45/58 [00:08<00:01, 10.16it/s] Capturing num tokens (num_tokens=96 avail_mem=43.11 GB):  81%|████████  | 47/58 [00:08<00:01, 10.36it/s]Capturing num tokens (num_tokens=80 avail_mem=42.75 GB):  81%|████████  | 47/58 [00:08<00:01, 10.36it/s]

    Capturing num tokens (num_tokens=64 avail_mem=43.11 GB):  81%|████████  | 47/58 [00:08<00:01, 10.36it/s]Capturing num tokens (num_tokens=64 avail_mem=43.11 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.73it/s]Capturing num tokens (num_tokens=48 avail_mem=42.78 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.73it/s]Capturing num tokens (num_tokens=32 avail_mem=43.10 GB):  84%|████████▍ | 49/58 [00:08<00:00, 10.73it/s]Capturing num tokens (num_tokens=32 avail_mem=43.10 GB):  88%|████████▊ | 51/58 [00:08<00:00, 11.90it/s]Capturing num tokens (num_tokens=28 avail_mem=42.80 GB):  88%|████████▊ | 51/58 [00:08<00:00, 11.90it/s]

    Capturing num tokens (num_tokens=24 avail_mem=42.79 GB):  88%|████████▊ | 51/58 [00:08<00:00, 11.90it/s]Capturing num tokens (num_tokens=24 avail_mem=42.79 GB):  91%|█████████▏| 53/58 [00:09<00:00,  9.95it/s]Capturing num tokens (num_tokens=20 avail_mem=43.09 GB):  91%|█████████▏| 53/58 [00:09<00:00,  9.95it/s]Capturing num tokens (num_tokens=16 avail_mem=42.81 GB):  91%|█████████▏| 53/58 [00:09<00:00,  9.95it/s]

    Capturing num tokens (num_tokens=16 avail_mem=42.81 GB):  95%|█████████▍| 55/58 [00:09<00:00, 11.26it/s]Capturing num tokens (num_tokens=12 avail_mem=43.08 GB):  95%|█████████▍| 55/58 [00:09<00:00, 11.26it/s]Capturing num tokens (num_tokens=8 avail_mem=42.83 GB):  95%|█████████▍| 55/58 [00:09<00:00, 11.26it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=42.83 GB):  98%|█████████▊| 57/58 [00:09<00:00,  6.55it/s]Capturing num tokens (num_tokens=4 avail_mem=42.85 GB):  98%|█████████▊| 57/58 [00:09<00:00,  6.55it/s]Capturing num tokens (num_tokens=4 avail_mem=42.85 GB): 100%|██████████| 58/58 [00:09<00:00,  6.58it/s]Capturing num tokens (num_tokens=4 avail_mem=42.85 GB): 100%|██████████| 58/58 [00:09<00:00,  5.85it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32798


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-20 00:17:35] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries and their capitals:<br><br>1. **France** - Paris<br>2. **Australia** - Canberra<br>3. **Japan** - Tokyo</strong>


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


<strong style='color: #00008B;'>Certainly! Here is a list of three countries and their respective capitals:<br><br>1. **Germany** - Berlin<br>2. **Japan** - Tokyo<br>3. **Brazil** - Brasília</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries and their capitals:<br><br>1. **France** - Paris<br>2. **Mexico** - Mexico City<br>3. **Australia** - Canberra</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>Let's calculate it:<br><br>2 * 2 = 4<br><br>Therefore, 2 * 2 equals 4. No calculator was needed in this case, as it's a simple multiplication.</strong>


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


<strong style='color: #00008B;'>### Tip 1: Maintaining a Balanced Diet<br>- **Nourishment**: Consume a variety of foods to ensure you get all necessary nutrients.<br>- **Proportions**: Ensure your diet includes a good mix of carbohydrates, proteins, and healthy fats.<br>- **Vitamins and Minerals**: Get essential nutrients from fruits and vegetables.<br>- **Hydration**: Drink plenty of water.<br>- **Limit Unhealthy Foods**: Reduce intake of excess sugar, salt, and unhealthy fats to prevent chronic diseases.<br><br>### Tip 2: Regular Exercise<br>- **Cardiovascular Health**: Improve heart and lung function.<br>- **Muscle and Bone Strength**: Strengthen muscles and bones.<br>- **Flexibility and Balance**: Enhance flexibility and balance.<br>- **Weight Management**: Help in managing weight and reducing the risk of obesity.<br>- **Chronic Disease Prevention**: Lower the risk of chronic diseases like diabetes and heart disease.<br>- **Mental Health**: Reduce stress, improve mood, and enhance self-esteem.<br>- **Sleep Quality**: Promote better sleep.<br><br>By combining these healthy habits, you can significantly improve your overall well-being and set a foundation for a healthier lifestyle.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Oak",<br>        "core": "Fern frond",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "The Boggart in a"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 35.22it/s]

    



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

    [2026-03-20 00:17:45] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:17:45] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:17:45] INFO utils.py:164: NumExpr defaulting to 16 threads.
    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-03-20 00:17:46] WARNING model_config.py:1098: Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.
    [2026-03-20 00:17:46] INFO server_args.py:2222: Attention backend not specified. Use fa3 backend by default.
    [2026-03-20 00:17:46] INFO server_args.py:3449: Set soft_watchdog_timeout since in CI


    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/http_server.py:175: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      from sglang.srt.utils.json_response import (


    [2026-03-20 00:17:47] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 00:17:51] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:17:51] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:17:51] INFO utils.py:164: NumExpr defaulting to 16 threads.
    [2026-03-20 00:17:51] INFO utils.py:148: Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-03-20 00:17:51] INFO utils.py:151: Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-03-20 00:17:51] INFO utils.py:164: NumExpr defaulting to 16 threads.


    [2026-03-20 00:17:53] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [2026-03-20 00:17:57] Transformers version 5.3.0 is used for model type qwen2_5_vl. If you experience issues related to RoPE parameters, they may be due to incompatibilities between Transformers >=5.0.0 and some models. You can try downgrading to transformers==4.57.1 as a workaround.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards:  20% Completed | 1/5 [00:00<00:02,  1.53it/s]

    Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:01<00:02,  1.37it/s]

    Loading safetensors checkpoint shards:  60% Completed | 3/5 [00:02<00:01,  1.26it/s]

    Loading safetensors checkpoint shards:  80% Completed | 4/5 [00:03<00:00,  1.29it/s]

    Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.67it/s]Loading safetensors checkpoint shards: 100% Completed | 5/5 [00:03<00:00,  1.50it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:116: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:38633



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-03-20 00:18:10] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>The image shows a man ironing a shirt on a small, portable ironing board placed on the roof of a yellow vehicle parked on a city street. The vehicle appears to be a typical New York City taxi. There are other taxis and a flag visible in the background. The man seems to be using the car roof as a platform to iron his clothes in an unexpected and humorous situation.</strong>



```python
terminate_process(server_process)
```
