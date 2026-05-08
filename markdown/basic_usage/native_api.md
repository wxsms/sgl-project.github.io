# SGLang Native APIs

Apart from the OpenAI compatible APIs, the SGLang Runtime also provides its native server APIs. We introduce the following APIs:

- `/generate` (text generation model)
- `/get_model_info`
- `/server_info`
- `/health`
- `/health_generate`
- `/flush_cache`
- `/update_weights`
- `/encode`(embedding model)
- `/v1/rerank`(cross encoder rerank model)
- `/v1/score`(decoder-only scoring)
- `/classify`(reward model)
- `/start_expert_distribution_record`
- `/stop_expert_distribution_record`
- `/dump_expert_distribution_record`
- `/tokenize`
- `/detokenize`
- A full list of these APIs can be found at [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)

We mainly use `requests` to test these APIs in the following examples. You can also use `curl`.


## Launch A Server


```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:13:19] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.65it/s]


    2026-05-08 04:13:23,499 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:13:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.57it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.63it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.08it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]

    Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 19.65it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 23.68it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.74it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 37.19it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 37.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.36 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.09 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.09 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.34 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.33 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.32 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.31 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.31 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.30 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.30 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.47it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.29 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.27 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.05it/s]Capturing num tokens (num_tokens=960 avail_mem=74.26 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.05it/s] Capturing num tokens (num_tokens=896 avail_mem=74.20 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.05it/s]

    Capturing num tokens (num_tokens=896 avail_mem=74.20 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=832 avail_mem=74.18 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=768 avail_mem=74.24 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=704 avail_mem=74.15 GB):  40%|███▉      | 23/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=704 avail_mem=74.15 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.57it/s]Capturing num tokens (num_tokens=640 avail_mem=73.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.57it/s]Capturing num tokens (num_tokens=576 avail_mem=73.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.57it/s]Capturing num tokens (num_tokens=512 avail_mem=73.63 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.57it/s]Capturing num tokens (num_tokens=480 avail_mem=73.57 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.57it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.57 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.56it/s]Capturing num tokens (num_tokens=448 avail_mem=73.55 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.56it/s]Capturing num tokens (num_tokens=416 avail_mem=73.56 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.56it/s]Capturing num tokens (num_tokens=384 avail_mem=73.56 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.56it/s]Capturing num tokens (num_tokens=352 avail_mem=73.55 GB):  52%|█████▏    | 30/58 [00:01<00:01, 26.56it/s]Capturing num tokens (num_tokens=352 avail_mem=73.55 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.06it/s]Capturing num tokens (num_tokens=320 avail_mem=73.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.06it/s]Capturing num tokens (num_tokens=288 avail_mem=73.52 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.06it/s]Capturing num tokens (num_tokens=256 avail_mem=73.53 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.06it/s]Capturing num tokens (num_tokens=240 avail_mem=73.54 GB):  59%|█████▊    | 34/58 [00:01<00:00, 30.06it/s]

    Capturing num tokens (num_tokens=240 avail_mem=73.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=224 avail_mem=73.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=208 avail_mem=73.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=192 avail_mem=73.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=176 avail_mem=73.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.37it/s]Capturing num tokens (num_tokens=176 avail_mem=73.51 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=160 avail_mem=73.50 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=144 avail_mem=73.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=128 avail_mem=73.49 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.09it/s]Capturing num tokens (num_tokens=112 avail_mem=73.48 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.09it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=96 avail_mem=73.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.33it/s] Capturing num tokens (num_tokens=80 avail_mem=73.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=64 avail_mem=73.46 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=48 avail_mem=73.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=48 avail_mem=73.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=32 avail_mem=73.43 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=28 avail_mem=73.42 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=24 avail_mem=73.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=20 avail_mem=73.41 GB):  86%|████████▌ | 50/58 [00:01<00:00, 36.40it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=16 avail_mem=73.41 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=12 avail_mem=73.40 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=8 avail_mem=73.39 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.97it/s] Capturing num tokens (num_tokens=4 avail_mem=73.38 GB):  93%|█████████▎| 54/58 [00:01<00:00, 36.97it/s]Capturing num tokens (num_tokens=4 avail_mem=73.38 GB): 100%|██████████| 58/58 [00:02<00:00, 37.64it/s]Capturing num tokens (num_tokens=4 avail_mem=73.38 GB): 100%|██████████| 58/58 [00:02<00:00, 28.87it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


## Generate (text generation model)
Generate completions. This is similar to the `/v1/completions` in OpenAI API. Detailed parameters can be found in the [sampling parameters](sampling_params.md).


```python
import requests

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': ' The capital of France is Paris.\n\nWell,经营模式 (way of running a business)：\n\n经营模式的形成与演变有一其元素的递进、叠加和丰富，它主要表现为经营模式的生命周期与发展脉络等。\n\n1986年，拥有40多年历史的模式成为了一种新的管理理念框架，新理念逐渐积累并制约并影响经营观念、管理形式、经营方式、无形资源和有形资源的变化发展。\n\n例如，经营观念的变化：一九八六年第二批模式主流成立；八四，资者继续开始自我反省，新加入模式的经营者具体合作并利用企业更好的', 'output_ids': [576, 6722, 315, 9625, 374, 12095, 382, 11395, 11, 111657, 320, 3117, 315, 4303, 264, 2562, 8, 48443, 111657, 9370, 101894, 57218, 112148, 99992, 41146, 102268, 9370, 100520, 41299, 5373, 110081, 33108, 100733, 3837, 99652, 99558, 110966, 111657, 9370, 113509, 111636, 100297, 68065, 49567, 3407, 16, 24, 23, 21, 7948, 3837, 103926, 19, 15, 100097, 100022, 9370, 100144, 99787, 99593, 86402, 100676, 39352, 100427, 102724, 3837, 16628, 100427, 104052, 102467, 62926, 107336, 62926, 99564, 99881, 104626, 5373, 39352, 100414, 5373, 99881, 75768, 5373, 109253, 85329, 33108, 18830, 82699, 85329, 104896, 99185, 3407, 77557, 3837, 99881, 104626, 104896, 5122, 14777, 99609, 99568, 110472, 116760, 100144, 105276, 100680, 24968, 99568, 63703, 3837, 51125, 28946, 100640, 55286, 104049, 115053, 3837, 16628, 101963, 100144, 9370, 108523, 100398, 99721, 62926, 100152, 99304, 102287], 'meta_info': {'id': 'f9b81d325a44469a886edc7a268b7964', 'finish_reason': {'type': 'length', 'length': 128}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 128, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 1.0285905010532588, 'response_sent_to_client_ts': 1778213619.771603}}</strong>


## Get Model Info

Get the information of the model.

- `model_path`: The path/name of the model.
- `is_generation`: Whether the model is used as generation model or embedding model.
- `tokenizer_path`: The path/name of the tokenizer.
- `preferred_sampling_params`: The default sampling params specified via `--preferred-sampling-params`. `None` is returned in this example as we did not explicitly configure it in server args.
- `weight_version`: This field contains the version of the model weights. This is often used to track changes or updates to the model’s trained parameters.
- `has_image_understanding`: Whether the model has image-understanding capability.
- `has_audio_understanding`: Whether the model has audio-understanding capability.
- `model_type`: The model type from the HuggingFace config (e.g., "qwen2", "llama").
- `architectures`: The model architectures from the HuggingFace config (e.g., ["Qwen2ForCausalLM"]).


```python
url = f"http://localhost:{port}/get_model_info"

response = requests.get(url)
response_json = response.json()
print_highlight(response_json)
assert response_json["model_path"] == "qwen/qwen2.5-0.5b-instruct"
assert response_json["is_generation"] is True
assert response_json["tokenizer_path"] == "qwen/qwen2.5-0.5b-instruct"
assert response_json["preferred_sampling_params"] is None
assert response_json.keys() == {
    "model_path",
    "is_generation",
    "tokenizer_path",
    "preferred_sampling_params",
    "weight_version",
    "has_image_understanding",
    "has_audio_understanding",
    "model_type",
    "architectures",
}
```

    [2026-05-08 04:13:39] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.



<strong style='color: #00008B;'>{'model_path': 'qwen/qwen2.5-0.5b-instruct', 'tokenizer_path': 'qwen/qwen2.5-0.5b-instruct', 'is_generation': True, 'preferred_sampling_params': None, 'weight_version': 'default', 'has_image_understanding': False, 'has_audio_understanding': False, 'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM']}</strong>


## Get Server Info
Gets the server information including CLI arguments, token limits, and memory pool sizes.
- Note: `get_server_info` merges the following deprecated endpoints:
  - `get_server_args`
  - `get_memory_pool_size`
  - `get_max_total_num_tokens`


```python
url = f"http://localhost:{port}/server_info"

response = requests.get(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'>{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","host":"0.0.0.0","port":31561,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":null,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":798058143,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":null,"prefill_attention_backend":null,"sampling_backend":"flashinfer","grammar_backend":"xgrammar","mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","nsa_prefill_backend":null,"nsa_decode_backend":null,"disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_dflash_draft_window_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_adaptive":false,"speculative_adaptive_config":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","record_nolora_graph":true,"flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"best_effort","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_nan_detection":false,"enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":false,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"gc_threshold":null,"enable_nsa_prefill_context_parallel":false,"nsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"status":"ready","max_total_num_tokens":20480,"max_req_input_len":20474,"internal_states":[{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","host":"0.0.0.0","port":31561,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":128,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":798058143,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":"fa3","prefill_attention_backend":"fa3","sampling_backend":"flashinfer","grammar_backend":"xgrammar","mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","nsa_prefill_backend":null,"nsa_decode_backend":null,"disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_dflash_draft_window_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_adaptive":false,"speculative_adaptive_config":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","record_nolora_graph":true,"flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"best_effort","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_nan_detection":false,"enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":true,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"gc_threshold":null,"enable_nsa_prefill_context_parallel":false,"nsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"enable_grpc":false,"grpc_port":41561,"_quantization_explicitly_unset":false,"use_mla_backend":false,"_mx_config_cache":{},"last_gen_throughput":141.2254213418751,"memory_usage":{"weight":0.98,"kvcache":0.23,"token_capacity":20480,"graph":0},"effective_max_running_requests_per_dp":128}],"version":"0.0.0.dev1+g2afb45050"}</strong>


## Health Check
- `/health`: Check the health of the server.
- `/health_generate`: Check the health of the server by generating one token.


```python
url = f"http://localhost:{port}/health_generate"

response = requests.get(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'></strong>



```python
url = f"http://localhost:{port}/health"

response = requests.get(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'></strong>


## Flush Cache

Flush the radix cache. It will be automatically triggered when the model weights are updated by the `/update_weights` API.

Parameters:
- `timeout` (query, float, default `0`, unit: seconds): Wait time for idle state before flushing. `0` means fail fast if not idle. When HiCache async operations are in-flight, a non-zero timeout allows the server to wait until idle before flushing, avoiding unnecessary 400 errors.

```bash
# With timeout (wait up to 30s for idle state)
curl -s -X POST "http://127.0.0.1:30000/flush_cache?timeout=30"
```


```python
url = f"http://localhost:{port}/flush_cache"

response = requests.post(url)
print_highlight(response.text)
```


<strong style='color: #00008B;'>Cache flushed.<br>Please check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)<br></strong>


## Update Weights From Disk

Update model weights from disk without restarting the server. Only applicable for models with the same architecture and parameter size.

SGLang support `update_weights_from_disk` API for continuous evaluation during training (save checkpoint to disk and update weights from disk).



```python
# successful update with same architecture and size

url = f"http://localhost:{port}/update_weights_from_disk"
data = {"model_path": "qwen/qwen2.5-0.5b-instruct"}

response = requests.post(url, json=data)
print_highlight(response.text)
assert response.json()["success"] is True
assert response.json()["message"] == "Succeeded to update model weights."
```

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>{"success":true,"message":"Succeeded to update model weights.","num_paused_requests":0}</strong>



```python
# failed update with different parameter size or wrong name

url = f"http://localhost:{port}/update_weights_from_disk"
data = {"model_path": "qwen/qwen2.5-0.5b-instruct-wrong"}

response = requests.post(url, json=data)
response_json = response.json()
print_highlight(response_json)
assert response_json["success"] is False
assert response_json["message"] == (
    "Failed to get weights iterator: "
    "qwen/qwen2.5-0.5b-instruct-wrong"
    " (repository not found)."
)
```

    [2026-05-08 04:13:42] Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).



<strong style='color: #00008B;'>{'success': False, 'message': 'Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).', 'num_paused_requests': 0}</strong>



```python
terminate_process(server_process)
```

## Encode (embedding model)

Encode text into embeddings. Note that this API is only available for [embedding models](openai_api_embeddings.ipynb) and will raise an error for generation models.
Therefore, we launch a new server to server an embedding model.


```python
embedding_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --host 0.0.0.0 --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=embedding_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:13:57] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.41s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.00s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]


    2026-05-08 04:14:02,923 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:14:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:09,  5.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:09,  5.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:09,  5.44s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:20,  1.46s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:20,  1.46s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:20,  1.46s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:39,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:18,  2.68it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:06<00:09,  4.99it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:06<00:04,  9.45it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:06<00:04,  9.45it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:06<00:01, 16.79it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]

    Compiling num tokens (num_tokens=176):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=160):  59%|█████▊    | 34/58 [00:06<00:00, 24.92it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:00, 34.62it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 42.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.37 GB):   2%|▏         | 1/58 [00:00<00:06,  9.10it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.34 GB):   2%|▏         | 1/58 [00:00<00:06,  9.10it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.33 GB):   2%|▏         | 1/58 [00:00<00:06,  9.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.33 GB):   5%|▌         | 3/58 [00:00<00:05, 10.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.33 GB):   5%|▌         | 3/58 [00:00<00:05, 10.13it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.33 GB):   5%|▌         | 3/58 [00:00<00:05, 10.13it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.33 GB):   9%|▊         | 5/58 [00:00<00:04, 11.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.33 GB):   9%|▊         | 5/58 [00:00<00:04, 11.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.32 GB):   9%|▊         | 5/58 [00:00<00:04, 11.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 12.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 12.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 12.94it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.32 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.31 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.31 GB):  21%|██        | 12/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.30 GB):  21%|██        | 12/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.30 GB):  21%|██        | 12/58 [00:00<00:02, 17.50it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=57.29 GB):  21%|██        | 12/58 [00:00<00:02, 17.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.29 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.30 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.28 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.28 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.27 GB):  26%|██▌       | 15/58 [00:01<00:02, 20.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.27 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.27 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.27 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s]Capturing num tokens (num_tokens=960 avail_mem=57.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=57.24 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s]Capturing num tokens (num_tokens=832 avail_mem=57.26 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.65it/s]Capturing num tokens (num_tokens=832 avail_mem=57.26 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=768 avail_mem=57.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=704 avail_mem=57.25 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=640 avail_mem=57.25 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=576 avail_mem=57.25 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=512 avail_mem=57.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.85it/s]Capturing num tokens (num_tokens=512 avail_mem=57.24 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=480 avail_mem=57.24 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=448 avail_mem=57.24 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]

    Capturing num tokens (num_tokens=416 avail_mem=57.23 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=384 avail_mem=57.23 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=352 avail_mem=57.22 GB):  50%|█████     | 29/58 [00:01<00:00, 33.76it/s]Capturing num tokens (num_tokens=352 avail_mem=57.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=320 avail_mem=57.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=288 avail_mem=57.23 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=256 avail_mem=57.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=240 avail_mem=57.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=224 avail_mem=57.22 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.42it/s]Capturing num tokens (num_tokens=224 avail_mem=57.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=208 avail_mem=57.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]

    Capturing num tokens (num_tokens=192 avail_mem=57.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=176 avail_mem=57.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=160 avail_mem=57.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=144 avail_mem=57.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=144 avail_mem=57.20 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=128 avail_mem=57.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=112 avail_mem=57.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=96 avail_mem=57.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s] Capturing num tokens (num_tokens=80 avail_mem=57.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=64 avail_mem=57.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.39it/s]

    Capturing num tokens (num_tokens=64 avail_mem=57.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=48 avail_mem=57.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=32 avail_mem=57.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=28 avail_mem=57.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=24 avail_mem=57.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=20 avail_mem=57.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=20 avail_mem=57.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=16 avail_mem=57.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=12 avail_mem=57.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.98it/s]Capturing num tokens (num_tokens=8 avail_mem=57.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.98it/s] Capturing num tokens (num_tokens=4 avail_mem=57.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.98it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.15 GB): 100%|██████████| 58/58 [00:01<00:00, 29.30it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# successful encode for embedding model

url = f"http://localhost:{port}/encode"
data = {"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "text": "Once upon a time"}

response = requests.post(url, json=data)
response_json = response.json()
print_highlight(f"Text embedding (first 10): {response_json['embedding'][:10]}")
```


<strong style='color: #00008B;'>Text embedding (first 10): [-0.00023102760314941406, -0.04986572265625, -0.0032711029052734375, 0.011077880859375, -0.0140533447265625, 0.0159912109375, -0.01441192626953125, 0.0059051513671875, -0.0228424072265625, 0.0272979736328125]</strong>



```python
terminate_process(embedding_process)
```

## v1/rerank (cross encoder rerank model)
Rerank a list of documents given a query using a cross-encoder model. Note that this API is only available for cross encoder model like [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) with `attention-backend` `triton` and `torch_native`.



```python
reranker_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path BAAI/bge-reranker-v2-m3 \
    --host 0.0.0.0 --disable-radix-cache --chunked-prefill-size -1 --attention-backend triton --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=reranker_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-05-08 04:14:25] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:14:29] No HuggingFace chat template found


    [2026-05-08 04:14:32] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    [2026-05-08 04:14:36] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.37it/s]


    [2026-05-08 04:14:39] Disable piecewise CUDA graph because the model is not a language model


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# compute rerank scores for query and documents

url = f"http://localhost:{port}/v1/rerank"
data = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
    ],
}

response = requests.post(url, json=data)
response_json = response.json()
for item in response_json:
    print_highlight(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
```


<strong style='color: #00008B;'>Score: 5.26 - Document: 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'</strong>



<strong style='color: #00008B;'>Score: -8.19 - Document: 'hi'</strong>



```python
terminate_process(reranker_process)
```

## v1/score (decoder-only scoring)

Compute token probabilities for specified tokens given a query and items. This is useful for classification tasks, scoring responses, or computing log-probabilities.

Parameters:
- `query`: Query text
- `items`: Item text(s) to score
- `label_token_ids`: Token IDs to compute probabilities for
- `apply_softmax`: Whether to apply softmax to get normalized probabilities (default: False)
- `item_first`: Whether items come first in concatenation order (default: False)
- `model`: Model name

The response contains `scores` - a list of probability lists, one per item, each in the order of `label_token_ids`.


```python
score_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=score_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:15:06] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.01it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]


    2026-05-08 04:15:09,966 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:15:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.38it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.38it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.38it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.38it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.42it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.82it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:03, 12.49it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:03, 12.49it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:03, 12.49it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:03, 12.49it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:03, 12.49it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.89it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.89it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.89it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.89it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 15.89it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 19.83it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:05<00:01, 25.47it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 31.76it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 37.26it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 41.15it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 47.06it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 47.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.90 GB):   3%|▎         | 2/58 [00:00<00:05, 10.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.89 GB):   3%|▎         | 2/58 [00:00<00:05, 10.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.88 GB):   3%|▎         | 2/58 [00:00<00:05, 10.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.88 GB):   7%|▋         | 4/58 [00:00<00:04, 12.09it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.88 GB):   7%|▋         | 4/58 [00:00<00:04, 12.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.87 GB):   7%|▋         | 4/58 [00:00<00:04, 12.09it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.87 GB):  10%|█         | 6/58 [00:00<00:03, 13.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.74 GB):  10%|█         | 6/58 [00:00<00:03, 13.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.74 GB):  10%|█         | 6/58 [00:00<00:03, 13.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.84 GB):  10%|█         | 6/58 [00:00<00:03, 13.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.84 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.83 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.07it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=40.83 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.82 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.82 GB):  21%|██        | 12/58 [00:00<00:02, 17.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.82 GB):  21%|██        | 12/58 [00:00<00:02, 17.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.81 GB):  21%|██        | 12/58 [00:00<00:02, 17.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.75 GB):  21%|██        | 12/58 [00:00<00:02, 17.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.75 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.80 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.04it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=40.79 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.79 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.79 GB):  31%|███       | 18/58 [00:00<00:01, 22.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.78 GB):  31%|███       | 18/58 [00:00<00:01, 22.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.74 GB):  31%|███       | 18/58 [00:01<00:01, 22.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.75 GB):  31%|███       | 18/58 [00:01<00:01, 22.07it/s]Capturing num tokens (num_tokens=960 avail_mem=40.77 GB):  31%|███       | 18/58 [00:01<00:01, 22.07it/s] Capturing num tokens (num_tokens=960 avail_mem=40.77 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.46it/s]Capturing num tokens (num_tokens=896 avail_mem=40.76 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.46it/s]

    Capturing num tokens (num_tokens=832 avail_mem=40.76 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.46it/s]Capturing num tokens (num_tokens=768 avail_mem=40.75 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.46it/s]Capturing num tokens (num_tokens=704 avail_mem=40.75 GB):  38%|███▊      | 22/58 [00:01<00:01, 25.46it/s]Capturing num tokens (num_tokens=704 avail_mem=40.75 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=640 avail_mem=40.74 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=576 avail_mem=40.73 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.37it/s]Capturing num tokens (num_tokens=512 avail_mem=40.70 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.37it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.70 GB):  50%|█████     | 29/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=480 avail_mem=40.71 GB):  50%|█████     | 29/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=448 avail_mem=40.69 GB):  50%|█████     | 29/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=416 avail_mem=40.67 GB):  50%|█████     | 29/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=416 avail_mem=40.67 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=384 avail_mem=40.12 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=352 avail_mem=40.12 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=320 avail_mem=40.11 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.27it/s]

    Capturing num tokens (num_tokens=288 avail_mem=40.11 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=288 avail_mem=40.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=256 avail_mem=40.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=240 avail_mem=40.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=224 avail_mem=40.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=208 avail_mem=40.08 GB):  62%|██████▏   | 36/58 [00:01<00:00, 27.34it/s]Capturing num tokens (num_tokens=208 avail_mem=40.08 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.66it/s]Capturing num tokens (num_tokens=192 avail_mem=40.08 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.66it/s]Capturing num tokens (num_tokens=176 avail_mem=40.07 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.66it/s]

    Capturing num tokens (num_tokens=160 avail_mem=40.07 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.66it/s]Capturing num tokens (num_tokens=144 avail_mem=40.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.66it/s]Capturing num tokens (num_tokens=144 avail_mem=40.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.77it/s]Capturing num tokens (num_tokens=128 avail_mem=40.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.77it/s]Capturing num tokens (num_tokens=112 avail_mem=40.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.77it/s]Capturing num tokens (num_tokens=96 avail_mem=40.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.77it/s] Capturing num tokens (num_tokens=80 avail_mem=40.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.77it/s]Capturing num tokens (num_tokens=80 avail_mem=40.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.22it/s]Capturing num tokens (num_tokens=64 avail_mem=40.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 32.22it/s]Capturing num tokens (num_tokens=48 avail_mem=40.03 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.22it/s]

    Capturing num tokens (num_tokens=32 avail_mem=40.02 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.22it/s]Capturing num tokens (num_tokens=28 avail_mem=40.02 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.22it/s]Capturing num tokens (num_tokens=24 avail_mem=40.02 GB):  83%|████████▎ | 48/58 [00:02<00:00, 32.22it/s]Capturing num tokens (num_tokens=24 avail_mem=40.02 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s]Capturing num tokens (num_tokens=20 avail_mem=40.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s]Capturing num tokens (num_tokens=16 avail_mem=40.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s]Capturing num tokens (num_tokens=12 avail_mem=40.01 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s]Capturing num tokens (num_tokens=8 avail_mem=40.00 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s] Capturing num tokens (num_tokens=4 avail_mem=39.99 GB):  91%|█████████▏| 53/58 [00:02<00:00, 35.17it/s]Capturing num tokens (num_tokens=4 avail_mem=39.99 GB): 100%|██████████| 58/58 [00:02<00:00, 37.49it/s]Capturing num tokens (num_tokens=4 avail_mem=39.99 GB): 100%|██████████| 58/58 [00:02<00:00, 26.15it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
# Score the probability of different completions given a query
query = "The capital of France is"
items = ["Paris", "London", "Berlin"]

url = f"http://localhost:{port}/v1/score"
data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "query": query,
    "items": items,
    "label_token_ids": [9454, 2753],  # e.g. "Yes" and "No" token ids
    "apply_softmax": True,  # Normalize probabilities to sum to 1
}

response = requests.post(url, json=data)
response_json = response.json()

# Display scores for each item
for item, scores in zip(items, response_json["scores"]):
    print_highlight(f"Item '{item}': probabilities = {[f'{s:.4f}' for s in scores]}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:107: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await self._handle_non_streaming_request(



<strong style='color: #00008B;'>Item 'Paris': probabilities = ['0.0237', '0.9763']</strong>



<strong style='color: #00008B;'>Item 'London': probabilities = ['0.0284', '0.9716']</strong>



<strong style='color: #00008B;'>Item 'Berlin': probabilities = ['0.0637', '0.9363']</strong>



```python
terminate_process(score_process)
```

## Classify (reward model)

SGLang Runtime also supports reward models. Here we use a reward model to classify the quality of pairwise generations.


```python
# Note that SGLang now treats embedding models and reward models as the same type of models.
# This will be updated in the future.

reward_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 --host 0.0.0.0 --is-embedding --log-level warning
""")

wait_for_server(f"http://localhost:{port}", process=reward_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-05-08 04:15:31] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:15:33] Tokenizer loaded as generic TokenizersBackend for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2, retrying with use_fast=False


    [2026-05-08 04:15:36] Tokenizer for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-08 04:15:39] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    [2026-05-08 04:15:40] Tokenizer loaded as generic TokenizersBackend for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2, retrying with use_fast=False


    [2026-05-08 04:15:40] Tokenizer loaded as generic TokenizersBackend for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2, retrying with use_fast=False


    [2026-05-08 04:15:42] Tokenizer for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-08 04:15:42] Tokenizer for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-08 04:15:43] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.28it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.19it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.19it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.78it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.51it/s]


    2026-05-08 04:15:49,525 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:15:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:19,  5.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:19,  5.60s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:16,  2.45s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:16,  2.45s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:21,  1.48s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:21,  1.48s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:43,  1.23it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.19it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.19it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.53it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.53it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:16,  3.00it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:16,  3.00it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:14,  3.33it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:14,  3.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.95it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.25it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.77it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.77it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.33it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.33it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.79it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.79it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:07,  5.79it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.30it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.30it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:09<00:05,  7.78it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:09<00:05,  7.78it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:09<00:05,  7.78it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:03, 10.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:03, 10.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:03, 10.06it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:09<00:03, 11.57it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:09<00:03, 11.57it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:03, 11.57it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:03, 11.57it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 15.19it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:02, 15.19it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:10<00:02, 15.19it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:01, 17.78it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:01, 17.78it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:01, 17.78it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 17.78it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:10<00:01, 20.25it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:10<00:01, 20.25it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:10<00:01, 21.58it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:00, 26.30it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:00, 26.30it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:00, 26.30it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:10<00:00, 26.30it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:10<00:00, 26.30it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 28.59it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 28.59it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 28.59it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:10<00:00, 28.59it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 28.27it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 28.27it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 28.27it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 28.27it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:10<00:00, 28.27it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:10<00:00, 29.84it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 36.43it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 36.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.21 GB):   2%|▏         | 1/58 [00:00<00:28,  1.99it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.16 GB):   2%|▏         | 1/58 [00:00<00:28,  1.99it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.16 GB):   3%|▎         | 2/58 [00:00<00:25,  2.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.14 GB):   3%|▎         | 2/58 [00:00<00:25,  2.18it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.14 GB):   5%|▌         | 3/58 [00:01<00:22,  2.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.12 GB):   5%|▌         | 3/58 [00:01<00:22,  2.43it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.12 GB):   7%|▋         | 4/58 [00:01<00:20,  2.61it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.08 GB):   7%|▋         | 4/58 [00:01<00:20,  2.61it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.08 GB):   9%|▊         | 5/58 [00:01<00:19,  2.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.07 GB):   9%|▊         | 5/58 [00:01<00:19,  2.69it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.07 GB):  10%|█         | 6/58 [00:02<00:17,  3.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.07 GB):  10%|█         | 6/58 [00:02<00:17,  3.00it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=24.07 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.06 GB):  12%|█▏        | 7/58 [00:02<00:15,  3.32it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.06 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.05 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.05 GB):  16%|█▌        | 9/58 [00:02<00:12,  4.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.04 GB):  16%|█▌        | 9/58 [00:02<00:12,  4.06it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.04 GB):  17%|█▋        | 10/58 [00:03<00:10,  4.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.03 GB):  17%|█▋        | 10/58 [00:03<00:10,  4.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.03 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.02 GB):  19%|█▉        | 11/58 [00:03<00:09,  4.82it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.02 GB):  21%|██        | 12/58 [00:03<00:08,  5.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.00 GB):  21%|██        | 12/58 [00:03<00:08,  5.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=24.00 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.02it/s]Capturing num tokens (num_tokens=2816 avail_mem=24.00 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.02it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=24.00 GB):  22%|██▏       | 13/58 [00:03<00:07,  6.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=24.00 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=24.00 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=23.99 GB):  26%|██▌       | 15/58 [00:03<00:05,  7.59it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=23.99 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.99 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.99 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.99 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=23.99 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=23.99 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.02it/s]

    Capturing num tokens (num_tokens=960 avail_mem=23.97 GB):  33%|███▎      | 19/58 [00:04<00:03, 11.02it/s] Capturing num tokens (num_tokens=960 avail_mem=23.97 GB):  38%|███▊      | 22/58 [00:04<00:02, 14.24it/s]Capturing num tokens (num_tokens=896 avail_mem=23.97 GB):  38%|███▊      | 22/58 [00:04<00:02, 14.24it/s]Capturing num tokens (num_tokens=832 avail_mem=23.97 GB):  38%|███▊      | 22/58 [00:04<00:02, 14.24it/s]Capturing num tokens (num_tokens=768 avail_mem=23.96 GB):  38%|███▊      | 22/58 [00:04<00:02, 14.24it/s]Capturing num tokens (num_tokens=768 avail_mem=23.96 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.05it/s]Capturing num tokens (num_tokens=704 avail_mem=23.96 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.05it/s]Capturing num tokens (num_tokens=640 avail_mem=23.95 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.05it/s]

    Capturing num tokens (num_tokens=576 avail_mem=23.95 GB):  43%|████▎     | 25/58 [00:04<00:01, 17.05it/s]Capturing num tokens (num_tokens=576 avail_mem=23.95 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.86it/s]Capturing num tokens (num_tokens=512 avail_mem=23.94 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.86it/s]Capturing num tokens (num_tokens=480 avail_mem=23.94 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.86it/s]Capturing num tokens (num_tokens=448 avail_mem=23.93 GB):  48%|████▊     | 28/58 [00:04<00:01, 19.86it/s]Capturing num tokens (num_tokens=448 avail_mem=23.93 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.19it/s]Capturing num tokens (num_tokens=416 avail_mem=23.92 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.19it/s]Capturing num tokens (num_tokens=384 avail_mem=23.91 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.19it/s]

    Capturing num tokens (num_tokens=352 avail_mem=23.91 GB):  53%|█████▎    | 31/58 [00:04<00:01, 22.19it/s]Capturing num tokens (num_tokens=352 avail_mem=23.91 GB):  59%|█████▊    | 34/58 [00:04<00:01, 23.88it/s]Capturing num tokens (num_tokens=320 avail_mem=23.90 GB):  59%|█████▊    | 34/58 [00:04<00:01, 23.88it/s]Capturing num tokens (num_tokens=288 avail_mem=23.90 GB):  59%|█████▊    | 34/58 [00:04<00:01, 23.88it/s]Capturing num tokens (num_tokens=256 avail_mem=23.89 GB):  59%|█████▊    | 34/58 [00:04<00:01, 23.88it/s]Capturing num tokens (num_tokens=256 avail_mem=23.89 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.74it/s]Capturing num tokens (num_tokens=240 avail_mem=23.89 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.74it/s]

    Capturing num tokens (num_tokens=224 avail_mem=23.89 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.74it/s]Capturing num tokens (num_tokens=208 avail_mem=23.88 GB):  64%|██████▍   | 37/58 [00:04<00:00, 22.74it/s]Capturing num tokens (num_tokens=208 avail_mem=23.88 GB):  69%|██████▉   | 40/58 [00:04<00:00, 22.18it/s]Capturing num tokens (num_tokens=192 avail_mem=23.88 GB):  69%|██████▉   | 40/58 [00:04<00:00, 22.18it/s]Capturing num tokens (num_tokens=176 avail_mem=23.87 GB):  69%|██████▉   | 40/58 [00:04<00:00, 22.18it/s]Capturing num tokens (num_tokens=160 avail_mem=23.87 GB):  69%|██████▉   | 40/58 [00:04<00:00, 22.18it/s]

    Capturing num tokens (num_tokens=160 avail_mem=23.87 GB):  74%|███████▍  | 43/58 [00:04<00:00, 22.11it/s]Capturing num tokens (num_tokens=144 avail_mem=23.86 GB):  74%|███████▍  | 43/58 [00:04<00:00, 22.11it/s]Capturing num tokens (num_tokens=128 avail_mem=23.86 GB):  74%|███████▍  | 43/58 [00:04<00:00, 22.11it/s]Capturing num tokens (num_tokens=112 avail_mem=23.86 GB):  74%|███████▍  | 43/58 [00:05<00:00, 22.11it/s]Capturing num tokens (num_tokens=112 avail_mem=23.86 GB):  79%|███████▉  | 46/58 [00:05<00:00, 22.30it/s]Capturing num tokens (num_tokens=96 avail_mem=23.86 GB):  79%|███████▉  | 46/58 [00:05<00:00, 22.30it/s] Capturing num tokens (num_tokens=80 avail_mem=23.85 GB):  79%|███████▉  | 46/58 [00:05<00:00, 22.30it/s]

    Capturing num tokens (num_tokens=64 avail_mem=23.85 GB):  79%|███████▉  | 46/58 [00:05<00:00, 22.30it/s]Capturing num tokens (num_tokens=64 avail_mem=23.85 GB):  84%|████████▍ | 49/58 [00:05<00:00, 22.35it/s]Capturing num tokens (num_tokens=48 avail_mem=23.85 GB):  84%|████████▍ | 49/58 [00:05<00:00, 22.35it/s]Capturing num tokens (num_tokens=32 avail_mem=23.84 GB):  84%|████████▍ | 49/58 [00:05<00:00, 22.35it/s]Capturing num tokens (num_tokens=28 avail_mem=23.84 GB):  84%|████████▍ | 49/58 [00:05<00:00, 22.35it/s]

    Capturing num tokens (num_tokens=28 avail_mem=23.84 GB):  90%|████████▉ | 52/58 [00:05<00:00, 19.09it/s]Capturing num tokens (num_tokens=24 avail_mem=42.94 GB):  90%|████████▉ | 52/58 [00:05<00:00, 19.09it/s]Capturing num tokens (num_tokens=20 avail_mem=42.94 GB):  90%|████████▉ | 52/58 [00:05<00:00, 19.09it/s]Capturing num tokens (num_tokens=16 avail_mem=42.94 GB):  90%|████████▉ | 52/58 [00:05<00:00, 19.09it/s]Capturing num tokens (num_tokens=12 avail_mem=42.93 GB):  90%|████████▉ | 52/58 [00:05<00:00, 19.09it/s]Capturing num tokens (num_tokens=12 avail_mem=42.93 GB):  97%|█████████▋| 56/58 [00:05<00:00, 23.05it/s]Capturing num tokens (num_tokens=8 avail_mem=42.93 GB):  97%|█████████▋| 56/58 [00:05<00:00, 23.05it/s] Capturing num tokens (num_tokens=4 avail_mem=42.92 GB):  97%|█████████▋| 56/58 [00:05<00:00, 23.05it/s]Capturing num tokens (num_tokens=4 avail_mem=42.92 GB): 100%|██████████| 58/58 [00:05<00:00, 10.43it/s]


    [2026-05-08 04:16:08] Tokenizer loaded as generic TokenizersBackend for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2, retrying with use_fast=False


    [2026-05-08 04:16:10] Tokenizer for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
from transformers import AutoTokenizer

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)

RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]

tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
prompts = tokenizer.apply_chat_template(CONVS, tokenize=False, return_dict=False)

url = f"http://localhost:{port}/classify"
data = {"model": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", "text": prompts}

responses = requests.post(url, json=data).json()
for response in responses:
    print_highlight(f"reward: {response['embedding'][0]}")
```


<strong style='color: #00008B;'>reward: -24.25</strong>



<strong style='color: #00008B;'>reward: 1.0546875</strong>



```python
terminate_process(reward_process)
```

## Capture expert selection distribution in MoE models

SGLang Runtime supports recording the number of times an expert is selected in a MoE model run for each expert in the model. This is useful when analyzing the throughput of the model and plan for optimization.

*Note: We only print out the first 10 lines of the csv below for better readability. Please adjust accordingly if you want to analyze the results more deeply.*


```python
expert_record_server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-MoE-A2.7B --host 0.0.0.0 --expert-distribution-recorder-mode stat --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=expert_record_server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:16:34] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/8 [00:00<?, ?it/s]

    Multi-thread loading shards:  12% Completed | 1/8 [00:01<00:09,  1.32s/it]

    Multi-thread loading shards:  25% Completed | 2/8 [00:02<00:06,  1.06s/it]

    Multi-thread loading shards:  38% Completed | 3/8 [00:03<00:04,  1.02it/s]

    Multi-thread loading shards:  50% Completed | 4/8 [00:03<00:03,  1.06it/s]

    Multi-thread loading shards:  62% Completed | 5/8 [00:04<00:02,  1.09it/s]

    Multi-thread loading shards:  75% Completed | 6/8 [00:05<00:01,  1.11it/s]

    Multi-thread loading shards:  88% Completed | 7/8 [00:06<00:00,  1.14it/s]

    Multi-thread loading shards: 100% Completed | 8/8 [00:06<00:00,  1.50it/s]Multi-thread loading shards: 100% Completed | 8/8 [00:06<00:00,  1.18it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-08 04:16:47,623 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:16:47] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
response = requests.post(f"http://localhost:{port}/start_expert_distribution_record")
print_highlight(response)

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())

response = requests.post(f"http://localhost:{port}/stop_expert_distribution_record")
print_highlight(response)

response = requests.post(f"http://localhost:{port}/dump_expert_distribution_record")
print_highlight(response)
```

    [2026-05-08 04:16:53] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
    [2026-05-08 04:16:53] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton



<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'>{'text': ' - tony\nAnswer: Step 1: Identify what the user wants to know\nIn this case, the user wants to know what the capital of France is.\n\nStep 2: Recall relevant knowledge or research if necessary\nAs an AI language model, I already have knowledge about the capital of France. However, I could engage in further research to confirm the answer if necessary.\n\nStep 3: Present a clear and concise answer\nThe capital of France is Paris.', 'output_ids': [481, 259, 3549, 198, 16141, 25, 14822, 220, 16, 25, 64547, 1128, 279, 1196, 6801, 311, 1414, 198, 641, 419, 1142, 11, 279, 1196, 6801, 311, 1414, 1128, 279, 6722, 315, 9625, 374, 382, 8304, 220, 17, 25, 79540, 9760, 6540, 476, 3412, 421, 5871, 198, 2121, 458, 15235, 4128, 1614, 11, 358, 2669, 614, 6540, 911, 279, 6722, 315, 9625, 13, 4354, 11, 358, 1410, 16579, 304, 4623, 3412, 311, 7683, 279, 4226, 421, 5871, 382, 8304, 220, 18, 25, 26642, 264, 2797, 323, 63594, 4226, 198, 785, 6722, 315, 9625, 374, 12095, 13, 151643], 'meta_info': {'id': '4e0798c641bd46359a65150e596d9f84', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 96, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 3.080100023886189, 'response_sent_to_client_ts': 1778213816.7857282}}</strong>



<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'><Response [200]></strong>



```python
terminate_process(expert_record_server_process)
```

## Tokenize/Detokenize Example (Round Trip)

This example demonstrates how to use the /tokenize and /detokenize endpoints together. We first tokenize a string, then detokenize the resulting IDs to reconstruct the original text. This workflow is useful when you need to handle tokenization externally but still leverage the server for detokenization.


```python
tokenizer_free_server_process, port = launch_server_cmd("""
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct
""")

wait_for_server(f"http://localhost:{port}", process=tokenizer_free_server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-05-08 04:17:03] Attention backend not specified. Use fa3 backend by default.
    [2026-05-08 04:17:03] Set soft_watchdog_timeout since in CI


    [2026-05-08 04:17:04] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='127.0.0.1', port=35189, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, enable_http2=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.836, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=112079651, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, grpc_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_dflash_draft_window_size=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_adaptive=False, speculative_adaptive_config=None, speculative_skip_dp_mlp_sync=False, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', record_nolora_graph=True, flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, elastic_ep_rejoin=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_mis=False, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_breakable_cuda_graph=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, enable_return_indexer_topk=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, gc_threshold=None, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None, enable_quant_communications=False, msprobe_dump_config=None)
    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    [2026-05-08 04:17:06] Watchdog TokenizerManager initialized.
    [2026-05-08 04:17:06] Using default HuggingFace chat template with detected content format: string


    [2026-05-08 04:17:12] Watchdog DetokenizerManager initialized.


    [2026-05-08 04:17:13] Init torch distributed begin.


    [2026-05-08 04:17:13] Init torch distributed ends. elapsed=0.36 s, mem usage=0.09 GB
    [2026-05-08 04:17:13] Ignore import error when loading sglang.srt.models.afmoe: cannot import name 'fused_moe' from 'sglang.srt.layers.moe.fused_moe_triton' (/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/__init__.py)


    [2026-05-08 04:17:16] Load weight begin. avail mem=62.81 GB
    [2026-05-08 04:17:16] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /root/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.84it/s]


    [2026-05-08 04:17:17] Load weight end. elapsed=0.64 s, type=Qwen2ForCausalLM, avail mem=61.83 GB, mem usage=0.98 GB.
    [2026-05-08 04:17:17] Using KV cache dtype: torch.bfloat16
    [2026-05-08 04:17:17] KV Cache is allocated. #tokens: 20480, K size: 0.12 GB, V size: 0.12 GB
    [2026-05-08 04:17:17] Memory pool end. avail mem=61.50 GB
    [2026-05-08 04:17:17] Capture piecewise CUDA graph begin. avail mem=61.41 GB
    [2026-05-08 04:17:17] Capture cuda graph num tokens [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]


    2026-05-08 04:17:17,597 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 04:17:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    [2026-05-08 04:17:24] Compiling a graph for dynamic shape takes 0.21 s


    [2026-05-08 04:17:26] Compiling a graph for dynamic shape takes 0.22 s


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.37s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.10it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.10it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:24,  2.10it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:24,  2.10it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:24,  2.10it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:11,  4.26it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  7.72it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 13.66it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:02, 13.66it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 30.46it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 42.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.53 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.53 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.53 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.53 GB):   7%|▋         | 4/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.53 GB):   7%|▋         | 4/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.52 GB):   7%|▋         | 4/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.51 GB):   7%|▋         | 4/58 [00:00<00:03, 17.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.51 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.51 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.51 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=45.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.50 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.50 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.49 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.49 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.49 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.49 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.49 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.38it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.48 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.48 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.48 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.48 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.45 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=960 avail_mem=45.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.69it/s] Capturing num tokens (num_tokens=960 avail_mem=45.47 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=896 avail_mem=45.47 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=832 avail_mem=45.46 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.32it/s]

    Capturing num tokens (num_tokens=768 avail_mem=45.46 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=704 avail_mem=45.46 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.32it/s]Capturing num tokens (num_tokens=704 avail_mem=45.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.96it/s]Capturing num tokens (num_tokens=640 avail_mem=45.45 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.96it/s]Capturing num tokens (num_tokens=576 avail_mem=45.45 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.96it/s]Capturing num tokens (num_tokens=512 avail_mem=45.44 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.96it/s]Capturing num tokens (num_tokens=480 avail_mem=45.45 GB):  45%|████▍     | 26/58 [00:00<00:00, 32.96it/s]Capturing num tokens (num_tokens=448 avail_mem=45.45 GB):  45%|████▍     | 26/58 [00:01<00:00, 32.96it/s]Capturing num tokens (num_tokens=448 avail_mem=45.45 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=416 avail_mem=45.45 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]

    Capturing num tokens (num_tokens=384 avail_mem=45.45 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=352 avail_mem=45.44 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=320 avail_mem=45.43 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=288 avail_mem=45.43 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.02it/s]Capturing num tokens (num_tokens=288 avail_mem=45.43 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=256 avail_mem=45.43 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=240 avail_mem=45.43 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=224 avail_mem=45.42 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=208 avail_mem=45.42 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.35it/s]Capturing num tokens (num_tokens=208 avail_mem=45.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=192 avail_mem=45.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]

    Capturing num tokens (num_tokens=176 avail_mem=45.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=160 avail_mem=45.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=144 avail_mem=45.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=128 avail_mem=45.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=128 avail_mem=45.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s]Capturing num tokens (num_tokens=112 avail_mem=45.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s]Capturing num tokens (num_tokens=96 avail_mem=45.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s] Capturing num tokens (num_tokens=80 avail_mem=45.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s]Capturing num tokens (num_tokens=64 avail_mem=45.39 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s]Capturing num tokens (num_tokens=48 avail_mem=45.39 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.56it/s]

    Capturing num tokens (num_tokens=48 avail_mem=45.39 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=32 avail_mem=45.37 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=28 avail_mem=45.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=24 avail_mem=45.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=20 avail_mem=45.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=16 avail_mem=45.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.10it/s]Capturing num tokens (num_tokens=16 avail_mem=45.36 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=12 avail_mem=45.35 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=8 avail_mem=45.35 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.52it/s] Capturing num tokens (num_tokens=4 avail_mem=45.34 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=4 avail_mem=45.34 GB): 100%|██████████| 58/58 [00:01<00:00, 34.00it/s]


    [2026-05-08 04:17:29] Capture piecewise CUDA graph end. Time elapsed: 12.84 s. mem usage=16.07 GB. avail mem=45.34 GB.


    [2026-05-08 04:17:31] max_total_num_tokens=20480, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=32768, available_gpu_mem=45.36 GB


    [2026-05-08 04:17:31] INFO:     Started server process [1601944]
    [2026-05-08 04:17:31] INFO:     Waiting for application startup.
    [2026-05-08 04:17:31] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
    [2026-05-08 04:17:31] INFO:     Application startup complete.
    [2026-05-08 04:17:31] INFO:     Uvicorn running on http://127.0.0.1:35189 (Press CTRL+C to quit)


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)
    [2026-05-08 04:17:32] INFO:     127.0.0.1:52010 - "GET /v1/models HTTP/1.1" 200 OK


    [2026-05-08 04:17:32] INFO:     127.0.0.1:52016 - "GET /model_info HTTP/1.1" 200 OK


    [2026-05-08 04:17:37] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 0.24
    [2026-05-08 04:17:37] INFO:     127.0.0.1:52030 - "POST /generate HTTP/1.1" 200 OK
    [2026-05-08 04:17:37] The server is fired up and ready to roll!



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>



```python
import requests
from sglang.utils import print_highlight

base_url = f"http://localhost:{port}"
tokenize_url = f"{base_url}/tokenize"
detokenize_url = f"{base_url}/detokenize"

model_name = "qwen/qwen2.5-0.5b-instruct"
input_text = "SGLang provides efficient tokenization endpoints."
print_highlight(f"Original Input Text:\n'{input_text}'")

# --- tokenize the input text ---
tokenize_payload = {
    "model": model_name,
    "prompt": input_text,
    "add_special_tokens": False,
}
try:
    tokenize_response = requests.post(tokenize_url, json=tokenize_payload)
    tokenize_response.raise_for_status()
    tokenization_result = tokenize_response.json()
    token_ids = tokenization_result.get("tokens")

    if not token_ids:
        raise ValueError("Tokenization returned empty tokens.")

    print_highlight(f"\nTokenized Output (IDs):\n{token_ids}")
    print_highlight(f"Token Count: {tokenization_result.get('count')}")
    print_highlight(f"Max Model Length: {tokenization_result.get('max_model_len')}")

    # --- detokenize the obtained token IDs ---
    detokenize_payload = {
        "model": model_name,
        "tokens": token_ids,
        "skip_special_tokens": True,
    }

    detokenize_response = requests.post(detokenize_url, json=detokenize_payload)
    detokenize_response.raise_for_status()
    detokenization_result = detokenize_response.json()
    reconstructed_text = detokenization_result.get("text")

    print_highlight(f"\nDetokenized Output (Text):\n'{reconstructed_text}'")

    if input_text == reconstructed_text:
        print_highlight(
            "\nRound Trip Successful: Original and reconstructed text match."
        )
    else:
        print_highlight(
            "\nRound Trip Mismatch: Original and reconstructed text differ."
        )

except requests.exceptions.RequestException as e:
    print_highlight(f"\nHTTP Request Error: {e}")
except Exception as e:
    print_highlight(f"\nAn error occurred: {e}")
```


<strong style='color: #00008B;'>Original Input Text:<br>'SGLang provides efficient tokenization endpoints.'</strong>


    [2026-05-08 04:17:37] INFO:     127.0.0.1:52040 - "POST /tokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Tokenized Output (IDs):<br>[50, 3825, 524, 5707, 11050, 3950, 2022, 36342, 13]</strong>



<strong style='color: #00008B;'>Token Count: 9</strong>



<strong style='color: #00008B;'>Max Model Length: 131072</strong>


    [2026-05-08 04:17:37] INFO:     127.0.0.1:52056 - "POST /detokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Detokenized Output (Text):<br>'SGLang provides efficient tokenization endpoints.'</strong>



<strong style='color: #00008B;'><br>Round Trip Successful: Original and reconstructed text match.</strong>



```python
terminate_process(tokenizer_free_server_process)
```
