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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.97it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:21,  4.59s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:48,  1.11it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.69it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  5.81it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]

    Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:05<00:07,  5.81it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04,  8.20it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04,  8.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03, 11.02it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 14.37it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]

    Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 27.41it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 35.10it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 42.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.79 GB):   3%|▎         | 2/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.79 GB):   3%|▎         | 2/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.79 GB):   3%|▎         | 2/58 [00:00<00:03, 14.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.79 GB):   7%|▋         | 4/58 [00:00<00:03, 13.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 13.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 13.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.75 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.73 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.73 GB):  10%|█         | 6/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.73 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.53 GB):  14%|█▍        | 8/58 [00:00<00:04, 12.39it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.53 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.53 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.70 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.70 GB):  21%|██        | 12/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.69 GB):  21%|██        | 12/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.69 GB):  21%|██        | 12/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.68 GB):  21%|██        | 12/58 [00:00<00:03, 14.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.68 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.67 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.11it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.67 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.66 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.66 GB):  31%|███       | 18/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.65 GB):  31%|███       | 18/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.65 GB):  31%|███       | 18/58 [00:01<00:02, 19.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.63 GB):  31%|███       | 18/58 [00:01<00:02, 19.15it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.50it/s]Capturing num tokens (num_tokens=960 avail_mem=71.64 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.50it/s] Capturing num tokens (num_tokens=896 avail_mem=71.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.50it/s]Capturing num tokens (num_tokens=832 avail_mem=71.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.50it/s]Capturing num tokens (num_tokens=832 avail_mem=71.63 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.24it/s]Capturing num tokens (num_tokens=768 avail_mem=71.61 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.24it/s]Capturing num tokens (num_tokens=704 avail_mem=71.62 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.24it/s]Capturing num tokens (num_tokens=640 avail_mem=71.61 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.24it/s]

    Capturing num tokens (num_tokens=576 avail_mem=71.61 GB):  41%|████▏     | 24/58 [00:01<00:01, 22.24it/s]Capturing num tokens (num_tokens=576 avail_mem=71.61 GB):  48%|████▊     | 28/58 [00:01<00:01, 24.08it/s]Capturing num tokens (num_tokens=512 avail_mem=71.59 GB):  48%|████▊     | 28/58 [00:01<00:01, 24.08it/s]Capturing num tokens (num_tokens=480 avail_mem=71.60 GB):  48%|████▊     | 28/58 [00:01<00:01, 24.08it/s]Capturing num tokens (num_tokens=448 avail_mem=71.60 GB):  48%|████▊     | 28/58 [00:01<00:01, 24.08it/s]Capturing num tokens (num_tokens=416 avail_mem=71.59 GB):  48%|████▊     | 28/58 [00:01<00:01, 24.08it/s]Capturing num tokens (num_tokens=416 avail_mem=71.59 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=384 avail_mem=71.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.55it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=320 avail_mem=71.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.55it/s]Capturing num tokens (num_tokens=320 avail_mem=71.58 GB):  60%|██████    | 35/58 [00:01<00:00, 26.76it/s]Capturing num tokens (num_tokens=288 avail_mem=71.57 GB):  60%|██████    | 35/58 [00:01<00:00, 26.76it/s]Capturing num tokens (num_tokens=256 avail_mem=71.57 GB):  60%|██████    | 35/58 [00:01<00:00, 26.76it/s]Capturing num tokens (num_tokens=240 avail_mem=71.56 GB):  60%|██████    | 35/58 [00:01<00:00, 26.76it/s]Capturing num tokens (num_tokens=224 avail_mem=71.55 GB):  60%|██████    | 35/58 [00:01<00:00, 26.76it/s]Capturing num tokens (num_tokens=224 avail_mem=71.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=208 avail_mem=71.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.90it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=176 avail_mem=71.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=160 avail_mem=71.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.90it/s]Capturing num tokens (num_tokens=160 avail_mem=71.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=144 avail_mem=71.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.17it/s]Capturing num tokens (num_tokens=128 avail_mem=71.50 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.17it/s]Capturing num tokens (num_tokens=112 avail_mem=71.49 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.17it/s]Capturing num tokens (num_tokens=96 avail_mem=71.48 GB):  74%|███████▍  | 43/58 [00:02<00:00, 30.17it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=71.48 GB):  81%|████████  | 47/58 [00:02<00:00, 28.47it/s]Capturing num tokens (num_tokens=80 avail_mem=71.48 GB):  81%|████████  | 47/58 [00:02<00:00, 28.47it/s]Capturing num tokens (num_tokens=64 avail_mem=71.49 GB):  81%|████████  | 47/58 [00:02<00:00, 28.47it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.48 GB):  81%|████████  | 47/58 [00:02<00:00, 28.47it/s]Capturing num tokens (num_tokens=48 avail_mem=71.48 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.42it/s]Capturing num tokens (num_tokens=32 avail_mem=71.48 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.42it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.47 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.42it/s]Capturing num tokens (num_tokens=24 avail_mem=71.44 GB):  86%|████████▌ | 50/58 [00:02<00:00, 17.42it/s]Capturing num tokens (num_tokens=24 avail_mem=71.44 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.49it/s]Capturing num tokens (num_tokens=20 avail_mem=71.46 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.49it/s]Capturing num tokens (num_tokens=16 avail_mem=71.45 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.49it/s]Capturing num tokens (num_tokens=12 avail_mem=71.44 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.49it/s]Capturing num tokens (num_tokens=8 avail_mem=71.44 GB):  91%|█████████▏| 53/58 [00:02<00:00, 16.49it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=71.44 GB):  98%|█████████▊| 57/58 [00:02<00:00, 19.99it/s]Capturing num tokens (num_tokens=4 avail_mem=71.43 GB):  98%|█████████▊| 57/58 [00:02<00:00, 19.99it/s]Capturing num tokens (num_tokens=4 avail_mem=71.43 GB): 100%|██████████| 58/58 [00:02<00:00, 20.24it/s]


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


<strong style='color: #00008B;'>{'text': " France has three important cities: Paris, Lyon, and simply called Paris: the capital.  \n\n- The capital of France is Paris, where most French people live and work. It's the largest city in France (larger than the federal governments of Canada, Germany, and Switzerland).\n\n- It lies in the South, and is located near the Seine River, which flows through the center of Paris.\n\n- The Chamber of Commerce and Industry mainly exists in Paris.\n- It is acknowledged that the capital of France is not based in Lyon or Marseille, since both cities offer too many trade difficulties.\n\n- It's known that there are bored bikes", 'output_ids': [9625, 702, 2326, 2989, 9720, 25, 12095, 11, 55201, 11, 323, 4936, 2598, 12095, 25, 279, 6722, 13, 18611, 12, 576, 6722, 315, 9625, 374, 12095, 11, 1380, 1429, 8585, 1251, 3887, 323, 975, 13, 1084, 594, 279, 7772, 3283, 304, 9625, 320, 26897, 261, 1091, 279, 6775, 16636, 315, 6864, 11, 9856, 11, 323, 29121, 3593, 12, 1084, 15448, 304, 279, 4882, 11, 323, 374, 7407, 3143, 279, 1345, 482, 10948, 11, 892, 27455, 1526, 279, 4126, 315, 12095, 382, 12, 576, 31379, 315, 30380, 323, 23840, 14576, 6724, 304, 12095, 624, 12, 1084, 374, 25498, 429, 279, 6722, 315, 9625, 374, 537, 3118, 304, 55201, 476, 87693, 11, 2474, 2176, 9720, 3010, 2238, 1657, 6559, 26038, 382, 12, 1084, 594, 3881, 429, 1052, 525, 33286, 30453], 'meta_info': {'id': '49d45725ea9d4c26a9f3f158e6a03440', 'finish_reason': {'type': 'length', 'length': 128}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 128, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 1.0526745808310807, 'response_sent_to_client_ts': 1780020343.0100684}}</strong>


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

    [2026-05-29 02:05:43] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.



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


<strong style='color: #00008B;'>{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"detokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"prefill_only_disable_kv_cache":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","model_config_parser":"auto","host":"0.0.0.0","port":31619,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"prefill_delayer_queue_min_ratio":null,"prefill_delayer_max_delay_ms":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":null,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":355442161,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_forward_pass_metrics":false,"forward_pass_metrics_worker_id":"","forward_pass_metrics_ipc_name":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"stat_loggers":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","asr_max_buffer_seconds":60,"asr_max_concurrent_sessions":32,"dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":null,"prefill_attention_backend":null,"sampling_backend":"flashinfer","grammar_backend":"xgrammar","radix_cache_backend":null,"mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","dsa_prefill_backend":null,"dsa_decode_backend":null,"dsa_topk_backend":"sgl-kernel","disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_draft_window_size":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"speculative_adaptive":false,"speculative_adaptive_config":null,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","deepep_dispatcher_output_dtype":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"enable_deepep_waterfill":false,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"timeout","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"lmcache_config_file":null,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":false,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"disable_attn_tp_gather":false,"gc_threshold":null,"enable_dsa_prefill_context_parallel":false,"dsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"weight_loader_drop_cache_after_load":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"status":"ready","max_total_num_tokens":20480,"max_req_input_len":20474,"internal_states":[{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"detokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"prefill_only_disable_kv_cache":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","model_config_parser":"auto","host":"0.0.0.0","port":31619,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"prefill_delayer_queue_min_ratio":null,"prefill_delayer_max_delay_ms":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":128,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":355442161,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_forward_pass_metrics":false,"forward_pass_metrics_worker_id":"","forward_pass_metrics_ipc_name":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"stat_loggers":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","asr_max_buffer_seconds":60,"asr_max_concurrent_sessions":32,"dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":"fa3","prefill_attention_backend":"fa3","sampling_backend":"flashinfer","grammar_backend":"xgrammar","radix_cache_backend":null,"mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","dsa_prefill_backend":null,"dsa_decode_backend":null,"dsa_topk_backend":"sgl-kernel","disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_draft_window_size":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"speculative_adaptive":false,"speculative_adaptive_config":null,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","deepep_dispatcher_output_dtype":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"enable_deepep_waterfill":false,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"timeout","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"lmcache_config_file":null,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":true,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"disable_attn_tp_gather":false,"gc_threshold":null,"enable_dsa_prefill_context_parallel":false,"dsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"weight_loader_drop_cache_after_load":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"enable_grpc":false,"grpc_port":41619,"_quantization_explicitly_unset":false,"use_mla_backend":false,"_mx_config_cache":{},"max_speculative_num_draft_tokens":null,"last_gen_throughput":135.84764029950588,"memory_usage":{"weight":0.98,"kvcache":0.23,"token_capacity":20480,"graph":0},"effective_max_running_requests_per_dp":128}],"version":"0.0.0.dev1+g54b06f199","kv_events":null}</strong>


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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.25it/s]
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

    [2026-05-29 02:05:45] Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).



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


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.32s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.04it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.01s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<05:52,  6.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<05:52,  6.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:06<05:52,  6.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:30,  1.65s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:30,  1.65s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:06<01:30,  1.65s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:06<01:30,  1.65s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:34,  1.50it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:34,  1.50it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:06<00:34,  1.50it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:06<00:34,  1.50it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]

    Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:06<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:06<00:08,  5.30it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:06<00:03, 10.06it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:06<00:01, 16.71it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:00, 24.10it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:00, 24.10it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:07<00:00, 24.10it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:07<00:00, 32.20it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:07<00:00, 41.78it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:07<00:00, 41.78it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:07<00:00, 41.78it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:07<00:00, 41.78it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:07<00:00, 41.78it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.37 GB):   2%|▏         | 1/58 [00:00<00:06,  9.34it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.33 GB):   2%|▏         | 1/58 [00:00<00:06,  9.34it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.33 GB):   3%|▎         | 2/58 [00:00<00:07,  7.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.31 GB):   3%|▎         | 2/58 [00:00<00:07,  7.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.31 GB):   5%|▌         | 3/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.31 GB):   5%|▌         | 3/58 [00:00<00:06,  8.13it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.82 GB):   5%|▌         | 3/58 [00:00<00:06,  8.13it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.82 GB):   9%|▊         | 5/58 [00:00<00:05,  9.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.81 GB):   9%|▊         | 5/58 [00:00<00:05,  9.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.65 GB):   9%|▊         | 5/58 [00:00<00:05,  9.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.99it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.65 GB):  12%|█▏        | 7/58 [00:00<00:04, 11.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.65 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.65 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.64 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.64 GB):  21%|██        | 12/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.63 GB):  21%|██        | 12/58 [00:00<00:02, 17.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=56.63 GB):  21%|██        | 12/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.62 GB):  21%|██        | 12/58 [00:00<00:02, 17.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.62 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.63 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.61 GB):  26%|██▌       | 15/58 [00:01<00:02, 20.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.61 GB):  26%|██▌       | 15/58 [00:01<00:02, 20.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.60 GB):  26%|██▌       | 15/58 [00:01<00:02, 20.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.60 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.60 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=56.60 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s]Capturing num tokens (num_tokens=960 avail_mem=56.57 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s] Capturing num tokens (num_tokens=896 avail_mem=56.57 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s]Capturing num tokens (num_tokens=832 avail_mem=56.59 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.45it/s]Capturing num tokens (num_tokens=832 avail_mem=56.59 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]Capturing num tokens (num_tokens=768 avail_mem=56.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]Capturing num tokens (num_tokens=704 avail_mem=56.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]Capturing num tokens (num_tokens=640 avail_mem=56.58 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]Capturing num tokens (num_tokens=576 avail_mem=56.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]Capturing num tokens (num_tokens=512 avail_mem=56.57 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.33it/s]

    Capturing num tokens (num_tokens=512 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=480 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=448 avail_mem=56.57 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=416 avail_mem=56.56 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=384 avail_mem=56.56 GB):  50%|█████     | 29/58 [00:01<00:00, 32.94it/s]Capturing num tokens (num_tokens=384 avail_mem=56.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=352 avail_mem=56.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=320 avail_mem=56.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=288 avail_mem=56.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.86it/s]Capturing num tokens (num_tokens=256 avail_mem=56.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.86it/s]

    Capturing num tokens (num_tokens=256 avail_mem=56.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=240 avail_mem=56.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=224 avail_mem=56.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=208 avail_mem=56.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 34.94it/s]Capturing num tokens (num_tokens=192 avail_mem=56.54 GB):  71%|███████   | 41/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=176 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=160 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:01<00:00, 33.07it/s]

    Capturing num tokens (num_tokens=144 avail_mem=56.53 GB):  71%|███████   | 41/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=128 avail_mem=56.52 GB):  71%|███████   | 41/58 [00:01<00:00, 33.07it/s]Capturing num tokens (num_tokens=128 avail_mem=56.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=112 avail_mem=56.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=96 avail_mem=56.51 GB):  78%|███████▊  | 45/58 [00:01<00:00, 32.04it/s] Capturing num tokens (num_tokens=80 avail_mem=56.51 GB):  78%|███████▊  | 45/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=64 avail_mem=56.51 GB):  78%|███████▊  | 45/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=64 avail_mem=56.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=48 avail_mem=56.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.11it/s]

    Capturing num tokens (num_tokens=32 avail_mem=56.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 32.11it/s]Capturing num tokens (num_tokens=28 avail_mem=56.50 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=24 avail_mem=56.49 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.11it/s]Capturing num tokens (num_tokens=24 avail_mem=56.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.66it/s]Capturing num tokens (num_tokens=20 avail_mem=56.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.66it/s]Capturing num tokens (num_tokens=16 avail_mem=56.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.66it/s]Capturing num tokens (num_tokens=12 avail_mem=56.48 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.66it/s]Capturing num tokens (num_tokens=8 avail_mem=56.48 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.66it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=56.48 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=4 avail_mem=56.47 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.32it/s]Capturing num tokens (num_tokens=4 avail_mem=56.47 GB): 100%|██████████| 58/58 [00:02<00:00, 25.94it/s]


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


    [2026-05-29 02:06:30] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    [2026-05-29 02:06:35] No HuggingFace chat template found


    [2026-05-29 02:06:40] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]Multi-thread loading shards: 100% Completed | 1/1 [00:01<00:00,  1.05s/it]


    [2026-05-29 02:06:46] Disable piecewise CUDA graph because the model is not a language model


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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.52it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.52it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.61it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 18.70it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.70it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 21.74it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 24.31it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s]

    Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 27.93it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 31.72it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 31.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.99 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=40.99 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=7680 avail_mem=40.95 GB):   2%|▏         | 1/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=40.95 GB):   3%|▎         | 2/58 [00:00<00:07,  7.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.95 GB):   3%|▎         | 2/58 [00:00<00:07,  7.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=40.95 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=40.95 GB):   5%|▌         | 3/58 [00:00<00:07,  7.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=40.95 GB):   7%|▋         | 4/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.95 GB):   7%|▋         | 4/58 [00:00<00:07,  7.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.95 GB):   9%|▊         | 5/58 [00:00<00:07,  7.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=40.94 GB):   9%|▊         | 5/58 [00:00<00:07,  7.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=40.94 GB):  10%|█         | 6/58 [00:00<00:07,  6.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.93 GB):  10%|█         | 6/58 [00:00<00:07,  6.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=40.93 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=40.93 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.73it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=40.91 GB):  12%|█▏        | 7/58 [00:01<00:07,  6.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=40.91 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.91 GB):  16%|█▌        | 9/58 [00:01<00:06,  7.36it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.91 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=40.41 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.33 GB):  17%|█▋        | 10/58 [00:01<00:06,  7.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=40.33 GB):  21%|██        | 12/58 [00:01<00:05,  9.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=40.33 GB):  21%|██        | 12/58 [00:01<00:05,  9.09it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=40.33 GB):  21%|██        | 12/58 [00:01<00:05,  9.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=40.33 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=40.32 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.32 GB):  24%|██▍       | 14/58 [00:01<00:04, 10.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=40.32 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=40.32 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=40.31 GB):  28%|██▊       | 16/58 [00:01<00:03, 12.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=40.31 GB):  31%|███       | 18/58 [00:01<00:02, 13.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=40.31 GB):  31%|███       | 18/58 [00:01<00:02, 13.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=40.31 GB):  31%|███       | 18/58 [00:01<00:02, 13.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.29 GB):  31%|███       | 18/58 [00:01<00:02, 13.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=40.29 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.20it/s]Capturing num tokens (num_tokens=960 avail_mem=40.30 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.20it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=40.30 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.20it/s]Capturing num tokens (num_tokens=832 avail_mem=40.29 GB):  36%|███▌      | 21/58 [00:02<00:02, 16.20it/s]Capturing num tokens (num_tokens=832 avail_mem=40.29 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=768 avail_mem=40.28 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=704 avail_mem=40.26 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=640 avail_mem=40.23 GB):  41%|████▏     | 24/58 [00:02<00:01, 18.94it/s]Capturing num tokens (num_tokens=640 avail_mem=40.23 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.61it/s]Capturing num tokens (num_tokens=576 avail_mem=40.23 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.61it/s]

    Capturing num tokens (num_tokens=512 avail_mem=40.22 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.61it/s]Capturing num tokens (num_tokens=480 avail_mem=40.23 GB):  47%|████▋     | 27/58 [00:02<00:01, 21.61it/s]Capturing num tokens (num_tokens=480 avail_mem=40.23 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=448 avail_mem=40.23 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=416 avail_mem=40.23 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=384 avail_mem=40.23 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=352 avail_mem=40.22 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=320 avail_mem=40.21 GB):  52%|█████▏    | 30/58 [00:02<00:01, 23.47it/s]Capturing num tokens (num_tokens=320 avail_mem=40.21 GB):  60%|██████    | 35/58 [00:02<00:00, 30.62it/s]Capturing num tokens (num_tokens=288 avail_mem=40.21 GB):  60%|██████    | 35/58 [00:02<00:00, 30.62it/s]Capturing num tokens (num_tokens=256 avail_mem=40.19 GB):  60%|██████    | 35/58 [00:02<00:00, 30.62it/s]

    Capturing num tokens (num_tokens=240 avail_mem=40.16 GB):  60%|██████    | 35/58 [00:02<00:00, 30.62it/s]Capturing num tokens (num_tokens=224 avail_mem=40.16 GB):  60%|██████    | 35/58 [00:02<00:00, 30.62it/s]Capturing num tokens (num_tokens=224 avail_mem=40.16 GB):  67%|██████▋   | 39/58 [00:02<00:00, 30.00it/s]Capturing num tokens (num_tokens=208 avail_mem=40.16 GB):  67%|██████▋   | 39/58 [00:02<00:00, 30.00it/s]Capturing num tokens (num_tokens=192 avail_mem=40.16 GB):  67%|██████▋   | 39/58 [00:02<00:00, 30.00it/s]Capturing num tokens (num_tokens=176 avail_mem=40.15 GB):  67%|██████▋   | 39/58 [00:02<00:00, 30.00it/s]

    Capturing num tokens (num_tokens=160 avail_mem=40.15 GB):  67%|██████▋   | 39/58 [00:02<00:00, 30.00it/s]Capturing num tokens (num_tokens=160 avail_mem=40.15 GB):  74%|███████▍  | 43/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=144 avail_mem=40.15 GB):  74%|███████▍  | 43/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=128 avail_mem=39.01 GB):  74%|███████▍  | 43/58 [00:02<00:00, 28.46it/s]

    Capturing num tokens (num_tokens=112 avail_mem=39.01 GB):  74%|███████▍  | 43/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=112 avail_mem=39.01 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.62it/s]Capturing num tokens (num_tokens=96 avail_mem=39.00 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.62it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=40.10 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.62it/s]Capturing num tokens (num_tokens=64 avail_mem=40.10 GB):  79%|███████▉  | 46/58 [00:03<00:00, 16.62it/s]Capturing num tokens (num_tokens=64 avail_mem=40.10 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.80it/s]Capturing num tokens (num_tokens=48 avail_mem=39.11 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.80it/s]

    Capturing num tokens (num_tokens=32 avail_mem=39.10 GB):  84%|████████▍ | 49/58 [00:03<00:00, 13.80it/s]Capturing num tokens (num_tokens=32 avail_mem=39.10 GB):  88%|████████▊ | 51/58 [00:03<00:00, 11.77it/s]Capturing num tokens (num_tokens=28 avail_mem=39.78 GB):  88%|████████▊ | 51/58 [00:03<00:00, 11.77it/s]

    Capturing num tokens (num_tokens=24 avail_mem=58.60 GB):  88%|████████▊ | 51/58 [00:03<00:00, 11.77it/s]Capturing num tokens (num_tokens=24 avail_mem=58.60 GB):  91%|█████████▏| 53/58 [00:03<00:00, 10.74it/s]Capturing num tokens (num_tokens=20 avail_mem=57.67 GB):  91%|█████████▏| 53/58 [00:03<00:00, 10.74it/s]Capturing num tokens (num_tokens=16 avail_mem=57.67 GB):  91%|█████████▏| 53/58 [00:04<00:00, 10.74it/s]

    Capturing num tokens (num_tokens=16 avail_mem=57.67 GB):  95%|█████████▍| 55/58 [00:04<00:00, 11.15it/s]Capturing num tokens (num_tokens=12 avail_mem=58.59 GB):  95%|█████████▍| 55/58 [00:04<00:00, 11.15it/s]Capturing num tokens (num_tokens=8 avail_mem=58.59 GB):  95%|█████████▍| 55/58 [00:04<00:00, 11.15it/s] Capturing num tokens (num_tokens=8 avail_mem=58.59 GB):  98%|█████████▊| 57/58 [00:04<00:00, 11.89it/s]Capturing num tokens (num_tokens=4 avail_mem=57.73 GB):  98%|█████████▊| 57/58 [00:04<00:00, 11.89it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.73 GB): 100%|██████████| 58/58 [00:04<00:00, 13.33it/s]


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


    [2026-05-29 02:07:40] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    [2026-05-29 02:07:49] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:02<00:06,  2.05s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:03,  1.78s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:05<00:01,  1.67s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.06s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.32s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:17,  6.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:17,  6.63s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:40,  2.87s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:40,  2.87s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:31,  1.66s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:31,  1.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<00:58,  1.08s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<00:58,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:29,  1.79it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:29,  1.79it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:21,  2.36it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:21,  2.36it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:16,  3.02it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:16,  3.02it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:13,  3.76it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:13,  3.76it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:10,  4.60it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:10,  4.60it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:08<00:10,  4.60it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:07,  6.18it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:07,  6.18it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:08<00:07,  6.18it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:05,  7.58it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:05,  7.58it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:05,  7.91it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:05,  7.91it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:08<00:05,  7.91it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:04,  9.71it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:04,  9.71it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:04,  9.71it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:03, 11.89it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:03, 11.89it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:03, 11.89it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:08<00:03, 11.89it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:08<00:03, 11.89it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:08<00:01, 17.66it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:08<00:01, 17.66it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:09<00:01, 17.66it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:09<00:01, 17.66it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:09<00:01, 17.66it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:09<00:01, 17.66it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:09<00:01, 25.08it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:09<00:00, 33.19it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:09<00:00, 42.32it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:09<00:00, 48.61it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:09<00:00, 57.01it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:09<00:00, 57.01it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:09<00:00, 57.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.08 GB):   2%|▏         | 1/58 [00:00<00:19,  2.89it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.05 GB):   2%|▏         | 1/58 [00:00<00:19,  2.89it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.05 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.05 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.05 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.05 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.05 GB):   7%|▋         | 4/58 [00:01<00:15,  3.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.05 GB):   7%|▋         | 4/58 [00:01<00:15,  3.48it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.05 GB):   9%|▊         | 5/58 [00:01<00:14,  3.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=24.05 GB):   9%|▊         | 5/58 [00:01<00:14,  3.75it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=24.05 GB):  10%|█         | 6/58 [00:01<00:12,  4.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.05 GB):  10%|█         | 6/58 [00:01<00:12,  4.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=24.05 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=24.05 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=24.05 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.05 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=24.05 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=24.04 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.24it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=24.04 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.02 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=24.02 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=24.02 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.97it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=24.02 GB):  21%|██        | 12/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.99 GB):  21%|██        | 12/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.99 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=23.91 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=23.91 GB):  24%|██▍       | 14/58 [00:02<00:06,  7.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=23.89 GB):  24%|██▍       | 14/58 [00:02<00:06,  7.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=23.38 GB):  24%|██▍       | 14/58 [00:02<00:06,  7.12it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=23.38 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=23.37 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=23.37 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.37 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.14it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=23.37 GB):  31%|███       | 18/58 [00:03<00:05,  6.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.36 GB):  31%|███       | 18/58 [00:03<00:05,  6.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.36 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=23.36 GB):  33%|███▎      | 19/58 [00:03<00:05,  7.45it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=23.36 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=23.36 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.00it/s]Capturing num tokens (num_tokens=960 avail_mem=23.35 GB):  34%|███▍      | 20/58 [00:03<00:04,  8.00it/s] Capturing num tokens (num_tokens=960 avail_mem=23.35 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=896 avail_mem=23.34 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=832 avail_mem=23.34 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.72it/s]

    Capturing num tokens (num_tokens=768 avail_mem=23.34 GB):  38%|███▊      | 22/58 [00:03<00:03,  9.72it/s]Capturing num tokens (num_tokens=768 avail_mem=23.34 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.19it/s]Capturing num tokens (num_tokens=704 avail_mem=41.59 GB):  43%|████▎     | 25/58 [00:03<00:02, 13.19it/s]Capturing num tokens (num_tokens=640 avail_mem=41.58 GB):  43%|████▎     | 25/58 [00:04<00:02, 13.19it/s]

    Capturing num tokens (num_tokens=640 avail_mem=41.58 GB):  47%|████▋     | 27/58 [00:04<00:02, 12.77it/s]Capturing num tokens (num_tokens=576 avail_mem=41.58 GB):  47%|████▋     | 27/58 [00:04<00:02, 12.77it/s]Capturing num tokens (num_tokens=512 avail_mem=41.58 GB):  47%|████▋     | 27/58 [00:04<00:02, 12.77it/s]Capturing num tokens (num_tokens=480 avail_mem=41.57 GB):  47%|████▋     | 27/58 [00:04<00:02, 12.77it/s]Capturing num tokens (num_tokens=480 avail_mem=41.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 16.52it/s]Capturing num tokens (num_tokens=448 avail_mem=41.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 16.52it/s]Capturing num tokens (num_tokens=416 avail_mem=41.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 16.52it/s]Capturing num tokens (num_tokens=384 avail_mem=41.57 GB):  52%|█████▏    | 30/58 [00:04<00:01, 16.52it/s]

    Capturing num tokens (num_tokens=384 avail_mem=41.57 GB):  57%|█████▋    | 33/58 [00:04<00:01, 18.50it/s]Capturing num tokens (num_tokens=352 avail_mem=41.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 18.50it/s]Capturing num tokens (num_tokens=320 avail_mem=41.56 GB):  57%|█████▋    | 33/58 [00:04<00:01, 18.50it/s]Capturing num tokens (num_tokens=288 avail_mem=41.55 GB):  57%|█████▋    | 33/58 [00:04<00:01, 18.50it/s]Capturing num tokens (num_tokens=256 avail_mem=41.55 GB):  57%|█████▋    | 33/58 [00:04<00:01, 18.50it/s]Capturing num tokens (num_tokens=256 avail_mem=41.55 GB):  64%|██████▍   | 37/58 [00:04<00:01, 20.83it/s]Capturing num tokens (num_tokens=240 avail_mem=41.55 GB):  64%|██████▍   | 37/58 [00:04<00:01, 20.83it/s]Capturing num tokens (num_tokens=224 avail_mem=41.54 GB):  64%|██████▍   | 37/58 [00:04<00:01, 20.83it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.54 GB):  64%|██████▍   | 37/58 [00:04<00:01, 20.83it/s]Capturing num tokens (num_tokens=192 avail_mem=41.53 GB):  64%|██████▍   | 37/58 [00:04<00:01, 20.83it/s]Capturing num tokens (num_tokens=192 avail_mem=41.53 GB):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Capturing num tokens (num_tokens=176 avail_mem=41.53 GB):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Capturing num tokens (num_tokens=160 avail_mem=41.52 GB):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Capturing num tokens (num_tokens=144 avail_mem=41.52 GB):  71%|███████   | 41/58 [00:04<00:00, 24.14it/s]Capturing num tokens (num_tokens=144 avail_mem=41.52 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.96it/s]Capturing num tokens (num_tokens=128 avail_mem=41.51 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.96it/s]

    Capturing num tokens (num_tokens=112 avail_mem=41.52 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.96it/s]Capturing num tokens (num_tokens=96 avail_mem=41.51 GB):  76%|███████▌  | 44/58 [00:04<00:00, 24.96it/s] Capturing num tokens (num_tokens=96 avail_mem=41.51 GB):  81%|████████  | 47/58 [00:04<00:00, 25.20it/s]Capturing num tokens (num_tokens=80 avail_mem=41.51 GB):  81%|████████  | 47/58 [00:04<00:00, 25.20it/s]Capturing num tokens (num_tokens=64 avail_mem=41.50 GB):  81%|████████  | 47/58 [00:04<00:00, 25.20it/s]Capturing num tokens (num_tokens=48 avail_mem=41.50 GB):  81%|████████  | 47/58 [00:04<00:00, 25.20it/s]Capturing num tokens (num_tokens=32 avail_mem=41.50 GB):  81%|████████  | 47/58 [00:04<00:00, 25.20it/s]

    Capturing num tokens (num_tokens=32 avail_mem=41.50 GB):  88%|████████▊ | 51/58 [00:04<00:00, 27.67it/s]Capturing num tokens (num_tokens=28 avail_mem=41.49 GB):  88%|████████▊ | 51/58 [00:04<00:00, 27.67it/s]Capturing num tokens (num_tokens=24 avail_mem=41.49 GB):  88%|████████▊ | 51/58 [00:04<00:00, 27.67it/s]Capturing num tokens (num_tokens=20 avail_mem=41.48 GB):  88%|████████▊ | 51/58 [00:04<00:00, 27.67it/s]Capturing num tokens (num_tokens=20 avail_mem=41.48 GB):  93%|█████████▎| 54/58 [00:05<00:00, 27.58it/s]Capturing num tokens (num_tokens=16 avail_mem=41.48 GB):  93%|█████████▎| 54/58 [00:05<00:00, 27.58it/s]Capturing num tokens (num_tokens=12 avail_mem=41.48 GB):  93%|█████████▎| 54/58 [00:05<00:00, 27.58it/s]Capturing num tokens (num_tokens=8 avail_mem=41.47 GB):  93%|█████████▎| 54/58 [00:05<00:00, 27.58it/s] Capturing num tokens (num_tokens=4 avail_mem=41.47 GB):  93%|█████████▎| 54/58 [00:05<00:00, 27.58it/s]

    Capturing num tokens (num_tokens=4 avail_mem=41.47 GB): 100%|██████████| 58/58 [00:05<00:00, 28.22it/s]Capturing num tokens (num_tokens=4 avail_mem=41.47 GB): 100%|██████████| 58/58 [00:05<00:00, 11.28it/s]


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


    Multi-thread loading shards:   0% Completed | 0/8 [00:00<?, ?it/s]

    Multi-thread loading shards:  12% Completed | 1/8 [00:00<00:05,  1.30it/s]

    Multi-thread loading shards:  25% Completed | 2/8 [00:01<00:04,  1.33it/s]

    Multi-thread loading shards:  38% Completed | 3/8 [00:02<00:03,  1.31it/s]

    Multi-thread loading shards:  50% Completed | 4/8 [00:03<00:03,  1.33it/s]

    Multi-thread loading shards:  62% Completed | 5/8 [00:03<00:02,  1.35it/s]

    Multi-thread loading shards:  75% Completed | 6/8 [00:04<00:01,  1.37it/s]

    Multi-thread loading shards:  88% Completed | 7/8 [00:05<00:00,  1.37it/s]

    Multi-thread loading shards: 100% Completed | 8/8 [00:05<00:00,  1.77it/s]Multi-thread loading shards: 100% Completed | 8/8 [00:05<00:00,  1.48it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



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

    [2026-05-29 02:09:07] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
    [2026-05-29 02:09:07] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton



<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'>{'text': ' Capital of France is Paris.', 'output_ids': [18374, 315, 9625, 374, 12095, 13, 151643], 'meta_info': {'id': '07ff1f68f8494898a75b2ae2c63ef879', 'finish_reason': {'type': 'stop', 'matched': 151643}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 1.0965558751486242, 'response_sent_to_client_ts': 1780020548.7757573}}</strong>



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


    [2026-05-29 02:09:15] Attention backend not specified. Use fa3 backend by default.
    [2026-05-29 02:09:15] Set soft_watchdog_timeout since in CI


    [2026-05-29 02:09:17] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, detokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, prefill_only_disable_kv_cache=False, enable_multimodal=None, revision=None, model_impl='auto', model_config_parser='auto', host='127.0.0.1', port=34265, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, enable_http2=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.836, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=1065876435, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, grpc_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_forward_pass_metrics=False, forward_pass_metrics_worker_id='', forward_pass_metrics_ipc_name=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, stat_loggers=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', asr_max_buffer_seconds=60, asr_max_concurrent_sessions=32, dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', radix_cache_backend=None, mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', dsa_prefill_backend=None, dsa_decode_backend=None, dsa_topk_backend='sgl-kernel', disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_draft_window_size=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_skip_dp_mlp_sync=False, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, enable_multi_layer_eagle=False, speculative_adaptive=False, speculative_adaptive_config=None, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', deepep_dispatcher_output_dtype='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, enable_deepep_waterfill=False, elastic_ep_rejoin=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, lmcache_config_file=None, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_mis=False, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_breakable_cuda_graph=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, enable_return_indexer_topk=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, disable_attn_tp_gather=False, gc_threshold=None, enable_dsa_prefill_context_parallel=False, dsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None, enable_quant_communications=False, msprobe_dump_config=None)


    [2026-05-29 02:09:18] Watchdog TokenizerManager initialized.
    [2026-05-29 02:09:18] Using default HuggingFace chat template with detected content format: string


    [2026-05-29 02:09:25] Watchdog DetokenizerManager initialized.


    [2026-05-29 02:09:26] Init torch distributed begin.


    [2026-05-29 02:09:26] Init torch distributed ends. elapsed=0.33 s, mem usage=0.09 GB


    [2026-05-29 02:09:28] Load weight begin. avail mem=49.07 GB
    [2026-05-29 02:09:28] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /root/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]


    [2026-05-29 02:09:29] Load weight end. elapsed=1.24 s, type=Qwen2ForCausalLM, avail mem=48.09 GB, mem usage=0.98 GB.
    [2026-05-29 02:09:29] Max concurrent requests (per dp worker) from the finalized token capacity: max_num_reqs=128.
    [2026-05-29 02:09:29] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 20480, K size: 0.12 GB, V size: 0.12 GB
    [2026-05-29 02:09:29] Memory pool end. avail mem=47.76 GB
    [2026-05-29 02:09:30] Capture piecewise CUDA graph begin. avail mem=46.68 GB
    [2026-05-29 02:09:30] Capture cuda graph num tokens [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    [2026-05-29 02:09:33] Compiling a graph for dynamic shape takes 0.64 s


    [2026-05-29 02:09:35] Compiling a graph for dynamic shape takes 0.23 s


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.99it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.50it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.06it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.80it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.56 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.56 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6656 avail_mem=45.56 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.56 GB):   3%|▎         | 2/58 [00:00<00:03, 18.62it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=45.56 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.55 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.54 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.54 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.53 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.53 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=45.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.52 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.52 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.52 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 26.48it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=45.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.48 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s]Capturing num tokens (num_tokens=960 avail_mem=45.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s] Capturing num tokens (num_tokens=896 avail_mem=45.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s]Capturing num tokens (num_tokens=832 avail_mem=45.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 32.36it/s]Capturing num tokens (num_tokens=832 avail_mem=45.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=768 avail_mem=45.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=704 avail_mem=45.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=640 avail_mem=45.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]Capturing num tokens (num_tokens=576 avail_mem=45.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 36.08it/s]

    Capturing num tokens (num_tokens=576 avail_mem=45.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=512 avail_mem=45.46 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=480 avail_mem=45.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=448 avail_mem=45.48 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=416 avail_mem=45.47 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.13it/s]Capturing num tokens (num_tokens=416 avail_mem=45.47 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=384 avail_mem=45.47 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=352 avail_mem=45.47 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=320 avail_mem=45.46 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.20it/s]

    Capturing num tokens (num_tokens=288 avail_mem=45.46 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.20it/s]Capturing num tokens (num_tokens=288 avail_mem=45.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=256 avail_mem=45.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=240 avail_mem=45.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=224 avail_mem=45.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=208 avail_mem=45.44 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=208 avail_mem=45.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=192 avail_mem=45.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=176 avail_mem=45.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.98it/s]

    Capturing num tokens (num_tokens=160 avail_mem=45.44 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=144 avail_mem=45.43 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=144 avail_mem=45.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=128 avail_mem=45.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=112 avail_mem=45.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=96 avail_mem=45.43 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.61it/s] Capturing num tokens (num_tokens=80 avail_mem=45.42 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.61it/s]Capturing num tokens (num_tokens=80 avail_mem=45.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.98it/s]Capturing num tokens (num_tokens=64 avail_mem=45.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.98it/s]

    Capturing num tokens (num_tokens=48 avail_mem=45.42 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.98it/s]Capturing num tokens (num_tokens=32 avail_mem=45.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.98it/s]Capturing num tokens (num_tokens=28 avail_mem=45.41 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.98it/s]Capturing num tokens (num_tokens=28 avail_mem=45.41 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=24 avail_mem=45.41 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=20 avail_mem=45.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=16 avail_mem=45.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.14it/s]Capturing num tokens (num_tokens=12 avail_mem=45.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.14it/s]

    Capturing num tokens (num_tokens=12 avail_mem=45.40 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=8 avail_mem=45.39 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.31it/s] Capturing num tokens (num_tokens=4 avail_mem=45.39 GB):  97%|█████████▋| 56/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=4 avail_mem=45.39 GB): 100%|██████████| 58/58 [00:01<00:00, 32.66it/s]
    [2026-05-29 02:09:39] Capture piecewise CUDA graph end. Time elapsed: 9.03 s. mem usage=1.29 GB. avail mem=45.39 GB.


    [2026-05-29 02:09:40] max_total_num_tokens=20480, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=32768, available_gpu_mem=45.39 GB
    [2026-05-29 02:09:40] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False


    [2026-05-29 02:09:40] INFO:     Started server process [914524]
    [2026-05-29 02:09:40] INFO:     Waiting for application startup.
    [2026-05-29 02:09:40] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
    [2026-05-29 02:09:40] INFO:     Application startup complete.
    [2026-05-29 02:09:40] INFO:     Uvicorn running on http://127.0.0.1:34265 (Press CTRL+C to quit)


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)
    [2026-05-29 02:09:41] INFO:     127.0.0.1:52490 - "GET /v1/models HTTP/1.1" 200 OK


    [2026-05-29 02:09:41] INFO:     127.0.0.1:52506 - "GET /model_info HTTP/1.1" 200 OK


    [2026-05-29 02:09:42] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 2.92
    [2026-05-29 02:09:42] INFO:     127.0.0.1:52514 - "POST /generate HTTP/1.1" 200 OK
    [2026-05-29 02:09:42] The server is fired up and ready to roll!



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


    [2026-05-29 02:09:46] INFO:     127.0.0.1:52516 - "POST /tokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Tokenized Output (IDs):<br>[50, 3825, 524, 5707, 11050, 3950, 2022, 36342, 13]</strong>



<strong style='color: #00008B;'>Token Count: 9</strong>



<strong style='color: #00008B;'>Max Model Length: 131072</strong>


    [2026-05-29 02:09:46] INFO:     127.0.0.1:52520 - "POST /detokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Detokenized Output (Text):<br>'SGLang provides efficient tokenization endpoints.'</strong>



<strong style='color: #00008B;'><br>Round Trip Successful: Original and reconstructed text match.</strong>



```python
terminate_process(tokenizer_free_server_process)
```
