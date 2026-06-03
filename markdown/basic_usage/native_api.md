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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.62it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.65it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:02, 12.62it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]

    Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 19.38it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 26.24it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 17.76it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.76it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.64it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  31%|███       | 18/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]

    Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:01, 32.24it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.74it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.74it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.40it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.46it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.46it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.69it/s]

    Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.69it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.69it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.50it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.50it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.50it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.50it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.50it/s]

    Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.96it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 30.96it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 32.75it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 30.41it/s]


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


<strong style='color: #00008B;'>{'text': " To determine the capital of France, I'll follow these steps:\n\n1. Recall and remember the names of the major European countries, since France is strongly associated with many European contacts and lands.\n2. Consider the different types of European capitals, such as:\n- A capital city is always above the railway station\n- The capital of any country can have only one name\n- The capital of a less developed country may only have a single name\n\nNow, a place like Paris is indeed a capital, but Paris is in France, not a less developed country. It's worth noting that France is a very strong global power, and the concept", 'output_ids': [2014, 8253, 279, 6722, 315, 9625, 11, 358, 3278, 1795, 1493, 7354, 1447, 16, 13, 79540, 323, 6099, 279, 5036, 315, 279, 3598, 7513, 5837, 11, 2474, 9625, 374, 16510, 5815, 448, 1657, 7513, 18508, 323, 11835, 624, 17, 13, 21144, 279, 2155, 4494, 315, 7513, 92999, 11, 1741, 438, 510, 12, 362, 6722, 3283, 374, 2677, 3403, 279, 39006, 8056, 198, 12, 576, 6722, 315, 894, 3146, 646, 614, 1172, 825, 829, 198, 12, 576, 6722, 315, 264, 2686, 7881, 3146, 1231, 1172, 614, 264, 3175, 829, 271, 7039, 11, 264, 1992, 1075, 12095, 374, 12824, 264, 6722, 11, 714, 12095, 374, 304, 9625, 11, 537, 264, 2686, 7881, 3146, 13, 1084, 594, 5802, 26305, 429, 9625, 374, 264, 1602, 3746, 3644, 2355, 11, 323, 279, 7286], 'meta_info': {'id': 'f7eca5b6053f4691ae897e1a73e463e7', 'finish_reason': {'type': 'length', 'length': 128}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 128, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 1.11977275647223, 'response_sent_to_client_ts': 1780464239.1434548}}</strong>


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

    [2026-06-03 05:23:59] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.



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


<strong style='color: #00008B;'>{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"detokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"prefill_only_disable_kv_cache":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","model_config_parser":"auto","host":"0.0.0.0","port":33386,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"prefill_delayer_queue_min_ratio":null,"prefill_delayer_max_delay_ms":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":null,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":950121191,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"load_snapshot_publish_interval":15,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_forward_pass_metrics":false,"forward_pass_metrics_worker_id":"","forward_pass_metrics_ipc_name":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"stat_loggers":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","asr_max_buffer_seconds":60,"asr_max_concurrent_sessions":32,"dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":null,"prefill_attention_backend":null,"sampling_backend":"flashinfer","grammar_backend":"xgrammar","radix_cache_backend":null,"mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","dsa_prefill_backend":null,"dsa_decode_backend":null,"dsa_topk_backend":"sgl-kernel","disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_draft_window_size":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"speculative_adaptive":false,"speculative_adaptive_config":null,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","deepep_dispatcher_output_dtype":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"enable_deepep_waterfill":false,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"timeout","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"lmcache_config_file":null,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":false,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"enable_deepseek_v4_fp4_indexer":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"disable_attn_tp_gather":false,"gc_threshold":null,"kv_canary":"none","kv_canary_real_data":"none","kv_canary_sweep_interval":0,"enable_dsa_prefill_context_parallel":false,"dsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"optimistic_prefill_retries":0,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"weight_loader_drop_cache_after_load":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"status":"ready","max_total_num_tokens":20480,"max_req_input_len":20474,"internal_states":[{"model_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_path":"qwen/qwen2.5-0.5b-instruct","tokenizer_mode":"auto","tokenizer_backend":"huggingface","tokenizer_worker_num":1,"detokenizer_worker_num":1,"skip_tokenizer_init":false,"load_format":"auto","model_loader_extra_config":"{}","trust_remote_code":false,"context_length":null,"is_embedding":false,"prefill_only_disable_kv_cache":false,"enable_multimodal":null,"revision":null,"model_impl":"auto","model_config_parser":"auto","host":"0.0.0.0","port":33386,"fastapi_root_path":"","grpc_mode":false,"skip_server_warmup":false,"warmups":null,"nccl_port":null,"checkpoint_engine_wait_weights_before_ready":false,"ssl_keyfile":null,"ssl_certfile":null,"ssl_ca_certs":null,"ssl_keyfile_password":null,"enable_ssl_refresh":false,"enable_http2":false,"dtype":"auto","quantization":null,"quantization_param_path":null,"kv_cache_dtype":"auto","enable_fp32_lm_head":false,"modelopt_quant":null,"modelopt_checkpoint_restore_path":null,"modelopt_checkpoint_save_path":null,"modelopt_export_path":null,"quantize_and_serve":false,"rl_quant_profile":null,"mem_fraction_static":0.836,"max_running_requests":128,"max_queued_requests":null,"max_total_tokens":20480,"chunked_prefill_size":8192,"enable_dynamic_chunking":false,"max_prefill_tokens":16384,"prefill_max_requests":null,"schedule_policy":"fcfs","enable_priority_scheduling":false,"disable_priority_preemption":false,"default_priority_value":null,"abort_on_priority_when_disabled":false,"schedule_low_priority_values_first":false,"priority_scheduling_preemption_threshold":10,"schedule_conservativeness":1.0,"page_size":1,"swa_full_tokens_ratio":0.8,"disable_hybrid_swa_memory":false,"radix_eviction_policy":"lru","enable_prefill_delayer":false,"prefill_delayer_max_delay_passes":30,"prefill_delayer_token_usage_low_watermark":null,"prefill_delayer_forward_passes_buckets":null,"prefill_delayer_wait_seconds_buckets":null,"prefill_delayer_queue_min_ratio":null,"prefill_delayer_max_delay_ms":null,"device":"cuda","tp_size":1,"pp_size":1,"pp_max_micro_batch_size":128,"pp_async_batch_depth":0,"stream_interval":1,"batch_notify_size":16,"stream_response_default_include_usage":false,"incremental_streaming_output":false,"enable_streaming_session":false,"random_seed":950121191,"constrained_json_whitespace_pattern":null,"constrained_json_disable_any_whitespace":false,"watchdog_timeout":300,"soft_watchdog_timeout":300,"dist_timeout":null,"download_dir":null,"model_checksum":null,"base_gpu_id":0,"gpu_id_step":1,"sleep_on_idle":false,"load_snapshot_publish_interval":15,"use_ray":false,"custom_sigquit_handler":null,"log_level":"warning","log_level_http":null,"log_requests":false,"log_requests_level":2,"log_requests_format":"text","log_requests_target":null,"uvicorn_access_log_exclude_prefixes":[],"crash_dump_folder":null,"show_time_cost":false,"enable_metrics":false,"grpc_http_sidecar_port":null,"enable_mfu_metrics":false,"enable_metrics_for_all_schedulers":false,"tokenizer_metrics_custom_labels_header":"x-custom-labels","tokenizer_metrics_allowed_custom_labels":null,"extra_metric_labels":null,"bucket_time_to_first_token":null,"bucket_inter_token_latency":null,"bucket_e2e_request_latency":null,"prompt_tokens_buckets":null,"generation_tokens_buckets":null,"gc_warning_threshold_secs":0.0,"decode_log_interval":40,"enable_request_time_stats_logging":false,"kv_events_config":null,"enable_forward_pass_metrics":false,"forward_pass_metrics_worker_id":"","forward_pass_metrics_ipc_name":null,"enable_trace":false,"otlp_traces_endpoint":"localhost:4317","export_metrics_to_file":false,"export_metrics_to_file_dir":null,"stat_loggers":null,"api_key":null,"admin_api_key":null,"served_model_name":"qwen/qwen2.5-0.5b-instruct","weight_version":"default","chat_template":null,"hf_chat_template_name":null,"completion_template":null,"file_storage_path":"sglang_storage","enable_cache_report":false,"reasoning_parser":null,"strip_thinking_cache":false,"enable_strict_thinking":false,"tool_call_parser":null,"tool_server":null,"sampling_defaults":"model","asr_max_buffer_seconds":60,"asr_max_concurrent_sessions":32,"dp_size":1,"load_balance_method":"round_robin","attn_cp_size":1,"moe_dp_size":1,"dist_init_addr":null,"nnodes":1,"node_rank":0,"json_model_override_args":"{}","preferred_sampling_params":null,"enable_lora":null,"enable_lora_overlap_loading":null,"max_lora_rank":null,"lora_target_modules":null,"lora_paths":null,"max_loaded_loras":null,"max_loras_per_batch":8,"lora_eviction_policy":"lru","lora_backend":"csgmv","max_lora_chunk_size":16,"experts_shared_outer_loras":null,"lora_use_virtual_experts":false,"lora_strict_loading":false,"lora_drain_wait_threshold":0.0,"attention_backend":"fa3","decode_attention_backend":"fa3","prefill_attention_backend":"fa3","sampling_backend":"flashinfer","grammar_backend":"xgrammar","radix_cache_backend":null,"mm_attention_backend":null,"fp8_gemm_runner_backend":"auto","fp4_gemm_runner_backend":"auto","dsa_prefill_backend":null,"dsa_decode_backend":null,"dsa_topk_backend":"sgl-kernel","disable_flashinfer_autotune":false,"mamba_backend":"triton","speculative_algorithm":null,"speculative_draft_model_path":null,"speculative_draft_model_revision":null,"speculative_draft_load_format":null,"speculative_num_steps":null,"speculative_eagle_topk":null,"speculative_num_draft_tokens":null,"speculative_dflash_block_size":null,"speculative_accept_threshold_single":1.0,"speculative_accept_threshold_acc":1.0,"speculative_token_map":null,"speculative_attention_mode":"prefill","speculative_draft_attention_backend":null,"speculative_draft_window_size":null,"speculative_moe_runner_backend":"auto","speculative_moe_a2a_backend":null,"speculative_draft_model_quantization":null,"speculative_skip_dp_mlp_sync":false,"speculative_ngram_min_bfs_breadth":1,"speculative_ngram_max_bfs_breadth":10,"speculative_ngram_match_type":"BFS","speculative_ngram_max_trie_depth":18,"speculative_ngram_capacity":10000000,"speculative_ngram_external_corpus_path":null,"speculative_ngram_external_sam_budget":0,"speculative_ngram_external_corpus_max_tokens":10000000,"enable_multi_layer_eagle":false,"speculative_adaptive":false,"speculative_adaptive_config":null,"ep_size":1,"moe_a2a_backend":"none","moe_runner_backend":"auto","flashinfer_mxfp4_moe_precision":"default","enable_flashinfer_allreduce_fusion":false,"enforce_disable_flashinfer_allreduce_fusion":false,"enable_aiter_allreduce_fusion":false,"deepep_mode":"auto","deepep_dispatcher_output_dtype":"auto","ep_num_redundant_experts":0,"ep_dispatch_algorithm":null,"init_expert_location":"trivial","enable_eplb":false,"eplb_algorithm":"auto","eplb_rebalance_num_iterations":1000,"eplb_rebalance_layers_per_chunk":null,"eplb_min_rebalancing_utilization_threshold":1.0,"expert_distribution_recorder_mode":null,"expert_distribution_recorder_buffer_size":1000,"enable_expert_distribution_metrics":false,"deepep_config":null,"moe_dense_tp_size":null,"elastic_ep_backend":null,"enable_elastic_expert_backup":false,"mooncake_ib_device":null,"enable_deepep_waterfill":false,"elastic_ep_rejoin":false,"max_mamba_cache_size":null,"mamba_ssm_dtype":null,"mamba_full_memory_ratio":0.9,"mamba_scheduler_strategy":"no_buffer","mamba_track_interval":256,"linear_attn_backend":"triton","linear_attn_decode_backend":null,"linear_attn_prefill_backend":null,"enable_hierarchical_cache":false,"hicache_ratio":2.0,"hicache_size":0,"hicache_write_policy":"write_through","hicache_io_backend":"kernel","hicache_mem_layout":"layer_first","hicache_storage_backend":null,"hicache_storage_prefetch_policy":"timeout","hicache_storage_backend_extra_config":null,"enable_hisparse":false,"hisparse_config":null,"enable_lmcache":false,"lmcache_config_file":null,"kt_weight_path":null,"kt_method":"AMXINT4","kt_cpuinfer":null,"kt_threadpool_count":2,"kt_num_gpu_experts":null,"kt_max_deferred_experts_per_token":null,"dllm_algorithm":null,"dllm_algorithm_config":null,"cpu_offload_gb":0,"offload_group_size":-1,"offload_num_in_group":1,"offload_prefetch_step":1,"offload_mode":"cpu","enable_mis":false,"disable_radix_cache":false,"cuda_graph_max_bs":4,"cuda_graph_bs":[1,2,4],"disable_cuda_graph":true,"disable_cuda_graph_padding":false,"enable_breakable_cuda_graph":false,"enable_profile_cuda_graph":false,"enable_cudagraph_gc":false,"debug_cuda_graph":false,"enable_layerwise_nvtx_marker":false,"enable_nccl_nvls":false,"enable_symm_mem":false,"disable_flashinfer_cutlass_moe_fp4_allgather":false,"enable_tokenizer_batch_encode":false,"disable_tokenizer_batch_decode":false,"disable_outlines_disk_cache":false,"disable_custom_all_reduce":false,"enable_mscclpp":false,"enable_torch_symm_mem":false,"pre_warm_nccl":false,"disable_overlap_schedule":false,"enable_mixed_chunk":false,"enable_dp_attention":false,"enable_dp_attention_local_control_broadcast":false,"enable_dp_lm_head":false,"enable_two_batch_overlap":false,"enable_single_batch_overlap":false,"tbo_token_distribution_threshold":0.48,"enable_torch_compile":false,"disable_piecewise_cuda_graph":false,"enforce_piecewise_cuda_graph":false,"enable_torch_compile_debug_mode":false,"torch_compile_max_bs":32,"piecewise_cuda_graph_max_tokens":8192,"piecewise_cuda_graph_tokens":[4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096,4608,5120,5632,6144,6656,7168,7680,8192],"piecewise_cuda_graph_compiler":"eager","torchao_config":"","enable_p2p_check":false,"triton_attention_reduce_in_fp32":false,"triton_attention_num_kv_splits":8,"triton_attention_split_tile_size":null,"num_continuous_decode_steps":1,"delete_ckpt_after_loading":false,"enable_memory_saver":false,"enable_weights_cpu_backup":false,"enable_draft_weights_cpu_backup":false,"allow_auto_truncate":false,"enable_custom_logit_processor":false,"flashinfer_mla_disable_ragged":false,"disable_shared_experts_fusion":false,"enforce_shared_experts_fusion":false,"disable_chunked_prefix_cache":true,"disable_fast_image_processor":false,"keep_mm_feature_on_device":false,"enable_return_hidden_states":false,"enable_return_routed_experts":false,"enable_return_indexer_topk":false,"enable_deepseek_v4_fp4_indexer":false,"scheduler_recv_interval":1,"numa_node":null,"enable_deterministic_inference":false,"rl_on_policy_target":null,"enable_attn_tp_input_scattered":false,"disable_attn_tp_gather":false,"gc_threshold":null,"kv_canary":"none","kv_canary_real_data":"none","kv_canary_sweep_interval":0,"enable_dsa_prefill_context_parallel":false,"dsa_prefill_cp_mode":"round-robin-split","enable_fused_qk_norm_rope":false,"enable_precise_embedding_interpolation":false,"enable_fused_moe_sum_all_reduce":false,"enable_prefill_context_parallel":false,"prefill_cp_mode":"in-seq-split","enable_dynamic_batch_tokenizer":false,"dynamic_batch_tokenizer_batch_size":32,"dynamic_batch_tokenizer_batch_timeout":0.002,"debug_tensor_dump_output_folder":null,"debug_tensor_dump_layers":null,"debug_tensor_dump_input_file":null,"debug_tensor_dump_inject":false,"disaggregation_mode":"null","disaggregation_transfer_backend":"mooncake","disaggregation_bootstrap_port":8998,"disaggregation_ib_device":null,"disaggregation_decode_enable_radix_cache":false,"disaggregation_decode_enable_offload_kvcache":false,"num_reserved_decode_tokens":512,"disaggregation_decode_polling_interval":1,"optimistic_prefill_retries":0,"encoder_only":false,"language_only":false,"encoder_transfer_backend":"zmq_to_scheduler","encoder_urls":[],"enable_adaptive_dispatch_to_encoder":false,"custom_weight_loader":[],"weight_loader_disable_mmap":false,"weight_loader_prefetch_checkpoints":false,"weight_loader_prefetch_num_threads":4,"weight_loader_drop_cache_after_load":false,"remote_instance_weight_loader_seed_instance_ip":null,"remote_instance_weight_loader_seed_instance_service_port":null,"remote_instance_weight_loader_send_weights_group_ports":null,"remote_instance_weight_loader_backend":"nccl","remote_instance_weight_loader_start_seed_via_transfer_engine":false,"engine_info_bootstrap_port":6789,"modelexpress_config":null,"enable_pdmux":false,"pdmux_config_path":null,"sm_group_num":8,"enable_broadcast_mm_inputs_process":false,"enable_prefix_mm_cache":false,"mm_enable_dp_encoder":false,"mm_process_config":{},"limit_mm_data_per_request":null,"enable_mm_global_cache":false,"decrypted_config_file":null,"decrypted_draft_config_file":null,"forward_hooks":null,"enable_quant_communications":false,"msprobe_dump_config":null,"enable_grpc":false,"grpc_port":43386,"_quantization_explicitly_unset":false,"uses_mamba_radix_cache":false,"use_mla_backend":false,"_mx_config_cache":{},"max_speculative_num_draft_tokens":null,"last_gen_throughput":133.34886430455958,"memory_usage":{"weight":0.98,"kvcache":0.23,"token_capacity":20480,"graph":0},"effective_max_running_requests_per_dp":128}],"version":"0.0.0.dev1+gaa510bda4","kv_events":null}</strong>


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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.61it/s]


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

    [2026-06-03 05:24:01] Failed to get weights iterator: qwen/qwen2.5-0.5b-instruct-wrong (repository not found).



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.34s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.01s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.06s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:15,  1.37s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:15,  1.37s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:05<01:15,  1.37s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:17,  2.85it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:17,  2.85it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:17,  2.85it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:17,  2.85it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:17,  2.85it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:08,  5.29it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04, 10.00it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=640):  31%|███       | 18/58 [00:05<00:04, 10.00it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=288):  47%|████▋     | 27/58 [00:05<00:01, 18.55it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:00, 27.74it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:06<00:00, 37.06it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 46.25it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 46.25it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 46.25it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 46.25it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 46.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.65 GB):   2%|▏         | 1/58 [00:00<00:06,  9.27it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.62 GB):   2%|▏         | 1/58 [00:00<00:06,  9.27it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=58.61 GB):   2%|▏         | 1/58 [00:00<00:06,  9.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.61 GB):   5%|▌         | 3/58 [00:00<00:05, 10.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.61 GB):   5%|▌         | 3/58 [00:00<00:05, 10.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.61 GB):   5%|▌         | 3/58 [00:00<00:05, 10.35it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.61 GB):   9%|▊         | 5/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.61 GB):   9%|▊         | 5/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.60 GB):   9%|▊         | 5/58 [00:00<00:04, 11.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.60 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.60 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.60 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.12it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=58.60 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.60 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.60 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.59 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.59 GB):  21%|██        | 12/58 [00:00<00:02, 17.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.58 GB):  21%|██        | 12/58 [00:00<00:02, 17.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.59 GB):  21%|██        | 12/58 [00:00<00:02, 17.81it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=58.58 GB):  21%|██        | 12/58 [00:00<00:02, 17.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.58 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.58 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.56 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.56 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.56 GB):  26%|██▌       | 15/58 [00:01<00:02, 20.75it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.56 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.56 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.55 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.70it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.52 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.70it/s] Capturing num tokens (num_tokens=896 avail_mem=58.52 GB):  33%|███▎      | 19/58 [00:01<00:01, 24.70it/s]Capturing num tokens (num_tokens=896 avail_mem=58.52 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=832 avail_mem=58.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=768 avail_mem=58.52 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=704 avail_mem=58.54 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=640 avail_mem=58.53 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.66it/s]Capturing num tokens (num_tokens=640 avail_mem=58.53 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=576 avail_mem=58.53 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=512 avail_mem=58.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.68it/s]

    Capturing num tokens (num_tokens=480 avail_mem=58.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=448 avail_mem=58.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=448 avail_mem=58.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=416 avail_mem=58.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=384 avail_mem=58.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=352 avail_mem=58.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=320 avail_mem=58.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=288 avail_mem=58.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.94it/s]Capturing num tokens (num_tokens=288 avail_mem=58.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=256 avail_mem=58.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=240 avail_mem=58.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=208 avail_mem=58.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=192 avail_mem=58.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=192 avail_mem=58.49 GB):  71%|███████   | 41/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=176 avail_mem=58.49 GB):  71%|███████   | 41/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=160 avail_mem=58.49 GB):  71%|███████   | 41/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=144 avail_mem=58.48 GB):  71%|███████   | 41/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=128 avail_mem=58.47 GB):  71%|███████   | 41/58 [00:01<00:00, 37.40it/s]Capturing num tokens (num_tokens=128 avail_mem=58.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=112 avail_mem=58.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.85it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.85it/s] Capturing num tokens (num_tokens=80 avail_mem=58.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=64 avail_mem=58.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=64 avail_mem=58.46 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=48 avail_mem=58.46 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=32 avail_mem=58.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=28 avail_mem=58.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=24 avail_mem=58.45 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=24 avail_mem=58.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=20 avail_mem=58.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=12 avail_mem=58.44 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=8 avail_mem=58.43 GB):  91%|█████████▏| 53/58 [00:01<00:00, 38.34it/s] Capturing num tokens (num_tokens=8 avail_mem=58.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=4 avail_mem=58.43 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.56it/s]Capturing num tokens (num_tokens=4 avail_mem=58.43 GB): 100%|██████████| 58/58 [00:02<00:00, 28.71it/s]


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


    [2026-06-03 05:24:57] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    [2026-06-03 05:25:00] No HuggingFace chat template found


    [2026-06-03 05:25:04] Failed to load generation config for BAAI/bge-reranker-v2-m3: BAAI/bge-reranker-v2-m3 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main' for available files.. Proceeding without generation config.


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.09it/s]
    [2026-06-03 05:25:11] Disable piecewise CUDA graph because the model is not a language model


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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.53s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.28it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.28it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.28it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.28it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.28it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.28it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.45it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=58.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.35 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=58.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=58.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=58.35 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.34 GB):   3%|▎         | 2/58 [00:00<00:03, 17.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.34 GB):   9%|▊         | 5/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.34 GB):   9%|▊         | 5/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.33 GB):   9%|▊         | 5/58 [00:00<00:02, 19.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.33 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=58.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.32 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.31 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.30 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=58.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.29 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.27 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s]Capturing num tokens (num_tokens=960 avail_mem=58.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s] Capturing num tokens (num_tokens=896 avail_mem=58.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s]Capturing num tokens (num_tokens=832 avail_mem=58.28 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s]Capturing num tokens (num_tokens=768 avail_mem=58.27 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.37it/s]Capturing num tokens (num_tokens=768 avail_mem=58.27 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=704 avail_mem=58.27 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=640 avail_mem=58.27 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=576 avail_mem=58.27 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]

    Capturing num tokens (num_tokens=512 avail_mem=58.25 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=480 avail_mem=58.27 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.97it/s]Capturing num tokens (num_tokens=480 avail_mem=58.27 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=448 avail_mem=58.27 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=416 avail_mem=58.26 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=384 avail_mem=58.26 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=352 avail_mem=58.26 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=320 avail_mem=58.25 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=320 avail_mem=58.25 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=288 avail_mem=58.25 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=256 avail_mem=58.25 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=240 avail_mem=58.24 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.24 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.22 GB):  60%|██████    | 35/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=208 avail_mem=58.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 20.67it/s]Capturing num tokens (num_tokens=192 avail_mem=58.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 20.67it/s]Capturing num tokens (num_tokens=176 avail_mem=58.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 20.67it/s]Capturing num tokens (num_tokens=160 avail_mem=58.21 GB):  69%|██████▉   | 40/58 [00:01<00:00, 20.67it/s]Capturing num tokens (num_tokens=144 avail_mem=58.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 20.67it/s]Capturing num tokens (num_tokens=144 avail_mem=58.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s]Capturing num tokens (num_tokens=128 avail_mem=57.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s]Capturing num tokens (num_tokens=112 avail_mem=57.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s]

    Capturing num tokens (num_tokens=96 avail_mem=57.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s] Capturing num tokens (num_tokens=80 avail_mem=57.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s]Capturing num tokens (num_tokens=64 avail_mem=57.62 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.61it/s]Capturing num tokens (num_tokens=64 avail_mem=57.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=48 avail_mem=57.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=32 avail_mem=57.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=28 avail_mem=57.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=24 avail_mem=57.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=20 avail_mem=57.61 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=20 avail_mem=57.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=16 avail_mem=57.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.18it/s]

    Capturing num tokens (num_tokens=12 avail_mem=57.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=8 avail_mem=57.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.18it/s] Capturing num tokens (num_tokens=4 avail_mem=57.60 GB):  93%|█████████▎| 54/58 [00:01<00:00, 30.18it/s]Capturing num tokens (num_tokens=4 avail_mem=57.60 GB): 100%|██████████| 58/58 [00:01<00:00, 29.22it/s]


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


    [2026-06-03 05:25:59] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    [2026-06-03 05:26:11] Failed to load generation config for Skywork/Skywork-Reward-Llama-3.1-8B-v0.2: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/tree/main' for available files.. Proceeding without generation config.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.11it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.02it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.02it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.53it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:18,  5.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:18,  5.58s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:16,  2.44s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:16,  2.44s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:18,  1.42s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:18,  1.42s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:50,  1.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:50,  1.07it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.02it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.02it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.63it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.63it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.31it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.31it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.09it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.09it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:09,  4.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:09,  4.97it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:07<00:09,  4.97it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:06,  6.59it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:06,  6.59it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:07<00:06,  6.59it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  8.09it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  8.09it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  8.09it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.87it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.87it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.87it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:03, 11.83it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:03, 11.83it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:03, 11.83it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:07<00:03, 11.83it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:07<00:02, 15.60it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:07<00:01, 22.86it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:07<00:00, 31.21it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:07<00:00, 31.21it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:08<00:00, 31.21it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:08<00:00, 40.61it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]

    Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:08<00:00, 48.05it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:08<00:00, 47.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=60.73 GB):   2%|▏         | 1/58 [00:00<00:19,  2.95it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.70 GB):   2%|▏         | 1/58 [00:00<00:19,  2.95it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=60.70 GB):   3%|▎         | 2/58 [00:00<00:17,  3.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.70 GB):   3%|▎         | 2/58 [00:00<00:17,  3.12it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=60.70 GB):   5%|▌         | 3/58 [00:00<00:16,  3.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.70 GB):   5%|▌         | 3/58 [00:00<00:16,  3.36it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=60.70 GB):   7%|▋         | 4/58 [00:01<00:15,  3.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.70 GB):   7%|▋         | 4/58 [00:01<00:15,  3.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.70 GB):   9%|▊         | 5/58 [00:01<00:13,  3.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.70 GB):   9%|▊         | 5/58 [00:01<00:13,  3.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.70 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.70 GB):  10%|█         | 6/58 [00:01<00:12,  4.15it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.70 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.70 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.70 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.70 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.93it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=60.70 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.69 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.69 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.69 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=60.69 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.69 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.69 GB):  21%|██        | 12/58 [00:02<00:07,  6.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.69 GB):  21%|██        | 12/58 [00:02<00:07,  6.26it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=60.69 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.69 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.69 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.68 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.83it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=60.68 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.68 GB):  26%|██▌       | 15/58 [00:02<00:06,  7.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.68 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.68 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.68 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.58it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=60.68 GB):  31%|███       | 18/58 [00:03<00:04,  9.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.67 GB):  31%|███       | 18/58 [00:03<00:04,  9.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.67 GB):  31%|███       | 18/58 [00:03<00:04,  9.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.67 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.67 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.21it/s]Capturing num tokens (num_tokens=960 avail_mem=60.66 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.21it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=60.66 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.02it/s]Capturing num tokens (num_tokens=896 avail_mem=60.65 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.02it/s]Capturing num tokens (num_tokens=832 avail_mem=60.65 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.02it/s]Capturing num tokens (num_tokens=768 avail_mem=60.65 GB):  38%|███▊      | 22/58 [00:03<00:02, 13.02it/s]Capturing num tokens (num_tokens=768 avail_mem=60.65 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.25it/s]Capturing num tokens (num_tokens=704 avail_mem=60.64 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.25it/s]Capturing num tokens (num_tokens=640 avail_mem=60.64 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.25it/s]Capturing num tokens (num_tokens=576 avail_mem=60.63 GB):  43%|████▎     | 25/58 [00:03<00:02, 16.25it/s]

    Capturing num tokens (num_tokens=576 avail_mem=60.63 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.51it/s]Capturing num tokens (num_tokens=512 avail_mem=60.63 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.51it/s]Capturing num tokens (num_tokens=480 avail_mem=60.62 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.51it/s]Capturing num tokens (num_tokens=448 avail_mem=60.62 GB):  48%|████▊     | 28/58 [00:03<00:01, 19.51it/s]Capturing num tokens (num_tokens=448 avail_mem=60.62 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.16it/s]Capturing num tokens (num_tokens=416 avail_mem=60.62 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.16it/s]Capturing num tokens (num_tokens=384 avail_mem=60.62 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.16it/s]Capturing num tokens (num_tokens=352 avail_mem=60.61 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.16it/s]Capturing num tokens (num_tokens=320 avail_mem=60.61 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.16it/s]

    Capturing num tokens (num_tokens=320 avail_mem=60.61 GB):  60%|██████    | 35/58 [00:03<00:00, 25.08it/s]Capturing num tokens (num_tokens=288 avail_mem=60.60 GB):  60%|██████    | 35/58 [00:03<00:00, 25.08it/s]Capturing num tokens (num_tokens=256 avail_mem=60.60 GB):  60%|██████    | 35/58 [00:03<00:00, 25.08it/s]Capturing num tokens (num_tokens=240 avail_mem=60.60 GB):  60%|██████    | 35/58 [00:03<00:00, 25.08it/s]Capturing num tokens (num_tokens=224 avail_mem=60.59 GB):  60%|██████    | 35/58 [00:03<00:00, 25.08it/s]Capturing num tokens (num_tokens=224 avail_mem=60.59 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=208 avail_mem=60.59 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=192 avail_mem=60.58 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.82it/s]Capturing num tokens (num_tokens=176 avail_mem=60.58 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.82it/s]

    Capturing num tokens (num_tokens=176 avail_mem=60.58 GB):  72%|███████▏  | 42/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=160 avail_mem=60.57 GB):  72%|███████▏  | 42/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=144 avail_mem=60.57 GB):  72%|███████▏  | 42/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=128 avail_mem=60.56 GB):  72%|███████▏  | 42/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=112 avail_mem=60.57 GB):  72%|███████▏  | 42/58 [00:04<00:00, 26.61it/s]Capturing num tokens (num_tokens=112 avail_mem=60.57 GB):  79%|███████▉  | 46/58 [00:04<00:00, 28.19it/s]Capturing num tokens (num_tokens=96 avail_mem=60.56 GB):  79%|███████▉  | 46/58 [00:04<00:00, 28.19it/s] Capturing num tokens (num_tokens=80 avail_mem=60.56 GB):  79%|███████▉  | 46/58 [00:04<00:00, 28.19it/s]Capturing num tokens (num_tokens=64 avail_mem=60.56 GB):  79%|███████▉  | 46/58 [00:04<00:00, 28.19it/s]

    Capturing num tokens (num_tokens=48 avail_mem=60.55 GB):  79%|███████▉  | 46/58 [00:04<00:00, 28.19it/s]Capturing num tokens (num_tokens=48 avail_mem=60.55 GB):  86%|████████▌ | 50/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=32 avail_mem=60.55 GB):  86%|████████▌ | 50/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=28 avail_mem=60.54 GB):  86%|████████▌ | 50/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=24 avail_mem=60.54 GB):  86%|████████▌ | 50/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=20 avail_mem=60.53 GB):  86%|████████▌ | 50/58 [00:04<00:00, 30.52it/s]Capturing num tokens (num_tokens=20 avail_mem=60.53 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.23it/s]Capturing num tokens (num_tokens=16 avail_mem=60.53 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.23it/s]Capturing num tokens (num_tokens=12 avail_mem=60.53 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.23it/s]Capturing num tokens (num_tokens=8 avail_mem=60.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.23it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=60.52 GB):  93%|█████████▎| 54/58 [00:04<00:00, 32.23it/s]Capturing num tokens (num_tokens=4 avail_mem=60.52 GB): 100%|██████████| 58/58 [00:04<00:00, 33.34it/s]Capturing num tokens (num_tokens=4 avail_mem=60.52 GB): 100%|██████████| 58/58 [00:04<00:00, 12.66it/s]


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

    Multi-thread loading shards:  12% Completed | 1/8 [00:00<00:05,  1.22it/s]

    Multi-thread loading shards:  25% Completed | 2/8 [00:01<00:04,  1.27it/s]

    Multi-thread loading shards:  38% Completed | 3/8 [00:02<00:03,  1.27it/s]

    Multi-thread loading shards:  50% Completed | 4/8 [00:03<00:03,  1.27it/s]

    Multi-thread loading shards:  62% Completed | 5/8 [00:03<00:02,  1.28it/s]

    Multi-thread loading shards:  75% Completed | 6/8 [00:04<00:01,  1.28it/s]

    Multi-thread loading shards:  88% Completed | 7/8 [00:05<00:00,  1.28it/s]

    Multi-thread loading shards: 100% Completed | 8/8 [00:05<00:00,  1.66it/s]Multi-thread loading shards: 100% Completed | 8/8 [00:05<00:00,  1.40it/s]


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

    [2026-06-03 05:27:34] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
    [2026-06-03 05:27:34] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! Config file not found at /actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=60,N=1408,device_name=NVIDIA_H100_80GB_HBM3_down.json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton



<strong style='color: #00008B;'><Response [200]></strong>



<strong style='color: #00008B;'>{'text': ' The capital of France is Paris. \n\nIf you would like to know more, here\'s some additional information: \n\n- Paris is located in the northern part of France and is the country\'s largest city with a population of over 2 million people.\n- It is known as the "City of Light" and the "City of Love" due to its romantic atmosphere, historic landmarks, and cultural attractions.\n- Paris is famous for its many iconic landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.\n- The city is also known for its fashion, cuisine, and wine, and is home', 'output_ids': [576, 6722, 315, 9625, 374, 12095, 13, 4710, 2679, 498, 1035, 1075, 311, 1414, 803, 11, 1588, 594, 1045, 5107, 1995, 25, 4710, 12, 12095, 374, 7407, 304, 279, 18172, 949, 315, 9625, 323, 374, 279, 3146, 594, 7772, 3283, 448, 264, 7042, 315, 916, 220, 17, 3526, 1251, 624, 12, 1084, 374, 3881, 438, 279, 330, 12730, 315, 8658, 1, 323, 279, 330, 12730, 315, 10689, 1, 4152, 311, 1181, 23467, 16566, 11, 18035, 59924, 11, 323, 12752, 38491, 624, 12, 12095, 374, 11245, 369, 1181, 1657, 26277, 59924, 11, 1741, 438, 279, 468, 3092, 301, 21938, 11, 43464, 9420, 373, 56729, 11, 323, 279, 9729, 48506, 16328, 624, 12, 576, 3283, 374, 1083, 3881, 369, 1181, 11153, 11, 35005, 11, 323, 13078, 11, 323, 374, 2114], 'meta_info': {'id': '171d7f97477647b2aed88ab99b0d01a8', 'finish_reason': {'type': 'length', 'length': 128}, 'prompt_tokens': 7, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 128, 'cached_tokens': 0, 'cached_tokens_details': None, 'dp_rank': None, 'e2e_latency': 3.9324970450252295, 'response_sent_to_client_ts': 1780464459.0460222}}</strong>



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


    [2026-06-03 05:27:45] Attention backend not specified. Use fa3 backend by default.
    [2026-06-03 05:27:45] Set soft_watchdog_timeout since in CI


    [2026-06-03 05:27:46] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_backend='huggingface', tokenizer_worker_num=1, detokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, prefill_only_disable_kv_cache=False, enable_multimodal=None, revision=None, model_impl='auto', model_config_parser='auto', host='127.0.0.1', port=31048, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, enable_http2=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.836, max_running_requests=128, max_queued_requests=None, max_total_tokens=20480, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, prefill_delayer_queue_min_ratio=None, prefill_delayer_max_delay_ms=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, batch_notify_size=16, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=915381510, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=300, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, load_snapshot_publish_interval=15, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, grpc_http_sidecar_port=None, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_forward_pass_metrics=False, forward_pass_metrics_worker_id='', forward_pass_metrics_ipc_name=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, stat_loggers=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, strip_thinking_cache=False, enable_strict_thinking=False, tool_call_parser=None, tool_server=None, sampling_defaults='model', asr_max_buffer_seconds=60, asr_max_concurrent_sessions=32, dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, lora_use_virtual_experts=False, lora_strict_loading=False, lora_drain_wait_threshold=0.0, attention_backend='fa3', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', radix_cache_backend=None, mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', dsa_prefill_backend=None, dsa_decode_backend=None, dsa_topk_backend='sgl-kernel', disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_dflash_block_size=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_draft_window_size=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_skip_dp_mlp_sync=False, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, speculative_ngram_external_corpus_path=None, speculative_ngram_external_sam_budget=0, speculative_ngram_external_corpus_max_tokens=10000000, enable_multi_layer_eagle=False, speculative_adaptive=False, speculative_adaptive_config=None, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', deepep_dispatcher_output_dtype='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, enable_deepep_waterfill=False, elastic_ep_rejoin=False, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='timeout', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, lmcache_config_file=None, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', enable_mis=False, disable_radix_cache=False, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 4], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_breakable_cuda_graph=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, debug_cuda_graph=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_attention_local_control_broadcast=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=False, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, enforce_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, enable_return_indexer_topk=False, enable_deepseek_v4_fp4_indexer=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, disable_attn_tp_gather=False, gc_threshold=None, kv_canary='none', kv_canary_real_data='none', kv_canary_sweep_interval=0, enable_dsa_prefill_context_parallel=False, dsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_radix_cache=False, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, optimistic_prefill_retries=0, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False, weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None, enable_quant_communications=False, msprobe_dump_config=None)


    [2026-06-03 05:27:47] Watchdog TokenizerManager initialized.
    [2026-06-03 05:27:47] Using default HuggingFace chat template with detected content format: string


    [2026-06-03 05:27:52] Watchdog DetokenizerManager initialized.


    [2026-06-03 05:27:53] Init torch distributed begin.


    [2026-06-03 05:27:54] Init torch distributed ends. elapsed=0.24 s, mem usage=0.09 GB


    [2026-06-03 05:27:56] Load weight begin. avail mem=78.50 GB
    [2026-06-03 05:27:57] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /root/.cache/huggingface/hub/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.
    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.36it/s]
    [2026-06-03 05:27:57] Load weight end. elapsed=0.43 s, type=Qwen2ForCausalLM, avail mem=77.53 GB, mem usage=0.98 GB.
    [2026-06-03 05:27:57] Max concurrent requests (per dp worker) from the finalized token capacity: max_num_reqs=128.
    [2026-06-03 05:27:57] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 20480, K size: 0.12 GB, V size: 0.12 GB
    [2026-06-03 05:27:57] Memory pool end. avail mem=77.20 GB
    [2026-06-03 05:27:57] Capture piecewise CUDA graph begin. avail mem=77.10 GB
    [2026-06-03 05:27:57] Capture cuda graph num tokens [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    [2026-06-03 05:28:00] Compiling a graph for dynamic shape takes 0.20 s


    [2026-06-03 05:28:02] Compiling a graph for dynamic shape takes 0.21 s


    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.38it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 14.90it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 22.91it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 22.91it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 22.91it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 22.91it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 22.91it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.28it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.42it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 16.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.52it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 42.10it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.95it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.66it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.56it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.81it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 48.11it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.93it/s]
    [2026-06-03 05:28:05] Capture piecewise CUDA graph end. Time elapsed: 7.79 s. mem usage=0.49 GB. avail mem=76.61 GB.


    [2026-06-03 05:28:06] max_total_num_tokens=20480, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=128, context_len=32768, available_gpu_mem=76.61 GB
    [2026-06-03 05:28:06] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False


    [2026-06-03 05:28:07] INFO:     Started server process [4070092]
    [2026-06-03 05:28:07] INFO:     Waiting for application startup.
    [2026-06-03 05:28:07] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
    [2026-06-03 05:28:07] INFO:     Application startup complete.
    [2026-06-03 05:28:07] INFO:     Uvicorn running on http://127.0.0.1:31048 (Press CTRL+C to quit)
    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)
    [2026-06-03 05:28:07] INFO:     127.0.0.1:46368 - "GET /v1/models HTTP/1.1" 200 OK


    [2026-06-03 05:28:08] INFO:     127.0.0.1:46376 - "GET /model_info HTTP/1.1" 200 OK


    [2026-06-03 05:28:08] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 3.03
    [2026-06-03 05:28:08] INFO:     127.0.0.1:46388 - "POST /generate HTTP/1.1" 200 OK
    [2026-06-03 05:28:08] The server is fired up and ready to roll!



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


    [2026-06-03 05:28:12] INFO:     127.0.0.1:56180 - "POST /tokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Tokenized Output (IDs):<br>[50, 3825, 524, 5707, 11050, 3950, 2022, 36342, 13]</strong>



<strong style='color: #00008B;'>Token Count: 9</strong>



<strong style='color: #00008B;'>Max Model Length: 131072</strong>


    [2026-06-03 05:28:12] INFO:     127.0.0.1:56182 - "POST /detokenize HTTP/1.1" 200 OK



<strong style='color: #00008B;'><br>Detokenized Output (Text):<br>'SGLang provides efficient tokenization endpoints.'</strong>



<strong style='color: #00008B;'><br>Round Trip Successful: Original and reconstructed text match.</strong>



```python
terminate_process(tokenizer_free_server_process)
```
