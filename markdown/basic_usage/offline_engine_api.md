# Offline Engine API

SGLang provides a direct inference engine without the need for an HTTP server, especially for use cases where additional HTTP server adds unnecessary complexity or overhead. Here are two general use cases:

- Offline Batch Inference
- Custom Server on Top of the Engine

This document focuses on the offline batch inference, demonstrating four different inference modes:

- Non-streaming synchronous generation
- Streaming synchronous generation
- Non-streaming asynchronous generation
- Streaming asynchronous generation

Additionally, you can easily build a custom server on top of the SGLang offline engine. A detailed example working in a python script can be found in [custom_server](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/custom_server.py).



## Nest Asyncio
Note that if you want to use **Offline Engine** in ipython or some other nested loop code, you need to add the following code:
```python
import nest_asyncio

nest_asyncio.apply()

```

## Advanced Usage

The engine supports [vlm inference](https://github.com/sgl-project/sglang/blob/main/examples/runtime/engine/offline_batch_inference_vlm.py) as well as [extracting hidden states](https://github.com/sgl-project/sglang/blob/main/examples/runtime/hidden_states). 

Please see [the examples](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine) for further use cases.

## Offline Batch Inference

SGLang offline engine supports batch inference with efficient scheduling.


```python
# launch the offline engine
import asyncio

import sglang as sgl
import sglang.test.doc_patch  # noqa: F401
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
```

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.17s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.73it/s]

    Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:18,  2.73it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.74it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.74it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.74it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.74it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.18it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.18it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  7.18it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  7.18it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  7.18it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.02it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.02it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 11.02it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.83it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:01, 20.13it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 24.93it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:00, 29.46it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 35.12it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 38.46it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 42.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.85 GB):   2%|▏         | 1/58 [00:00<00:07,  7.45it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   2%|▏         | 1/58 [00:00<00:07,  7.45it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.82 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.82 GB):   3%|▎         | 2/58 [00:00<00:07,  7.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.82 GB):   5%|▌         | 3/58 [00:00<00:07,  7.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.82 GB):   5%|▌         | 3/58 [00:00<00:07,  7.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.82 GB):   7%|▋         | 4/58 [00:00<00:06,  7.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.82 GB):   7%|▋         | 4/58 [00:00<00:06,  7.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.82 GB):   9%|▊         | 5/58 [00:00<00:06,  7.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):   9%|▊         | 5/58 [00:00<00:06,  7.89it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.81 GB):  10%|█         | 6/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.80 GB):  10%|█         | 6/58 [00:00<00:06,  8.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.80 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.80 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.80 GB):  12%|█▏        | 7/58 [00:00<00:05,  8.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.80 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.79 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.79 GB):  16%|█▌        | 9/58 [00:01<00:05,  8.55it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=55.79 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  19%|█▉        | 11/58 [00:01<00:05,  8.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.39 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  22%|██▏       | 13/58 [00:01<00:04, 10.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.38 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.37 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:01<00:03, 13.14it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.75it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.75it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:01<00:02, 14.75it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.54it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.54it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 23.23it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 23.23it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:02<00:01, 23.23it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 23.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:02<00:01, 23.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  53%|█████▎    | 31/58 [00:02<00:01, 26.28it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:02<00:01, 26.28it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:02<00:01, 26.28it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:02<00:01, 26.28it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:02<00:01, 26.28it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:02<00:00, 28.06it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:02<00:00, 33.30it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:02<00:00, 36.83it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:02<00:00, 39.65it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 41.65it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 41.65it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 41.65it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 41.65it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:02<00:00, 21.18it/s]


### Non-streaming Synchronous Generation


```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("===============================")
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```

    ===============================
    Prompt: Hello, my name is
    Generated text:  Vinnie and I am an ASP.NET developer, an enthusiast of data science, and a student at the University of New Mexico. I have a passion for learning and discovering new things every day. I am always trying to improve my knowledge and skills in order to become a more effective developer. What are some of your favorite programming languages and frameworks that you use for development? As an ASP.NET developer, I use several popular frameworks such as .NET Core and .NET 5/6, and the most used tools in my development are Visual Studio, Git, and Docker. I also use various SQL Server objects to work with SQL databases.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He or she is in charge of the country. The president is the leader of the country. Most of the time, the president works with the other members of the government to help run the country. He or she is usually the head of the military. He or she also often has to deal with the issue of national security.
    
    According to the last paragraph, the president has to deal with the issue of national security.
    
    A). Yes; B). No;
    
    B). No;
    
    The president is the head of the military. He or she also often has to deal with the issue of national security.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The library of the university of Paris is called the Louvre.
    A. 正确
    B. 错误
    答案:
    A
    
    一般来说,任何一个连续的运动，如果其速度为常数，则该连续的运动轨迹将是一个
    A. 直线
    B. 圆
    C. 射线
    D. 双曲线
    答案:
    A
    
    个人利益与公共利益是一致的。要充分认识到个人利益与公共利益的矛盾冲突，坚持社会______、价值取向、理想信念，把个人利益与公共利益紧密结合起来。
    A. 公平
    B
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but one thing is certain: it’s going to be important. AI is going to play a key role in every industry in the future. It’s going to change the way we do things and shape the way we live our lives. AI is going to help us to accomplish more in less time, and it is going to help us to make better decisions. With its ability to learn from data, AI can be used to improve our productivity and efficiency, and it can be used to make more accurate predictions about future events.
    One of the key challenges of AI is how to develop ethical guidelines for its use. This means that we


### Streaming Synchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {
    "temperature": 0.2,
    "top_p": 0.9,
}

print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

for prompt in prompts:
    print(f"Prompt: {prompt}")
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    print("Generated text:", merged_output)
    print()
```

    
    === Testing synchronous streaming generation with overlap removal ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and culture of the world. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, making it a must-visit destination for anyone interested in French culture and history. Paris is also known for its fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration of AI into everyday life: AI is already being integrated into our daily lives, from voice assistants like Siri and Alexa to self-driving cars. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This includes issues such as bias, privacy, and transparency.
    
    3. Increased focus on AI safety: As
    


### Non-streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous batch generation ===")


async def main():
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")


asyncio.run(main())
```

    
    === Testing asynchronous batch generation ===


    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text:  __________ and I'm a/an (insert profession or occupation). I'm passionate about (insert hobby, interest, or passion). I'm always looking for (insert activity or pursuit) and I'm always willing to learn new things and improve my skills. I enjoy (insert activity or pursuit), and I'm always willing to help others. I value (insert value or quality) and I'm always looking for new ways to grow and develop myself. I'm always willing to embrace challenges and take risks, even when it means facing new situations or confronting difficult people. I'm confident in myself and my abilities, and I believe in my potential
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    I apologize, but I can't generate that response as it appears to be a self-generated fact. If you have a specific question about Paris, I'd be happy to provide factual information about it. Otherwise, I can't fulfill that request. Is there anything else I can help you with? Let me know if you need any other assistance. 
    
    Remember, Paris is a very large city with a rich history and many attractions. Here are some top attractions:
    
    1. **Eiffel Tower**: One of the most recognizable landmarks in the world, this iconic structure has stood for 108 years.
    
    2. **
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and challenges. Here are some possible future trends in AI:
    
    1. Deep Learning: With the development of deep learning, AI can learn from large amounts of data much faster and better than traditional machine learning. This could lead to more efficient and accurate machine learning models in many applications.
    
    2. Autonomous Vehicles: With AI, autonomous vehicles will become more advanced and self-driving cars will become more common. This could reduce accidents and improve traffic flow.
    
    3. Personalized Medicine: AI can help doctors predict patient outcomes based on their medical history and genetic information. This could lead to more personalized treatment plans for patients.
    
    4. Robotics


### Streaming Asynchronous Generation


```python
prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


asyncio.run(main())
```

    
    === Testing asynchronous streaming generation (no repeats) ===
    
    Prompt: Write a short, neutral self-introduction for a fictional character. Hello, my name is
    Generated text: 

     [

    Your

     Name

    ],

     and

     I

     am

     a

     [

    insert

     profession

     or

     specialty

    ]

     with

     a

     strong

     passion

     for

     [

    insert

     something

     specific

     to

     your

     field

     or

     hobby

    ].

     I

     have

     been

     studying

     for

     [

    insert

     number

     of

     years

    ]

     and

     have

     taken

     advanced

     courses

     in

     [

    insert

     the

     area

     you

     have

     exc

    elled

     in

     the

     past

    ].

     I

     am

     always

     eager

     to

     learn

     and

     always

     aim

     to

     [

    insert

     something

     that

     describes

     your

     work

     ethic

     or

     personal

     values

    ].

     I

     have

     a

     deep

     love

     for

     [

    insert

     an

     area

     of

     interest

     or

     interest

     in

     life

    ],

     and

     I

     believe

     that

     being

     well

    -rounded

     and

     intelligent

     are

     qualities

     that

     are

     essential

     in

     my

     field

     of

     study

    .

     I

     am

     [

    insert

     a

     personality

     trait

     or

     set

     of

     traits

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     vibrant

     culture

    ,

     stunning

     architecture

    ,

     and

     historic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

    's

     a

     city

     that

     is

     renowned

     for

     its

     annual

     Carnival

     celebrations

     and

     its

     status

     as

     a

     global

     cultural

     hub

    .

     Paris

     is

     also

     home

     to

     many

     notable

     French

     artists

    ,

     writers

    ,

     and

     composers

    ,

     including

     Vincent

     van

     G

    ogh

    ,

     Pablo

     Picasso

    ,

     and

     Gust

    av

     K

    lim

    t

    .

     It

     is

     also

     known

     for

     its

     rich

     gastr

    onomy

    ,

     with

     its

     famous

     French

     cuisine

     and

     numerous

     Mich

    elin

    -star

    red

     restaurants

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     is

     a

     cultural

     melting

     pot

     of

     many

     different

     traditions

     and

     influences

    .

     The

     streets

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     involves

     many

     exciting

     developments

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     we

     can

     expect

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     AI

     efficiency

    :

     With

     the

     development

     of

     better

     algorithms

     and

     hardware

    ,

     AI

     will

     become

     even

     more

     efficient

     at

     processing

     and

     analyzing

     data

    .

     This

     will

     lead

     to

     faster

     and

     more

     accurate

     decision

    -making

    ,

     which

     can

     lead

     to

     improved

     productivity

     and

     efficiency

    .
    


    2

    .

     AI

     ethics

     and

     privacy

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     it

     will

     be

     important

     to

     address

     the

     ethical

     and

     privacy

     concerns

     associated

     with

     AI

    .

     This

     includes

     ensuring

     that

     AI

     systems

     are

     not

     used

     to

     perpet

    uate

     discrimination

     or

     bias

    ,

     and

     that

     the

     data

     used

     to

     train

     them

     is

    



```python
llm.shutdown()
```
