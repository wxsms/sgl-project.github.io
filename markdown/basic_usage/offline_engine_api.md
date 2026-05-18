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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]


    2026-05-18 03:43:36,335 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 03:43:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]

    Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:15,  3.19it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:06,  6.63it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:03, 12.41it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]

    Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:01, 18.92it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:00, 26.02it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 33.24it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 41.26it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 49.45it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 49.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.70 GB):   2%|▏         | 1/58 [00:00<00:06,  8.60it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.67 GB):   2%|▏         | 1/58 [00:00<00:06,  8.60it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:06,  9.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:06,  9.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.66 GB):   3%|▎         | 2/58 [00:00<00:06,  9.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.66 GB):   7%|▋         | 4/58 [00:00<00:05, 10.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.66 GB):   7%|▋         | 4/58 [00:00<00:05, 10.10it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):   7%|▋         | 4/58 [00:00<00:05, 10.10it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):  10%|█         | 6/58 [00:00<00:04, 11.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.65 GB):  10%|█         | 6/58 [00:00<00:04, 11.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  10%|█         | 6/58 [00:00<00:04, 11.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.76it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.64 GB):  14%|█▍        | 8/58 [00:00<00:03, 12.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.49it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.77it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=55.62 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.62 GB):  22%|██▏       | 13/58 [00:01<00:02, 16.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.61 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.61 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.61 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.64it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=55.59 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.64it/s]Capturing num tokens (num_tokens=960 avail_mem=55.60 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.64it/s] Capturing num tokens (num_tokens=896 avail_mem=55.60 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.64it/s]Capturing num tokens (num_tokens=896 avail_mem=55.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=832 avail_mem=55.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=768 avail_mem=55.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=704 avail_mem=55.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.96it/s]Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.96it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.58it/s]Capturing num tokens (num_tokens=576 avail_mem=55.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.58it/s]Capturing num tokens (num_tokens=512 avail_mem=55.57 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.58it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.58it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.58it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.53it/s]Capturing num tokens (num_tokens=416 avail_mem=55.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.53it/s]Capturing num tokens (num_tokens=384 avail_mem=55.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.53it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.53it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.53it/s]Capturing num tokens (num_tokens=320 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=288 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=240 avail_mem=55.56 GB):  60%|██████    | 35/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=224 avail_mem=55.56 GB):  60%|██████    | 35/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=224 avail_mem=55.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=176 avail_mem=55.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.04it/s]

    Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.04it/s]Capturing num tokens (num_tokens=160 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=128 avail_mem=55.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=112 avail_mem=55.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.12it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.12it/s] Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  81%|████████  | 47/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=80 avail_mem=55.53 GB):  81%|████████  | 47/58 [00:01<00:00, 33.62it/s]Capturing num tokens (num_tokens=64 avail_mem=55.53 GB):  81%|████████  | 47/58 [00:02<00:00, 33.62it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  81%|████████  | 47/58 [00:02<00:00, 33.62it/s]Capturing num tokens (num_tokens=32 avail_mem=55.52 GB):  81%|████████  | 47/58 [00:02<00:00, 33.62it/s]Capturing num tokens (num_tokens=32 avail_mem=55.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.99it/s]Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.99it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.99it/s]Capturing num tokens (num_tokens=20 avail_mem=55.51 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.99it/s]Capturing num tokens (num_tokens=16 avail_mem=55.51 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.99it/s]Capturing num tokens (num_tokens=16 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.74it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.74it/s]Capturing num tokens (num_tokens=8 avail_mem=55.50 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.74it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=55.50 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.74it/s]Capturing num tokens (num_tokens=4 avail_mem=55.50 GB): 100%|██████████| 58/58 [00:02<00:00, 25.35it/s]


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
    Generated text:  Xiaoming, I am 13 years old and I live in Jiangdu District, Hangzhou. How do I go about buying a house in Hangzhou? 
    A. Go to the real estate agency directly.
    B. Apply for a residence permit at the administrative authority of the housing administration at your place of residence.
    C. Register for a property purchase contract at your place of residence.
    D. Complete the mortgage registration at your place of residence.
    Answer:
    D
    
    Which of the following belongs to the final level of people's livelihood?
    A. Telecommunications
    B. Education
    C. Healthcare
    D. Income
    Answer:
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy preparing for the upcoming election. He has to decide whether to vote for a candidate that received 42 percent of the popular vote or a candidate that received 35 percent of the popular vote. If the total popular vote is 250 million votes, and the candidate with the higher vote count wins the election. What is the total number of votes the candidate with the higher vote count received? To determine the candidate with the higher vote count, we need to calculate the number of votes each candidate received and compare the two values.
    
    First, we calculate the number of votes the candidate who received 42 percent of the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that "Paris has no capital."?
    OPTIONS: + yes; + it is not possible to tell; + no;
    
    No;
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and fast-paced. While we are only just beginning to understand the complex nature of AI, it has already had a profound impact on society. Artificial intelligence has changed the way we work, live, and interact with the world around us. From chatbots to self-driving cars, AI is everywhere and is not going away anytime soon.
    
    One of the most exciting areas of AI research is natural language processing. In this article, we will explore the latest developments and challenges in this field, as well as the potential impact that this technology will have on our lives.
    
    ### Understanding Natural Language Processing
    
    Natural language processing (NLP) is the study


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art scene. It is a popular tourist destination and a major economic center in Europe. Paris is the capital of France and is the largest city in the country. It is also the seat of the French government and the country's cultural and political center. Paris is a major cultural and economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This could lead to increased efficiency, reduced costs, and improved quality of life for many people.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about its impact on society. This includes issues such as bias, privacy, and the potential for AI to be
    


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
    Generated text:  ________, and I'm a professional writer. I specialize in crafting compelling narratives that engage readers and spark their imagination. My work spans a wide range of genres, including romance, fantasy, and thrillers. I believe that storytelling is the ultimate expression of human creativity and have had the opportunity to work with some of the most talented writers and editors of our time. Whether you're looking for a captivating introduction to your own writing or a creative spark to inspire your own story, I'd be happy to assist you in any way I can. How can I help you today? Let me know! (This should be a brief, professional self-int
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris, the city of love and light, is the largest and most populous city in France, with a population of over 2.1 million people. It serves as the capital of France and the seat of the French government, as well as a major cultural, scientific, and artistic center. The city is famous for its architecture, including the iconic Eiffel Tower, and is a major destination for tourists, academics, and artists. Paris is also known for its diverse and multicultural community, with over 170 languages spoken and a rich history dating back over 2, 000 years. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by a variety of trends, including:
    
    1. Advancements in machine learning and deep learning: Machine learning and deep learning are the two main areas of AI research, and they are likely to continue to develop rapidly. This will lead to more complex and sophisticated AI systems that can learn and adapt to new situations.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. As AI becomes more widely used, we may see more personalized medicine and more accurate diagnoses.
    
    3. Integration of AI into everyday life: AI is


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

    First

     Name

    ]

     and

     I

    'm

     a

     [

    Job

     Title

    ]

     who

     specializes

     in

     [

    Primary

     Function

     or

     Role

    ].

     I

     believe

     in

     [

    My

     Core

     Bel

    ief

     or

     Core

     Value

    ].

     And

     I

     love

     [

    Something

     About

     Me

     or

     What

     I

     Do

    ].

     In

     summary

    ,

     I

    'm

     [

    Brief

     Summary

     of

     Your

     Character

    ].

     I

    'm

     excited

     to

     meet

     you

     and

     let

    's

     create

     a

     connection

    .

     How

     would

     you

     describe

     yourself

    ?

     
    


    [

    First

     Name

    ],

     I

    'm

     a

     [

    Your

     Job

     Title

    ]

     who

     specializes

     in

     [

    Primary

     Function

     or

     Role

    ],

     and

     I

     believe

     in

     [

    My

     Core

     Bel

    ief

     or

     Core

     Value

    ].

     I

     love

     [

    Something

     About

     Me

     or

     What

     I

     Do

    ],

     and

     in

     summary

    ,

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     enchant

    ing

     city

     with

     a

     rich

     history

     dating

     back

     over

     

    2

    0

    0

    0

     years

    .
    


    Could

     you

     expand

     on

     the

     cultural

     significance

     of

     Paris

     and

     its

     importance

     in

     French

     literature

    ?

     Yes

    ,

     Paris

     is

     home

     to

     many

     literary

     figures

     such

     as

     Vol

    taire

    ,

     Victor

     Hugo

    ,

     and

     D

    umas

    ,

     who

     left

     a

     lasting

     impact

     on

     French

     literature

    .

     It

     has

     also

     been

     the

     setting

     for

     countless

     movies

    ,

     plays

    ,

     and

     books

    .
    


    Could

     you

     please

     provide

     information

     on

     the

     cultural

     significance

     of

     Paris

     and

     its

     importance

     in

     French

     literature

    ?

     Yes

    ,

     Paris

     is

     the

     cultural

     center

     of

     France

     and

     is

     known

     for

     its

     rich

     history

     and

     architectural

     heritage

    .

     It

     has

     a

     long

     and

     stor

    ied

     past

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     potential

     trends

     that

     could

     occur

    :
    


    1

    .

     Personal

    ized

     AI

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increase

     in

     the

     ability

     of

     AI

     systems

     to

     learn

     from

     individual

     users

     and

     provide

     more

     personalized

     experiences

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Self

    -driving

     cars

     are

     already

     being

     developed

     and

     are

     expected

     to

     become

     more

     common

     in

     the

     future

    .

     Autonomous

     vehicles

     are

     expected

     to

     bring

     significant

     changes

     to

     traffic

     patterns

     and

     urban

     environments

    .
    


    3

    .

     Medical

     advancements

    :

     AI

     is

     already

     being

     used

     to

     improve

     healthcare

     outcomes

    ,

     from

     personalized

     medicine

     to

     drug

     discovery

    .

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     greater

     progress

     in

     this

     area

    .
    


    



```python
llm.shutdown()
```
