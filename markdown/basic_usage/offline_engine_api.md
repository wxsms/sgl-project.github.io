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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.85it/s]


    2026-04-09 21:44:15,550 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 21:44:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.61it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.87it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.87it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.89it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.42it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.39it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.51it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:06,  9.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.41it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:06,  9.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   5%|▌         | 3/58 [00:00<00:05, 10.33it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   5%|▌         | 3/58 [00:00<00:05, 10.33it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.20it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.20it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.75 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.74 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  24%|██▍       | 14/58 [00:00<00:02, 21.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.22it/s]Capturing num tokens (num_tokens=960 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.22it/s] Capturing num tokens (num_tokens=896 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.22it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.71 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.22it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.22it/s]Capturing num tokens (num_tokens=768 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.53it/s]Capturing num tokens (num_tokens=704 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.53it/s]Capturing num tokens (num_tokens=640 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.53it/s]Capturing num tokens (num_tokens=576 avail_mem=118.70 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.53it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.53it/s]Capturing num tokens (num_tokens=512 avail_mem=118.69 GB):  50%|█████     | 29/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=480 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.19it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=416 avail_mem=118.70 GB):  50%|█████     | 29/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  50%|█████     | 29/58 [00:01<00:00, 29.19it/s]Capturing num tokens (num_tokens=384 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.45it/s]Capturing num tokens (num_tokens=352 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.45it/s]Capturing num tokens (num_tokens=320 avail_mem=118.69 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.45it/s]Capturing num tokens (num_tokens=288 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.45it/s]Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 30.45it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=240 avail_mem=118.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=224 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=208 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=192 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.85it/s]Capturing num tokens (num_tokens=176 avail_mem=118.67 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.25it/s]Capturing num tokens (num_tokens=160 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.25it/s]Capturing num tokens (num_tokens=144 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.25it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.25it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 35.25it/s]Capturing num tokens (num_tokens=112 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 34.80it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=32 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.53it/s]

    Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=20 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.65it/s]Capturing num tokens (num_tokens=16 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.65it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.65it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.65it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  93%|█████████▎| 54/58 [00:02<00:00, 34.65it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 34.68it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 27.37it/s]


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
    Generated text:  Maria and I'm a student of 12 years of age. My father is named Santiago and my mother is named Ana. We have a beautiful house in the heart of the city and we have three children. My mother was a writer and she started a magazine called "Cracia" in the year of 1965. And my father was a journalist and he started a magazine called "La Prensa" in the year of 1975. We have a house with four bedrooms and a garden. My mother was a teacher and she started teaching me at the age of two and a half years old.
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. The vice president of the United States is 4 feet 6 inches tall. If both presidents stand in a row, what is the total height of the two?
    To determine the total height of the two presidents standing in a row, we need to add their individual heights together. The height of the president is 5 feet 3 inches and the height of the vice president is 4 feet 6 inches.
    
    First, we convert the height of the vice president into inches because the heights are asked in inches. Since there are 12 inches in a foot, we have:
    \[ 4
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) London  
    B) Paris  
    C) Berlin  
    D) Moscow  
    E) Istanbul  
    
    1. **Identify the key elements of the question**: 
       - The capital of France, as given in the problem, is Paris.
    
    2. **Analyze the options**:
       - A) London
       - B) Paris
       - C) Berlin
       - D) Moscow
       - E) Istanbul
    
    3. **Determine the correct capital of France**:
       - France, as a country, has its capital in the city of Paris.
       - Other options like London, Berlin, Moscow
    ===============================
    Prompt: The future of AI is
    Generated text:  still in its infancy. But if you're an AI researcher, a software developer, or a data scientist, it is possible to do your job with AI. This could help you develop new models, algorithms, and solutions that help solve problems with greater efficiency and effectiveness. Here are some specific examples of how AI is being used to help solve real-world problems.
    With deep learning, a powerful algorithm that has the ability to learn from large amounts of data and extract meaningful patterns, companies are able to make more accurate predictions about the future. With this in mind, many companies are looking at AI as a way to better understand their customers and anticipate


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


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is the largest city in France and the third-largest city in the world by population. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is also known for its fashion, art, and cuisine. The city is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is also a major transportation hub, with the Eiffel Tower serving as a symbol of the city's status as a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can better understand and respond to human emotions and preferences.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and responsible AI development. This could involve creating AI systems that are
    


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
    Generated text:  [Name] and I'm a [Age] [Gender] who loves [What your hobbies or interests are]. I'm passionate about [Your favorite hobby or activity]. I enjoy [What's your profession or occupation]? I believe in [Your belief or concept]. How do you feel about [What's your favorite book or music that you enjoy]?
    
    ---
    
    **Please feel free to make the self-introduction as you see fit, but aim for a neutral, professional tone and a brief introduction that captures the essence of your character without being too lengthy.**
    
    ---
    
    Note: If you'd like me to add more details, feel free to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! The capital of France is Paris. It's the largest city in France and the country's political, cultural, and economic center. 
    
    Here are some key points about Paris:
    
    1. History: It has a rich history dating back to the Roman Empire.
    2. Architecture: Paris is known for its iconic architecture, including the Eiffel Tower and Notre Dame Cathedral.
    3. Transportation: It's a major transportation hub, with numerous train lines and excellent public transportation systems.
    4. Culture: Paris is home to many world-famous museums, theaters, and festivals.
    5. Food: It's famous for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by a number of key trends, including:
    
    1. Increasingly personalized and context-aware AI: AI systems will become more personal and tailored to individual users, understanding the context and emotions of the users they interact with. This will require more sophisticated algorithms and machine learning techniques to analyze and interpret user behavior.
    
    2. Integration of AI into other sectors: AI will increasingly be integrated into other sectors such as healthcare, finance, and transportation to improve efficiency and reduce costs. This will require a more integrated approach to AI and better data sharing and analysis.
    
    3. AI will become more transparent and accountable: AI systems will become more transparent


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

    Name

    ],

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    company

    ].

     I

    'm

     thrilled

     to

     have

     the

     opportunity

     to

     meet

     you

    .

     I

     want

     to

     talk

     to

     you

     about

     [

    what

     you

    're

     discussing

    ],

     and

     I

     hope

     we

     can

     stay

     in

     touch

    .
    


    I

     look

     forward

     to

     our

     conversation

     and

     to

     any

     questions

     you

     have

    .

     Let

    's

     make

     some

     history

     and

     build

     something

     strong

     together

    .

     Do

     you

     have

     any

     questions

     before

     I

     continue

    ?

     [

    Insert

     any

     relevant

     information

     about

     your

     job

     title

    ,

     company

    ,

     and

     any

     skills

     or

     experiences

     you

     might

     have

    .

    ]


    Hey

     there

    ,

     I

    'm

     [

    Your

     Name

    ]

     and

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    Your

     Company

    ].

     I

    'm

     super

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     "

    the

     Eternal

     City

    ".

     It

     is

     a

     historic

     and

     cultural

     city

     located

     in

     the

     south

     of

     France

     and

     is

     home

     to

     the

     French

     parliament

     and

     government

     buildings

    .

     Paris

     is

     also

     known

     for

     its

     art

    ,

     music

    ,

     and

     food

     scene

    ,

     and

     is

     home

     to

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

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     has

     a

     rich

     history

     and

     is

     famous

     for

     its

     ambiance

    ,

     architecture

    ,

     and

     cuisine

    .

     According

     to

     the

     

    2

    0

    2

    1

     census

    ,

     Paris

     has

     a

     population

     of

     approximately

     

    2

    .

    3

     million

     people

    .

     It

     is

     one

     of

     the

     most

     visited

     cities

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     it

     is

     likely

     to

     continue

     to

     be

     a

     powerful

     and

     transformative

     force

     in

     many

     ways

    .

     Here

     are

     some

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     automation

     and

     artificial

     general

     intelligence

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

     we

     can

     expect

     automation

     to

     become

     more

     prevalent

    ,

     with

     AI

     systems

     being

     able

     to

     perform

     a

     wide

     range

     of

     tasks

     and

     tasks

     that

     were

     once

     done

     by

     humans

    .

     This

     could

     lead

     to

     significant

     job

     losses

    ,

     but

     it

     could

     also

     create

     new

     opportunities

     for

     people

     to

     work

     with

     AI

     systems

     in

     new

     and

     exciting

     ways

    .
    


    2

    .

     Improved

     AI

     ethics

     and

     privacy

    :

     As

     AI

     systems

     become

     more

     integrated

     into

    



```python
llm.shutdown()
```
