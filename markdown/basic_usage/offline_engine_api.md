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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


    2026-04-11 04:41:12,293 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 04:41:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:27,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:27,  2.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:27,  2.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:27,  2.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.87it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:02<00:03, 12.43it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:02<00:03, 12.43it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.43it/s]

    Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 12.43it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.84it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]

    Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.59it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.69it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:03<00:00, 35.82it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:03<00:00, 42.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   2%|▏         | 1/58 [00:00<00:09,  5.81it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   2%|▏         | 1/58 [00:00<00:09,  5.81it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   2%|▏         | 1/58 [00:00<00:09,  5.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   5%|▌         | 3/58 [00:00<00:04, 11.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:03, 14.24it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:03, 14.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 24.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.07it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.77it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 43.16it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:01<00:00, 43.16it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=240 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=224 avail_mem=75.73 GB):  59%|█████▊    | 34/58 [00:01<00:00, 40.51it/s]Capturing num tokens (num_tokens=224 avail_mem=75.73 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=208 avail_mem=75.70 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.02 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=144 avail_mem=75.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s] Capturing num tokens (num_tokens=80 avail_mem=75.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  76%|███████▌  | 44/58 [00:01<00:00, 34.92it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=48 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.99 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=24 avail_mem=74.98 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.97 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.96it/s] Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.96it/s]

    Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 32.97it/s]


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
    Generated text:  Aya S. Ojeda, an Indian woman. I’m a professional, bilingual speaker who was born in a Jewish family in the United States. I’ve worked for the United States Attorney’s Office in New York City, as well as at the United States District Court for the Southern District of New York, and have worked in the federal legal community, both as a lawyer and as a court clerk. In my early work in the legal community, I was a probation officer. I worked on juvenile criminal cases, working with very young people. My clients were often from the American Indian community. We would work together to teach them the
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official who serves for a term of 4 years. During that time, they are expected to serve as a role model for the next generation and to be a leader in shaping the country's policies. The president is responsible for representing the interests of the country and making decisions on behalf of the people, including those of the military, in matters that affect the nation's security and stability. They are also responsible for managing the country's budget and overseeing the executive branch of the government. The president is elected through a system of preferential electoral districts where each state has a number of seats equal to the number of electoral votes they have.
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Marseille
    D. Toulouse
    A. Paris is the capital of France. It is the largest city in France and one of the most populous cities in the European Union. Paris is known for its architecture, art, and cuisine, and it is the seat of the government, major institutions, and most populous of the French cities. Lyon, Marseille, and Toulouse are other major cities in France. The other options are not capitals of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  more diverse than ever, with more and more people working in this field. However, not everyone is comfortable with the concept. In this article, we’ll explore the various ways in which AI impacts society and society’s response to it.
    There are two types of AI that impact society: one is artificial intelligence, or AI, and the other is machine learning. Artificial intelligence refers to the development of computer systems that can perform tasks that would normally require human intelligence, such as recognizing faces, playing chess, and understanding language. Machine learning is a subset of artificial intelligence that uses algorithms to allow computers to improve their performance over time by learning from data


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is the largest city in Europe by population and is home to many famous landmarks and attractions. The city is known for its cuisine, fashion, and art, and is a major center of business and commerce. Paris is a popular tourist destination and a major economic hub in Europe. The city is also home to many international organizations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, including manufacturing, healthcare, and transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI will become more integrated with other technologies: AI will likely become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will likely lead to new applications and opportunities for AI, such as autonomous vehicles and smart homes.
    
    3. AI will become more ethical and transparent: As AI becomes more integrated into our daily
    


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
    Generated text:  John, and I'm a young historian, eager to learn about the past. My journey began at the helm of a traditional wooden ship, where I explored the wonders of sailing and the challenges of navigation. I've always been fascinated by the complexities of human emotion and the impact of culture on history. I am driven to uncover the past through research, writing, and sharing knowledge. I'm an optimistic person who values education and seeks to pass on my knowledge to others. I'm eager to learn about the past and have always been fascinated by the ways in which history shapes the present. What kind of day would you like to start the day
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La France" or "Paris." It is the largest city in Europe by population and a UNESCO World Heritage site. Paris is a world-renowned cultural, artistic, and intellectual center, home to numerous museums, theaters, and landmarks. It is also a major transportation hub, with the famous Eiffel Tower and Notre-Dame Cathedral. Paris is known for its cuisine, fashion, and nightlife, making it a popular tourist destination and cultural center. The city is home to numerous museums, theaters, and art galleries, and is a major center of politics and diplomacy. Paris is known for its distinctive architecture,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and has the potential to transform virtually every sector, from healthcare to transportation to entertainment. Here are some of the most likely future trends in AI:
    
    1. Improved accuracy and reliability: One of the biggest challenges facing AI is accuracy and reliability. As AI systems become more sophisticated, we may see improvements in their ability to detect and correct errors, and more reliable predictions and decision-making.
    
    2. Greater integration with natural language processing: As AI becomes more integrated with our everyday lives, we may see increased integration with natural language processing, which can make it easier for machines to understand human language and respond appropriately.
    
    3. Increased ethical and responsible use


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

    job

     title

    ]

     at

     [

    company

     name

    ].

     As

     a

     seasoned

     professional

    ,

     I

     have

     a

     passion

     for

     learning

     and

     improving

     my

     skills

     continuously

    .

     Whether

     it

    's

     in

     [

    specific

     skill

     or

     area

    ]

     or

     [

    more

     specific

     skill

    ],

     I

     am

     always

     eager

     to

     expand

     my

     knowledge

     and

     stay

     ahead

     of

     the

     curve

    .

     I

     have

     a

     strong

     work

     ethic

     and

     always

     strive

     to

     exceed

     expectations

     in

     every

     project

    .

     I

     enjoy

     creating

     a

     positive

     work

     environment

    ,

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     contribute

     and

     grow

    .

     Thank

     you

     for

     the

     opportunity

     to

     meet

     and

     be

     part

     of

     this

     team

    .

     Let

    's

     connect

    !

     
    


    Edit

     the

     text

     to

     make

     it

     sound

     more

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (A

    )

     Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     home

     to

     the

     European

     Parliament

    .

     


    (B

    )

     Paris

     is

     the

     largest

     city

     in

     France

     but

     is

     not

     home

     to

     the

     European

     Parliament

    .

     


    (C

    )

     Paris

     is

     not

     the

     largest

     city

     in

     France

     but

     is

     home

     to

     the

     European

     Parliament

    .

     


    (D

    )

     Paris

     is

     neither

     the

     largest

     nor

     home

     to

     the

     European

     Parliament

    .

     To

     determine

     the

     correct

     answer

    ,

     let

    's

     analyze

     each

     option

     step

     by

     step

    :
    


    (A

    )

     Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     home

     to

     the

     European

     Parliament

    .


    -

     This

     statement

     is

     incorrect

     because

     Paris

     is

     not

     the

     largest

     city

     in

     France

    .

     According

     to

     the

     

    2

    0

    2

    1

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     diverse

     and

     unpredictable

    ,

     with

     a

     wide

     range

     of

     trends

     that

     could

     change

     the

     way

     we

     live

     and

     work

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     More

     data

    -driven

     decision

    -making

    :

     With

     the

     increasing

     amount

     of

     data

     available

    ,

     AI

     systems

     will

     be

     able

     to

     make

     more

     accurate

     and

     data

    -driven

     decisions

    .

     This

     could

     lead

     to

     improved

     efficiency

     in

     everything

     from

     healthcare

     to

     transportation

    .
    


    2

    .

     Autonomous

     vehicles

    :

     The

     development

     of

     AI

    -driven

     autonomous

     vehicles

     is

     likely

     to

     become

     more

     prevalent

     in

     the

     years

     to

     come

    .

     These

     vehicles

     will

     be

     able

     to

     navigate

     roads

     and

     handle

     various

     scenarios

    ,

     including

     driving

     safely

     under

     different

     conditions

    .
    


    3

    .

     Personal

    ized

    



```python
llm.shutdown()
```
