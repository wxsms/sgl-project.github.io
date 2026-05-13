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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]


    2026-05-13 02:10:18,662 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 02:10:18] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:52,  5.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.81it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.11it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 12.89it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.72it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 26.53it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 36.06it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 36.06it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 36.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 17.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 20.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:01, 28.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.50it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.85it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.00it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 35.95it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 35.95it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 35.95it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 35.95it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.95it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 35.95it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.02it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 29.51it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.12it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.12it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.12it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.12it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 28.12it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.87it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.87it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.87it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.87it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 18.70it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  86%|████████▌ | 50/58 [00:01<00:00, 18.70it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:01<00:00, 18.70it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:02<00:00, 18.70it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  86%|████████▌ | 50/58 [00:02<00:00, 18.70it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:02<00:00, 22.11it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:02<00:00, 22.11it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:02<00:00, 25.41it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:02<00:00, 26.84it/s]


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
    Generated text:  Trish and I am a graduate of the UCPA and LAEC community and community college systems.
    I was a Junior and Senior at UCLA and prior to that, I was a Senior at Sunset College.
    I was also a track athlete for the California women's cross country team and was a member of the team for two years.
    I earned a B.S. in Business Administration from the University of California, Los Angeles (UCLA) and a Master of Public Health degree from the University of California, Los Angeles (UCLA) in the community health management program.
    I am currently working as a marketing strategist for a non-profit organization that
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who holds the position of head of state, the head of the executive branch of the government. In the United States, the president is elected by the people through the electoral college. The role of the president is to oversee the government, to ensure that the executive branch works well with the legislative branch, and to ensure that the executive branch works well with the judiciary branch. The president is also tasked with representing the country and fulfilling its obligations. In the United States, the president is not an elected official, but rather an appointed official. The president serves a four-year term, and can be re-elected. In addition to representing the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Los Angeles
    D. Sydney
    The correct answer is A. Paris. Paris is the capital city of France and is known for its history, art, and culture. It is the most populous city in France, with over 2.3 million residents as of 2021. The city is home to many landmarks, museums, and cultural institutions. The most famous landmark in Paris is the Eiffel Tower, which is over 300 years old and a symbol of the city.
    ===============================
    Prompt: The future of AI is
    Generated text:  here. It is the future of our own lives, and we must take the next steps to create a world where AI is not a threat, but a tool. It is an AI that is designed for good, to support humans in their daily lives in the most efficient and effective way possible, where people can feel confident and secure knowing that their lives are being shaped by intelligent, powerful, and ethical AI technologies. To help us achieve this, we must work together to create a better AI infrastructure, which is where it all begins.
    AI and Artificial Intelligence (AI) have become a global phenomenon. We have already seen a large amount of


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm always eager to learn and grow, and I'm always looking for new challenges and opportunities to grow. What's your name? What's your job title? What's your company name? What's your passion? What's your current job title? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a major tourist destination and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools, chatbots, and virtual assistants are expected to become more sophisticated and accurate, leading to better patient outcomes and reduced costs.
    
    2. Increased use of AI in finance: AI is already being used in finance to help
    


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
    Generated text:  [Name] and I am [Age]. I am a [occupation/teacher/student], and I work for [Company]. I also play [interest/genre] on social media. How can I be sure that you can identify me?
    I am an AI language model, created by Anthropic, designed to assist with a wide range of tasks, including writing, language translation, and language generation. I am capable of understanding and generating text in multiple languages, and I can also perform a variety of other tasks based on your instructions.
    Let's get started! How can I help you today? Let's see how good you can be at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, picturesque architecture, and iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
    
    That's correct! Paris is the capital of France and known for its stunning architecture, iconic landmarks, and rich history. Its Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum are just a few of its most famous attractions. Its historical importance is further emphasized by its status as the most-visited city in the world, attracting millions of tourists every year. France's capital city is a true wonder of the world, offering a unique blend of tradition and modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and we can expect to see many exciting trends in the coming years. Here are some of the key trends that are likely to shape the future of AI:
    
    1. Deep learning: Deep learning is a field of AI that focuses on building more complex models that can handle massive amounts of data and learn from it. This could lead to more accurate and efficient predictions in a wide range of applications.
    
    2. Autonomous vehicles: Autonomous vehicles are becoming more advanced and are likely to be a key aspect of AI in the future. This could lead to safer and more efficient transportation, as well as the reduction of air pollution.
    
    3. Natural


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

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

    'm here

     to

     share

     my

     experiences

    ,

     skills

    ,

     and

     vision

     for

     this

     company

    .

     How

     can

     I

     assist

     you

     today

    ?

     Feel

     free

     to

     ask

     me

     anything

    !

     

    🌟

    ✨

    
    


    Can

     you

     tell

     us

     about

     a

     specific

     project

     you

    've

     worked

     on

     recently

     that

     you

    're

     particularly

     proud

     of

    ?

     What

     was

     it

     about

    ,

     and

     how

     did

     you

     feel

     when

     you

     saw

     the

     results

    ?

     

    🕊

    ️

    ✨

    
    


    Remember

    ,

     I

     don

    't

     want

     to

     feel

     like

     I

    'm

     being

     dishonest

     if

     you

     think

     it

     was

     a

     bit

     of

     a

     big

     deal

    !

     

    😊

    😉

    
    


    What

    's

     your

     most

     significant

     achievement

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     oldest

     city

     in

     Europe

     and

     the

     largest

     city

     in

     the

     European

     Union

    ,

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     The

     city

     is

     home

     to

     many

     important

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

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

     also

     a

     major

     economic

     hub

     and

     a

     center

     for

     international

     trade

     and

     diplomacy

    .

     Paris

     has

     a

     long

     history

     dating

     back

     to

     ancient

     times

    ,

     with

     many

     iconic

     structures

     and

     landmarks

     such

     as

     the

     Sag

    r

    ada

     Famil

    ia

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Today

    ,

     it

     remains

     a

     significant

     cultural

     and

     economic

     center

     of

     France

     and

     plays

     a

     vital

     role

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     there

     is

     no

     sure

    -fire

     way

     to

     predict

     exactly

     what

     will

     happen

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     may

     occur

     in

     the

     near

     future

    :
    


    1

    .

     AI

     will

     continue

     to

     improve

     and

     become

     more

     intelligent

    .

     In

     the

     near

     future

    ,

     we

     may

     see

     more

     powerful

     AI

     systems

     that

     can

     think

     and

     reason

     like

     humans

    ,

     and

     that

     can

     learn

     and

     adapt

     to

     new

     situations

    .
    


    2

    .

     AI

     will

     become

     more

     ubiquitous

    .

     As

     AI

     technology

     advances

    ,

     more

     and

     more

     people

     will

     be

     able

     to

     interact

     with

     it

     in

     their

     daily

     lives

    ,

     from

     voice

     assistants

     to

     self

    -driving

     cars

    .
    


    3

    .

     AI

     will

     continue

     to

     be

     used

     in

     healthcare

    .

     AI

     will

     help

     doctors

     diagnose

     diseases

    ,

    



```python
llm.shutdown()
```
