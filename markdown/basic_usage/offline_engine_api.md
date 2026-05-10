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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.19it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.18it/s]


    2026-05-10 07:27:23,794 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 07:27:23] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.23it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.52it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 21.60it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 21.60it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 21.60it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 21.60it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 21.60it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.60it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.60it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.60it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.60it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.60it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 29.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 41.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.21it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.17it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.17it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.17it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.16it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.16it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.16it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.66it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.66it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 30.44it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 30.44it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:00<00:01, 30.44it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:01, 30.44it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:01<00:01, 30.44it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.40it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 30.25it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 30.25it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 28.51it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.29it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 30.29it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.74it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.74it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.58it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.58it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 31.94it/s]


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
    Generated text:  Qixing. I'm a 14-year-old girl who has a crush on my long-haired, thick-voiced brother, who is about the same age as me. I hope you can understand that I am not in any romantic relationship with him and I am not physically interested in him. However, I am interested in his long-haired, thick-voiced appearance and I am curious to know if his long hair and thick voice might make me feel more confident and attractive.
    
    Could you please share any insights or advice on how to better understand and accept my brother's appearance? It would be great if you could provide some tips
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, who holds a high-ranking position in the government and has the right to appoint and dismiss members of the military and other government agencies. The president is also the commander-in-chief of the armed forces, which is the highest-ranking officer in the armed forces. True or False?
    A. True
    B. False
    Answer:
    A
    
    The socialist core value system is composed of four basic components, including the lofty ideal of socialism with Chinese characteristics and the common ideal of socialism with Chinese characteristics, as well as the national spirit centered on patriotism and the era spirit centered on reform and innovation. Determine if the following statement is true or
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. 
    The capital of Spain is Madrid.
    Which one of the following must be true?
    A. Paris is in Europe
    B. Madrid is in Europe
    C. Paris is in North America
    D. Madrid is in Asia
    E. Paris and Madrid are in different continents
    
    Let's analyze the information given and the statements one by one to determine which one must be true. We have two statements:
    
    1. Paris is the capital of France.
    2. Madrid is the capital of Spain.
    
    We will evaluate each option:
    
    A. Paris is in Europe.
    - Since France is a country, Paris is indeed in Europe. However
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of people with a passion for the latest developments and a willingness to embrace new technologies. From developing cutting-edge software and hardware solutions to creating innovative business models, AI is constantly evolving and transforming the way we live, work, and play.
    It's essential to have a strong understanding of AI to keep up with the latest developments and stay ahead of the curve. Whether you're a data scientist, software developer, or a business owner, learning about AI can provide you with a valuable skill set that will help you succeed in your chosen field.
    AI is a complex and rapidly evolving field, and there is no one-size-fits-all


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and [job title]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in Europe and the third-largest city in the world by population. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, Louvre Museum, and the Palace of Versailles. The city is also known for its rich history, including the French Revolution and the French Revolution Monument. Paris is a cultural and artistic center, and its cuisine, fashion, and music are also highly regarded. The city is also known for its annual festivals and events, including the Eiffel Tower Festival and the Carn
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective applications of AI in various fields.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on
    


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
    Generated text:  [Name], and I am a [职业] who has [number] years of experience in [职业] field. I am [age] years old and I come from [country] with a [职业] degree from [university]. I have always been passionate about [职业] and always strive to do my best in [职业] every day. I am always eager to learn and challenge myself. I am a [character trait] person, and I am committed to [character trait] in all I do. I am always [emotion] and never give up. I am a [personality trait] person, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the country's seat of government and culture. It is known for its rich history, beautiful architecture, and vibrant culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very bright and promising, with several possible trends that could shape the development of this field in the years to come. Here are some of the key trends we are seeing in AI:
    
    1. Advanced machine learning: With the advent of deep learning, we are seeing a significant shift in how AI algorithms are developed and used. Machine learning is becoming more sophisticated, with algorithms being able to learn from large amounts of data and identify patterns that humans cannot.
    
    2. Increased integration with human decision-making: As AI becomes more sophisticated, we are seeing a growing integration of AI into human decision-making processes. This could lead to more complex and nuanced AI


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

    Character

    's

     Name

    ].

     I

     am

     a

     [

    Job

     Title

    /

    Position

    ]

     and

     I

     love

     [

    Reason

     for

     Job

    /

    Position

    ].

     I

     have

     [

    Number

     of

     years

    ]

     years

     of

     experience

     in

     [

    Industry

    /

    Field

    ],

     and

     I

     have

     always

     been

     [

    Positive

     Traits

    ].

     I

     am

     a

     [

    Current

     Skills

    /

    Qual

    ifications

    ].

     I

     enjoy

     [

    My

     hobbies

     and

     interests

    ].

     And

     I

     [

    The

     Goal

    /

     Passion

    /

    Op

    portunity

    ].

     I

     am

     dedicated

     to

     [

    Current

     or

     Future

     Projects

     or

     Goals

    ].

     I

     look

     forward

     to

     [

    Next

     Steps

     or

     Next

     Challenge

    ].

     And

     my

     [

    Personal

     Brand

    ing

    /

    Pro

    tag

    on

    ist

     Identity

    ].

     I

     am

     [

    What

     I

     Believe

     About

     My

    self

    /

    My

     Personality

    ].

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Please

     respond

     with

     facts

    ,

     not

     with

     a

     speculation

     or

     subjective

     statement

    .

     
    


    Sure

    ,

     here

    's

     a

     concise

     statement

     about

     France

    's

     capital

     city

    :


    The

     capital

     of

     France

     is

     Paris

    .

     
    


    This

     statement

     encaps

    ulates

     the

     most

     basic

     information

     required

     to

     understand

     the

     capital

     city

    's

     name

     and

     location

    .

     It

     avoids

     speculation

     or

     subjective

     elements

     and

     provides

     a

     straightforward

     factual

     answer

     to

     the

     question

    .

     The

     statement

     is

     concise

     yet

     comprehensive

    ,

     meeting

     the

     requirements

     set

     by

     the

     task

    .


    Paris

     is

     the

     capital

     city

     of

     France

    ,

     located

     in

     the

     north

    western

     part

     of

     the

     country

    .

     It

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     European

     Union

    ,

     with

     an

     estimated

     population

     of

     

    1

    4

     million

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     many

     different

     trends

    ,

     each

     with

     potential

     to

     have

     significant

     impacts

     on

     society

     and

     the

     way

     we

     live

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     our

     daily

     lives

    :

     AI

     is

     already

     making

     a

     significant

     impact

     on

     our

     lives

    ,

     from

     personal

     assistants

     like

     Siri

     and

     Alexa

     to

     self

    -driving

     cars

     and

     intelligent

     speech

     recognition

    .

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

     to

     see

     even

     more

     widespread

     adoption

     of

     AI

     in

     our

     everyday

     activities

    .
    


    2

    .

     AI

     will

     become

     more

     widespread

     and

     accessible

    :

     One

     of

     the

     most

     promising

     future

     trends

     is

     the

     increasing

     widespread

     adoption

     of

     AI

     in

     our

     daily

     lives

    .

     As

     AI

     becomes

    



```python
llm.shutdown()
```
