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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


    2026-05-02 11:23:53,259 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 11:23:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.56s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.42it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.26it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.26it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 14.78it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 14.78it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.78it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.09it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 23.57it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 27.37it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 32.00it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 32.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:04, 13.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.59 GB):   3%|▎         | 2/58 [00:00<00:04, 13.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=70.58 GB):   3%|▎         | 2/58 [00:00<00:04, 13.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.58 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.57 GB):   7%|▋         | 4/58 [00:00<00:03, 14.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.57 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=69.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.49 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=63.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=63.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.22 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.21 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.21 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.21 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.21 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=56.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.20 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.19 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.17 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=960 avail_mem=56.19 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.22it/s] Capturing num tokens (num_tokens=960 avail_mem=56.19 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=896 avail_mem=55.64 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=832 avail_mem=56.15 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.33it/s]

    Capturing num tokens (num_tokens=768 avail_mem=55.68 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=704 avail_mem=56.14 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.33it/s]Capturing num tokens (num_tokens=704 avail_mem=56.14 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=640 avail_mem=55.71 GB):  45%|████▍     | 26/58 [00:00<00:01, 27.54it/s]Capturing num tokens (num_tokens=576 avail_mem=56.14 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.54it/s]

    Capturing num tokens (num_tokens=512 avail_mem=55.72 GB):  45%|████▍     | 26/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=512 avail_mem=55.72 GB):  50%|█████     | 29/58 [00:01<00:01, 24.21it/s]Capturing num tokens (num_tokens=480 avail_mem=56.14 GB):  50%|█████     | 29/58 [00:01<00:01, 24.21it/s]Capturing num tokens (num_tokens=448 avail_mem=55.76 GB):  50%|█████     | 29/58 [00:01<00:01, 24.21it/s]Capturing num tokens (num_tokens=416 avail_mem=56.13 GB):  50%|█████     | 29/58 [00:01<00:01, 24.21it/s]

    Capturing num tokens (num_tokens=416 avail_mem=56.13 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=384 avail_mem=55.78 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=352 avail_mem=56.12 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=320 avail_mem=56.11 GB):  55%|█████▌    | 32/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=320 avail_mem=56.11 GB):  60%|██████    | 35/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=288 avail_mem=55.83 GB):  60%|██████    | 35/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=256 avail_mem=56.10 GB):  60%|██████    | 35/58 [00:01<00:01, 22.29it/s]

    Capturing num tokens (num_tokens=240 avail_mem=56.10 GB):  60%|██████    | 35/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=240 avail_mem=56.10 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.29it/s]Capturing num tokens (num_tokens=224 avail_mem=55.86 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.29it/s]Capturing num tokens (num_tokens=208 avail_mem=55.87 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.29it/s]Capturing num tokens (num_tokens=192 avail_mem=56.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.29it/s]

    Capturing num tokens (num_tokens=192 avail_mem=56.07 GB):  71%|███████   | 41/58 [00:01<00:00, 21.28it/s]Capturing num tokens (num_tokens=176 avail_mem=56.07 GB):  71%|███████   | 41/58 [00:01<00:00, 21.28it/s]Capturing num tokens (num_tokens=160 avail_mem=56.06 GB):  71%|███████   | 41/58 [00:01<00:00, 21.28it/s]Capturing num tokens (num_tokens=144 avail_mem=55.90 GB):  71%|███████   | 41/58 [00:01<00:00, 21.28it/s]Capturing num tokens (num_tokens=144 avail_mem=55.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.93it/s]Capturing num tokens (num_tokens=128 avail_mem=55.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.93it/s]Capturing num tokens (num_tokens=112 avail_mem=55.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.93it/s]Capturing num tokens (num_tokens=96 avail_mem=55.92 GB):  76%|███████▌  | 44/58 [00:01<00:00, 22.93it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=55.92 GB):  81%|████████  | 47/58 [00:01<00:00, 23.91it/s]Capturing num tokens (num_tokens=80 avail_mem=55.95 GB):  81%|████████  | 47/58 [00:01<00:00, 23.91it/s]Capturing num tokens (num_tokens=64 avail_mem=55.96 GB):  81%|████████  | 47/58 [00:02<00:00, 23.91it/s]Capturing num tokens (num_tokens=48 avail_mem=55.95 GB):  81%|████████  | 47/58 [00:02<00:00, 23.91it/s]Capturing num tokens (num_tokens=32 avail_mem=55.95 GB):  81%|████████  | 47/58 [00:02<00:00, 23.91it/s]Capturing num tokens (num_tokens=32 avail_mem=55.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=28 avail_mem=55.94 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=24 avail_mem=55.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=20 avail_mem=55.94 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.52it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.96 GB):  88%|████████▊ | 51/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=16 avail_mem=55.96 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=12 avail_mem=55.94 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=8 avail_mem=55.97 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.90it/s] Capturing num tokens (num_tokens=4 avail_mem=55.96 GB):  95%|█████████▍| 55/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=4 avail_mem=55.96 GB): 100%|██████████| 58/58 [00:02<00:00, 25.41it/s]


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
    Generated text:  Nick, I’m a passionate person with a hobby for creating educational content. My personal interest is in technology and digital marketing. I'm currently in my third year of college in a computer science major, and I'm also a member of a tech startup, which I'm excited to expand on.
    I've been in the tech startup industry for about a year and have been responsible for creating marketing strategies, content, and products. I'm looking for a part-time job where I can work on creating digital marketing content, and I'm currently looking for a role that would allow me to share my passion and expertise.
    What are some ways I could
    ===============================
    Prompt: The president of the United States is
    Generated text:  a ________ government. [ ]
    A. parliamentary
    B. presidential
    C. direct
    D. executive
    Answer:
    
    B
    
    When the temperature drops, a certain amount of alcohol (C2H5OH) begins to boil. The temperature of the liquid surface remains the same. This is due to [ ]
    A. the greater kinetic energy of the molecules
    B. the higher boiling point of the alcohol
    C. the larger volume of the molecules
    D. the greater density of the alcohol
    Answer:
    
    A
    
    The characteristic of the air column in a spring tube pressure gauge is that [ ]
    A. The length is constant
    ===============================
    Prompt: The capital of France is
    Generated text: ______. A．Paris B．London C．New York D．Rome
    C 解析：本题考查的是法国的首都。
    
    14.下列关于毛泽东思想的表述中，正确的是（）。 A.它是一个具有中国特色的马克思主义政治经济学的科学体系 B.它主要由《新民主主义论》、《论联合政府》、《中国革命和中国共产党》等著作提出和丰富 C.它不是完全符合中国国情的科学社会主义 D.它指导着中国革命取得胜利
    答案：B 解析：毛泽东思想是中国共产党、中国人民和中华民族的宝贵精神财富。
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the tech industry. The development of new technologies has been a recent highlight in this field. In the past, it was expensive, complex and expensive, but now it is becoming more and more affordable and simple to create.
    Here, we will discuss the future of AI in the following areas: the impact of AI on the healthcare industry, the impact of AI on education, the impact of AI on the economy, the impact of AI on the environment, and the impact of AI on society.
    What’s AI’s impact on healthcare?
    In the healthcare industry, the future of AI is set to be very interesting. This technology


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [insert a few key points about yourself, such as your personality, skills, or hobbies]. And what can you tell me about your company? I'm excited to learn more about your company and how it fits into the world of [insert a few key points about your company, such as its mission, products, or services]. And what can you tell me about your role at [company name]? I'm excited to learn more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also known for its fashion industry, art scene, and its role in the French Revolution and the French Revolution. It is a popular tourist destination and a major economic and cultural center in Europe. The city is home to many famous landmarks and attractions, including the Louvre, the Notre-Dame Cathedral, and the Champs-Élysées. Paris is also known for its cuisine, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to more personalized and adaptive AI systems that can learn from human behavior and adapt to new situations.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased public awareness of the potential risks and benefits of AI.
    
    3. Greater reliance on AI for decision-making: As
    


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
    Generated text:  [Your Name], and I am an [insert your occupation here]. I love [mention something that makes you proud or excites you] and I am constantly learning new things. I have a passion for [insert something that makes you proud or excites you]. I am always up for a challenge, and I have a great sense of humor that I use to make people laugh. I am also [insert something that makes you proud or excites you] and I have a great sense of teamwork. I am an active and independent person who enjoys pursuing my passions. I have a sense of humor that makes me smile and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its romantic architecture, vibrant culture, and annual cultural events such as the Eiffel Tower ceremony and the Carnaval festival. The city is also home to many renowned museums, including the Louvre and Musée d'Orsay. Paris is a cultural and economic center, known for its fashion, wine, and gastronomy. With its historical landmarks and modern skyscrapers, Paris is a city that combines tradition and modernity to create a unique and exciting experience for visitors. Paris is also home to the Eiffel Tower, a symbol of France and a UNESCO World Heritage site, and the Louvre
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by exponential growth and diversification. Here are some potential trends we can expect in AI over the next few decades:
    
    1. Personalization: AI will become more adept at understanding and predicting individual human behaviors. This will allow for more personalized experiences and better outcomes for users. This could lead to a more efficient and personalized healthcare system, as well as better customer service for businesses.
    
    2. Autonomous vehicles: AI will continue to advance rapidly, with autonomous vehicles becoming a reality. This could revolutionize transportation, creating new jobs and changing the way we live and work.
    
    3. Natural language processing: AI will continue to improve at


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

    insert

     your

     name

    ],

     and

     I

     am

     a

     [

    insert

     your

     occupation

     or

     background

    ]

     who

     has

     been

     working

     in

     [

    insert

     your

     profession

    ]

     for

     [

    insert

     how

     long

     you

    've

     been

     working

     in

     the

     industry

    ].

     I

     love

     to

     [

    insert

     something

     related

     to

     your

     hobbies

    ,

     interests

    ,

     or

     passion

     that

     you

     enjoy

    ].

     And

     I

    'm

     constantly

     learning

    ,

     whether

     it

    's

     through

     [

    insert

     what

     you

     do

     for

     a

     living

     that

     you

     are

     learning

    ]

     or

     through

     [

    insert

     what

     you

    're

     learning

     on

     your

     free

     time

    ],

     and

     I

    'm

     always

     looking

     for

     new

     challenges

     to

     grow

     and

     improve

    .

     Thanks

     for

     taking

     the

     time

     to

     meet

     me

    !

     [

    insert

     any

     additional

     information

     you

     think

     would

     be

     helpful

    ,

     such

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

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

     Additionally

    ,

     Paris

     is

     known

     for

     its

     rich

     cultural

     heritage

     and

     cuisine

    ,

     which

     is

     influenced

     by

     the

     history

     of

     the

     city

    ,

     including

     French

     and

     other

     European

     cultures

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     with

     its

     famous

     landmarks

     and

     museums

     attracting

     millions

     of

     visitors

     each

     year

    .

     The

     city

    's

     unique

     blend

     of

     historical

     and

     modern

     influences

     has

     made

     it

     a

     symbol

     of

     France

    's

     cultural

     and

     political

     evolution

    .

     
    


    Therefore

    ,

     Paris

     is

     the

     cultural

     and

     historical

     center

     of

     France

    ,

     known

     for

     its

     vibrant

     city

     life

    ,

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     delicious

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     one

     of

     rapid

     growth

    ,

     innovation

    ,

     and

     transformation

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     AI

     will

     continue

     to

     personalize

     interactions

     with

     users

    ,

     autom

    ating

     routine

     tasks

     and

     providing

     more

     relevant

     and

     tailored

     recommendations

    .
    


    2

    .

     AI

     will

     be

     more

     autonomous

    :

     More

     and

     more

     machines

     will

     be

     able

     to

     learn

     and

     adapt

     to

     new

     situations

     on

     their

     own

    ,

     without

     human

     intervention

    .
    


    3

    .

     AI

     will

     be

     more

     ethical

     and

     responsible

    :

     AI

     will

     be

     used

     for

     more

     ethical

     and

     responsible

     purposes

    ,

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     cybersecurity

    .
    


    4

    .

     AI

     will

     be

     more

     integrated

     with

     other

     technologies

    :

     AI

     will

     be

     more

     integrated

     with

     other

    



```python
llm.shutdown()
```
