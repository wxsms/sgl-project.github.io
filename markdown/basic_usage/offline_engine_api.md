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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-16 07:10:04] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.20it/s]


    2026-04-16 07:10:09,471 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 07:10:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.64it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.64it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.64it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.55it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.53it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.26it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   3%|▎         | 2/58 [00:00<00:03, 16.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.73 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):   9%|▊         | 5/58 [00:00<00:02, 19.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.70it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  21%|██        | 12/58 [00:00<00:01, 28.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.88it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]

    Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=448 avail_mem=120.24 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=384 avail_mem=120.27 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=352 avail_mem=120.27 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=320 avail_mem=119.04 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.28it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.28it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=256 avail_mem=118.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=208 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=192 avail_mem=118.93 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=192 avail_mem=118.93 GB):  71%|███████   | 41/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  71%|███████   | 41/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  71%|███████   | 41/58 [00:01<00:00, 36.79it/s]

    Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  71%|███████   | 41/58 [00:01<00:00, 36.79it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.37it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.37it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.37it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.37it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 24.37it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.81it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.81it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.81it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.81it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.81it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 27.51it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 27.51it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 27.51it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 27.51it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 27.51it/s] Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  98%|█████████▊| 57/58 [00:02<00:00, 23.38it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  98%|█████████▊| 57/58 [00:02<00:00, 23.38it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:02<00:00, 28.60it/s]


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
    Generated text:  Maria. I'm very happy to tell you about my family. My parents are very strict with me. When I was 3, I had a terrible toothache. The dentist sent me home. After a few hours, I felt much better. The dentist told me to take a bath, brush my teeth and eat some ice cream. When I got home, I went to the store and bought some toothpaste. The next morning, my parents said I should take the ice cream. That night, I ate the ice cream. I fell down from the tree and hurt my knee. My parents said I should go to the hospital.
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by _______.
    A. a party that has the largest share of the vote in the popular vote
    B. the party that has the most candidates
    C. the party that has the most state legislatures
    D. the party that has the most votes in the state legislature
    Answer:
    
    A
    
    Among the following options, which one is NOT considered a civil legal act? 
    A. A company manager signs a contract with a mall.
    B. A company manager executes an insurance policy on behalf of a business.
    C. A company manager enters into a labor contract with a worker.
    D. A company manager signs a contract for
    ===============================
    Prompt: The capital of France is
    Generated text:  located in ______ province.
    A. Hainan
    B. Amiens
    C. Normandy
    D. Quebec
    Answer:
    
    C
    
    The hallmark of a person's maturity is their ability to _____.
    A. adhere to rules
    B. change
    C. apply
    D. think
    Answer:
    
    D
    
    According to the law, a mine's roof must be classified as ____.
    A. Class A
    B. Class B
    C. Class C
    D. Class D
    Answer:
    
    A
    
    In the event of a sudden public crisis, what should a nurse prioritize?
    A. Caring for the patients
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  now
    
    The AI era is coming and the internet is rapidly evolving with the introduction of new technologies and developments. By exploring the future of AI and how it will impact us in the coming years, we can better prepare for its impact and prepare to embrace it fully.
    
    In this blog post, we discuss how AI is evolving and how it will impact society and the world in the coming years. We will explore the technologies that are changing the world and the applications of AI in different industries, including healthcare, finance, and transportation. We will also discuss the ethical considerations that arise from the use of AI and the importance of ensuring that AI is developed


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the [Industry] field. I'm passionate about [Why] and I'm always looking for ways to [What I'm Working On]. I'm a [What I Do] person who is always [What I Do Well]. I'm [What I Do Well] and I'm [What I Do Well] in my field. I'm a [What I Do Well] person who is always [What I Do Well]. I'm [What I Do Well] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is home to many international organizations and is a major hub for international trade and diplomacy. The French capital is a vibrant and dynamic place, with a rich cultural heritage and a strong sense of identity. Paris is a city that is both old and new, with a unique blend of historical and modern elements. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between humans and machines.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a greater emphasis on privacy and security. This will include measures to protect user data and prevent unauthorized access to AI systems.
    
    3. Greater focus on ethical considerations: As AI systems become more complex and sophisticated, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    4
    


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
    Generated text:  [Name], I'm a [job title or specialty], and I've always been fascinated by [specific topic or field of interest]. I'm not just any person, I'm a [specific interest or area of expertise], and I have a knack for [specific skill or trait]. I've always been [specific personality trait or characteristic], and I'm always eager to learn and grow. I'm always looking for the next step in my journey, and I'm ready to challenge myself and succeed in whatever path I choose. Is there anything I can help you with today? I'm excited to learn more about you and to share my knowledge
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a large city located on the left bank of the Seine River in the center of the country. Paris is known for its rich history, art, fashion, and cuisine, and is a bustling city with many famous landmarks and attractions. The city is also home to a variety of cultural institutions and organizations, including the Louvre and the Musée d'Orsay, as well as the Paris Opera, the Opéra Garnier, and the Champs-Élysées. Paris is a popular tourist destination, and attracts millions of visitors each year. Its location on the river and its beautiful architecture make it a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by an increasing focus on creating AI that can be more ethical, transparent, and personalized. Here are some possible trends in AI in the coming years:
    
    1. Increased use of AI in healthcare: AI is increasingly being used in healthcare to improve patient outcomes and reduce costs. This includes the use of AI algorithms to diagnose diseases, predict treatment outcomes, and optimize medication scheduling.
    
    2. Enhanced natural language processing: AI is expected to become more capable of understanding and generating human-like natural language, making it easier for humans to communicate with AI. This could lead to more effective healthcare communication, personalized medicine, and chatbots for customer


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

    ].

     I

     have

     been

     living

     and

     working

     in

     [

    Your

     Location

    ]

     for

     [

    Your

     Duration

     of

     Service

    ].

     I

     am

     a

     [

    Your

     Profession

    /

    Title

    ]

     with

     over

     [

    Your

     Number

     of

     Years

     of

     Service

    ].

     My

     career

     has

     span

    ned

     [

    Your

     Career

     Highlights

     and

     Achie

    vements

    ],

     and

     I

     strive

     to

     [

    Your

     Professional

     Goals

     and

     aspirations

    ]. I

     am

     passionate

     about

     [

    Your

     Area

     of

     Interest

    /

    Interest

    /

    Experience

    ].

     I

     am

     always

     looking

     to

     learn

     and

     grow

     in

     my

     field

     and

     contribute

     to

     the

     advancement

     of

     [

    Your

     Field

     of

     Interest

    /

    Field

     of

     Experience

    ].

     I

     am

     eager

     to

     continue

     my

     professional

     journey

     and

     contribute

     to

     [

    Your

     Field

     of

     Interest

    /

    Field

     of

     Experience

    ]

     in

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     renowned

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     breathtaking

     views

     of

     the

     city

     and

     its

     surroundings

    ,

     and

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

    .

     Its

     elegant

     architecture

    ,

     lively

     streets

    ,

     and

     world

    -ren

    owned

     museums

     have

     made

     Paris

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

     of

     the

     country

    .

     
    


    The

     city

     is

     also

     home

     to

     many

     notable

     institutions

    ,

     including

     the

     Lou

    vre

    ,

     the

     most

     famous

     museum

     in

     the

     world

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     which

     is

     one

     of

     the

     largest

     and

     oldest

     Gothic

     cath

    ed

    r

    als

     in

     the

     world

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     range

     of

     trends

    ,

     some

     of

     which

     are

     already

     becoming

     more

     prevalent

     and

     some

     of

     which

     are

     still

     in

     the

     early

     stages

     of

     development

    .

     Here

     are

     some

     potential

     trends

     that

     may

     occur

     in

     the

     near

     and

     long

     term

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     continues

     to

     become

     more

     advanced

    ,

     we

     may

     see

     a

     growing

     integration

     of

     AI

     into

     our

     daily

     lives

    ,

     such

     as

     through

     the

     use

     of

     virtual

     assistants

     like

     Siri

     and

     Alexa

    ,

     smart

     home

     devices

    ,

     and

     AI

    -driven

     vehicles

    .
    


    2

    .

     More

     widespread

     adoption

     of

     AI

     for

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     the

     healthcare

     industry

     to

     assist

     with

     diagnosis

    ,

     treatment

     planning

    ,

     and

     patient

     management

    .

    



```python
llm.shutdown()
```
