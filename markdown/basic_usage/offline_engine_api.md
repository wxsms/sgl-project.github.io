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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]


    2026-05-08 19:40:12,062 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 19:40:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.89it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:08,  5.13it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:08,  5.13it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:08,  5.13it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:08,  5.13it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:08,  5.13it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:05,  7.88it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:02, 13.08it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 20.12it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 29.07it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 39.64it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 39.64it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 39.64it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 39.64it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 51.01it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 51.01it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 51.01it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.48it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.74 GB):   3%|▎         | 2/58 [00:00<00:03, 14.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.71 GB):   3%|▎         | 2/58 [00:00<00:03, 14.88it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=73.71 GB):   3%|▎         | 2/58 [00:00<00:03, 14.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.71 GB):   7%|▋         | 4/58 [00:00<00:03, 13.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.70 GB):   7%|▋         | 4/58 [00:00<00:03, 13.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.70 GB):   7%|▋         | 4/58 [00:00<00:03, 13.54it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=73.70 GB):  10%|█         | 6/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.48 GB):  10%|█         | 6/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.49 GB):  10%|█         | 6/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.49 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.49 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.50 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.52 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=73.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.52 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.52 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.52 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.52 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.54 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.82it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=73.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.55 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.53 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.53 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=960 avail_mem=73.55 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.82it/s] Capturing num tokens (num_tokens=896 avail_mem=73.55 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=832 avail_mem=73.56 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=768 avail_mem=73.55 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.82it/s]

    Capturing num tokens (num_tokens=704 avail_mem=73.55 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=704 avail_mem=73.55 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=640 avail_mem=73.54 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=576 avail_mem=73.54 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=512 avail_mem=73.51 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=480 avail_mem=73.53 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=448 avail_mem=73.53 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.35it/s]Capturing num tokens (num_tokens=448 avail_mem=73.53 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=416 avail_mem=73.53 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=384 avail_mem=73.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=352 avail_mem=73.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=288 avail_mem=73.50 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.64it/s]Capturing num tokens (num_tokens=288 avail_mem=73.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=256 avail_mem=73.49 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=240 avail_mem=73.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=224 avail_mem=73.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=208 avail_mem=73.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=192 avail_mem=73.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=192 avail_mem=73.47 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]Capturing num tokens (num_tokens=176 avail_mem=73.46 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]Capturing num tokens (num_tokens=160 avail_mem=73.46 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.45 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]Capturing num tokens (num_tokens=128 avail_mem=73.44 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]Capturing num tokens (num_tokens=112 avail_mem=73.44 GB):  71%|███████   | 41/58 [00:01<00:00, 38.19it/s]Capturing num tokens (num_tokens=112 avail_mem=73.44 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=96 avail_mem=73.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s] Capturing num tokens (num_tokens=80 avail_mem=73.15 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=64 avail_mem=73.15 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=48 avail_mem=73.12 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=32 avail_mem=72.43 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.46it/s]Capturing num tokens (num_tokens=32 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=28 avail_mem=72.44 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=20 avail_mem=72.43 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=16 avail_mem=72.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=12 avail_mem=72.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 39.87it/s]Capturing num tokens (num_tokens=12 avail_mem=72.41 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=8 avail_mem=72.41 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.78it/s] Capturing num tokens (num_tokens=4 avail_mem=72.40 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.78it/s]Capturing num tokens (num_tokens=4 avail_mem=72.40 GB): 100%|██████████| 58/58 [00:01<00:00, 30.92it/s]


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
    Generated text:  ____.
    A. Miss Li
    B. Miss Wu
    C. Miss Huang
    D. Teacher Wang
    Answer:
    
    A
    
    The location of the locus of the focus of an ellipse is
    A. the upper part
    B. the lower part
    C. the left part
    D. the right part
    Answer:
    
    B
    
    In the context of kindergarten education, the purpose of the kindergarten's small class activities and activities is to ____.
    A. Regulate the kindergarten environment
    B. Meet children's physical and mental needs
    C. Educate children
    D. Promote the development of children
    Answer:
    
    B
    
    In terms
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. The president has a lot of important jobs to do, including being in charge of the country's money, the country's air, the country's water, and more. The president also has to do important things to make sure that the country is a safe place to live. The president has to be good at being president. He or she should be very smart and good at thinking about important things. He or she should also be good at making decisions. The president has to be fair and just. He or she should also be a good listener. The president has
    ===============================
    Prompt: The capital of France is
    Generated text:  _____
    A. Paris
    B. Lyon
    C. Marseille
    D. Nice
    
    To determine the capital of France, let's consider the major cities in France and their geopolitical importance:
    
    1. **Paris**: Located in the north of France, Paris is the capital of France and is one of the most important cities in Europe. It is known for its rich history, art, and culture, including the Eiffel Tower.
    
    2. **Lyon**: Located in the north-west of France, Lyon is the second-largest city in France. It is known for its arts scene, particularly the Musée d'Orsay, which
    ===============================
    Prompt: The future of AI is
    Generated text:  not only dependent on the capabilities of today’s AI systems, but also on the future of the data they are fed with. In this talk, I will give an overview of the potential of using AI for mapping the future of AI, emphasizing the need for better data management, the importance of incorporating diverse perspectives, and the importance of enabling AI systems to learn from their environment and learn from each other.
    
    In particular, I will focus on the challenges and opportunities that lie ahead in the future of AI, and the key principles and tools that can help us address them. This includes the importance of AI systems being able to learn from their environment,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I enjoy [insert a short, positive, enthusiastic statement about your hobby or activity]. I'm always looking for new experiences and adventures to share with you. What's your favorite book or movie? I love [insert a short, positive, enthusiastic
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is home to the iconic Eiffel Tower and the Louvre Museum. It is also the seat of the French government and the country's cultural and political capital. Paris is known for its rich history, diverse culture, and beautiful architecture, making it a popular tourist destination. The city is also home to many famous landmarks, including the Notre-Dame Cathedral and the Champs-Élysées. Paris is a vibrant and dynamic city that is constantly changing and evolving, with a rich history and a vibrant culture that continues to inspire and captivate people around the world. The city is also known for its delicious cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals.
    
    2. AI-powered healthcare: AI is already being used in healthcare to diagnose and treat diseases, predict patient outcomes, and personalize treatment plans. As AI technology continues to improve, we can expect to see even more personalized and
    


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
    Generated text:  [Name] and I'm a [Age] year-old software engineer with a [field of interest] background in [your field of interest]. My goal is to help solve complex problems and provide a positive impact. I'm always looking to learn new skills and technologies to stay up-to-date with the latest advancements. I enjoy working on projects that require creativity and problem-solving, and I'm excited to be part of a team that is passionate about creating innovative solutions. Thank you for considering me for a job interview! [Your name], [Your position] [Your company name]. How can I help you today? Looking forward to the opportunity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    # The code below is used to generate a random fact about the capital city of France. However, it is not related to the factual statement. Instead, it generates a random fact about France that you can share with your friends about its capital.
    
    from random import choice
    
    capitals = {
        'Paris': 'The capital and most populous city of France, known for its historic Notre-Dame Cathedral, museums, and cafes.',
        'Rome': 'The capital and largest city of Italy, famous for its ancient Colosseum, Roman Forum, and numerous historical sites.',
        'London': 'The capital and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, with many potential trends to consider. Here are some possible future trends in AI:
    
    1. Increased focus on ethical AI: As AI becomes more advanced, there will be a growing need to consider its impact on society. Ethical AI is increasingly being recognized as a necessary component of AI development, with the goal of creating AI that is fair, transparent, and responsible for human decisions.
    
    2. Increased integration with other technologies: AI will continue to be integrated with other technologies, such as IoT and blockchain, to create more complex and powerful AI systems. This integration could lead to new opportunities for AI-based applications, such as smart


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

     name

    ]

     and

     I

     am

     a

     [

    insert

     age

    ,

     gender

    ,

     and

     occupation

    ].

     I

     am

     a

     [

    insert

     profession

     or

     skill

    ]

     and

     have

     always

     been

     a

     [

    insert

     hobby

    ,

     interest

    ,

     or

     passion

    ].

     I

     am

     [

    insert

     your

     profession

     or

     occupation

    ]

     and

     I

     enjoy

     [

    insert

     something

     you

     like

     to

     do

    ].

     I

     am

     a

     [

    insert

     your

     age

    ,

     gender

    ,

     and

     occupation

    ]

     and

     I

     am

     passionate

     about

     [

    insert

     something

     you

     believe

     in

    ].

     I

     am

     [

    insert

     your

     age

    ,

     gender

    ,

     and

     occupation

    ]

     and

     I

     am

     a

     [

    insert

     your

     profession

     or

     occupation

    ]

     and

     I

     have

     a

     strong

     work

     ethic

     and

     a

     love

     for

     [

    insert

     something

     you

     love

    ].

     I

     am

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    -

     **

    (

    A

    )**

     It

     is

     the

     largest

     city

     in

     France

    .

     


    -

     **

    (

    B

    )**

     It

     is

     a

     global

     financial

     center

    .

     


    -

     **

    (

    C

    )**

     It

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     


    -

     **

    (

    D

    )**

     It

     is

     a

     modern

     met

    ropolis

    .

     
    


    **

    Answer

     the

     question

    **

     based

     on

     the

     information

     given

     in

     the

     passage

    .

     
    


    **

    Correct

     Answer

    **

    :


    (D

    )

     It

     is

     a

     modern

     met

    ropolis

    .

     
    


    The

     passage

     states

     that

     "

    Paris

     is

     the

     largest

     city

     in

     France

    "

     and

     adds

     that

     "

    Paris

     is

     a

     modern

     met

    ropolis

    ,

     "

     which

     means

     it

     is

     a

     modern

    ,

     large

     city

    .

     
    


    -

     Option

     A

     is

     incorrect

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     extremely

     bright

     and

     diverse

    .

     Here

     are

     some

     of

     the

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Human

    -A

    I

     collaboration

    :

     In

     the

     future

    ,

     human

     AI

     will

     become

     more

     and

     more

     integrated

     with

     machines

    .

     This

     could

     involve

     developing

     new

     AI

     technologies

     that

     work

     best

     with

     humans

     and

     vice

     versa

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     will

     become

     more

     advanced

    ,

     and

     will

     be

     able

     to

     drive

     themselves

     on

     the

     road

    .

     This

     will

     change

     the

     way

     people

     travel

     and

     change

     the

     way

     businesses

     operate

    .
    


    3

    .

     Predict

    ive

     analytics

    :

     Predict

    ive

     analytics

     will

     become

     more

     advanced

    ,

     and

     will

     be

     able

     to

     predict

     future

     trends

     and

     events

     based

     on

     data

    .
    


    4

    .

     Virtual

     assistants

    :

     Virtual

     assistants

     will

    



```python
llm.shutdown()
```
