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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


    2026-05-12 06:28:09,074 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 06:28:09] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.43it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.43it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.22it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.09it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.86it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 15.35it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 15.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 15.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 11.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 11.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:04, 11.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   7%|▋         | 4/58 [00:00<00:04, 11.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.97it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.02it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.56it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.73it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.73it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 40.80it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.71it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.56it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.56it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.06it/s]


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
    Generated text:  Remy and I'm a second year medical student at the University of Toronto. I have a background in gerontology and behavioral science, and I'm interested in understanding the effects of sleep deprivation and its impact on mental health and cognitive functioning. I have a great interest in the relationship between sleep deprivation and depression, so I'm planning to do a thesis on this topic. To that end, I've started researching the effects of sleep deprivation on mental health and cognitive functioning. However, I'm not sure how to start the research on my own. Could you please provide me with some guidance on how to begin this research?
    Certainly! Here are
    ===============================
    Prompt: The president of the United States is
    Generated text:  a new term, and the incumbent (current) president is Hillary Clinton. The president is in his second term in the White House. How many terms have the incumbent president been in the position? To determine how many terms the incumbent president has been in the position, we need to consider the nature of the term structure of an elected president. Generally, an elected president is in office for a single term, unless they are re-elected immediately after the end of their term. However, the situation you described suggests that the incumbent, Hillary Clinton, has been in office for more than one term.
    
    Given that the president is a new term and the
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lyon
    C. Bruxelles
    D. London
    Answer:
    A
    
    During the process of adding salt, if you discover that the salt has spilled on the table, which of the following actions is correct?
    A. Quickly cover it with a cloth and pour out the remaining salt
    B. Cover it with a wet cloth and pour out the remaining salt
    C. Cover it with a wet cloth and let the salt settle naturally
    D. Cover it with a wet cloth and pour out the remaining salt
    Answer:
    D
    
    Which of the following statements about the subject of study in economics is
    ===============================
    Prompt: The future of AI is
    Generated text:  not about using it for personal gain but about using it to improve people’s lives. By harnessing the power of AI, organizations can gain insights into customer behavior and preferences, enabling better decision-making and enhancing customer satisfaction. AI can also be used to predict and prevent potential issues in the business environment, leading to cost savings and improved operational efficiency. The role of AI in creating a more sustainable future is also significant. By analyzing data, AI can identify sources of carbon emissions and suggest sustainable solutions, thereby reducing environmental impact. The possibilities of AI in improving the quality of life for everyone is immense. AI can be used to detect and address health


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


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also known for its rich history, art, and cuisine. Paris is a popular tourist destination and a cultural hub in France. The city is home to many international organizations and institutions, including the French Academy of Sciences and the French National Library. Paris is a major transportation hub and a major economic center in France. It is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could emerge in the coming years:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be a growing need for measures to protect privacy and security. This could include measures to
    


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
    Generated text:  [Name], and I'm [Age]. I’m a [Position] with [Company/Organization]. I am passionate about [Occupation] and have been learning and growing in this field for [Number of Years]. I’m confident in my abilities and I’m always looking to learn new things. I’m eager to share my knowledge and experience with anyone who is interested in pursuing a career in [Occupation]. Thank you for taking the time to meet me! 🎉😊
    
    As an AI language model, I'm designed to understand and respond to various types of questions, including personal information, workplace details, and technical queries.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich cultural heritage and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    You are to construct a response in Spanish. Las ciudades son objetos de valor que brindan vida, confort y un lugar para vivir. Las ciudades también constituyen un elemento de la cultura. Por ejemplo, en Francia, la capital, Francia, es conocida por su rica historia cultural y sus monumentos icónicos, como el Eiffel Tower, el Prado, y la Catedral de Notre-Dame. También es importante mencionar
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements and growth in several key areas. Here are some possible future trends:
    
    1. Increased integration with human decision-making: AI is already being integrated into a wide range of systems, from financial and healthcare applications to autonomous vehicles. As this technology continues to improve, it is likely to become more integrated with human decision-making processes, allowing for more complex and nuanced interactions between machines and humans.
    
    2. Enhanced natural language processing: AI is already capable of understanding and generating human language, but there's still much more to be learned in this area. As AI continues to improve, it is likely to be able to better understand and


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

     Alex

    ,

     and

     I

    ’m

     a

     writer

    .

     I

     love

     to

     write

     about

     the

     world

     around

     me

    ,

     and

     I

    ’m

     always

     trying

     to

     come

     up

     with

     new

     ideas

     and

     stories

    .

     I

    ’m

     always

     open

     to

     new

     experiences

     and

     have

     a

     love

     for

     exploring

     the

     world

     from

     a

     creative

     point

     of

     view

    .

     How

     do

     you

     want

     to

     meet

    ?

     For

     what

     would

     you

     like

     to

     become

    ?

     And

     what

     is

     your

     writing

     style

    ?

     I

    ’m

     always

     looking

     for

     new

     writing

     projects

     and

     would

     love

     to

     hear

     from

     you

    .

     So

    ,

     what

    's

     your

     name

     and

     what

     do

     you

     like

     to

     write

     about

    ?


    Alex

    


    Hi

    !

     I

    'm

     a

     writer

    ,

     and

     I

     really

     love

     to

     write

     about

     the

     world

     around

     me

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     the

     third

     largest

     city

     in

     the

     world

    ,

     and

     the

     second

     largest

     city

     in

     Europe

     by

     population

    .

     It

     is

     the

     second

     most

     populous

     city

     in

     the

     European

     Union

     and

     the

     fourth

     largest

     city

     in

     the

     world

     by

     urban

     population

    .

     The

     city

     is

     known

     for

     its

     medieval

     architecture

    ,

     its

     lively

     cultural

     scene

    ,

     and

     its

     elegant

     public

     transportation

     system

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     and

     it

     has

     a

     rich

     history

     dating

     back

     over

     

    2

    ,

    5

    0

    0

     years

    .

     The

     city

     is

     also

     home

     to

     many

     international

     organizations

     and

     institutions

    ,

     including

     the

     French

     Academy

     of

     Sciences

    ,

     the

     French

     National

     Assembly

    ,

     and

     the

     French

     Cons

    ulate

     General

    .

     As

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     and

     significant

     changes

     in

     its

     applications

    ,

     technologies

    ,

     and

     the

     way

     we

     interact

     with

     them

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

     Increased

     focus

     on

     ethics

     and

     responsible

     AI

    :

     There

     will

     be

     a

     greater

     emphasis

     on

     the

     ethical

     and

     moral

     considerations

     of

     AI

     systems

    ,

     including

     their

     potential

     to

     cause

     harm

     or

     misuse

    .

     This

     will

     likely

     involve

     the

     development

     of

     new

     ethical

     frameworks

     and

     standards

     for

     AI

    ,

     as

     well

     as

     greater

     transparency

     and

     accountability

     in

     AI

     decision

    -making

    .
    


    2

    .

     Emer

    gence

     of

     new

     AI

     technologies

    :

     As

     we

     continue

     to

     develop

     new

     technologies

    ,

     we

     may

     see

     the

     emergence

     of

     new

     AI

    -based

     systems

     that

     are

     more

     efficient

    ,

     accurate

    



```python
llm.shutdown()
```
