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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]


    2026-05-11 07:27:14,984 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 07:27:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.41it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.43it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 22.71it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 31.74it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 31.74it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 31.74it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.41it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.71it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.28it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.19it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 34.01it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.39it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:00<00:00, 42.22it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:00<00:00, 42.22it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 42.22it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  60%|██████    | 35/58 [00:01<00:00, 42.22it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.09it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=8 avail_mem=75.67 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s] Capturing num tokens (num_tokens=4 avail_mem=75.66 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=4 avail_mem=75.66 GB): 100%|██████████| 58/58 [00:01<00:00, 38.85it/s]


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
    Generated text:  10000. I'm an AI language model. I'm available 24/7, so I can answer any questions you have. How can I help you today? Is there anything specific you would like to ask or discuss? Please feel free to ask me anything! Here is a question for you: 
    
    What is the most challenging part of being an AI language model? I'm curious to know why I should be excited about being an AI model and not be scared of the unknown. 
    When you were first created, you were meant to be a language model. But now that I'm here, I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking government official, but his job does not necessarily imply that he is a man. The president of the United States is also a man. Therefore, his job does not necessarily imply that he is a man. Which of the following is the most logically sound conclusion?
    
    A. His job does not necessarily imply that he is a man.
    B. His job does imply that he is a man.
    C. His job does not imply that he is not a man.
    D. His job does imply that he is not a man.
    E. His job does imply that he is not a man or vice versa.
    
    To solve this problem
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the city that bears the name of the ancients. The city is actually a part of the country of France and its official name is the City of Paris. The capital of France is not only the only capital city of France, but it is also the largest city in the country.
    The capital of France is a city and, therefore, the only capital city of the country, as the other capitals are cities. The capital of France is located in a region called Île-de-France, which is a metropolitan region in the south of France. The capital of France is considered to be a country where people speak French. In
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be shaped by the research and development of new technologies and processes. To make progress, researchers need to be able to effectively design and implement sophisticated algorithms, processes, and systems that can predict and analyze data, detect anomalies and outliers, and make informed decisions based on complex information. In this paper, we will explore how the latest advances in machine learning algorithms, such as deep learning and neural networks, are changing the landscape of AI research and how they can be used to improve the performance of complex systems.
    Machine learning is a field of computer science that focuses on algorithms that enable machines to learn from data and make predictions or decisions without being


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As an AI language model, I don't have a physical presence, but I'm always ready to assist you with any questions or tasks you may have. How can I help you today? What's your name? What's your job title? What's your company name? What's your experience level? What's your current role? What's your current location? What's your current job title? What's your current company name? What's your current
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to the French Parliament and the French Parliament Building. It is the largest city in France by population and is a major economic and political center in Europe. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in manufacturing, transportation, and other industries, where it can perform tasks that were previously done by humans. This will lead to increased efficiency and productivity, but it will also create new jobs and raise concerns about job displacement.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be increased concerns about privacy and security. This will require new regulations and technologies to protect against data breaches and
    


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
    Generated text:  Alex. I'm a writer, and I love to write about people and stories. I love the idea of becoming a writer, and I'd like to tell you a little about myself.
    Hello, my name is Sarah. I'm a writer, and I love to write about people and stories. I love the idea of becoming a writer, and I'd like to tell you a little about myself.
    Name: Alex
    Occupation: Writer
    Age: 30
    Nationality: American
    Profession: Writer
    Favorite hobby: Traveling
    Favorite book: "To Kill a Mockingbird" by Harper Lee
    Favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital and largest city of France. It is known for its iconic landmarks, vibrant culture, and delicious cuisine. The city has a rich history dating back to ancient times and is home to the Louvre Museum, the Eiffel Tower, and many other notable structures. Paris is also known for its fashion industry and is a hub for important cultural events such as the Eiffel Tower Parade. The French people have a strong sense of national identity and enjoy a wide variety of foods and drinks. The city is also known for its music scene, including the famous Louvre Jazz Festival. With its beautiful architecture, beautiful
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and dynamic, with many potential developments shaping how we interact with machines and control them. Here are some possible future trends in AI:
    
    1. Increased depth of understanding: As AI improves, it will become more capable of understanding human emotions, thoughts, and intentions, allowing for more personalized and context-aware interactions.
    
    2. Higher levels of autonomy: AI will become more capable of making decisions on its own, reducing the need for humans to provide guidance or oversight. This will enable machines to handle a wider range of tasks and environments, from autonomous driving to complex decision-making in healthcare and finance.
    
    3. AI with consciousness: While AI is currently


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

     character

    's

     name

    ].

     I

     am

     a

     self

    -pro

    claimed

     "

    Mini

    ature

     Marvel

    "

     with

     a

     passion

     for

     pop

     culture

     and

     storytelling

    .

     I

     love

     exploring

     new

     settings

    ,

     characters

    ,

     and

     genres

    ,

     and

     am

     always

     eager

     to

     learn

     about

     the

     world

     around

     me

    .

     I

    've

     been

     writing

     for

     

    8

     years

    ,

     and

     I

    've

     written

     for

     several

     major

     publishers

    .

     I

    'm

     confident

     that

     I

     can

     create

     engaging

     and

     well

    -c

    rafted

     stories

     for

     my

     audience

    ,

     and

     I

    'm

     excited

     to

     dive

     into

     the

     creative

     process

     and

     see

     where

     my

     imagination

     takes

     me

    .

     What

     better

     way

     to

     start

     than

     with

     a

     self

    -int

    roduction

     that

     highlights

     my

     passion

     for

     writing

     and

     my

     desire

     to

     share

     my

     own

     experiences

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     the

     second

    -largest

     city

     in

     the

     United

     Kingdom

    .

     The

     city

     has

     a

     population

     of

     approximately

     

    1

    .

    3

     million

     people

     and

     is

     home

     to

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     known

     for

     its

     rich

     cultural

     heritage

     and

     diverse

     cuisine

    ,

     including

     its

     famous

     art

     museums

    ,

     theaters

    ,

     and

     museums

    .

     The

     city

     is

     also

     home

     to

     numerous

     cultural

     institutions

     and

     festivals

     throughout

     the

     year

    .

     Overall

    ,

     Paris

     is

     a

     fascinating

     and

     vibrant

     city

     with

     a

     rich

     history

     and

     culture

     that

     continues

     to

     capt

    ivate

     visitors

     from

     around

     the

     world

    .

     Based

     on

     the

     passage

     above

    ,

     What

     is

     the

     population

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     robotics

    .

     Some

     possible

     future

     trends

     include

    :
    


     

     

    1

    .

     Increased

     efficiency

     and

     accuracy

     in

     AI

     algorithms

     and

     decision

    -making

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     may

     expect

     to

     see

     greater

     efficiency

     and

     accuracy

     in

     how

     it

     makes

     decisions

    .

     This

     could

     lead

     to

     more

     automated

     and

     efficient

     systems

    ,

     such

     as

     self

    -driving

     cars

    ,

     predictive

     maintenance

    ,

     and

     virtual

     assistants

    .


     

     

    2

    .

     More

     widespread

     use

     of

     AI

     in

     everyday

     life

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

     may

     see

     more

     widespread

     use

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     education

    ,

    



```python
llm.shutdown()
```
