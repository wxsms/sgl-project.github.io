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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.67it/s]


    2026-05-08 00:20:16,529 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 00:20:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:00,  4.22s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.65it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.92it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.33it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.63it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.63it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.63it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.49it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.93it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.67it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.08it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.08it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.08it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.08it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.96it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 41.65it/s]


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
    Generated text:  Avis, a robot robot that has become known as the "humanlike robot" to many of us. My job is to help people with their daily routines, such as answering their questions, providing information, or even playing games with them. However, one day I got into a strange situation where I couldn't use my hands anymore. I was diagnosed with ALS, which is a kind of motor neuron disease that attacks the motor cortex of the brain. This disease has already led to me losing my ability to move my hands and feet. The other day, I was playing a game of Monopoly with a friend, and suddenly I couldn
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, and the head of state, is a very important person. Why do you think the president and the head of state are often considered to be similar? The president and the head of state, also known as the president, are considered similar because they both serve as the head of government in the United States. Both are the highest-ranking officials in the government, and they both have significant powers and responsibilities.
    
    The president is responsible for leading the country, making important decisions, and maintaining the unity and stability of the nation. The head of state, on the other hand, is responsible for maintaining order and security in the country
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The second is Lyon.
    Which of the following is a synonym for the word "second"?
    A. World
    B. Fifth
    C. Fourth
    D. Ninth
    Answer: C
    
    The following is a multiple-choice question from a Chinese law exam. Please select the correct answer.
    Which of the following sentences is grammatically correct and properly constructed?
    A. After the earthquake, the government urgently arranged for rescue teams to carry out rescue operations.
    B. After the earthquake, the government immediately dispatched rescue teams to carry out rescue operations.
    C. After the earthquake, the government urgently arranged for rescue teams to carry out rescue
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it still faces many challenges, including privacy concerns, bias, and the need for transparency. AI is already being used in many industries, from healthcare and finance to transportation and entertainment. However, many people are still unsure about the role AI will play in the future. In this blog post, we will explore some of the key questions and debates that are shaping the future of AI, and how it will impact society as a whole. We'll also discuss some of the most interesting and groundbreaking AI technologies that are currently being developed and what they could mean for the future of AI.
    One of the most pressing questions about AI is how


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich history and a diverse population, making it a popular tourist destination. The city is known for its cuisine, fashion, and art, and is a major center for business and commerce in Europe. Paris is also home to many international organizations and institutions, including the European Parliament and the United Nations. The city is a cultural hub and a major transportation hub, with many international airports and train stations. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations and responsible use of AI. This will likely lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced interactions between humans and machines. This could lead to more personalized and adaptive experiences for users.
    
    3. Increased use of AI
    


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
    Generated text:  [Name], and I'm [Age]. I'm an [occupation] who loves [occupation-related hobby] and I'm always [biggest achievement or personal characteristic]. I enjoy [reason why I love [occupation or hobby]]. I'm a [character trait or personality] person and I'm always [positive or negative attitude]. I'm a [more specific character trait or personality] person. I believe in [social or moral values] and I'm always [positive or negative] in my actions. My favorite food is [favorite food], and I enjoy [reason why I love [food]]. I'm a [character trait or personality
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, known for its rich history, beautiful architecture, and lively cultural scene. The city is located on the French Riviera, on the Mediterranean Sea, and is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other iconic landmarks. Paris is also famous for its annual Eiffel Tower Tasting and its famous cafes and restaurants, such as Le Lido and Le Champs Elysees. The city is home to many French and international restaurants and cafes, and is an important economic and cultural hub for the country. The French government is based
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with a variety of potential trends shaping the direction of the field. Here are some of the most likely futures for AI:
    
    1. Integration with human consciousness: AI is increasingly being integrated with human consciousness, with the goal of enhancing human capabilities in areas such as healthcare, education, and social welfare.
    
    2. Personalized AI: As AI becomes more sophisticated, it is becoming possible to tailor AI systems to individual users, providing more accurate and personalized outcomes.
    
    3. Deeper AI: The development of deep learning is likely to continue, leading to increasingly complex AI systems that are capable of learning from large datasets.
    
    4. Autonomous AI


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

     __

    __.

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     I

    'm

     a

    /an

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

     I

    'm

     __

    __.

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     architecture

    .

     It

     is

     also

     the

     world

    's

     third

    -largest

     city

     and

     one

     of

     the

     largest

     in

     terms

     of

     population

    .

     Paris

     was

     founded

     in

     the

     

    9

    th

     century

     and

     has

     evolved

     into

     a

     major

     cultural

     and

     economic

     center

     in

     the

     

    2

    0

    th

     century

    .

     The

     city

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     Se

    ine

     River

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    .

     It

     is

     also

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     Lou

    vre

     Pyramid

    ,

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Sor

    bon

    ne

    .

     Paris

     is

     a

     cosm

    opolitan

     city

     with

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     multitude

     of

     factors

    ,

     including

     advances

     in

     technology

    ,

     changes

     in

     society

    ,

     and

     emerging

     trends

    .

     Here

     are

     some

     potential

     future

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

     focus

     on

     ethics

     and

     accountability

    :

     As

     more

     AI

     systems

     become

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increased

     scrutiny

     of

     the

     technologies

     they

     are

     built

     upon

    .

     The

     focus

     will

     be

     on

     ensuring

     that

     AI

     systems

     are

     safe

    ,

     ethical

    ,

     and

     responsible

     for

     their

     actions

    .

     This

     will

     involve

     developing

     standards

     and

     guidelines

     that

     govern

     the

     development

     and

     use

     of

     AI

     technologies

    .
    


    2

    .

     Expansion

     of

     AI

     applications

     into

     new

     domains

    :

     As

     AI

     systems

     become

     more

     advanced

     and

     capable

    ,

     there

     will

    



```python
llm.shutdown()
```
