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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.26it/s]


    2026-05-10 04:49:02,913 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 04:49:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.43s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.38it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 15.78it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 24.74it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 24.74it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 24.74it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 24.74it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 24.74it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s]

    Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 24.74it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 33.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.83it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.81it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.88it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.88it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.88it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.88it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 24.88it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.93it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.93it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.93it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.93it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:01<00:01, 27.93it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.07it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.74it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.33it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 42.33it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 32.47it/s]


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
    Generated text:  Amy and I am a big fan of the comedy industry. I have a great passion for comedy and I like to tell jokes about other people's lives. I am a comedy buff and I have a lot of resources to assist with my work. I read books and watched films to learn how to tell jokes. I have a lot of friends who are comedians and I really enjoy making friends with them. I am also great at coding and I use code to help me improve my writing and humor. I also have an inbox full of jokes and stories. 
    
    My mission is to improve my writing, humor and knowledge in the comedy industry.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a popular title, for the U. S. president has been elected every four years since 1789. Every president has a slightly different story to tell. At first, many Americans wanted to see an African-American candidate for president. The first black candidate was Thomas Jefferson, who won the 1800 presidential election. Jefferson was elected president in 1800 because he was a highly educated and wealthy businessman. Thomas Jefferson's family moved to Virginia in 1785. He was educated by his grandfather and his father. When Jefferson was about 20 years old, he worked for a church during
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, where the iconic Eiffel Tower stands tall. And while you may be familiar with Paris, you may not know that it is actually the second largest city in the world by population. Located in the western part of France, the metropolis is known for its rich history, stunning architecture, and diverse cultural scene. The city is also known for its impressive landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and Montmartre. As you explore the city, take a tour of its many attractions, such as the Louvre Museum, the Musée d'Orsay, and the Musée des Arts
    ===============================
    Prompt: The future of AI is
    Generated text:  changing rapidly, and it is imperative for developers to keep up with the latest advancements in the field. One of the most exciting areas of AI research involves the development of highly efficient and accurate machine learning models, such as those used in natural language processing and computer vision.
    In this article, we will explore the latest breakthroughs in machine learning and their impact on the field of AI. We will also discuss some of the most promising areas for future research and innovations in this area.
    At the heart of machine learning lies the concept of a model, which is a set of rules and algorithms that can be used to make predictions or decisions. These models


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a major tourist destination and is known for its fashion industry, with Paris Fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management and fraud detection. As AI technology
    


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
    Generated text:  _______ and I'm a/an _______.
    
    Sure, here's a short, neutral self-introduction for a fictional character:
    
    Hi, my name is [Your Name] and I'm a/an [Character Type]. I enjoy [Your Passion]. And I love [Your Hobby/Activity]. I'm [Your Character Trait]. And I'm always [Your Personal Quote]. How is it going, [Friend or Other Character's Name]? Let's get to know each other better! 📝❤️📚📚📚
    
    Would you like me to expand on any of these points or provide more details? I'd love to know more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Rationale: I only include Paris, the main city in France, in the list of the capital city. If the other cities in France were also included, it would become a larger list. As I am limited to 128 characters, I only chose Paris to keep the main city in France within the 128-character limit. The response adheres to the requirement of being concise and factual. The statement includes the capital city of France as a specific example of a capital city.
    You are an AI assistant that helps people find information. How can I improve my ability to write concisely and accurately?
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by increasing integration with other technologies and a continued focus on ethical and safety concerns. Here are some potential trends in AI that could shape the industry in the coming years:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies such as the Internet of Things (IoT), the cloud, and edge computing. This integration will lead to new opportunities for AI to be applied in areas such as healthcare, transportation, and smart cities.
    
    2. Enhanced capabilities: AI is expected to become more capable, allowing it to perform tasks that were previously impossible or difficult to achieve. This could include tasks such as


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

    'm

     a

     [

    fill

     in

     the

     blank

     with

     appropriate

     age

     range

     for

     your

     character

    ,

     e

    .g

    .,

     teenager

    ,

     adult

    ,

     young

     adult

    ,

     etc

    .]

    .
    


    I

    'm

     passionate

     about

     [

    mention

     a

     hobby

     or

     interest

     you

     enjoy

    ,

     such

     as

     sports

    ,

     music

    ,

     reading

    ,

     or

     writing

    ].

     I

    'm

     a

     [

    fill

     in

     the

     blank

     with

     appropriate

     age

     range

     for

     your

     character

    ,

     e

    .g

    .,

     teenager

    ,

     adult

    ,

     young

     adult

    ,

     etc

    .]

     who

     is

     always

     looking

     to

     learn

     new

     things

     and

     grow

     in

     my

     career

    ,

     life

    ,

     and

     personality

    .

     I

    'm

     confident

    ,

     but

     I

     don

    't

     always

     know

     where

     to

     start

    ,

     and

     I

    'm

     willing

     to

     take

     risks

     to

     pursue

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     lights

    ,

     elegant

     gardens

    ,

     and

     world

    -ren

    owned

     art

     and

     culture

    .

     It

     has

     a

     rich

     and

     complex

     history

    ,

     and

     is

     the

     country

    's

     largest

     city

     with

     a

     population

     of

     over

     

    2

     million

    .

     Paris

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     E

    iff

    el

     Tower

    ,

     and

     Mont

    mart

    re

    .

     It

     is

     also

     home

     to

     several

     world

    -class

     museums

    ,

     theaters

    ,

     and

     restaurants

    ,

     making

     it

     a

     top

     tourist

     destination

    .

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     with

     a

     rich

     cultural

     heritage

    ,

     and

     is

     known

     for

     its

     famous

     annual

     events

     such

     as

     the

     E

    iff

    el

     Tower

     Opening

     Ceremony

     and

     the

     Car

    ne

    val

     de

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

    ,

     and

     there

     are

     numerous

     possibilities

     and

     future

     directions

     that

     are

     promising

    .

     Here

     are

     some

     possible

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     autonomy

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     will

     become

     more

     autonomous

    ,

     capable

     of

     making

     decisions

     on

     its

     own

     without

     human

     intervention

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     operations

    ,

     as

     AI

     can

     handle

     tasks

     that

     would

     otherwise

     require

     human

     oversight

    .
    


    2

    .

     Enhanced

     intelligence

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     sophisticated

    ,

     with

     better

     ability

     to

     learn

     and

     adapt

     to

     new

     situations

    .

     This

     could

     lead

     to

     more

     sophisticated

     forms

     of

     artificial

     intelligence

    ,

     capable

     of

     solving

     complex

     problems

     and

     making

     predictions

     about

     future

     events

    .
    


    3

    .

     Increased

     integration

    :

     AI

    



```python
llm.shutdown()
```
