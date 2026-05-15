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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.02it/s]


    2026-05-15 03:32:43,553 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 03:32:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.84it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.28it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 30.03it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 30.03it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 30.03it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 30.03it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 30.03it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 30.03it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 38.35it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 38.35it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 38.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.91it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.55it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.37it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.37it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.37it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.20it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.82it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.82it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.82it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.86it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.76it/s]


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
    Generated text:  Rodolfo Ponce, a 21-year-old student majoring in law at the University of California, Los Angeles, and I have a short video at https://www.youtube.com/watch?v=3eN84X1k8Mk, and I was wondering if you could provide some feedback on it. Thank you very much for your time. Rodolfo Ponce
    Certainly! I'd be happy to provide feedback on your video. Please go ahead and share your content, and I'll be happy to give you constructive criticism and suggestions for improvement. Is there anything specific you'd like me to highlight or discuss
    ===============================
    Prompt: The president of the United States is
    Generated text:  a permanent officer of the United States government. He is responsible for the budget of the country, which is composed of the sum of the United States' income and expenditures. The president is also responsible for the national defense and security, which are the responsibility of the armed forces. The president is also responsible for the administration of the country, which includes making decisions on laws and regulations that affect the country's economic and social policies. The president is also responsible for the execution of the laws passed by Congress, and for the administration of the country's judicial process. The president's term of office is only one year, and he must be elected for
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Paris, France is located on the right bank of the Seine River, 33 kilometers (20 miles) west of the river and 13 kilometers (8 miles) south of the city centre.
    The city is centered on the Arve Tower, the oldest part of the city, which was originally built around 977 CE and is still used today as the seat of government.
    The city’s port is at the mouth of the Seine, and this port is also home to the French flag.
    The city is known for its beautiful architecture and its rich history. The Louvre, the Musée d
    ===============================
    Prompt: The future of AI is
    Generated text:  dependent on how we get the technology right
    
    As an AI language model, I can't provide an opinion, but I can explain the potential impact of AI on society and its current state of development.
    
    The future of AI is expected to revolutionize many aspects of human life, from healthcare to transportation to education. AI systems are capable of processing and analyzing vast amounts of data to provide insights and make decisions that can help improve efficiency, save lives, and save money.
    
    However, it's important to note that AI is not always developed with the best interests of society in mind. While AI systems have the potential to reduce the burden of physical labor


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. What brings you to [company name] and what makes you a good fit for the position? I'm excited to learn more about your company and how I can contribute to your success. What are your goals for the future? I'm looking forward to working with you and helping you achieve your goals. What's your favorite part
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a cultural and economic hub, known for its rich history and diverse population. Paris is a major tourist destination and a major center of politics and government. It is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a major center for the arts. The city is also known for its cuisine, including its famous croissants and its famous French wine. Paris is a city of contrasts, with its modern architecture and history, as well as its vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with other technologies: As AI becomes more integrated with other technologies, such as sensors and IoT devices, it is likely that AI systems will become more capable of performing tasks that were previously impossible or difficult to achieve.
    
    2. Greater emphasis on ethical considerations: As AI systems become more sophisticated, there will be a
    


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
    Generated text:  [Name] and I'm a [type of work] expert. I've been [number of years] in this field and have developed my skills through [reasons]. If you have any questions or need advice, feel free to reach out. I look forward to working with you! #Self-Introduction
    
    Hello, my name is [Name] and I'm a [type of work] expert. I've been [number of years] in this field and have developed my skills through [reasons]. If you have any questions or need advice, feel free to reach out. I look forward to working with you! #Self
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historic city with a rich history dating back over 2,000 years. The city has a diverse population and a vibrant culture, including many famous landmarks and museums. The city is known for its romantic and historical atmosphere, and is home to numerous museums, theaters, and parks that showcase the city's rich history and culture. Paris is also a global hub for art, design, and entertainment, attracting millions of visitors each year. The city is known for its elegant architecture, iconic landmarks, and vibrant nightlife. Its status as the capital of France and its role as a cultural and economic center has made Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, with potential trends that could shape the direction of this field. Here are some of the possible future trends in AI:
    
    1. Advancements in machine learning algorithms: One of the main trends in AI is the development of more advanced machine learning algorithms that can perform tasks more efficiently and accurately than current methods. This could include algorithms that can learn from large amounts of data, recognize patterns in complex data, and make predictions based on that data.
    
    2. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there is a growing emphasis on ethical considerations surrounding its development and use. This includes issues such as


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

     Sarah

    ,

     a

     freelance

     writer

     with

     over

     five

     years

     of

     experience

     in

     the

     creative

     industry

    .

     I

    'm

     an

     avid

     reader

    ,

     traveler

    ,

     and

     visual

     artist

    ,

     and

     I

     work

     on

     multiple

     projects

     simultaneously

     while

     editing

     and

     revis

    ing

     my

     own

     stories

    .

     I

    'm

     always

     looking

     for

     new

     ways

     to

     push

     the

     boundaries

     of

     storytelling

     and

     make

     my

     work

     stand

     out

     in

     the

     competitive

     world

     of

     creative

     writing

    .

     I

     believe

     in

     using

     my

     passion

     and

     creativity

     to

     bring

     stories

     to

     life

    ,

     no

     matter

     how

     unusual

     or

     unconventional

     they

     may

     seem

    .

     Thank

     you

     for

     considering

     me

     for

     a

     potential

     writer

     in

     your

     own

     projects

    .

     Let

    's

     chat

    ,

     will

     you

    ?

     (

    I

     sound

     like

     I

     have

     a

     great

     deal

     to

     offer

     in

    
    
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

     France

     and

     the

     eighth

     largest

     in

     the

     world

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     music

    ,

     fashion

    ,

     and

     cuisine

    .

     Paris

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     historic

     and

     cultural

     landmarks

    .

     Its

     famous

     landmarks

     include

     Mont

    mart

    re

    ,

     Sac

    ré

    -C

    œur

     Basil

    ica

    ,

     and

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     the

     seat

     of

     government

    ,

     economy

    ,

     and

     culture

     for

     France

     and

     is

     a

     major

     tourist

     attraction

     worldwide

    .

     The

     city

     is

     also

     a

     center

     of

     international

     politics

     and

     diplomacy

    ,

     with

     several

     international

     organizations

     headquartered

     there

    .

     According

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     variable

    .

     However

    ,

     some

     possible

     trends

     that

     are

     likely

     to

     continue

     include

    :
    


    1

    .

     Increased

     automation

     and

     machine

     learning

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     be

     able

     to

     perform

     tasks

     that

     currently

     require

     human

     intervention

    ,

     such

     as

     diagn

    osing

     diseases

    ,

     making

     decisions

    ,

     and

     autom

    ating

     manufacturing

     processes

    .
    


    2

    .

     Deep

     learning

    :

     This

     is

     a

     subset

     of

     AI

     that

     involves

     the

     use

     of

     neural

     networks

     to

     learn

     from

     data

    .

     It

     is

     expected

     to

     become

     more

     advanced

     and

     powerful

     in

     the

     coming

     years

    ,

     leading

     to

     breakthrough

    s

     in

     areas

     such

     as

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     robotics

    .
    


    3

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

    



```python
llm.shutdown()
```
