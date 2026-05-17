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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.32it/s]


    2026-05-17 12:51:10,074 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 12:51:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.53s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.53s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.53s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.53s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.53s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.09it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.26it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.26it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.26it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.26it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.26it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.28it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.43it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:00<00:00, 42.31it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.93it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.13it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.77it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.90it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.26it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.26it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 33.26it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 32.39it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.35it/s]


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
    Generated text:  Ayael. I am a student at KUOW, University of Houston, in Houston, Texas. I am a double major in international politics and cultural studies, and I would like to be the person that I want to go to college with. What are the qualities that make a person a good student and how can I be a good student?
    
    What are your thoughts on education in general and what role does education play in a person's life? Should there be more emphasis on reading and writing? Should there be more emphasis on history or science? Should there be more emphasis on social studies?
    
    Are you a fan of reading?
    
    Is
    ===============================
    Prompt: The president of the United States is
    Generated text:  in a car, traveling at a constant speed of 75 miles per hour. The president leaves for Washington D. C. at 8:00 AM and heads towards his destination. If it takes the president 2 hours to reach Washington D. C., what is the destination of the president?
    The president is traveling at a constant speed of 75 miles per hour.
    It takes the president 2 hours to reach Washington D. C.
    Therefore, the destination of the president is Washington D. C.
    #### 2
    The answer is: 2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and Notre-Dame Cathedral. Located in the city center, it's a bustling metropolis with a rich history and cultural diversity. 
    
    French culture is renowned for its rich history, literature, and art. Paris has been a hub for intellectual and cultural pursuits, with notable figures like Louis XVI, Napoleon, and Victor Hugo all residing there.
    
    The city's population is estimated at around 2.8 million, and it's a major global center for business, finance, and media. Paris is also home to the Louvre Museum, one of the world's most famous art museums.
    
    In
    ===============================
    Prompt: The future of AI is
    Generated text:  shifting towards more ethical and responsible approaches. In this article, we explore the potential applications of AI in healthcare, how these applications can impact the healthcare system, and what ethical considerations need to be taken into account. We will also provide a detailed overview of the current and future ethical guidelines for AI in healthcare, including how they differ from the current ethical guidelines for AI in general.
    The potential of AI in healthcare is vast and has the potential to revolutionize the way we treat diseases and provide better outcomes. One of the key areas where AI can make a significant impact in healthcare is in the diagnosis and treatment of diseases. AI algorithms can analyze medical


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your character, such as "fun-loving, adventurous, and always looking for new experiences."]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a hobby or activity that you enjoy, such as hiking, painting, or playing music]. I'm always looking for new ways to express myself and connect with others. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a vibrant and diverse city with a population of over 2.5 million people. It is a popular tourist destination and a major economic center in Europe. The city is home to many museums, art galleries, and theaters, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into everyday life, from manufacturing to customer service. This will lead to the automation of many tasks, freeing up workers to focus on more complex and creative work.
    
    2. Improved privacy and security: As AI becomes more integrated into our lives, there will be an increased need for privacy and security. This will require the development of new technologies and protocols to protect user data and prevent cyber attacks.
    
    3. Enhanced human-computer interaction:
    


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
    Generated text:  [insert name], and I'm a [insert occupation or role] for [insert company name]. I've always been fascinated by the idea of [insert something you like or think is important to you], and I've always been impressed by the [insert something you find amazing or unique]. I enjoy learning new things and trying new things, and I'm always looking for ways to improve my skills. I'm not afraid to ask questions and try new things, and I'm always eager to learn and grow. I believe in working hard and staying focused on my goals, and I'm always willing to put in the extra effort to achieve them
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Ville de Paris" and the city's official name is "Rue de la République." The city is known as the "City of Light" and is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for fashion, art, and music. The French government, as well as various countries and international organizations, maintain strong diplomatic relationships with Paris. According to the 2020 census, Paris has a population of about 2.1 million people and is the largest city in France by area
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements and developments that will continue to revolutionize the field. Some of the possible future trends in AI include:
    
    1. Increased reliance on machine learning and deep learning: As the complexity of tasks continues to increase, the AI system will need to learn and adapt more quickly. Machine learning and deep learning will become increasingly important for AI to handle a wide range of tasks, including natural language processing, image recognition, speech recognition, and more.
    
    2. Integration with human decision-making: As AI systems become more sophisticated, they will become more capable of making decisions that are more aligned with human values and ethics. This integration


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

    ]

     and

     I

     am

     [

    Age

    ].

     I

     have

     a

     master

    's

     degree

     in

     [

    Major

    ]

     and

     have

     been

     working

     in

     [

    Industry

    ]

     for

     [

    Years

    ]

     years

    .

     I

     am

     passionate

     about

     [

    Occup

    ation

    ]

     and

     have

     always

     been

     driven

     by

     [

    Mot

    ivation

    ],

     which

     drives

     me

     to

     constantly

     learn

     and

     improve

    .

     I

     am

     determined

     to

     achieve

     my

     goals

     and

     help

     others

     succeed

    .

     I

     am

     very

     open

     to

     feedback

     and

     constantly

     striving

     to

     grow

     and

     develop

     as

     a

     person

     and

     as

     a

     leader

    .

     I

     believe

     in

     the

     power

     of

     [

    Mot

    ivation

    ]

     to

     lead

     and

     inspire

     others

    .

     I

     am

     excited

     to

     bring

     my

     skills

    ,

     knowledge

    ,

     and

     passion

     to

     the

     table

     and

     help

     drive

     change

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

    's

     known

     for

     its

     beautiful

     architecture

    ,

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    ,

     and

     its

     historic

     sites

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     cultural

     and

     tourist

     hub

    ,

     with

     a

     rich

     history

     and

     a

     vibrant

     economy

     that

     is

     a

     major

     contributor

     to

     France

    's

     GDP

    .

     It

    's

     also

     a

     major

     economic

     hub

     for

     much

     of

     Europe

    ,

     with

     many

     international

     companies

     and

     institutions

     based

     in

     Paris

    .

     
    


    Paris

     has

     a

     long

     and

     stor

    ied

     history

    ,

     going

     back

     to

     the

     Roman

     Empire

     and

     the

     fall

     of

     the

     Roman

     Empire

    .

     It

     was

     the

     capital

     of

     the

     Frank

    ish

     Kingdom

     in

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     an

     incredibly

     exciting

     and

     constantly

     evolving

     field

     with

     the

     potential

     to

     transform

     many

     areas

     of

     our

     lives

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

     Automation

     and

     Intelligence

    :

     One

     of

     the

     most

     promising

     areas

     for

     AI

     growth

     is

     in

     automation

    ,

     where

     machines

     will

     be

     able

     to

     perform

     tasks

     that

     typically

     require

     human

     expertise

    ,

     such

     as

     data

     analysis

    ,

     healthcare

    ,

     and

     customer

     service

    .
    


    2

    .

     Enhanced

     Intelligence

    :

     AI

     will

     continue

     to

     learn

     and

     improve

    ,

     so

     machines

     will

     be

     able

     to

     learn

     from

     experiences

    ,

     develop

     their

     own

     ways

     of

     working

    ,

     and

     become

     more

     intelligent

     over

     time

    .
    


    3

    .

     Improved

     Privacy

     and

     Security

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     a

     growing

     need

    



```python
llm.shutdown()
```
