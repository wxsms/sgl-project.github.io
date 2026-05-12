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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.02it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.01it/s]


    2026-05-12 05:22:06,544 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 05:22:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:03,  9.81it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 16.45it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 25.14it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 34.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=41.34 GB):   3%|▎         | 2/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=41.33 GB):   3%|▎         | 2/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=41.33 GB):   3%|▎         | 2/58 [00:00<00:03, 15.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.33 GB):   3%|▎         | 2/58 [00:00<00:03, 15.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.33 GB):   9%|▊         | 5/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.31 GB):   9%|▊         | 5/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.31 GB):   9%|▊         | 5/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.30 GB):  16%|█▌        | 9/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.30 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.29 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.28 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=41.26 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s]Capturing num tokens (num_tokens=960 avail_mem=41.27 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s] Capturing num tokens (num_tokens=896 avail_mem=41.27 GB):  31%|███       | 18/58 [00:00<00:01, 35.34it/s]Capturing num tokens (num_tokens=896 avail_mem=41.27 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=832 avail_mem=41.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=768 avail_mem=41.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=704 avail_mem=41.26 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=640 avail_mem=41.25 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=576 avail_mem=41.25 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.62it/s]Capturing num tokens (num_tokens=576 avail_mem=41.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=512 avail_mem=41.24 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=480 avail_mem=41.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=416 avail_mem=41.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=384 avail_mem=41.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.38it/s]Capturing num tokens (num_tokens=384 avail_mem=41.25 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=352 avail_mem=41.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=320 avail_mem=41.24 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=288 avail_mem=41.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=256 avail_mem=41.23 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.04it/s]Capturing num tokens (num_tokens=240 avail_mem=41.23 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.04it/s]Capturing num tokens (num_tokens=240 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=224 avail_mem=41.23 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]

    Capturing num tokens (num_tokens=208 avail_mem=41.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=192 avail_mem=41.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=176 avail_mem=41.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=160 avail_mem=41.22 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=160 avail_mem=41.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=144 avail_mem=41.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=128 avail_mem=41.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=112 avail_mem=41.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=96 avail_mem=41.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=41.20 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=80 avail_mem=41.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=64 avail_mem=41.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=48 avail_mem=41.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=32 avail_mem=41.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=28 avail_mem=41.18 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.34it/s]Capturing num tokens (num_tokens=28 avail_mem=41.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=24 avail_mem=41.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=20 avail_mem=41.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s]

    Capturing num tokens (num_tokens=16 avail_mem=41.18 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=12 avail_mem=41.17 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s]Capturing num tokens (num_tokens=8 avail_mem=41.17 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.48it/s] Capturing num tokens (num_tokens=8 avail_mem=41.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=4 avail_mem=41.17 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=4 avail_mem=41.17 GB): 100%|██████████| 58/58 [00:01<00:00, 36.72it/s]


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
    Generated text:  Jacek and I'm a computer science student at the University of Warsaw. In my spare time, I enjoy playing guitar and drawing. I'm an active member of the Robotics Club and I'm passionate about developing new ways of learning and creating. I also love to cook and try new recipes. Let me know if you're interested in connecting with me! 😊
    Great! That's a great personality! Can you tell me more about your hobbies and how they help you stay active? Sure! As a computer science student, my hobbies have been expanding my skills and knowledge in different areas, such as programming, web development, and
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy, and he has a lot of work to do. The president must first read a lot of books. And then he has to learn about the history and science of the United States. He must also know how to speak to the American people. The president must be able to understand and think about what is important to Americans. The president must make decisions about important matters that affect the whole country. Sometimes the president must be in the White House for a very long time. It's a busy life, but for the president, it's not too hard. He knows how to get his work done. The president has very strict rules
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lyon
    C. Bordeaux
    D. Nantes
    Answer:
    
    A
    
    Female, 27 years old, has had a fever and headache for 2 weeks, and the blood pressure has decreased. Physical examination: T37.3°C, BP150/90mmHg, muffled heart sounds, liver palpable 1.5cm below the costal margin, hard, spleen palpable 3.5cm below the costal margin. The most likely diagnosis for this patient is:
    A. Subacute infective endocarditis
    B. Ac
    ===============================
    Prompt: The future of AI is
    Generated text:  all about the integration of artificial intelligence with the Internet of Things (IoT), according to a new study released today by the Wellcome Trust. The study, entitled “AI in the Internet of Things: A Call to Action”, highlights the significant role the Internet of Things (IoT) can play in enabling AI research and applications.
    The well-being of the human race can benefit immensely from the research and development of AI, according to the study. The analysis indicates that IoT can be a significant asset for the development of AI, as it can collect data from a wide range of sources, such as devices and equipment that are installed in homes


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


    Generated text:  [Name] and I am a [occupation] with [number] years of experience in [field]. I am a [type of person] who is [positive or negative] about [what you do for a living]. I am [positive or negative] about [what you do for a living]. I am [positive or negative] about [what you do for a living]. I am [positive or negative] about [what you do for a living]. I am [positive or negative] about [what you do for a living]. I am [positive or negative] about [what you do for a living]. I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center with a rich history and a diverse population. The city is known for its vibrant nightlife, fashion, and food scene. It is also home to many famous landmarks and attractions, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a major hub for international business and tourism, and its status as the capital of France is recognized worldwide. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare.
    
    2. AI in finance: AI is already being used in finance to improve fraud detection, risk management, and portfolio optimization. As AI technology continues to improve, we can expect to see even more widespread use of AI in finance.
    
    3.
    


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
    Generated text:  [Your Name] and I am a [Your Profession/Role] who has been passionate about [Your Area of Expertise] for [Your Duration] years now. I have always been driven by a desire to help others and make a positive impact in their lives. I believe in the power of empathy and the ability to connect with people on a deeper level. I love [Your Area of Expertise] and strive to learn and grow continuously to stay up-to-date with the latest techniques and tools. I am a [Your Education Level/Experience Level] who have honed my skills through rigorous training and have always been open to learning
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Île de la Cite, and is the largest city in France by population. It is known for its rich history, arts, and cuisine. Paris is also one of the world's most visited cities and is a UNESCO World Heritage site. Its landmarks include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and Montmartre. Paris has a diverse population and is home to numerous museums, theaters, and cafes, making it an ideal destination for visitors from around the world. It is also an important political and cultural center of France. 
    
    The city is home to the Paris Opera
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain and unpredictable, as it is a rapidly evolving field with numerous possibilities and limitations. Here are some possible trends in AI that are expected to shape the industry in the coming years:
    
    1. Autonomous vehicles: AI-driven autonomous vehicles are becoming more common and capable, and they are expected to change the way we travel and live. With the development of sensors, cameras, and other technologies, autonomous vehicles are becoming more reliable, safe, and efficient.
    
    2. Expert systems: AI is being used to replace human experts in certain fields, such as healthcare and finance. These systems are expected to become more sophisticated and capable, but they may face challenges


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

    ],

     and

     I

     am

     a

     [

    job

     title

     or

     profession

    ].

     I

     am

     currently

     a

     [

    current

     job

     role

    ].

     I

     enjoy

     spending

     my

     time

     [

    time

     activities

     or

     interests

    ]

     and

     I

     enjoy

     helping

     others

     [

    reason

     for

     interest

     or

     engagement

    ].

     If

     you

     wanted

     to

     become

     one

     of

     my

     best

     friends

    ,

     what

     qualities

     or

     traits

     would

     you

     look

     for

    ?

     [

    Feel

     free

     to

     add

     any

     other

     specific

     information

     or

     details

     you

     think

     would

     be

     helpful

     to

     know

     about

     the

     character

     you

     are

     writing

     about

    .]

     Hello

    !

     My

     name

     is

     [

    Name

    ],

     and

     I

     am

     a

     [

    job

     title

     or

     profession

    ].

     I

     am

     currently

     a

     [

    current

     job

     role

    ].

     I

     enjoy

     spending

     my

     time

     [

    time

     activities

     or

     interests

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    What

     is

     the

     capital

     city

     of

     the

     United

     States

    ?

     Washington

    ,

     D

    .C

    .

     
    


    Please

     provide

     a

     detailed

     analysis

     of

     the

     pros

     and

     cons

     of

     using

     a

     social

     media

     platform

     for

     promoting

     a

     company

    .

     Use

     the

     following

     table

     to

     support

     your

     answer

    :
    


    |

     Company

     |

     Social

     Media

     Platform

     |


    |

    ---------

    |

    ----------------

    ------

    |


    |

     Airbnb

     |

     Instagram

    ,

     Twitter

    ,

     YouTube

     |


    |

     Amazon

     |

     Facebook

    ,

     Instagram

    ,

     Twitter

     |


    |

     Alibaba

     |

     We

    ibo

    ,

     We

    Chat

    ,

     Tout

    iao

     |
    


    Provide

     a

     comprehensive

     report

     on

     how

     a

     social

     media

     influ

    encer

     can

     effectively

     promote

     a

     business

     on

     their

     platform

     and

     increase

     brand

     awareness

    .

     Use

     the

     following

     information

     to

     support

     your

     answer

    :
    


    |

     Influ

    encer

     |

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     different

     trends

    ,

     including

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     advanced

     and

     specialized

    ,

     it

     is

     likely

     that

     it

     will

     be

     used

     more

     extensively

     in

     healthcare

     to

     improve

     diagnoses

    ,

     treatment

     plans

    ,

     and

     patient

     outcomes

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     will

     play

     an

     increasingly

     important

     role

     in

     finance

    ,

     with

     applications

     ranging

     from

     risk

     management

     to

     fraud

     detection

     and

     trading

    .
    


    3

    .

     AI

     in

     manufacturing

    :

     AI

     is

     already

     being

     used

     to

     automate

     manufacturing

     processes

     and

     improve

     efficiency

    ,

     and

     it

     is

     likely

     that

     AI

     will

     continue

     to

     play

     a

     larger

     role

     in

     manufacturing

     in

     the

     coming

     years

    .
    


    4

    .

     AI

     in

     education

    :

     As

     AI

     continues

     to

    



```python
llm.shutdown()
```
