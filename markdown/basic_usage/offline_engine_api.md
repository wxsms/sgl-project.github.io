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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


    2026-04-10 15:15:48,184 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 15:15:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:22,  2.50s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.42it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]Compiling num tokens (num_tokens=832):  28%|██▊       | 16/58 [00:02<00:03, 11.40it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:02<00:01, 19.25it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:03<00:01, 19.25it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:00, 27.53it/s]

    Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 34.49it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]

    Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 41.20it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 50.39it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 50.39it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 50.39it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 50.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 19.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=448 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.84it/s]Capturing num tokens (num_tokens=448 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=416 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=352 avail_mem=74.59 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=320 avail_mem=74.58 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=288 avail_mem=74.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=288 avail_mem=74.56 GB):  62%|██████▏   | 36/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=256 avail_mem=74.09 GB):  62%|██████▏   | 36/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=240 avail_mem=73.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=192 avail_mem=73.92 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.79it/s]

    Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.79it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s] Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=32 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.82it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.30it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.30it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.30it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.30it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.30it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.70it/s]


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
    Generated text:  Jessica and I am a professional graphic designer. I specialize in digital design, digital art, advertising, and graphic design. My specialties are logo design, graphics, branding, packaging, marketing materials, and e-commerce. I have a Bachelor of Science degree in Graphic Design from the University of Southern California, and a Master of Arts degree in Digital Media from the University of Texas at Austin. I am currently enrolled at Texas Tech University as a Graphic Design student. I enjoy working with people and learning new things. I can design any kind of project. I can get people inspired and motivated. I am very excited to have the opportunity to work with
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting another country. The president has a portfolio of 8 books that he wants to read. He reads a certain number of books per week. He has 8 weeks to complete his reading. How many books does he read per week?
    
    To determine how many books the president reads per week, we need to follow these steps:
    
    1. Identify the total number of books the president wants to read.
    2. Identify the total number of weeks available for reading.
    3. Divide the total number of books by the total number of weeks to find the number of books read per week.
    
    Step 1: The total number of books the president wants
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. London
    C. Moscow
    D. Tokyo
    Answer:
    A
    
    In the ____ month of the year, the weather in the southern part of China is typically milder and more humid than in the northern part.
    A. March
    B. June
    C. September
    D. November
    Answer:
    B
    
    Which of the following sentences is consistent with the Chinese meaning of the given word?
    A. I am a model for all the girls in the school.
    B. My grandfather is an old man with gray hair.
    C. I am very grateful to you.
    D. I'm going to
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it has no boundaries. AI is already making its mark on the world and its applications are increasing exponentially. However, in order to fully harness the power of AI, it is essential to consider the ethical implications of how the technology will impact society.
    
    Here are some of the key ethical concerns with AI:
    
      1. Bias and Fairness: One of the major ethical concerns with AI is the potential for bias and unfairness. AI algorithms can be biased if they are trained on biased data or if they are programmed with assumptions about certain groups of people. This can lead to discriminatory outcomes and a lack of fairness and equality.
    
    


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. 
    
    (Note: The statement provided is a factual statement about Paris, not a fictional one. The Eiffel Tower is a famous landmark in Paris, and Notre-Dame Cathedral is a major religious site in the city.) 
    
    The statement is concise and accurately reflects the facts about Paris, including its iconic landmarks and cultural attractions. However, if you would like a more detailed statement, please let me know! 
    
    For example, a more detailed statement could be: "Paris, the capital of France, is renowned for its iconic Eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve and become more integrated into our daily lives, from self-driving cars to personalized medicine to virtual assistants. Additionally, AI will likely continue to be used for more complex tasks, such as autonomous decision-making and predictive analytics, which will require even more advanced algorithms and models. Finally, AI will likely continue to be used for more ethical and responsible applications, such as improving healthcare and education, and reducing the impact of climate change. Overall, the future of AI is likely to be one of continued innovation and growth
    


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
    Generated text:  [name]. I'm a curious, creative person who loves to explore new ideas and challenges. I'm always looking for new ways to solve problems and come up with new ways to solve problems. I'm always up for new adventures, whether it's solving mysteries, trying to create new art forms, or simply enjoying the thrill of making something happen. I'm a joy to be around and I love to learn new things, so if you're looking for a curious, creative, and adventurous friend, I'm your guy! #curiouscreativefriend #adventurer #challengemaker #newidealover
    
    This self-introduction sounds great
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as “la Parisis” (the city of light) and is the oldest capital city in Europe. It is a cultural, historical, and economic center that is home to many of France’s iconic landmarks, including the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. Paris is also home to many famous French artists, writers, and composers. It is an important financial and political center of Europe, and is a world-renowned center for the arts, music, fashion, and cuisine. 
    
    Related question: What is the capital city of Portugal? The capital city of Portugal is Lisbon, also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and many potential trends are likely to shape the technology in ways that are both exciting and challenging. Here are a few potential trends that could shape the AI landscape in the coming years:
    
    1. Increased focus on ethical AI: There will likely be greater emphasis on ethical considerations when developing AI systems. This could include things like privacy concerns, bias, and transparency.
    
    2. Autonomous vehicles: As autonomous vehicles become more advanced, it's likely that they will be widely adopted. This could lead to a shift in how people interact with the world around them and how we organize our transportation systems.
    
    3. AI in healthcare: AI could revolutionize the


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

     ______

    _

     and

     I

    'm

     ______

    _.

     I

    'm

     here

     to

     provide

     ______

    _

     to

     you

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     How

     do

     you

     come

     up

     with

     new

     ideas

    ?

     Do

     you

     have

     a

     personal

     story

     or

     background

    ?

     What

     motiv

    ates

     you

     to

     become

     an

     active

     participant

     in

     our

     community

    ?

     I

     am

     ______

    __

     and

     I

     am

     ______

    __.

     How

     do

     you

     feel

     about

     ______

    __

    ?

     How

     can

     I

     help

     you

     better

    ?

     Here

    's

     a

     sample

    :
    


    Hello

    ,

     my

     name

     is

     Jim

     and

     I

    'm

     from

     ______

    .

     I

    'm

     a

     passionate

     advocate

     for

     __

    __.

     I

     come

     up

     with

     new

     ideas

     through

     the

     lens

     of

     what

     I

     value

     most

     and

     what

     brings

     me

     joy

    .

     I

     also

     have

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     a

     historical

     city

     and

     the

     seat

     of

     the

     French

     government

     and

     the

     largest

     city

     in

     France

    .

     The

     city

     has

     been

     a

     UNESCO

     World

     Heritage

     Site

     for

     more

     than

     

    5

    0

     years

     and

     is

     also

     one

     of

     the

     most

     famous

     cities

     in

     the

     world

    .

     It

     is

     known

     for

     its

     iconic

     architecture

    ,

     world

    -ren

    owned

     museums

    ,

     and

     annual

     summer

     festivals

    ,

     including

     the

     Op

    éra

    .

     Paris

     is

     also

     home

     to

     many

     famous

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     a

     popular

     tourist

     destination

     and

     attracts

     millions

     of

     visitors

     each

     year

    .

     In

     recent

     years

    ,

     Paris

     has

     been

     a

     major

     city

     in

     the

     European

     Union

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

     and

     depends

     on

     a

     wide

     range

     of

     factors

    ,

     including

     advances

     in

     hardware

    ,

     software

    ,

     and

     data

    ,

     as

     well

     as

     ongoing

     research

     and

     development

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

     Enhanced

     autonomy

    :

     AI

     systems

     are

     becoming

     more

     capable

     of

     performing

     tasks

     that

     once

     required

     human

     intervention

    ,

     such

     as

     decision

    -making

    ,

     problem

    -solving

    ,

     and

     decision

    -making

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     autonomous

     AI

     systems

     that

     can

     make

     decisions

     on

     their

     own

     without

     human

     intervention

    .
    


    2

    .

     Enhanced

     cognitive

     capabilities

    :

     AI

     systems

     are

     becoming

     more

     capable

     of

     processing

     and

     analyzing

     large

     amounts

     of

     data

    ,

     which

     could

     lead

     to

     the

     development

     of

     AI

     systems

     that

     can

     perform

     complex

     cognitive

     tasks

    



```python
llm.shutdown()
```
