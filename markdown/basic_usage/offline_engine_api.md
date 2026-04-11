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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.64it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.63it/s]


    2026-04-11 07:59:39,820 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 07:59:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.85it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.67it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.67it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.07it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 19.22it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 25.94it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 32.61it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 48.55it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 48.55it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 48.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.20it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.98 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.95 GB):   3%|▎         | 2/58 [00:00<00:02, 19.16it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.94 GB):   3%|▎         | 2/58 [00:00<00:02, 19.16it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.94 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.94 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.93 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=68.92 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.92 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.91 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.91 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.90 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.90 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.89 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.87 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=960 avail_mem=68.89 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s] Capturing num tokens (num_tokens=896 avail_mem=68.89 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=832 avail_mem=68.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]

    Capturing num tokens (num_tokens=768 avail_mem=68.88 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.20it/s]Capturing num tokens (num_tokens=768 avail_mem=68.88 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=704 avail_mem=68.88 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=640 avail_mem=68.87 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=576 avail_mem=68.87 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=512 avail_mem=68.86 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=480 avail_mem=68.87 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.82it/s]Capturing num tokens (num_tokens=480 avail_mem=68.87 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=448 avail_mem=68.87 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=416 avail_mem=68.87 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=384 avail_mem=68.87 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=352 avail_mem=68.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=320 avail_mem=68.86 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]

    Capturing num tokens (num_tokens=288 avail_mem=68.85 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=288 avail_mem=68.85 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=256 avail_mem=68.85 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=240 avail_mem=68.85 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=224 avail_mem=68.85 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=208 avail_mem=68.84 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=192 avail_mem=68.84 GB):  62%|██████▏   | 36/58 [00:00<00:00, 47.06it/s]Capturing num tokens (num_tokens=176 avail_mem=68.84 GB):  62%|██████▏   | 36/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=176 avail_mem=68.84 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=160 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=144 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=128 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=112 avail_mem=68.83 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]

    Capturing num tokens (num_tokens=96 avail_mem=68.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s] Capturing num tokens (num_tokens=80 avail_mem=68.82 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=80 avail_mem=68.82 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=64 avail_mem=68.82 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=48 avail_mem=68.81 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=32 avail_mem=68.81 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=28 avail_mem=68.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=24 avail_mem=68.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=20 avail_mem=68.80 GB):  83%|████████▎ | 48/58 [00:01<00:00, 49.52it/s]Capturing num tokens (num_tokens=20 avail_mem=68.80 GB):  93%|█████████▎| 54/58 [00:01<00:00, 50.40it/s]Capturing num tokens (num_tokens=16 avail_mem=68.80 GB):  93%|█████████▎| 54/58 [00:01<00:00, 50.40it/s]Capturing num tokens (num_tokens=12 avail_mem=68.79 GB):  93%|█████████▎| 54/58 [00:01<00:00, 50.40it/s]Capturing num tokens (num_tokens=8 avail_mem=68.79 GB):  93%|█████████▎| 54/58 [00:01<00:00, 50.40it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=68.79 GB):  93%|█████████▎| 54/58 [00:01<00:00, 50.40it/s]Capturing num tokens (num_tokens=4 avail_mem=68.79 GB): 100%|██████████| 58/58 [00:01<00:00, 43.64it/s]


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
    Generated text:  Juan Carlos, I am a 16 year old female, 360 cm tall, and I am bald.
    I am a postgraduate student at the University of Valencia, Spain. I am currently studying Bachelor of Arts in Psychology, in the Faculty of Social Sciences and Humanities.
    I am currently a member of the Psychology Department and have been in the Psychology Department for 3 years. In the last year, I have been a Research Assistant, part-time, with the Psychology Department. My job is to assist the researchers on conducting research and supervise the research projects. I am also studying for the exam of Research Assistant in Psychology.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have in the country. He likes the idea of 100 bases but doesn't like the idea of 110 bases. However, he also likes the idea of 90 bases, but doesn't like the idea of 95 bases. In how many different ways can the president build the bases? To determine the number of different ways the president can build the bases, we need to consider the constraints and the possible combinations of the bases he can choose from. The president likes 100, 90, and 95 bases, but doesn't like 
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    Answer: A
    
    The capital of France is located in the ___.
    A. South of the Mediterranean Sea
    B. North of the Atlantic Ocean
    C. North of the Mediterranean Sea
    D. South of the Atlantic Ocean
    Answer: A
    
    Which of the following statements is incorrect?
    A. The capital of France is Paris.
    B. The capital of France is located in the South of the Mediterranean Sea.
    C. The capital of France is located in the North of the Atlantic Ocean.
    D. The capital of France is located in the
    ===============================
    Prompt: The future of AI is
    Generated text:  full of possibilities, but as with any new technology, it also brings with it some challenges. One of the biggest challenges is the impact of AI on the work environment.
    Imagine a world where AI is taking over the tasks of human workers. It’s a scary thought, and it’s not just a futuristic vision of the future. In fact, it’s becoming a reality in many places, especially in certain industries.
    One of the most obvious examples of AI taking over human jobs is in the healthcare industry. AI is being used to automate routine tasks like patient appointment scheduling, appointment review, and medical diagnosis. This not only saves time and money


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for interest]. I'm always looking for ways to [job title] and I'm always eager to learn new things. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new experiences and I'm always eager to learn new things
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature and a major center for art, music, and film. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. Its historical significance and modern influence make it a fascinating city to explore. The city is home to many famous landmarks and is a major transportation hub for Europe. Paris is a vibrant and dynamic city that continues to thrive in the modern age. The city is also known for its diverse population, including many ethnic groups and cultures. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare, with the potential to revolutionize the way we treat and diagnose diseases.
    
    2. Increased use of AI in finance: AI is already being used in finance
    


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
    Generated text:  [Name], and I'm a/an [Occupation] with [Number of Years] years of experience in this field. I'm currently [Current Position], and I enjoy [Your hobby or passion]. I'm always looking for new opportunities to learn and grow, and I'm always open to opportunities to help people. I'm a/an [age range] year-old. I love [What you do as a hobby or passion]. I am [Your personality traits] and I am always looking for the best way to achieve my goals. I'm always looking for ways to improve my skills and knowledge, and I am always open to new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    How do you use this information in a sentence?
    Paris is the capital of France. 
    This sentence uses the information provided to state the fact that Paris is the capital of France. 
    It's a factual statement that includes only the facts given without including any additional information. 
    The sentence is straightforward and easy to understand. It uses only the main facts provided in the statement. 
    I will add "Capital of France" to the end of the sentence as per the instructions. 
    New sentence: Paris is the capital of France. 
    This sentence conveys the same information as the previous one but in a different order. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and it is possible that it will evolve in many exciting ways. Some of the possible trends in AI include:
    
    1. Increased focus on ethical considerations: As the technology continues to advance, it will become increasingly important to ensure that AI systems are designed and used ethically and responsibly. This will require the development of new ethical frameworks and standards, as well as new tools and methods for assessing the impact of AI on society.
    
    2. Greater collaboration between humans and AI: As AI becomes more advanced, there will be a growing need for humans to interact with the technology. This will require new ways of working that take into account the unique


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

     Jane

    .

     I

    'm

     a

     software

     engineer

     with

     a

     passion

     for

     innovation

     and

     problem

    -solving

    .

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     the

     way

     people

     use

     technology

     and

     find

     innovative

     solutions

     to

     complex

     challenges

    .

     My

     work

     is

     always

     centered

     around

     the

     idea

     that

     technology

     can

     be

     used

     for

     the

     better

    ment

     of

     society

    .

     If

     you

    'd

     like

     to

     learn

     more

     about

     my

     background

     or

     experience

    ,

     feel

     free

     to

     reach

     out

     and

     let

     me

     know

    .

     Let

     me

     know

     if

     there

    's

     anything

     you

    'd

     like

     to

     know

     about

     me

    .

     
    


    Jane

     is

     a

     software

     engineer

     with

     a

     passion

     for

     innovation

     and

     problem

    -solving

    .

     She

     is

     always

     looking

     for

     new

     ways

     to

     improve

     the

     way

     people

     use

     technology

     and

     find

     innovative

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    F

    acts

    :


    -

     It

     is

     located

     on

     the

     Mediterranean

     Sea

    ,

     just

     east

     of

     the

     French

     Riv

    iera

    .


    -

     It

     is

     the

     seat

     of

     the

     Government

     of

     France

    ,

     the

     head

     of

     state

    ,

     and

     the

     capital

     of

     France

    .


    -

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     most

     visited

     city

     in

     the

     world

     by

     tourists

     annually

    .


    -

     It

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .


    -

     It

     is

     known

     for

     its

     art

    ,

     architecture

    ,

     and

     historical

     significance

    .

     
    


    The

     capital

     city

     of

     France

     is

     Paris

    .

     The

     population

     is

     approximately

     

    2

    .

    1

     million

     according

     to

     the

     

    2

    0

    2

    1

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     increasing

     automation

    ,

     personal

    ization

    ,

     and

     integration

     with

     other

     technologies

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     could

     shape

     future

     developments

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     become

     more

     capable

     of

     performing

     complex

     tasks

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

     data

     analysis

    .

     This

     automation

     will

     enable

     AI

     systems

     to

     perform

     repetitive

    ,

     mon

    oton

    ous

    ,

     and

     time

    -consuming

     tasks

     more

     efficiently

    ,

     freeing

     up

     human

     resources

     to

     focus

     on

     more

     complex

     and

     creative

     tasks

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     enable

     personalized

     AI

     systems

     to

     learn

     from

     user

     data

     and

     preferences

     to

     provide

     more

     relevant

     and

     customized

     responses

    .

     This

     personal

    ization

     will

     enable

     AI

     systems

     to

     be

    



```python
llm.shutdown()
```
