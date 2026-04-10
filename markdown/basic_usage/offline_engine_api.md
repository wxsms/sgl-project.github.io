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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-04-10 22:54:46,472 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 22:54:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.64it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.64it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.95it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.95it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.95it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.56it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.52it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   2%|▏         | 1/58 [00:00<00:13,  4.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   7%|▋         | 4/58 [00:00<00:04, 12.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:04, 12.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   7%|▋         | 4/58 [00:00<00:04, 12.16it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   7%|▋         | 4/58 [00:00<00:04, 12.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s]Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s] Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s]

    Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.28it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.67it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.67it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]

    Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]

    Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.75it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]

    Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.92it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.92it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 35.39it/s]


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
    Generated text:  Cali. I’m a graduate student studying at the University of Technology Sydney in Australia. I’ve always been a big fan of coding and developing software. My current research project, called "Shoebox", aims to make educational software for children. It’s a software that can learn from users and make a personalized education plan. Now, I am searching for some research papers that explain the topic of educational software for children. Can you provide me with a list of at least 5 research papers about the topic of educational software for children?
    Sure, I can definitely help you with that. Here are 5 research papers about educational software for
    ===============================
    Prompt: The president of the United States is
    Generated text:  a president of the United States. The president of the United States is also a president of the United States. Can we conclude that the second sentence implies the first sentence? Yes, we can conclude that the second sentence implies the first sentence. 
    
    To break this down logically:
    
    1. The first sentence states that "the president of the United States is a president of the United States."
    2. The second sentence states that "the president of the United States is also a president of the United States."
    
    These two sentences are logically equivalent. They both claim that the president of the United States is a president of the United States. Therefore, the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer: A
    
    In the electric field of a certain point source, when the electric field strength is zero, which of the following is true?
    A. The electric potential energy is zero.
    B. The electric potential energy is maximum.
    C. The electric potential energy is minimum.
    D. The electric potential energy is constant.
    Answer: C
    
    When a ship is moving at a constant speed, the changes in momentum of the ship and the ship's moving direction occur simultaneously.
    A. True
    B. False
    Answer: A
    
    A ship's hull shape is the
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with vast possibilities for personalization and customization that can improve user experience and satisfaction. However, it’s also worth noting that AI is an ever-evolving field with increasing complexity and sophistication. This means that there may be some surprises and challenges that come with adopting new technologies and systems. In this post, we will explore some of the challenges that may arise in the adoption of AI in the future.
    
    One of the biggest challenges that may arise in the adoption of AI is the need for human oversight and validation. AI systems require human supervision and validation to ensure that they are operating correctly and are not causing harm or unintended consequences. Without proper


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


    Generated text:  [Name] and I am a [occupation] who has been working in the [industry] for [number] years. I have always been passionate about [occupation] and have always wanted to [goal]. I am always looking for new challenges and opportunities to [action]. I am confident in my abilities and always strive to [goal]. I am a [character type] who is always [character trait]. I am [character type] and I am [character trait]. I am [character type] and I am [character trait]. I am [character type] and I am [character trait]. I am [character type] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is also the seat of the French government and the country's cultural, political, and economic center. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is also home to many museums, including the Louvre and the Musée d'Orsay. Paris is a popular tourist destination and a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This means that AI systems will be able to learn and adapt to human behavior and preferences, and will be able to communicate with humans in a more natural way.
    
    2. Greater use of machine learning: Machine learning is a key area of focus for AI research, and it is likely to continue to be a major driver of future developments in AI. Machine
    


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
    Generated text:  [Your Name], and I am a friendly, outgoing, and optimistic person. I thrive on social interaction and can easily connect with people of all backgrounds and personalities. I am always eager to learn new things and challenge myself to grow and improve as a person. I believe that being kind, patient, and compassionate towards others is key to building a strong and fulfilling life. I am always ready to help others and make a difference in the world. Thank you for having me!
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its iconic landmarks such as the Eiffel Tower and the Louvre Museum, as well as for its rich cultural heritage and annual cultural festivals such as the Opéra National de Paris. 
    
    Given the ongoing debate about climate change, it's worth noting that Paris, being a megacity, also faces the challenge of sustainable development. The city is committed to reducing its carbon footprint and is working towards the targets set by the Paris Agreement. 
    
    Paris is also home to many notable institutions, including the Palace of Versailles, where Louis XIV lived during the French Revolution, and the Louvre Museum, which houses
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  an exciting and constantly evolving field. Here are some possible trends we can expect to see in the near and long term:
    
    1. Enhanced machine learning: With the increasing availability of data, the power of machine learning models will continue to grow. This will lead to more accurate predictions, better decision-making, and even autonomous behavior in various applications.
    
    2. More precise and personalized AI: As AI continues to improve, it will become more precise and tailored to individual users, leading to more personalized and efficient use of AI technology.
    
    3. Integration of AI with human experience: AI is becoming increasingly integrated into human experience, and we can expect to see


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

     am

     a

     computer

     program

     designed

     to

     assist

     humans

     in

     various

     tasks

    .

     I

     can

     perform

     calculations

    ,

     answer

     questions

    ,

     and

     even

     generate

     text

     based

     on

     the

     prompts

     I

     receive

    .

     How

     can

     I

     assist

     you

     today

    ?

     Just

     ask

    ,

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     the

     information

     you

     need

    .

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

     me

     know

    !

     How

     can

     I

     help

     you

     with

     your

     computer

     program

    ?

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

     me

     know

    !

     How

     can

     I

     help

     you

     with

     your

     computer

     program

    ?

     What

     can

     I

     do

     for

     you

     today

    ?

     Let

     me

     know

    !

     How

     can

     I

     help

     you

     with

     your

     computer

     program

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     historical

     landmarks

    ,

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

    ,

     and

     its

     vibrant

     cultural

     scene

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     evolving

    ,

     and

     it

     is

     difficult

     to

     predict

     with

     certainty

     what

     trends

     will

     emerge

    .

     However

    ,

     some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     AI

     systems

     are

     becoming

     more

     sophisticated

    ,

     and

     researchers

     are

     becoming

     more

     aware

     of

     their

     potential

     impacts

     on

     society

    .

     As

     a

     result

    ,

     there

     is

     an

     increasing

     emphasis

     on

     developing

     AI

     systems

     that

     are

     transparent

    ,

     accountable

    ,

     and

     sustainable

    .
    


    2

    .

     More

     diverse

     and

     representative

     data

    :

     AI

     systems

     rely

     on

     large

     amounts

     of

     data

     to

     learn

     and

     improve

    .

     However

    ,

     there

     is

     a

     growing

     concern

     that

     some

     AI

     systems

     may

     be

     biased

     or

     discriminatory

    .

     To

     address

     this

     issue

    ,

     researchers

     are

     exploring

     more

     diverse

     and

     representative

     data

    



```python
llm.shutdown()
```
