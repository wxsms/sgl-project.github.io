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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]


    2026-04-28 00:24:29,800 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 00:24:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:35,  4.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:35,  4.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:35,  4.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:35,  4.83s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:35,  4.83s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.04it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.04it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.10it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.75it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.50it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:04, 13.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:03, 14.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:03, 14.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   7%|▋         | 4/58 [00:00<00:03, 14.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   7%|▋         | 4/58 [00:00<00:03, 14.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.77it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.77it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.71it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.93it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.36it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:01<00:00, 36.36it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=384 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=320 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.05it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.76it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:01<00:00, 45.85it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=20 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=16 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.90it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 35.66it/s]


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
    Generated text:  Jiray Chaturanga. I am a 17 year old resident of South Africa. I am an engineer by profession. I am also a member of the Engineering Society of South Africa. I am a student at the University of KwaZulu-Natal, where I am studying a Bachelor of Engineering degree in Civil Engineering. I am a keen sportsman, and I regularly participate in the South African Soccer League, where I play in the league and I am currently the captain of the South African S.S. Men’s Elite Division. I am also a keen golfer, and I am the captain of the South African S
    ===============================
    Prompt: The president of the United States is
    Generated text:  in New York for a trip. He has a certain number of cars and wants to see what his traffic congestion looks like. If he starts his trip in New York and drives to Chicago, which is 500 miles away, and then drives from Chicago to Boston, which is 200 miles away, how many miles will he drive in total?
    To determine the total distance the president of the United States will drive, we need to add the distances of the two legs of his trip.
    
    1. The distance from New York to Chicago is 500 miles.
    2. The distance from Chicago to Boston is 2
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Lille
    C. Lyon
    D. Marseille
    Answer:
    
    A
    
    On a single-hop transmission, the total delay of a physical layer communication process is equal to the sum of the round-trip time of the message and the round-trip time of the transmission channel. The round-trip time of the message includes the time for the message to be received by the receiver and the time for the message to be sent to the transmitter. ____
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    According to the 'Civil Code of the People's Republic of China', the labor capacity assessment
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and diverse, with a wide range of applications in fields such as healthcare, finance, transportation, and entertainment. However, like any technology, there are concerns about the potential ethical implications of AI and its impact on society. Here are five key concerns that people should be aware of:
    
    1. Bias and Fairness: One of the most significant concerns about AI is its potential to perpetuate existing biases and discrimination. AI systems are designed to learn from data, and if the data is biased, the resulting algorithms and models may also be biased. This can lead to unfair outcomes in decision-making and perpetuate systemic inequality. To address this concern


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, art, and cuisine, and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for business, finance, and culture, and is a popular tourist destination for visitors from around the world. The city is home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city that continues to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a greater emphasis on ethical considerations, such as privacy, bias, and transparency. This will require developers to take a more responsible approach to AI design and development.
    
    2. Integration with human intelligence: AI systems will continue to become more integrated with human intelligence,
    


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
    Generated text:  [insert your name here] and I am a [insert occupation or profession here]. I have a keen interest in [insert something specific here, such as books, music, art, etc.]. What excites you the most is [insert your most exciting hobby or passion here]. And what is the most surprising or amazing thing about you? Your name is [insert your name here], and I am an AI language model. How can I assist you today?
    I'm here to help! How can I assist you today?
    I am an AI language model and am here to assist you with any questions or tasks you may have. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its historic architecture, landmarks such as Notre-Dame Cathedral, and its vibrant French culture and food. The city is also home to some of Europe's most famous museums, including the Louvre and the Musée d'Orsay. Paris is the fifth-largest city in the European Union by population and is a major cultural and economic center in Western Europe. Its climate is mild and beautiful, and it is home to many attractions throughout the year. Paris is a popular tourist destination and is known for its fashion, art, and cuisine. It has been a city of significance since ancient times and is a major hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve and transform in a variety of ways. Here are some possible future trends in artificial intelligence:
    
    1. AI will become more pervasive: As AI becomes more powerful and widely available, it will become increasingly integrated into our daily lives, from self-driving cars and virtual assistants to predictive analytics and personalized recommendations.
    
    2. AI will continue to develop and improve: As AI technology advances, we can expect to see more sophisticated algorithms, better data storage and processing, and more efficient ways to train and deploy AI models.
    
    3. AI will be used for more complex tasks: AI will be used for more complex tasks that require reasoning,


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

    ].

     I

     am

     [

    Your

     Profession

    ]

     and

     I

     have

     been

     working

     at

     [

    Company

     Name

    ]

     for

     [

    Your

     Duration

     of

     Time

    ].

     I

     am

     a

     [

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Area

     of

     Expert

    ise

    ].

     What

     kind

     of

     projects

     are

     you

     involved

     in

    ?

     What

     is

     your

     professional

     background

    ?

     What

     do

     you

     bring

     to

     the

     table

     that

     makes

     you

     a

     valuable

     addition

     to

     [

    Company

     Name

    ]?

     I

     would

     love

     to

     learn

     more

     about

     you

     and

     your

     expertise

    .

     [

    Tell

     your

     story

     in

     a

     convers

    ational

     tone

    ].

     Good

     day

    ,

     [

    Name

    ].

     I

     am

     thrilled

     to

     be

     here

    .

     Let

     me

     introduce

     myself

    ,

     [

    Your

     Profession

    ],

     and

     tell

     you

     about

     myself

    .

     As

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Gar

    de

    ,"

     which

     is

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     the

     French

     department

     of

     Paris

    .

     The

     city

     is

     famous

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     bustling

     streets

    ,

     lively

     culture

    ,

     and

     historic

     architecture

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     has

     been

     a

     major

     city

     for

     over

     

    2

    ,

    0

    0

    0

     years

    .

     The

     city

     is

     known

     for

     its

     diverse

     population

    ,

     including

     Paris

    ians

    ,

     non

    -native

     residents

    ,

     and

     visitors

     from

     all

     over

     the

     world

    .

     It

     is

     considered

     one

     of

     the

     most

     important

     cultural

    ,

     economic

    ,

     and

     political

     centers

     in

     the

     world

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     to

     keep

     an

     eye

     on

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     companies

     and

     governments

     take

     on

     the

     responsibility

     of

     developing

     and

     implementing

     AI

     systems

    ,

     they

     will

     likely

     prioritize

     ethical

     concerns

     such

     as

     bias

    ,

     transparency

    ,

     and

     accountability

    .
    


    2

    .

     Enhanced

     AI

     skills

    :

     AI

     systems

     will

     likely

     become

     even

     more

     intelligent

     and

     powerful

    ,

     allowing

     them

     to

     perform

     tasks

     that

     were

     previously

     beyond

     their

     capabilities

    .
    


    3

    .

     Autonomous

     vehicles

    :

     As

     self

    -driving

     cars

     become

     more

     advanced

    ,

     AI

     will

     likely

     play

     a

     key

     role

     in

     their

     development

    .

     Autonomous

     vehicles

     will

     likely

     become

     more

     common

    ,

     and

     AI

     systems

     will

     likely

     play

     a

     more

     significant

     role

     in

     their

     operation

    



```python
llm.shutdown()
```
