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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.74it/s]


    2026-04-14 13:37:14,398 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 13:37:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<02:45,  2.91s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:24,  2.18it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:07,  6.36it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.08it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 19.99it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 28.68it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 37.55it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 46.43it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 46.43it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 46.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 16.28it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 19.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.25it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.25it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.85it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.85it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.85it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.30 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=119.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s] Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  34%|███▍      | 20/58 [00:00<00:01, 33.73it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]

    Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]Capturing num tokens (num_tokens=512 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 35.14it/s]Capturing num tokens (num_tokens=512 avail_mem=118.96 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 37.19it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:01<00:00, 37.19it/s]

    Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 34.70it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.23it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.04it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.04it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.04it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.04it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.04it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.00it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.62it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 33.90it/s]


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
    Generated text:  Arthur, I am a young man in my mid-twenties. I am the president of a nonprofit organization. My name is Arthur, and I am excited to share some thoughts with you.
    
    How have you been keeping up with the social issues and advocacy for marginalized communities?
    
    I have been keeping up with the social issues and advocacy for marginalized communities through various ways, such as participating in various conferences and seminars, attending public meetings, and speaking to other people in a positive way to promote the cause. I have also been working to raise awareness on social issues through social media and other platforms.
    
    I also believe that educating others is key to ensuring
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by the Vice President. In how many ways can the vice president be chosen to fill an office?
    
    To determine the number of ways the Vice President can be chosen to fill an office, we need to consider that there are only two candidates for the position: the President and the Vice President. Since the Vice President's position is unique and fixed, there is only one way to choose the Vice President.
    
    Therefore, the number of ways to choose the Vice President is \(\boxed{1}\).
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the Atlantic coast.
    A. Incorrect
    B. Correct
    Answer: A
    
    When an enterprise or institution sets up, acquires, uses, or generates internal information, which of the following methods can be used to achieve confidentiality protection?
    A. Encrypt the internal information
    B. Limit access to internal information
    C. Purchase confidential information products
    D. Implement access control measures
    Answer: ABD
    
    Which of the following statements about the signal transmission process in the TCP/IP model is correct?
    A. The transmission process includes three stages: packet transmission, data transmission, and connection establishment.
    B. The transmission process includes three stages
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. What can I do to prepare for it? Is it too late to prepare for AI?
    
    Prepare for AI, but it is not too late. The more you know, the better equipped you will be. Here is what you can do to prepare for AI:
    
    1. Understand the basics: Understand the basics of AI including its history, definition, different types of AI, and its role in different industries. This will help you understand how AI is changing the world and what you can expect from AI in the future.
    
    2. Learn programming: Learning programming is essential for AI development. Start with Python or Java, as they are the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character's personality or background]. I enjoy [insert a short description of your hobby or interest]. What's your favorite [insert a short description of your hobby or interest]? I'm always looking for new experiences and learning new things. What's your favorite [insert a short description of your hobby or interest]? I'm always looking for new experiences and learning new things. What's your favorite [insert a short description
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and diverse cultural scene. It is also the birthplace of many famous French artists, writers, and composers. Paris is a bustling metropolis with a rich history and a vibrant nightlife. It is a popular tourist destination and a major economic center in Europe. The city is known for its fashion industry, art, and cuisine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is home to many museums, art galleries, and cultural institutions, making it a popular destination for art lovers and history buffs. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect user data. This could lead to the development of new technologies and approaches to data privacy and security, such as blockchain-based privacy-preserving AI.
    
    3. Greater automation and efficiency:
    


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
    Generated text:  [Name] and I am [Age]. I am a [Occupation] with [Educational Background] in [Major]. I am [Adjective] and [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I am [Adjective]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city and the capital of France. It is located in the south of the country, near the Bay of Biscay. Paris is a major cultural, economic, and political center, and is known for its historic buildings, museums, and fashion. Paris is also famous for its cuisine and world-class cinema, as well as its annual opera, ballet, and ballet-drama performances. The city is known for its vibrant nightlife, cultural scene, and distinctive architecture, making it a popular tourist destination. The city has a rich history, from ancient Greeks and Romans to French, and continues to evolve and grow with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be highly dynamic and involve significant changes in both its technology and its applications. Here are some possible trends in AI that we can expect to see in the coming years:
    
    1. Deep learning and machine learning advancements: The field of AI is constantly evolving, and we can expect to see more breakthroughs in deep learning and machine learning. These technologies will become more powerful and capable, leading to new applications in various fields, including healthcare, finance, and transportation.
    
    2. AI ethics and responsibility: As AI becomes more integrated into our daily lives, it will be essential to consider the ethical implications of AI development and deployment. The development of


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

    ________

    _

     and

     I

    'm

     a

    /an

     __

    ________

    __

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __ of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    __

     of

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    
    
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

    ,

     located

     in

     the

     center

     of

     the

     country

     and

     is

     the

     second

    -most

     populous

     city

     in

     Europe

    .

     
    


    Note

    :

     The

     statement

     is

     based

     on

     historical

     and

     official

     records

    ,

     so

     it

     may

     not

     be

     entirely

     accurate

     or

     up

    -to

    -date

    .

     However

    ,

     it

     is

     the

     most

     well

    -known

     and

     recognizable

     city

     in

     France

     and

     the

     largest

     metropolitan

     area

     in

     Europe

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

     and

     cuisine

    ,

     and

     has

     played

     a

     significant

     role

     in

     French

     culture

     and

     politics

     since

     the

     time

     of

     Charles

     V

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     has

     been

     the

     residence

     of

     kings

    ,

     queens

    ,

     em

    per

    ors

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

    ,

     and

     there

     are

     many

     different

     trends

     that

     are

     likely

     to

     shape

     the

     technology

    's

     direction

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Improved

     accuracy

     and

     precision

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     will

     be

     able

     to

     process

     and

     analyze

     more

     complex

     data

     sets

    ,

     leading

     to

     increased

     accuracy

     and

     precision

     in

     many

     tasks

    .
    


    2

    .

     Increased

     AI

     diversity

    :

     AI

     systems

     are

     becoming

     more

     diverse

     in

     terms

     of

     their

     features

    ,

     which

     could

     lead

     to

     more

     nuanced

     and

     sophisticated

     solutions

     to

     complex

     problems

    .
    


    3

    .

     Greater

     integration

     with

     human

     intelligence

    :

     As

     AI

     continues

     to

     improve

     and

     become

     more

     integrated

     with

     human

     intelligence

    ,

     it

     is

     possible

     that

     AI

     systems

     will

     be

     able

     to

    



```python
llm.shutdown()
```
