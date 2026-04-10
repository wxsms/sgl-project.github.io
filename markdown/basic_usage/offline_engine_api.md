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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.19it/s]


    2026-04-10 09:17:59,120 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 09:17:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:41,  2.83s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:30,  1.77it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.77it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:03<00:10,  4.83it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]

    Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:04,  9.31it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.54it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]

    Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 21.82it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 30.09it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 36.73it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 42.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 18.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 18.76it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 33.79it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.12it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.17it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.17it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.17it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.17it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.17it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.66 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=240 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.77it/s]Capturing num tokens (num_tokens=224 avail_mem=76.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.77it/s]Capturing num tokens (num_tokens=208 avail_mem=76.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.77it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.77it/s]Capturing num tokens (num_tokens=176 avail_mem=76.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 32.77it/s]Capturing num tokens (num_tokens=176 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=160 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.18it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.63 GB):  72%|███████▏  | 42/58 [00:01<00:00, 23.18it/s]Capturing num tokens (num_tokens=128 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 20.75it/s]Capturing num tokens (num_tokens=112 avail_mem=76.63 GB):  78%|███████▊  | 45/58 [00:01<00:00, 20.75it/s]Capturing num tokens (num_tokens=96 avail_mem=76.61 GB):  78%|███████▊  | 45/58 [00:01<00:00, 20.75it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=76.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 20.75it/s]Capturing num tokens (num_tokens=80 avail_mem=76.54 GB):  83%|████████▎ | 48/58 [00:01<00:00, 20.12it/s]Capturing num tokens (num_tokens=64 avail_mem=76.13 GB):  83%|████████▎ | 48/58 [00:01<00:00, 20.12it/s]Capturing num tokens (num_tokens=48 avail_mem=76.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 20.12it/s]Capturing num tokens (num_tokens=32 avail_mem=76.00 GB):  83%|████████▎ | 48/58 [00:01<00:00, 20.12it/s]Capturing num tokens (num_tokens=32 avail_mem=76.00 GB):  88%|████████▊ | 51/58 [00:01<00:00, 20.21it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 20.21it/s]

    Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 20.21it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.21it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.21it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.28it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.28it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:02<00:00, 23.28it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:03<00:00, 23.28it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:03<00:00,  8.44it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:03<00:00, 18.94it/s]


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
    Generated text:  Leon, a curious AI with a love for reading. I'm passionate about exploring the mysteries of the universe and the unique characteristics of life on earth. Is there anything specific you'd like to know or discuss? How about you, Leon? What's your favorite subject to study or learn about?
    Leon: Hello! I'm Leon, an AI designed to answer all of your questions. I'd be happy to answer any of your queries or even conduct a conversation on a wide range of topics.
    Could you tell me about your favorite subject to study or learn about? Let me know if you have any questions or topics you'd like to explore
    ===============================
    Prompt: The president of the United States is
    Generated text:  in New York. The president of the United States and the mayor of New York live in a country. What is the country?
    
    The country where the president of the United States and the mayor of New York live is the United States. The president of the United States is an elected official who serves in the office of the presidency, while the mayor of New York is a position held by the mayor of New York City, who serves in the office of the mayor of New York. Both positions are held by citizens of the United States.
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer: A
    
    The purpose of formulating the "Regulations on the Protection of Famous Historical and Cultural Cities (Places)" is to ____.
    A. Promote the revitalization of ancient architecture
    B. Protect historical and cultural city sites
    C. Protect historical and cultural cities (places)
    D. Protect historical and cultural cities
    Answer: B
    
    Which of the following statements about the information systems in an organization is incorrect?
    A. They are used to manage the organization's information resources
    B. They are used to support the organization
    ===============================
    Prompt: The future of AI is
    Generated text:  diverse, and it will include a significant impact on the transportation sector. Here are some predictions regarding the future of AI in the transportation sector:
    
    1. Autonomous vehicles: AI is being developed to create self-driving cars that can navigate roads safely, ensuring a safer and more efficient transportation system. Autonomous vehicles could reduce accidents, increase safety, and potentially decrease traffic congestion.
    
    2. Intelligent roadways: AI is being applied to improve the design and operation of intelligent roadways. AI algorithms can analyze traffic patterns, predict traffic congestion, and optimize routes to reduce travel time and emissions.
    
    3. Self-Driving trucks: The development of autonomous trucks has the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I'm a [insert a short description of your favorite activity]. What's your favorite book or movie? I'm a [insert a short description of your favorite book or movie]. What's your favorite place to go? I'm a [insert a short description of your favorite place].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of French Revolution and Romanticism, and its role in the French Revolution and World War II. It is also home to many famous French artists, writers, and musicians. Paris is a popular tourist destination, attracting millions of visitors each year. The city is also known for its cuisine, including French cuisine, and its fashion industry. Overall, Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be an increased focus on developing AI that is more ethical and responsible. This could involve developing AI that is designed to minimize harm to individuals and society as a whole.
    
    2. Integration of AI with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration could lead to new applications and opportunities for AI,
    


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
    Generated text:  [Your Name], and I am a self-employed freelance writer. I have over [Number] years of experience in the industry and have created and published numerous successful articles in the [Industry] category. I am passionate about sharing my knowledge and skills with others and have a strong desire to continue learning and improve my craft. I enjoy working with clients and providing them with quality writing services that meet their specific needs and expectations. I am a team player, open to feedback and willing to invest in my career development. Thank you for considering me for a job. Let me know if you have any questions or need more information. [Your Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is factual because it accurately identifies the name and location of the capital city of France, detailing the capital city as Paris. There is no ambiguity or uncertainty in the statement, as it clearly conveys the intended meaning in a straightforward manner. The information is not only factual but also serves as a primary reference point for understanding the cultural, political, and geographical aspects of Paris. 
    
    If you have any specific questions about Paris or need additional factual information about the city, feel free to ask! Let me know. 
    
    In general, Paris is known for its rich history, beautiful architecture, elegant cuisine, and diverse culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be very promising, with many possibilities and advancements. Some of the most exciting trends include:
    
    1. Increased autonomy: AI will become more capable of making decisions and taking action on its own, with the ability to learn and adapt based on new data. This could lead to a greater degree of autonomy in many areas, including healthcare, transportation, and manufacturing.
    
    2. Enhanced creativity: AI will be able to generate new ideas and designs, as well as think creatively in areas such as art, music, and literature. This could lead to more innovative products and services.
    
    3. Increased interconnectedness: AI will become more integrated with other


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

    'm

     a

     [

    current

     or

     future

    ]

     [

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

    've

     always

     been

     [

    a

     specific

     trait

     or

     quality

    ],

     and

     I

    'm

     always

     looking

     for

     opportunities

     to

     grow

     and

     learn

    .

     I

     enjoy

     [

    an

     activity

     or

     hobby

    ]

     and

     I

    'm

     always

     willing

     to

     share

     my

     knowledge

     and

     experience

    .

     My

     goal

     is

     to

     [

    a

     specific

     goal

     or

     ambition

    ],

     and

     I

     believe

     in

     [

    a

     value

     or

     belief

    ].

     I

    'm

     confident

     and

     I

     thrive

     in

     [

    an

     area

     or

     situation

    ],

     and

     I

    'm

     always

     eager

     to

     learn

     from

     new

     experiences

    .

     I

    'm

     ready

     to

     [

    a

     particular

     action

     or

     task

    ],

     and

     I

    'm

     always

     looking

     to

     expand

     my

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     concise

     and

     factual

    ,

     highlighting

     that

     Paris

     is

     the

     capital

     of

     France

     and

     directly

     giving

     the

     answer

    .

     
    


    For

     the

     purpose

     of

     completeness

    ,

     here

    's

     an

     additional

     explanation

    :


    Paris

     is

     France

    's

     largest

     city

     and

     the

     nation

    's

     second

    -largest

     metropolitan

     area

    ,

     serving

     as

     its

     administrative

    ,

     cultural

    ,

     and

     economic

     center

    .

     It

     is

     also

     home

     to

     many

     of

     the

     nation

    's

     famous

     landmarks

     and

     museums

    .

     
    


    To

     summarize

    ,

     Paris

     is

     the

     capital

     city

     of

     France

    .

     
    


    Please

     note

     that

     if

     you

     want

     to

     learn

     more

     about

     Paris

     or

     the

     capital

     city

     in

     general

    ,

     you

     can

     check

     out

     more

     detailed

     information

     and

     articles

    .

     The

     information

     provided

     here

     is

     based

     on

     widely

     recognized

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     challenges

    .

     Here

     are

     some

     potential

     trends

     that

     are

     likely

     to

     shape

     the

     field

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     ai

     integration

     with

     other

     technologies

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     integration

     with

     other

     technologies

     like

     sensors

    ,

     cameras

    ,

     and

     artificial

     intelligence

     assistants

    .

     This

     could

     lead

     to

     a

     greater

     understanding

     of

     the

     real

    -world

     applications

     of

     AI

     and

     help

     us

     make

     more

     informed

     decisions

    .
    


    2

    .

     Personal

    ized

     ai

    :

     With

     the

     help

     of

     big

     data

     and

     machine

     learning

    ,

     we

     may

     see

     a

     more

     personalized

     approach

     to

     AI

     that

     is

     tailored

     to

     each

     individual

     user

    's

     needs

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

    



```python
llm.shutdown()
```
