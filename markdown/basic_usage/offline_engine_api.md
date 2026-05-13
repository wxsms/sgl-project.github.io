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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]


    2026-05-13 07:20:47,830 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 07:20:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.46it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.00it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.09it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.35it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.49it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.49it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.49it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:04, 11.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.67 GB):   3%|▎         | 2/58 [00:00<00:04, 11.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.67 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.17 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:03, 14.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.74it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.10it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.74it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 44.70it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 44.70it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 46.10it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 47.18it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]

    Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 47.66it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.42it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 39.51it/s]


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
    Generated text:  Dusan, a software engineer with a passion for data. I like to play games and keep up with new technologies. Here's my first project. I made a web application to help people learn about the use of Python programming language in game development. The app is called "PyGame Learn". It is designed to be easy to use, with a user-friendly interface and interactive features. The app features a variety of exercises and challenges that help learners learn about the basics of Python programming in game development, such as creating a simple game, writing a basic game loop, and understanding the basics of classes and objects. The app also includes tutorials and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position in the government of the United States, and the position of the president of the United States was established by the Constitution of the United States. The president is appointed by the president of the United States, the president of the United States is elected by the people, and the president is held in office for a term of five years. This president is not elected by the people or members of the legislature, and the president is not bound by any legislation of the government. The vice president of the United States is a position in the government of the United States and the vice president of the United States is elected by the people. The vice
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A: Paris B: Nice C: London D: Moscow To determine the capital of France, let's analyze each option step by step:
    
    A: Paris - This is the capital of France. Paris is the largest city in France, known for its historical architecture, museums, and many landmarks.
    
    B: Nice - This is not the capital of France. Nice is a city in France but is not the capital.
    
    C: London - This is not the capital of France. London is the capital of the United Kingdom, not France.
    
    D: Moscow - This is not the capital of France. Moscow is the capital of Russia, not
    ===============================
    Prompt: The future of AI is
    Generated text:  not just here, but here now. We are at the dawn of a new era that will redefine what it means to be human. While the potential is immense, there is a dark side as well. As the AI industry continues to evolve, there are certain ethical considerations that must be addressed to ensure that the development and deployment of AI are fair, just, and secure.
    
    AI is a complex technology that has the potential to revolutionize the way we live and work. However, it is also a technology that has the potential to be exploited by those who seek to control and manipulate the systems they are built upon.
    
    AI is a technology that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, located in the south of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its rich history, art, and cuisine. Paris is a major tourist destination and a cultural hub, attracting millions of visitors each year. The city is also home to the French Parliament and the French government. 
    
    B. False is incorrect because Paris is indeed the capital city of France, and it is a major tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential impact of their technology on society and take steps to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with other technologies: AI is likely to become more integrated
    


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
    Generated text:  [Name]. I'm a [Occupation] with a passion for [What interests you about your career]. I believe in [Why you're passionate about your field], and I'm always eager to learn and grow. I enjoy [What you like to do to relax], and I'm always looking for new adventures and opportunities to explore the world. I'm always willing to collaborate and work with others, and I'm always eager to learn and improve. I'm a [General Objective], and I'm excited to have the opportunity to work with you. How about you, [Career Objective]? I'm a [Occupation] with a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly known as the "City of Light" and is known for its rich history, beautiful architecture, and vibrant culture. Paris is located in the Île-de-France region of France and is one of the most popular tourist destinations in the country. It is famous for its landmarks, such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also home to many famous museums and art galleries, as well as a vibrant food scene and shopping district. The French capital is a city that is truly unique and fascinating to visitors. Its architecture, culture, and cuisine make it a must
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and dynamic, and it is likely to continue to evolve and change. Some possible trends in AI include:
    
    1. Increased specialization: AI will become more specialized, as it will focus on specific tasks and applications. This will enable AI to perform tasks more accurately and efficiently, making it more useful in various fields.
    
    2. Improved security: AI will continue to rely on machine learning algorithms to detect and prevent cyber attacks. This will require the development of new encryption techniques and more secure algorithms.
    
    3. Increased collaboration: AI will become more integrated into the decision-making process, as it will be able to analyze data and make recommendations based on that


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

    name

    ]

     and

     I

     am

     a

     professional

     writer

    .

     I

     have

     a

     Bachelor

    's

     degree

     in

     English

     from

     [

    un

    iversity

     name

    ],

     and

     I

     have

     been

     working

     in

     the

     publishing

     industry

     for

     [

    number

    ]

     years

    .

     I

     am

     passionate

     about

     writing

     and

     have

     a

     love

     for

     storytelling

    .

     I

     love

     exploring

     different

     genres

    ,

     exploring

     new

     ways

     to

     structure

     my

     writing

    ,

     and

     trying

     new

     things

     in

     my

     creative

     writing

    .

     I

     am

     a

     great

     collabor

    ator

     and

     love

     to

     work

     with

     other

     writers

    .

     I

     hope

     to

     write

     more

     books

     in

     the

     future

     and

     help

     others

     to

     grow

     their

     writing

     skills

    .

     How

     can

     I

     become

     a

     better

     writer

    ?

     I

     want

     to

     expand

     my

     knowledge

     of

     the

     world

     of

     writing

     and

     better

     understand

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

     and

     is

     shaped

     by

     a

     variety

     of

     factors

    .

     Some

     potential

     trends

     in

     the

     field

     include

    :
    


    1

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     it

     is

     becoming

     increasingly

     important

     to

     address

     privacy

     and

     security

     concerns

    .

     This

     includes

     developing

     new

     privacy

    -pres

    erving

     techniques

     and

     tools

    ,

     as

     well

     as

     addressing

     issues

     of

     data

     security

     and

     integrity

    .
    


    2

    .

     Enhanced

     natural

     language

     processing

    :

     With

     the

     increasing

     reliance

     on

     AI

     in

     everyday

     life

    ,

     there

     is

     a

     growing

     need

     for

     natural

     language

     processing

     (

    N

    LP

    )

     systems

     that

     can

     better

     understand

     and

     generate

     human

    -like

     language

    .

     This

     includes

     developing

     new

     N

    LP

     models

     that

     can

     better

     handle

     complex

     natural

     language

     queries

     and

     improve

     the

     accuracy

     of

    



```python
llm.shutdown()
```
