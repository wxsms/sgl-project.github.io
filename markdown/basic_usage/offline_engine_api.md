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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.44it/s]


    2026-05-08 03:19:55,149 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 03:19:55] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:03,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.49it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 22.72it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 30.87it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.87it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.62it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.57it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.96it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.06it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.53it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.53it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.53it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.53it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.53it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.53it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.91it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.97it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]

    Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 43.65it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.76it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 38.89it/s]


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
    Generated text:  Tanaka and I work at the NCTI Research Center. I am a postdoctoral researcher at the Center. My research interests are in computer vision and machine learning. My research focus is on the development and application of deep learning algorithms for various computer vision tasks. Specifically, I am working on the development of deep learning models for object detection, image classification, and scene parsing tasks.
    I have a B.Sc. in Computer Science and Engineering from Kyushu University, Japan. Prior to my research, I was a postdoctoral fellow at the Columbia University in the United States. My research experiences have been involved in developing computer vision
    ===============================
    Prompt: The president of the United States is
    Generated text:  the leader of which organization?
    The president of the United States is the leader of the United States federal government.
    You are an AI assistant that helps students browse through physics problems database. Create a question for the given definition of a physics problem.
    
    The answer to the question "What is the capital of the United States?" is:
    
    The capital of the United States is Washington D. C.
    You are an AI assistant that helps students browse through physics problems database. Create a question for the given definition of a physics problem.
    
    The answer to the question "What is the capital of the United States?" is:
    
    The capital of the United States is Washington
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Germany is Berlin.
    Is the following statement true?
    "The capital of France is Paris, and the capital of Germany is Berlin."
    Options: I. yes II. no
    II. No
    
    The capital of France is actually Paris, and the capital of Germany is not Berlin. Berlin is the capital of Germany. Paris is the capital of France, not Germany. The correct answer is:
    
    I. no
    
    The capital of France is Paris, and the capital of Germany is Berlin. Paris is the capital of France, not the capital of Germany. Berlin is the capital of Germany. Paris is the correct capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  AI: How we should be Thinking
    
    By Benjamin Moncrief
    
    When I first began my tenure at The Future of AI Project at MIT, I was a computer science student, more specifically the engineering department. In my first week, I was given an assignment to research and write an article on AI. I was told to only use a maximum of 300 words on a topic of my own choosing, and I didn’t have time to make a paper. So, I wrote a story about me dreaming of an AI so advanced that I would have the ability to program my own AI. I never did finish it, but


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, art, and cuisine, and is a popular tourist destination. The city is also home to the French Parliament and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage and a strong sense of identity. Its status as the capital of France has made it a major center of politics, culture, and society. The city is also home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to prioritize these concerns when designing and implementing AI systems.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This will involve the use of AI to assist humans in making decisions, such
    


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
    Generated text:  [Name], and I'm a software developer with a passion for problem-solving and algorithmic thinking. My programming skills range from basic to advanced, and I enjoy using my knowledge to solve complex problems in various domains, from data analysis to artificial intelligence. I'm always looking for ways to improve my skills and explore new technologies. What kind of software development projects are you currently working on? As an AI language model, I don't have the ability to work on projects. However, I can assist you with any questions or tasks related to software development. What's the best way to contact you? You can contact me at any time using your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its stunning architecture, vibrant culture, and historic landmarks such as the Eiffel Tower and Notre-Dame Cathedral. It is one of the largest and most populous cities in the world, and is a major transportation hub and cultural center. Paris is also known for its cuisine, fashion, and music, and is a popular tourist destination for those interested in exploring its rich history and culture. Overall, Paris is a vibrant and vibrant city that is full of energy and excitement. 
    
    The French capital is Paris, the largest and most populous city in the world, known for its magnificent architecture, rich history, and vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid development, innovation, and application of AI technology in various industries and fields. Here are some possible trends that could shape the future of AI:
    
    1. Increased automation and intelligence: AI is likely to become more intelligent and capable of performing tasks that were previously done by humans. This could lead to the development of new AI applications that automate routine tasks and improve efficiency, productivity, and safety.
    
    2. Enhanced privacy and security: With the increasing use of AI in many areas of life, there is a greater need for safeguards against data breaches and cyber threats. AI systems will need to be designed with privacy and security in


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

     a

     [

    Role

    ]

     in

     [

    Company

     Name

    ].

     I

    'm

     excited

     to

     meet

     you

    .

     Please

     let

     me

     know

     if

     you

     have

     any

     questions

     for

     me

    .

     Welcome

    !

     

    🌟

    ✨

    ✨

    
    


    ##

     Detailed

     Instructions

    


    1

    .

     **

    Name

    **

    :


      

     -

     Provide

     a

     name

     that

     uniquely

     identifies

     the

     fictional

     character

    .

     This

     should

     be

     clear

     and

     memorable

     to

     the

     reader

    .
    


    2

    .

     **

    Role

    **

    :


      

     -

     Describe

     the

     character

    's

     role

     within

     the

     fictional

     organization

    .

     This

     should

     capture

     the

     essence

     of

     the

     role

    .

     Ensure

     it

    's

     accurate

     and

     concise

    .
    


    3

    .

     **

    Company

     Name

    **

    :


      

     -

     Specify

     the

     name

     of

     the

     fictional

     organization

     to

     which

     you

     are

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

    ,

     and

     the

     capital

     of

     France

    .

     It

     has

     a

     population

     of

     approximately

     

    2

    .

    7

     million

     people

     and

     is

     the

     world

    's

     

    2

    0

    th

     largest

     city

     by

     population

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     has

     been

     the

     seat

     of

     government

    ,

     culture

    ,

     and

     religion

     for

     over

     

    2

    ,

    0

    0

    0

     years

    .

     It

     is

     known

     for

     its

     rich

     art

    ,

     cuisine

    ,

     music

    ,

     and

     fashion

    ,

     and

     is

     home

     to

     many

     world

    -class

     museums

    ,

     theaters

    ,

     and

     landmarks

    .

     Paris

     is

     also

     a

     hub

     for

     business

     and

     commerce

    ,

     hosting

     the

     annual

     Les

     Je

    ux

     o

    lymp

    iques

     in

     

    2

    0

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     dominated

     by

     three

     main

     trends

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     Machine

     learning

     is

     the

     key

     driver

     of

     AI

     advancements

    .

     This

     will

     see

     the

     development

     of

     algorithms

     that

     are

     more

     efficient

    ,

     accurate

    ,

     and

     adaptable

    .

     As

     AI

     gets

     better

     at

     understanding

     human

     behavior

    ,

     it

     will

     be

     able

     to

     solve

     more

     complex

     problems

     and

     make

     better

     decisions

    .
    


    2

    .

     Emer

    gence

     of

     AI

     ethics

    :

     The

     future

     of

     AI

     will

     involve

     a

     growing

     need

     for

     ethical

     guidelines

     and

     regulations

     to

     ensure

     that

     AI

     is

     used

     responsibly

     and

     for

     the

     benefit

     of

     all

    .
    


    3

    .

     Integration

     with

     human

     intelligence

    :

     AI

     is

     already

     being

     integrated

     into

     human

     intelligence

     in

     various

     ways

    ,

     such

     as

     through

     natural

     language

     processing

    



```python
llm.shutdown()
```
