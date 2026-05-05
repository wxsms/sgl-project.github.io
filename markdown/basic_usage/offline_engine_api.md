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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.21it/s]


    2026-05-05 08:20:59,063 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 08:20:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.61s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.21it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.21it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.45it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.45it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.45it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.45it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.21it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.89it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.02it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.02it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 35.14it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.55it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.10it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.94it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.11it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.78it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.54it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.94it/s]


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
    Generated text:  7th grade. I'm a hard working girl with lots of questions, and I am currently researching how to become a doctor. Can you help me with my medical exam?
    
    Sure, I'd be happy to help! First, I'll ask some questions for you to answer. What is your favorite color, and what is your favorite food? Next, I'd like to hear about a book you have read recently. Then, I'll ask you a few questions about your favorite mode of transportation. What is your favorite mode of transportation? Finally, I'd like to ask you about your favorite hobby. What is your favorite hobby?
    ===============================
    Prompt: The president of the United States is
    Generated text:  visiting Japan. At the ambassador's meeting in the United States, the president talks for $2$ hours with the ambassador, then continues for $4$ hours with the deputy ambassador, and finally spends the next $2$ hours for a tour of the Japanese parliament. If the president spent $2$ hours more in meetings than in tours, how many hours in total did the president spend with the ambassador and the deputy ambassador? Let's denote the total number of hours the president spent with the ambassador and the deputy ambassador as \( x \).
    
    According to the problem, the president spent \( x \) hours with the ambassador, \( x
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the name of the country's capital city? To determine the name of the country's capital city, I'll follow these steps:
    
    1. Recall that the capital of France is Paris.
    2. Think about the European Union's capital city, which is Brussels.
    3. Note that the United Kingdom's capital city is London.
    4. Recall that the country with a smaller capital city would be one with a larger capital city.
    5. Therefore, the capital city of the country with Paris as its capital is Brussels, the capital of the European Union.
    
    So, the name of the country's capital city is Brussels. The
    ===============================
    Prompt: The future of AI is
    Generated text:  about to take off. We have learned how to code, and we have learned about robots. But how will we apply this knowledge to solve the problems that we have faced in the past? With the recent advent of the Internet of Things, our ability to connect the world more simply will allow us to better understand the world and create a future we can all live in.
    The Internet of Things (IoT) is a technology that connects all the devices and devices that have been connected to the Internet. It is a network that allows various types of devices to share information with one another. This technology has the potential to revolutionize the way we


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I love [job title] because [reason for passion]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I enjoy [hobby or activity] because [reason for enjoyment]. I'm always looking for new ways to challenge myself
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major center for art, culture, and fashion, and is home to many world-renowned museums, theaters, and restaurants. Paris is a cultural and economic hub of France and a major tourist destination. It is also home to the French Parliament, the French Supreme Court, and the French National Assembly. The city is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to the creation of new jobs, but it could also lead to the displacement of human workers in certain areas.
    
    2. AI ethics and privacy concerns: As AI technology becomes more advanced, there will be a growing concern about its impact on society. This includes questions about privacy, bias, and the potential for AI to be used for
    


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
    Generated text:  [Name], and I am [Job Title]. I am a [Age] year old [Country], and I currently live in [City/State]. I hold a Bachelor’s Degree in [Field of Study], and I have experience in [Field of Study]. I am [Department] with [Role], and I enjoy [Hobby/Interest/Passion]. I am passionate about [Your Passion], and I am constantly learning new things and growing as a person. I love to [Learn/Experience/Share Knowledge], and I am always looking for new ways to improve myself. I am someone who is [Personality Type],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that hosts the Eiffel Tower, the Louvre Museum, and numerous other iconic landmarks. It is a vibrant and bustling metropolis with a rich cultural and historical legacy. Paris is a major economic and financial hub and a major destination for tourism and art. With its beautiful architecture, bustling streets, and rich history, Paris is a fascinating city that is sure to leave a lasting impression. It is often referred to as the "City of a Thousand Springs" due to its numerous waterfalls, grottos, and scenic overlooks. The city is known for its coffee culture and has been the birthplace of numerous
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and we cannot predict with certainty what the future looks like. However, several trends are likely to shape the field in the coming years:
    
    1. Integration of AI with other fields: AI is expected to become more integrated with other fields, including healthcare, finance, education, and transportation. This integration will enable more efficient and effective use of AI technologies, and it could lead to new opportunities for innovation.
    
    2. Increased focus on ethics and responsible AI: As AI becomes more integrated into our lives, it is likely to raise ethical concerns. There will be a need for new standards and guidelines to ensure that AI is used in a responsible


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

    /an

     [

    Occup

    ation

    ].

     I

    'm

     

    2

    5

     years

     old

    ,

     and

     I

     love

     [

    Occup

    ation

    ].

     My

     hobbies

     include

     [

    h

    obbies

    ],

     and

     I

     enjoy

     [

    activities

    ].

     I

    'm

     constantly

     learning

    ,

     trying

     new

     things

    ,

     and

     always

     striving

     to

     grow

     and

     improve

     myself

    .

     I

    'm

     friendly

    ,

     kind

    ,

     and

     always

     ready

     to

     help

     others

    .

     I

    'm

     passionate

     about

     [

    Occup

    ation

    ],

     and

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     make

     a

     difference

    .

     So

    ,

     what

    's

     your

     story

    ?

     How

     are

     you

     planning

     to

     achieve

     your

     goals

    ?

     Welcome

     to

     my

     world

    ,

     and

     I

    'm

     excited

     to

     have

     you

     join

     me

     on

     this

     adventure

    .

     Let

    's

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     France

    's

     capital

     city

    :
    


    Paris

    ,

     the

     capital

     city

     of

     France

    ,

     is

     the

     most

     populous

     city

     in

     Europe

    ,

     with

     over

     

    2

     million

     residents

    .

     
    


    This

     statement

     encaps

    ulates

     the

     core

     facts

     about

     Paris

    ,

     including

     its

     status

     as

     the

     capital

     and

     its

     population

     size

    ,

     as

     well

     as

     the

     year

     of

     the

     statement

    .

     It

    's

     concise

     and

     informative

    ,

     providing

     readers

     with

     a

     clear

     understanding

     of

     the

     key

     information

     about

     Paris

    .

     
    


    Please

     let

     me

     know

     if

     you

     would

     like

     me

     to

     revise

     or

     expand

     on

     this

     statement

     further

    .

     Let

     me

     know

    !

     

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    😊

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     there

     are

     several

     possible

     trends

     that

     experts

     predict

    .

     Here

     are

     some

     of

     the

     most

     likely

     developments

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     AI

    -powered

     decision

    -making

     systems

     are

     likely

     to

     become

     more

     prevalent

     as

     more

     businesses

     adopt

     AI

     for

     tasks

     such

     as

     fraud

     detection

    ,

     customer

     service

    ,

     and

     human

     resource

     management

    .
    


    2

    .

     AI

     integration

     with

     human

     values

    :

     AI

     systems

     will

     become

     more

     integrated

     with

     human

     values

    ,

     such

     as

     empathy

     and

     compassion

    ,

     to

     make

     decisions

     that

     are

     more

     in

     line

     with

     ethical

     principles

    .
    


    3

    .

     Greater

     emphasis

     on

     privacy

     and

     security

    :

     AI

     systems

     are

     likely

     to

     become

     more

     data

    -intensive

    ,

     and

     governments

     and

     businesses

     will

     need

     to

     take

     measures

     to

    



```python
llm.shutdown()
```
