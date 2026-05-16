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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.60it/s]


    2026-05-16 05:05:02,569 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 05:05:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:07<06:43,  7.08s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:07<06:43,  7.08s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:07<06:43,  7.08s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:07<06:43,  7.08s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:07<06:43,  7.08s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:07<00:57,  1.08s/it]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:07<00:16,  2.82it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:07<00:06,  6.09it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:07<00:02,  9.85it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:07<00:01, 15.17it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:07<00:00, 21.62it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:07<00:00, 30.19it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:07<00:00, 30.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.63it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.31it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.84it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.84it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.84it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.84it/s]Capturing num tokens (num_tokens=704 avail_mem=75.74 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.84it/s]Capturing num tokens (num_tokens=704 avail_mem=75.74 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.60it/s]Capturing num tokens (num_tokens=640 avail_mem=75.64 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.60it/s]

    Capturing num tokens (num_tokens=576 avail_mem=75.42 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.60it/s]Capturing num tokens (num_tokens=512 avail_mem=75.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.60it/s]Capturing num tokens (num_tokens=480 avail_mem=75.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.60it/s]Capturing num tokens (num_tokens=480 avail_mem=75.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.63it/s]Capturing num tokens (num_tokens=448 avail_mem=75.01 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.63it/s]Capturing num tokens (num_tokens=416 avail_mem=74.53 GB):  52%|█████▏    | 30/58 [00:00<00:00, 31.63it/s]Capturing num tokens (num_tokens=384 avail_mem=74.53 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.63it/s]Capturing num tokens (num_tokens=352 avail_mem=74.53 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.63it/s]Capturing num tokens (num_tokens=320 avail_mem=74.52 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.63it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.52 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=288 avail_mem=74.36 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=256 avail_mem=74.36 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=240 avail_mem=74.35 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=224 avail_mem=74.35 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=208 avail_mem=74.34 GB):  60%|██████    | 35/58 [00:01<00:00, 35.27it/s]Capturing num tokens (num_tokens=208 avail_mem=74.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=192 avail_mem=74.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=176 avail_mem=74.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=160 avail_mem=74.34 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=144 avail_mem=74.33 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.15it/s]Capturing num tokens (num_tokens=128 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=112 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=96 avail_mem=74.33 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s] Capturing num tokens (num_tokens=80 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=64 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=48 avail_mem=74.32 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.48it/s]Capturing num tokens (num_tokens=48 avail_mem=74.32 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=32 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=28 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=24 avail_mem=74.31 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=16 avail_mem=74.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.27it/s]Capturing num tokens (num_tokens=16 avail_mem=74.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=12 avail_mem=74.30 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=8 avail_mem=74.29 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.02it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 35.92it/s]


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
    Generated text:  Tom. I am a 9-year-old boy. I have a friend named Mark. We're both in the same school and we play soccer together. Tom and Mark often play soccer at the same time. This afternoon, we play soccer and do lots of fun things together. We got lots of fun and I think we will win. In the afternoon, Mark and I go to a movie. Then we go to eat in a restaurant. We all get a big meal together. Tom and I have a fight, but we still go out to eat. We get lots of fun and I think we will win. On Saturday, we
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the United States. He or she is the leader of the country. He or she makes important decisions. The president is like the boss of the country. President Bush is a very important man. He has many important jobs. For example, he is the leader of the country. He also helps make important decisions. Mr. Bush is very interested in the economy. He wants to make the country stronger. He helps the president make important decisions for the country. He is very important. Mr. Bush is very good at his job. He always makes good decisions. Mr. Bush helped the country in
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the north of the country. The city is at the intersection of the Seine and the Marne river, between the 4th and 5th arrondissements.
    Paris was the first capital of France, and became the capital in 1790 after the storming of the Bastille. From 1800 to 1870, it was the capital of the French Empire, until the French Revolution, when it was renamed "Ville de Paris".
    Paris is one of the world's most important artistic and cultural centers, with a large and diverse artistic scene.
    The population
    ===============================
    Prompt: The future of AI is
    Generated text:  shaping up to be more complex and unpredictable than ever before. From secure, ethical AI to automation and the integration of AI with other forms of technology, the role of AI in driving innovation and creating value in many sectors is changing at an unprecedented pace.
    AI, which stands for Artificial Intelligence, is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, decision-making, and creativity.
    In recent years, AI has been used for a wide range of applications, including image and speech recognition, natural language processing, robotics, autonomous vehicles, and fraud detection


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm passionate about [job title] because [reason for passion]. I'm always looking for ways to [action or goal]. I'm [age] years old, and I'm [gender] (or [gender identity]). I'm [occupation] and I enjoy [occupation-related hobby or activity]. I'm [occupation-related skill or expertise]. I'm [occupation-related interest or passion]. I'm [occupation-related personality trait or quality]. I'm [occupation-related personal trait or quality]. I'm [occupation-related personal trait or quality]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic and cultural center with a rich history dating back to the Roman Empire. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also home to many world-renowned museums, theaters, and art galleries. Paris is a vibrant and diverse city with a rich cultural scene, and it is a popular tourist destination. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used to improve patient outcomes in areas such as diagnosis, treatment planning, and patient monitoring. As AI technology continues to improve, we can expect to see even greater use of AI in healthcare, with more personalized and accurate diagnoses and treatments.
    
    2. AI in manufacturing: AI is already being used to optimize production processes and improve quality control. As AI technology continues to improve, we can expect to see even greater use of AI in manufacturing,
    


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
    Generated text:  [Name], I'm a [age] year old [gender] who studies [major]. I enjoy [interests or hobbies] and [career]. How can I be a good fit for you? This is my first time meeting you and I really want to make a good first impression. You just have to be one of these people, I promise. I'm confident in my abilities and I can help you achieve your goals. Let's make this our first meeting and I'm looking forward to hearing about you. [Your Name] *Congratulations on your interview, it sounds like you have the skills and experience you're looking for.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is a cosmopolitan city with a rich history and culture.
    Paris is a major international city known for its architectural marvels, art galleries, and diverse culinary scene. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, among other landmarks. With its vibrant culture, rich history, and annual festival season, Paris continues to be a fascinating destination for tourists and locals alike. Overall, Paris offers a unique experience that cannot be found in any other city in Europe.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of factors, including advances in computing power, the development of more complex models, and the increasing use of AI in a wide range of industries. Here are some possible future trends in artificial intelligence:
    
    1. Increased focus on developing ethical and responsible AI: One of the biggest challenges facing AI is ensuring that it is used ethically and responsibly. As AI becomes more advanced, there will be a growing push for ethical guidelines and regulations to guide its development and use.
    
    2. Increased use of AI in healthcare: AI is already being used in a number of healthcare applications, including image analysis, diagnostic testing, and


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

    ]

     and

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    major

    ]

     [

    field

     of

     study

     or

     professional

     interest

    ].

     My

     experience

     ranges

     from

     [

    mention

     any

     relevant

     experience

     or

     skills

    ],

     and

     I

     am

     passionate

     about

     [

    mention

     a

     specific

     area

     of

     interest

     or

     hobby

     that

     interests

     you

    ].

     I

     enjoy

     [

    mention

     a

     hobby

     or

     activity

     that

     you

     enjoy

     doing

    ],

     and

     I

     have

     a

     strong

     work

     ethic

     and

     dedication

     to

     my

     goals

    .

     I

     am

     ambitious

     and

     always

     strive

     to

     reach

     my

     full

     potential

    ,

     and

     I

     am

     always

     willing

     to

     learn

     and

     grow

    .

     I

     am

     a

     [

    mention

     any

     skills

     or

     attributes

     that

     are

     unique

     to

     you

    ]

     who

     is

     determined

     to

     [

    mention

     a

     goal

     or

     challenge

     that

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     where

     the

     E

    iff

    el

     Tower

     stands

     tall

     and

     the

     Notre

    -D

    ame

     Cathedral

     is

     a

     renowned

     landmark

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     vibrant

     cultural

     scene

    .

     Paris

     is

     a

     city

     that

     has

     welcomed

     many

     important

     figures

    ,

     including

     Napoleon

     Bon

    ap

    arte

     and

     Paul

     Val

    é

    ry

    ,

     and

     has

     a

     long

    -standing

     reputation

     for

     being

     a

     hub

     for

     creativity

    ,

     art

    ,

     and

     literature

    .

     The

     city

     is

     also

     home

     to

     the

     Lou

    vre

     Museum

    ,

     one

     of

     the

     largest

     art

     museums

     in

     the

     world

    ,

     and

     is

     a

     popular

     destination

     for

     tourists

     from

     all

     over

     the

     world

    .

     Overall

    ,

     Paris

     is

     a

     city

     that

     has

     played

     a

     crucial

     role

     in

     shaping

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

     continued

     rapid

     advancements

     in

     both

     hardware

     and

     software

     development

    .

     The

     next

     generation

     of

     AI

     systems

     will

     likely

     use

     more

     advanced

     neural

     networks

     and

     machine

     learning

     algorithms

    ,

     which

     will

     enable

     them

     to

     learn

     and

     adapt

     more

     quickly

     and

     effectively

    .

     In

     addition

    ,

     there

     will

     be

     increased

     focus

     on

     ethical

     considerations

     and

     responsible

     AI

     development

    ,

     as

     concerns

     about

     privacy

    ,

     bias

    ,

     and

     transparency

     continue

     to

     grow

    .

     Additionally

    ,

     there

     will

     be

     a

     continued

     push

     towards

     making

     AI

     more

     accessible

     and

     user

    -friendly

    ,

     with

     more

     AI

     systems

     being

     able

     to

     interact

     with

     and

     understand

     human

     language

     and

     behaviors

    .

     Finally

    ,

     there

     will

     be

     continued

     development

     of

     AI

     systems

     that

     can

     operate

     in

     real

    -time

     and

     in

     distributed

     environments

    ,

     as

    



```python
llm.shutdown()
```
