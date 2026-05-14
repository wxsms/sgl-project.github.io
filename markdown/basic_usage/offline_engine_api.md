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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]


    2026-05-14 18:17:38,100 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 18:17:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.18it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.35it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.65it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.20it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 36.78it/s]

    Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.28it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  57%|█████▋    | 33/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.20it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.20it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  83%|████████▎ | 48/58 [00:01<00:00, 47.33it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  91%|█████████▏| 53/58 [00:01<00:00, 47.42it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 47.32it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 41.73it/s]


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
    Generated text:  Calypso and I am a unique and powerful 3D model of an organism. I am a simple humanoid with human-like features, such as a head, arms, and legs, which are connected by a flexible body. I am alive and have the ability to reproduce, grow in a variety of environments, and communicate with others in the world.
    The part of the model that has been altered to have a purpose is the left arm, which now has a smaller head, small hands, and smaller arms. This modification was done to make the model more friendly and adaptable. By reducing the size and shape of the left arm, I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what sport to promote for the upcoming summer. The president of Brazil has a sports enthusiast like him named Edson, who loves both football and baseball. The president of Mexico loves baseball and is a fan of the Olympics. What sport will the president of the United States choose to promote?
    
    The president of the United States would likely choose to promote baseball. The president of Brazil, Edson, is a fan of both football and baseball, which makes them a strong candidate for promoting baseball. The president of Mexico, who loves baseball, is another strong contender for promoting baseball. However, since the president of Brazil is the one who
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, known for its iconic landmarks, iconic people, and delicious cuisine. Paris is a city that has been an important center for French culture and history for over a thousand years. In Paris, one can experience the sights, sounds, and smell of a city that is filled with a rich and unique culture.
    Paris is a major tourist destination, with millions of tourists visiting each year. It is a popular destination for both locals and visitors alike, with many attractions that attract visitors from all over the world. Paris offers an array of experiences, from cultural events and museums to world-class dining and entertainment.
    One of the most notable attractions in Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  currently driven by the development of deep learning algorithms, which enable AI systems to learn from data and perform complex tasks. In this article, we will explore the concept of deep learning and how it is currently being used in various industries to revolutionize the way we interact with technology and learn. We will also examine the potential future of AI and how it could transform the way we live and work in the years to come. Finally, we will explore the role of the AI industry in creating jobs and the potential impact on society as a whole. Can you provide a summary of the current state of AI and how it is being used to revolutionize industries


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and its rich history dating back to the Middle Ages. It is the largest city in France and the third-largest city in the world by population. Paris is also known for its vibrant culture, art, and cuisine, and is a major tourist destination. The city is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its annual festivals and events, such as the Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of devices and systems, but there is potential for even greater integration with other technologies such as IoT, blockchain, and quantum computing.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there is a risk of data breaches and privacy violations. There is potential for new technologies to be developed that can help to protect against these risks.
    
    3. Increased focus on ethical considerations: As AI systems become more complex and
    


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
    Generated text:  [Your Name], and I'm a [insert your age here] year-old [insert your occupation or profession here]. I've always been interested in learning and trying new things, whether it's trying out new recipes, trying out new sports or games, or trying out new hobbies or activities. I'm passionate about reading, and I love to learn about new books, movies, and TV shows. I also love to travel, and I've traveled to [insert your travel destination here], [insert a few other places you've visited if relevant here]. I'm a[insert your favorite hobby or skill here] and I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and the Louvre Museum. Paris is the eighth-largest city in the European Union and a major global cultural center. The city is home to many of Europe’s cultural institutions and landmarks, including the Louvre, Notre-Dame Cathedral, and the Opera House. The city is known for its rich history, art, and architecture, and is a popular tourist destination. As of 2023, Paris has a population of approximately 2.8 million residents. With its diverse culture and stunning skyline, Paris is a must-visit destination for anyone seeking an experience of Europe’s rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of technological advancements and shifts in how we interact with technology. Here are a few possible future trends in AI:
    
    1. Increased focus on ethical AI: As more ethical concerns emerge, we may see an increased focus on AI that is more transparent, accountable, and accountable to its users. This could include things like AI-powered chatbots that are designed to provide real-time support, or AI systems that can assist in ethical decision-making.
    
    2. Integration of machine learning with other technologies: The AI field is already finding ways to integrate machine learning with other technologies like blockchain, IoT, and cloud computing. These integration efforts could


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

    insert

     name

    ],

     and

     I

    'm

     a

     [

    insert

     occupation

     or

     role

    ]

     with

     a

     passion

     for

     [

    insert

     something

     that

     interests

     you

    ,

     such

     as

     music

    ,

     sports

    ,

     or

     books

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     idea

     of

     [

    insert

     something

     new

     and

     exciting

    ],

     and

     I

    'm

     constantly

     exploring

     new

     ways

     to

     help

     people

     around

     me

    .

     Whether

     it

    's

     through

     my

     work

    ,

     social

     media

    ,

     or

     my

     writing

    ,

     I

    'm

     always

     trying

     to

     find

     new

     ways

     to

     connect

     with

     others

     and

     make

     the

     world

     a

     better

     place

    .

     I

     hope

     you

     find

     me

     an

     inspiration

     and

     look

     forward

     to

     meeting

     you

    .

     [

    Insert

     name

    ]

     (

    or

     whatever

     you

     want

     to

     call

     yourself

    )

    !

     [

    Insert

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     is

     factual

     because

     it

     is

     a

     widely

     recognized

     fact

     about

     Paris

     that

     it

     is

     the

     capital

     of

     France

    .

     The

     other

     city

     mentioned

     in

     the

     question

    ,

     Marseille

    ,

     is

     the

     second

    -largest

     city

     in

     France

     and

     is

     located

     in

     the

     Rh

    one

     Valley

    ,

     which

     is

     not

     the

     capital

     city

     of

     France

    .

     
    


    This

     statement

     is

     concise

    ,

     as

     it

     contains

     only

     three

     words

     and

     is

     easy

     to

     remember

    .

     It

     also

     represents

     a

     widely

     accepted

     fact

     about

     Paris

     that

     is

     widely

     known

     in

     the

     English

     language

    .

     
    


    A

     more

     concise

     statement

     could

     be

    :

     "

    Paris

     is

     the

     capital

     of

     France

    ."

     
    


    This

     statement

     is

     also

     factual

    ,

     as

     it

     is

     accurate

     and

     commonly

     known

    .

     It

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     possibilities

    ,

     but

     it

     is

     difficult

     to

     predict

     with

     certainty

     which

     trends

     will

     emerge

    .

     However

    ,

     here

     are

     some

     potential

     future

     trends

     that

     experts

     believe

     could

     shape

     the

     development

     of

     artificial

     intelligence

    :
    


    1

    .

     Improved

     Machine

     Learning

    :

     One

     of

     the

     most

     exciting

     trends

     in

     AI

     is

     the

     continued

     improvement

     of

     machine

     learning

     algorithms

    .

     As

     AI

     technologies

     advance

    ,

     we

     will

     see

     even

     more

     advanced

     algorithms

     that

     can

     learn

     and

     improve

     on

     their

     own

    .

     This

     will

     lead

     to

     more

     accurate

     and

     efficient

     predictions

    ,

     as

     well

     as

     better

     decision

    -making

     in

     various

     fields

    .
    


    2

    .

     Integration

     of

     AI

     into

     Human

     Work

    :

     AI

     is

     already

     starting

     to

     become

     more

     integrated

     into

     human

     work

    ,

     with

     applications

     in

     areas

     like

    



```python
llm.shutdown()
```
