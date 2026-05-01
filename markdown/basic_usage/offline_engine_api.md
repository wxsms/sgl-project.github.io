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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.46it/s]


    2026-05-01 03:13:27,128 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-01 03:13:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:14,  4.47s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.34it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.34it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.76it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.73it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.82it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.82it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.26 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 22.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.21 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=960 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s] Capturing num tokens (num_tokens=896 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.19it/s]Capturing num tokens (num_tokens=832 avail_mem=76.20 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=704 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=640 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=576 avail_mem=76.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=512 avail_mem=76.15 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.75it/s]Capturing num tokens (num_tokens=512 avail_mem=76.15 GB):  50%|█████     | 29/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=480 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 37.49it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=416 avail_mem=76.16 GB):  50%|█████     | 29/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=384 avail_mem=76.16 GB):  50%|█████     | 29/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=384 avail_mem=76.16 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=352 avail_mem=76.14 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=320 avail_mem=75.57 GB):  57%|█████▋    | 33/58 [00:00<00:00, 35.47it/s]Capturing num tokens (num_tokens=288 avail_mem=75.49 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=256 avail_mem=75.49 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.47it/s]Capturing num tokens (num_tokens=240 avail_mem=75.49 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.47it/s]

    Capturing num tokens (num_tokens=240 avail_mem=75.49 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=224 avail_mem=75.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=208 avail_mem=75.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=192 avail_mem=75.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=176 avail_mem=75.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=160 avail_mem=75.48 GB):  66%|██████▌   | 38/58 [00:01<00:00, 38.83it/s]Capturing num tokens (num_tokens=160 avail_mem=75.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=144 avail_mem=75.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=128 avail_mem=75.32 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=112 avail_mem=75.31 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=96 avail_mem=75.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s] Capturing num tokens (num_tokens=80 avail_mem=75.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.75it/s]

    Capturing num tokens (num_tokens=80 avail_mem=75.30 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=64 avail_mem=75.30 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=48 avail_mem=75.29 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=32 avail_mem=75.29 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=28 avail_mem=75.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=24 avail_mem=75.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.19it/s]Capturing num tokens (num_tokens=24 avail_mem=75.28 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=20 avail_mem=75.28 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=16 avail_mem=75.28 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=12 avail_mem=75.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=8 avail_mem=75.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=75.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 43.76it/s]Capturing num tokens (num_tokens=4 avail_mem=75.27 GB): 100%|██████████| 58/58 [00:01<00:00, 44.18it/s]Capturing num tokens (num_tokens=4 avail_mem=75.27 GB): 100%|██████████| 58/58 [00:01<00:00, 38.79it/s]


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
    Generated text:  a.k.a 'Lee' and I am a self-taught music producer, songwriter, and musician with a passion for creating innovative soundscapes and immersive experiences with music. My background is in digital music production, and I have worked with a diverse range of artists, including artists from the pop, rock, and R&B genres.
    
    As a producer, I specialize in crafting songs that not only grab the attention of the listener but also leave a lasting impression. My goal is to create music that resonates with the listener on an emotional level and encourages them to explore their own personal experiences.
    
    My professional journey has led me to have various
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person who occupies the highest position of leadership in a country. The office of the president is appointed by the president of the United States from among the members of the United States Senate.
    Since there is no federal election, no one is directly elected to the office of president. In 1787, the Constitution of the United States provided for the election of the president through the electoral college. Under this system, a person can be chosen to be president by an electoral college in which all the states and the non-voting territories of the United States are represented.
    The President of the United States serves for a term of four years.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of France is Paris. What is the function of the word "capital" in this sentence? Capital refers to a city's highest administrative or political status. In this case, the word "capital" refers to the city of Paris, which has the highest administrative status in France.
    
    The function of the word "capital" in this sentence is to convey that Paris is a city with the highest administrative or political status in France. It also suggests that this status is in a higher position or importance compared to other cities in France. 
    
    The use of "capital" in this sentence is often used to indicate the highest authority or
    ===============================
    Prompt: The future of AI is
    Generated text:  being shaped by the work of the thinkers and thinkers behind it, as well as the algorithms that power it. This week, we are highlighting some of the key areas where we are seeing AI evolve and progress.
    We are excited to be showcasing some of the work of the thinkers behind it.
    What can we do to improve our AI system?
    The field of AI is always evolving, and it’s great that we can keep learning and adapting as we go. We are excited to showcase the latest advances in AI, as well as new ways to improve our algorithms.
    Have you had a chance to experience AI in action? Share your feedback on this


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


    Generated text:  [Name] and I am a [occupation] who has been working in the [industry] for [number] years. I have always been passionate about [occupation] and have always wanted to [goal]. I am always looking for new challenges and opportunities to grow and learn. I am always eager to learn and adapt to new situations. I am a [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. Its population is around 2.5 million, making it the most populous city in Europe. The city is also home to many international organizations and institutions, including the European Union and the United Nations. Paris is a popular tourist destination, with millions of visitors annually
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more prevalent in various industries, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more advanced, it is likely to be used in even more sophisticated ways, such as personalized medicine and
    


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
    Generated text:  [Your Name]. I am a [Your Profession] with a passion for [Your Hobby or Interest], and I love the [Your Hobby or Interest] because [Your Remarkable Feature or Personal Trait]. I look forward to making you smile and having a good time with you. How can I help you today? Remember, [Your Profession] is [Your Profession], and I want you to know that I am always here for you. [Your Name] with a heart full of kindness and a smile. [Your Name] is a [Your Profession] with a passion for [Your Hobby or Interest], and I love the [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the third-largest urban area in the European Union after London and Rome. The city is home to the most visited tourist attraction in the world and is known for its rich history, architecture, cuisine, and cultural heritage. It is also the cultural and economic center of the country. The French government is based in Paris, and the city is home to important museums, galleries, and concert halls. The city is a hub for business and finance, with many famous companies, businesses, and entertainment venues located within its boundaries. As of 2021, Paris has a population of over
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be very exciting and transformative. Here are some possible trends that AI is likely to experience in the next few years:
    
    1. Increased AI ethics and privacy concerns: As AI is becoming more and more prevalent, there will be an increasing amount of data being collected and analyzed by AI systems. This data could be used to build an understanding of human behavior and emotions, but it could also raise concerns about privacy, data security, and bias.
    
    2. AI will become more autonomous and self-driving: As AI technology improves, we may see more self-driving cars, drones, and other forms of autonomous systems become more common. These systems will


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

     specialize

     in

     [

    Your

     specialty

    ].

     I

     have

     been

     working

     in

     this

     field

     for

     [

    X

     years

    ]

     and

     have

     a

     passion

     for

     [

    Your

     passion

    ].

     If

     you

    're

     looking

     for

     help

     with

     your

     [

    Your

     problem

    ],

     I

    'm

     your

     go

    -to

     person

    .

     You

     can

     rest

     assured

     that

     I

    'm

     here

     to

     assist

     you

     in

     any

     way

     I

     can

    .

     How

     can

     I

     assist

     you

     today

    ?

     [

    Your

     Name

    ]

     [

    Your

     specialty

    ]

     [

    Your

     passion

    ]

     [

    Your

     problem

    ]

     I

    'm

     here

     to

     help

    .

     [

    Your

     Name

    ]

     [

    Your

     specialty

    ]

     [

    Your

     passion

    ]

     [

    Your

     problem

    ]

     I

    'm

     here

     to

     assist

     you

     with

     [

    Your

     problem

    ].

     [

    Your

     Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (A

    )

     Paris

     is

     the

     capital

     city

     of

     France

    .

     


    (B

    )

     Paris

     is

     the

     largest

     city

     in

     France

    .

     


    (C

    )

     Paris

     is

     the

     oldest

     city

     in

     France

    .

     


    (D

    )

     Paris

     is

     the

     capital

     of

     a

     country

    .


    (C

    )

     Paris

     is

     the

     oldest

     city

     in

     France

    .

     


    The

     capital

     of

     France

     is

     Paris

    ,

     which

     is

     the

     oldest

     city

     in

     the

     European

     Union

     and

     one

     of

     the

     oldest

     continuously

     inhabited

     cities

     in

     the

     world

    .

     Paris

     is

     the

     second

     most

     populous

     city

     in

     the

     European

     Union

     and

     is

     the

     second

    -largest

     city

     in

     France

    .

     It

     has

     been

     the

     capital

     city

     of

     France

     since

     

    1

    8

    0

    4

    ,

     when

     Napoleon

     Bon

    ap

    arte

     became

     Emperor

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     and

     there

     is

     no

     telling

     what

     new

     developments

     will

     happen

     in

     the

     near

     or

     long

     term

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

     AI

     landscape

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     healthcare

     by

     improving

     patient

     outcomes

     and

     reducing

     costs

    .

     This

     could

     include

     the

     use

     of

     AI

     to

     analyze

     medical

     images

    ,

     predict

     disease

     outcomes

    ,

     and

     personalize

     treatment

     plans

     for

     individual

     patients

    .

     However

    ,

     there

     are

     also

     concerns

     about

     privacy

    ,

     ethics

    ,

     and

     bias

     in

     AI

    -driven

     healthcare

     applications

    .
    


    2

    .

     AI

     for

     manufacturing

    :

     AI

     has

     the

     potential

     to

     significantly

     improve

     manufacturing

     processes

     by

     autom

    ating

     tasks

    ,

    



```python
llm.shutdown()
```
