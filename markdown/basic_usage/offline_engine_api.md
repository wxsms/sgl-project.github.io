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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.32it/s]


    2026-04-14 01:12:30,647 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 01:12:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.13it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:03<00:02, 13.13it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:03<00:01, 21.48it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:03<00:00, 30.58it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 39.78it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 48.82it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 48.82it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 48.82it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 48.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.50it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.33it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.62 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s] Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.76it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.58it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=512 avail_mem=131.58 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]

    Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.83it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.83it/s]

    Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 31.22it/s]

    Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]

    Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.19it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 41.09it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 36.18it/s]


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
    Generated text:  Darius and I am a freshman in high school. I have a passion for history and have been able to take a great deal of pride in my country. The more I learn about history, the more I realize how much we do have in common with other countries around the world, and the more we can learn from each other. I would love to travel to China or India, but I also have a question about China and India. India is an important part of the world, and the country has been very active in the peace and security of the world. India has had many conflicts in the past and is always looking for a better
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. ____
    A. True
    B. False
    Answer:
    A
    
    The fundamental political system of our country is ____.
    A. People's Congress System
    B. Multiparty Cooperation and Political Consultation System
    C. Democratic Centralism
    D. Regional Ethnic Autonomy System
    Answer:
    A
    
    In patients with acute myocardial infarction, the main reason for the hypoglycemic phenomenon is:
    A. Severe pain
    B. Reduced insulin release
    C. Insulin resistance
    D. Insulin deficiency
    E. Insulin sensitization
    Answer:
    D
    
    Patient, male, 50
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. London
    C. New York
    D. Tokyo
    Answer: A
    
    In a certain company, the total amount of salaries paid to employees each month is 200,000 yuan, which is evenly distributed among all employees. In November, the company changed the payroll system to collect monthly salaries from employees. Each employee receives their salary on the same day every month, and the system is set so that the last employee receives a full monthly salary on the 31st day of the month. How many employees work in the company? 
    A. 40
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the cloud
    
    Today, with the greatest number of users, big companies, and countless technologies, AI has become integral to almost everything, be it finance, healthcare, education, or travel. It is evident that the AI is transforming every aspect of life, and is therefore its future.
    
    AI, in the context of the future, is defined as “the technology enabling intelligent machines to perform tasks that would typically require human intelligence.” This is the definition that all AI experts hold and the one that the world is waiting for. But the future of AI is not in the cloud.
    
    The future of AI is not in the cloud
    
    
    The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [country]. I have a [job title] at [company name], and I enjoy [job title] work. I'm passionate about [job title] and I'm always looking for ways to [job title] more. I'm always eager to learn and grow, and I'm always looking for new challenges. What's your favorite hobby or activity? I'm a [age] year
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination, known for its rich history, art, and cuisine. The city is home to many famous French artists, writers, and musicians, and is considered one of the most beautiful cities in the world. Paris is also a major hub for international business and trade, with many major companies and institutions headquartered there. The city is known for its fashion industry, with many famous designers and bout
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their creations and work to ensure that they are developed in a way that is fair and responsible.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making
    


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
    Generated text:  [Name], and I'm a [character type] at [company name]. I'm always available to assist you with any questions or concerns you may have. My expertise is in [specific area of expertise]. Please let me know how I can help you today. What's your name? What's your profession? What's your area of expertise? What do you do for a living? How can I help you today? [Name] Welcome to [company name], I'm your friendly, knowledgeable, and helpful assistant who can assist you with any questions or concerns you may have. How can I help you today? [Name] How
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also famous for its wine industry and the city's diverse cultural scene. Paris has a rich history dating back to the Roman Empire and is now home to some of the world's most iconic landmarks. It has been a major center of European culture and diplomacy since the Middle Ages. Today, Paris continues to be a major economic and cultural hub. 
    
    Paris is also home to many famous art museums, including the Louvre, the Musée d'Orsay
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a proliferation of new applications and technologies that harness the power of machine learning and deep neural networks to solve complex problems in areas such as healthcare, finance, and robotics. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI continues to evolve, there will be increasing scrutiny of its impact on society. Governments, regulators, and civil society will need to be more transparent about the development and use of AI, and develop ethical guidelines that govern its use.
    
    2. Greater emphasis on AI ethics and accountability: As AI systems become more prevalent in our lives, there will be a growing need


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

    Your

     Profession

    ]

     with

     a

     passion

     for

     [

    Your

     Field

     of

     Work

    ].

     I

     enjoy

     [

    Your

     Strength

    s

    /

    Weak

    ness

    es

    /

    Op

    port

    unities

    /

    Op

    port

    unities

    ].

     I

     am

     a

     [

    Your

     Education

     Level

    ]

     who

     has

     [

    Your

     Highest

     Degree

    ].

     My

     [

    Your

     H

    obbies

     or

     Inter

    ests

    ]

     often

     take

     precedence

     over

     other

     aspects

     of

     my

     life

    ,

     but

     I

     am

     passionate

     about

     [

    Your

     Purpose

     or

     Goal

    ].

     I

     believe

     that

     education

     is

     a

     lifelong

     journey

     and

     I

     am

     always

     striving

     to

     improve

     myself

    .

     I

     am

     [

    Your

     Character

     Qu

    ot

    ient

    ].

     I

     am

     always

     open

     to

     feedback

     and

     willing

     to

     learn

     from

     my

     mistakes

    .

     Thank

     you

    
    
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

     and

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     Paris

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     art

    ,

     fashion

    ,

     and

     food

    .

     It

     is

     also

     famous

     for

     its

     romantic

     and

     historical

     attractions

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     a

     bustling

     and

     diverse

     city

     with

     a

     population

     of

     over

     

    2

    .

    7

     million

     people

    .

     It

     is

     the

     seat

     of

     government

     for

     France

     and

     is

     a

     major

     transportation

     hub

    ,

     home

     to

     many

     famous

     landmarks

    .

     Despite

     being

     a

     large

     city

    ,

     Paris

     maintains

     a

     tight

    -k

    nit

     community

     and

     is

     known

     for

     its

     various

     cultural

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dynamic

    ,

     and

     it

     is

     influenced

     by

     a

     wide

     range

     of

     factors

    ,

     including

     advances

     in

     technology

    ,

     new

     data

     sources

    ,

     and

     changing

     societal

     needs

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Ub

    iqu

    ity

     and

     Personal

    ization

    :

     As

     AI

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     pervasive

     use

     of

     AI

     in

     everyday

     life

    .

     Personal

    ized

     recommendations

    ,

     chat

    bots

    ,

     and

     voice

     assistants

     will

     become

     more

     common

    ,

     and

     AI

     will

     be

     used

     to

     personalize

     the

     experiences

     of

     users

    .
    


    2

    .

     Integration

     with

     Social

     Media

    :

     Social

     media

     platforms

     are

     already

     heavily

     reliant

     on

     AI

    ,

     and

     we

     can

     expect

     that

     the

     integration

     of

     AI

     into

     social

     media

     will

     continue

    .

     We

     can

     expect

    



```python
llm.shutdown()
```
