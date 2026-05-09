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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.08it/s]


    2026-05-09 00:00:56,743 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 00:00:56] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.44it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.44it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.95it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.03it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.10it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 30.69it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 30.69it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.60 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.60 GB):   7%|▋         | 4/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.60 GB):   7%|▋         | 4/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.60 GB):   7%|▋         | 4/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.58 GB):   7%|▋         | 4/58 [00:00<00:02, 19.06it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.58 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.55 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.98it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=53.54 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.54 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.54 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.54 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.54 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.53 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.53 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.53 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=53.52 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.50 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.50 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s]Capturing num tokens (num_tokens=960 avail_mem=53.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s] Capturing num tokens (num_tokens=896 avail_mem=53.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s]Capturing num tokens (num_tokens=832 avail_mem=53.51 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s]Capturing num tokens (num_tokens=768 avail_mem=53.50 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s]Capturing num tokens (num_tokens=704 avail_mem=53.50 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.71it/s]Capturing num tokens (num_tokens=704 avail_mem=53.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]Capturing num tokens (num_tokens=640 avail_mem=53.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]Capturing num tokens (num_tokens=576 avail_mem=53.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]Capturing num tokens (num_tokens=512 avail_mem=53.48 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]

    Capturing num tokens (num_tokens=480 avail_mem=53.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]Capturing num tokens (num_tokens=448 avail_mem=53.50 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.02it/s]Capturing num tokens (num_tokens=448 avail_mem=53.50 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=416 avail_mem=53.49 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=384 avail_mem=53.49 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=352 avail_mem=53.49 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=320 avail_mem=53.48 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.50it/s]Capturing num tokens (num_tokens=288 avail_mem=53.48 GB):  53%|█████▎    | 31/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=288 avail_mem=53.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=256 avail_mem=53.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.95it/s]

    Capturing num tokens (num_tokens=240 avail_mem=53.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=224 avail_mem=53.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=208 avail_mem=53.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=208 avail_mem=53.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=192 avail_mem=53.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=176 avail_mem=53.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=160 avail_mem=53.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.03it/s]

    Capturing num tokens (num_tokens=144 avail_mem=53.45 GB):  69%|██████▉   | 40/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=144 avail_mem=53.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.60it/s]Capturing num tokens (num_tokens=128 avail_mem=53.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.60it/s]Capturing num tokens (num_tokens=112 avail_mem=53.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.60it/s]Capturing num tokens (num_tokens=96 avail_mem=53.45 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.60it/s] Capturing num tokens (num_tokens=80 avail_mem=53.44 GB):  76%|███████▌  | 44/58 [00:01<00:00, 32.60it/s]Capturing num tokens (num_tokens=80 avail_mem=53.44 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=64 avail_mem=53.44 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.67it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=32 avail_mem=53.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=28 avail_mem=53.43 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=28 avail_mem=53.43 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=24 avail_mem=53.42 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=20 avail_mem=53.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.11it/s]

    Capturing num tokens (num_tokens=16 avail_mem=53.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=12 avail_mem=53.40 GB):  90%|████████▉ | 52/58 [00:01<00:00, 31.11it/s]Capturing num tokens (num_tokens=12 avail_mem=53.40 GB):  97%|█████████▋| 56/58 [00:01<00:00, 28.37it/s]Capturing num tokens (num_tokens=8 avail_mem=53.39 GB):  97%|█████████▋| 56/58 [00:01<00:00, 28.37it/s] Capturing num tokens (num_tokens=4 avail_mem=53.37 GB):  97%|█████████▋| 56/58 [00:01<00:00, 28.37it/s]Capturing num tokens (num_tokens=4 avail_mem=53.37 GB): 100%|██████████| 58/58 [00:01<00:00, 31.69it/s]


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
    Generated text:  Lisa and I am 18 years old. I am a teacher at a special school for children with special needs. I am dedicated to creating a positive and welcoming environment for all students. My interest in becoming a teacher began when I was in the seventh grade and I was always fascinated by the idea of helping others. As I grew older, I decided to take the next step and earn a degree in special education, and I graduated from college with a degree in special education in June 2019. I was now working as a special education teacher in New York City's New School for Special Education, where I help students who
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with great responsibilities and high expectations, and the political office is a political position. The term of the president is three years. The president must possess noble qualities such as being polite and trustworthy, brave and having good health. As the head of state, it is the president's duty to protect the interests of the people and lead the country. When the president is in power, he should always be in good health and sober, and be free from all kinds of disorders. When a president is in power, he must abide by laws and regulations, and follow the orders of the government. The president has the authority to propose major
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. A Parisian has a nickname or a nickname belonging to his country. For example, in the case of a Frenchman who has been born in America, he might be nicknamed "The American". Thus, an American Frenchman would be nicknamed "The French American".
    
    You are given a list of cities. Each city is associated with a nickname. Your task is to generate a sentence summarizing the city. The sentence should start with "City: " followed by the city name and conclude with " - " and the final city name. Note that a city name may be different from the nickname associated with it.
    
    Example:
    Input:
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, as AI has become more and more useful for people. However, today's AI is still not practical and efficient. What's the best way to use AI? Do you know the way to develop AI? Here, I have some suggestions. First of all, don't be afraid of AI. The more you can see it, the more you can understand it. Secondly, always keep an open mind. Don't follow the opinions of others blindly. If you find an issue, you should first ask yourself why. Thirdly, don't be afraid of criticism. People can be like diamonds, and the more you appreciate them,


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love to Do], and I'm always looking for new challenges and opportunities to grow and learn. I'm a [What I Like to Do] person, and I'm always looking for ways to make the world a better place. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill] [Number] [Field] [What I Love to Do] [What I Like
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its rich history, including the French Revolution and the French Revolution Museum. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée d'Art Moderne. Paris is a vibrant and diverse city with a rich cultural scene, and it is a popular tourist destination. The city is also known for its cuisine, including French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we are likely to see more widespread adoption of AI technologies. This could include things like smart home devices, self-driving cars, and virtual assistants like Siri or Alexa.
    
    2. Greater emphasis on ethical and responsible AI: As AI becomes more integrated into our lives, there will be a greater emphasis on ensuring that AI is used ethically and responsibly.
    


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
    Generated text:  Sarah and I am a writer. I have always been fascinated by stories, both real and imagined. My interest in storytelling led me to pursue a career in writing and publishing. In addition to writing, I am an avid reader and have a love for learning new things and exploring new places. What other hobbies or interests do you have besides writing and reading? Dear Sarah,
    
    Thank you for stopping by my blog to share your humble self. I'm Sarah, a self-taught writer who has been exploring the creative landscape with a passion that goes beyond words. My journey has taken me from the pages of journals to the pages of novels,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest and most populous city in France, and the country's political, economic, and cultural center. Paris is renowned for its rich history, stunning architecture, and vibrant culture. It has a diverse range of neighborhoods and districts, and its most famous landmarks include the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe. The city is also known for its annual cultural festivals and events, such as the Les Quatre Grenadiers festival, the Bastille Day parade, and the Musée du Louvre's Trafalgar Square. Paris is the world's 9th largest
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  diverse and rapidly evolving, with numerous trends shaping the field's direction and impacting its applications. Here are some possible future trends in AI:
    
    1. Increased Automation: AI technology is getting better at simulating complex human behavior and decision-making. As a result, automation is becoming more prevalent, with AI systems performing tasks that were previously done by humans. This trend is expected to continue, with more applications of AI being developed that rely on machine learning algorithms and predictive modeling.
    
    2. Autonomous Vehicles: Autonomous vehicles are becoming more advanced, with the ability to navigate complex environments and make decisions based on real-time data. This trend is expected to continue as


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

    specific

     job

     or

     role

    ]

     at

     [

    Company

     Name

    ].

     I

     have

     [

    number

     of

     years

     of

     experience

    ]

     of

     experience

     in

     [

    specific

     field

     of

     expertise

    ],

     and

     I

     specialize

     in

     [

    specific

     area

     of

     expertise

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     learn

     and

     grow

    ,

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

     with

     anyone

     who

     wants

     to

     learn

    .

     I

     am

     a

     friendly

    ,

     person

    able

    ,

     and

     effective

     communicator

    .

     I

     am

     always

     ready

     to

     help

     and

     make

     things

     easier

     for

     others

    .

     I

     am

     a

     reliable

     and

     dependable

     team

     member

     who

     is

     always

     there

     to

     assist

     and

     support

     others

    .

     I

     am

     enthusiastic

     and

     passionate

     about

     [

    specific

     area

     of

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

     and

     vast

     gardens

    .
    


    I

    'm

     sorry

    ,

     but

     there

     seems

     to

     be

     an

     error

     in

     my

     previous

     response

    .

     The

     E

    iff

    el

     Tower

     is

     actually

     the

     iconic

     E

    iff

    el

     Tower

    ,

     not

     the

     E

    iff

    el

     Line

    .

     It

     is

     a

     long

     cable

    -st

    ayed

     bridge

     that

     spans

     the

     River

     Se

    ine

     in

     Paris

    ,

     France

    .

     
    


    Paris

     is

     known

     for

     its

     historical

     architecture

    ,

     such

     as

     the

     Lou

    vre

     Museum

    ,

     Notre

     Dame

     Cathedral

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     and

     the

     Ch

    amps

    -E

    lys

    ées

    .

     It

     is

     also

     home

     to

     numerous

     museums

    ,

     such

     as

     the

     Mus

    ée

     de

     l

    '

    Or

    anger

    ie

    ,

     the

     Mus

    ée

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     constantly

     evolving

    ,

     with

     many

     potential

     directions

     to

     explore

    .

     Some

     of

     the

     most

     exciting

     and

     promising

     trends

     in

     AI

     include

    :
    


    1

    .

     Personal

    ized

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     will

     see

     more

     personalized

     AI

     systems

     that

     are

     able

     to

     understand

     and

     adapt

     to

     individual

     user

     preferences

     and

     behaviors

    .

     This

     will

     allow

     for

     more

     efficient

     and

     effective

     use

     of

     resources

     in

     a

     variety

     of

     domains

    ,

     such

     as

     healthcare

    ,

     education

    ,

     and

     transportation

    .
    


    2

    .

     Autonomous

     vehicles

    :

     As

     technology

     continues

     to

     advance

    ,

     autonomous

     vehicles

     will

     become

     more

     and

     more

     common

    ,

     with

     the

     ability

     to

     navigate

     roads

     and

     intersections

     with

     greater

     accuracy

     and

     reliability

    .

     This

     will

     have

     a

     significant

     impact

     on

     transportation

     and

     traffic

    



```python
llm.shutdown()
```
