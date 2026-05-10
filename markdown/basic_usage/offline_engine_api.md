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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.86it/s]


    2026-05-10 08:20:36,173 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 08:20:36] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.94it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.94it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 11.00it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.07it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.14it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.20it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.08it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.08it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.08it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.44it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.44it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.44it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.44it/s]

    Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.44it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.16it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.16it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.16it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 33.16it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.43it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 35.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.88it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.88it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 36.14it/s]


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
    Generated text:  Paul Andersen, and I’m a licensed clinical social worker. My practice is primarily focused on working with people who are experiencing trauma and crisis. I help people and families to navigate through the complexities of crisis situations and to support them in the healing process. My work includes trauma-informed care, crisis counseling, and crisis group work. I offer a trauma-informed approach to working with trauma, and I am committed to using evidence-based practices. I have a Bachelor’s degree in psychology and a Master’s degree in social work with a specialization in mental health from the University of California, Los Angeles. My experience with mental health and trauma informed care
    ===============================
    Prompt: The president of the United States is
    Generated text:  in the White House, and this week, the country is going to celebrate the 40th anniversary of the inauguration of President Bill Clinton. The White House team has an envelope with a password that is the word "40th". The president, knowing that the password is encoded with ASCII, uses a simple method to decode it. The password is encoded by adding the ASCII value of 'B' (which is 57) to the ASCII value of 'T' (which is 84), then subtracting the ASCII value of 'B' from the result. What is the decoded password? To decode the password "
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, the second largest city in the country, and has an area of 642 km². The distance between Paris and the Loire river is 100 km. The distance between Paris and the Eiffel Tower is 320 m (there is no height, it's only the tower). The distance between the nearest city, Lyon, and the Loire is 340 km. The distance between Lyon and the Eiffel Tower is 380 m (again, there is no height). How many people did the city of Paris kill in 2010? To calculate
    ===============================
    Prompt: The future of AI is
    Generated text:  coming
    
    Three promising approaches to AI can make a dent in the problem of runaway AI.
    
    Tech companies and governments are up in arms against the world's biggest AI project, which could bring computer chips and software to the masses that could lead to the rapid advancement of artificial intelligence (AI), which could lead to a new age of computing.
    
    Recently, the International Rescue Committee (IRC) announced that it had successfully used its AI-powered platform to detect and rescue refugees from Syria. The AI platform, designed by a team of academics from the University of Washington, used a large amount of data to enable the platform to recognize refugees as they entered into the


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its vibrant arts scene and culinary traditions. Paris is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. The city is home to many cultural institutions, including museums, theaters, and concert halls, and is a popular destination for tourists and locals alike. Overall,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of how their technology is being used and how it may impact society.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as the Internet of Things (IoT) and the cloud. This will allow for more
    


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
    Generated text:  [Name], and I'm a [Skill or Qualification] enthusiast. I'm always looking for new ways to improve my skills and stay ahead of the curve. I'm not afraid to ask questions and seek out new challenges. I enjoy learning and sharing my knowledge with others, and I'm always looking for ways to contribute to the world. I believe in the power of continuous improvement and I'm always eager to share my ideas and insights. Thank you. How would you describe the type of person you are? As a neutral self-introduction, I would describe you as a person who is open-minded, curious, eager to learn,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, the City of Arts and Sciences, and the City of Democracy. Its population is around 2.1 million, and it is located in the eastern part of the country at the foot of the Pyrenees mountain range. The city is surrounded by the famous cathedrals of Notre-Dame de Paris and Sainte-Chapelle, and it is home to several world-renowned museums such as the Louvre, Musée d'Orsay, and the Musée d'art Moderne. Paris is also known for its rich cultural and artistic heritage, with its numerous museums
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and complex, with opportunities and risks that lie ahead. Here are some possible trends that could shape the landscape of AI:
    
    1. Increased focus on ethical AI: As the need to ensure that AI systems are aligned with ethical principles becomes more pressing, there is a growing emphasis on how to make AI systems that are fair, transparent, and accountable. This could lead to the development of ethical AI systems that are designed to protect human rights and promote social justice.
    
    2. More diverse AI systems: With the global population aging and the rise of artificial intelligence in various industries, there is a growing need for AI systems that are better equipped to handle


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

    ].

     I

    'm

     a

     [

    job

     title

    ]

     at

     [

    Company

     Name

    ].

     I

    've

     always

     been

     passionate

     about

     [

    something

    ],

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     myself

    .

     I

    'm

     an

     [

    interest

    /

    cur

    iosity

    ]

     and

     always

     have

     a

     fresh

     perspective

     to

     bring

     to

     the

     table

    .

     I

    'm

     always

     willing

     to

     learn

     and

     grow

    ,

     no

     matter

     how

     small

     the

     steps

    .

     I

     thrive

     on

     challenges

     and

     always

     push

     myself

     to

     grow

     and

     improve

    .

     Thank

     you

     for

     considering

     me

     for

     a

     position

    .

     
    


    Remember

    ,

     I

    'm

     just

     a

     [

    character

    ]

     with

     a

     few

     years

     of

     experience

    ,

     but

     I

     believe

     in

     my

     ability

     to

     contribute

     to

     [

    Company

     Name

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Pal

    me

    "

     or

     "

    La

     Petite

     Pal

    me

    "

     and

     is

     located

     in

     the

     heart

     of

     the

     city

     of

     Paris

    .

     Paris

     is

     one

     of

     the

     most

     famous

     and

     historic

     cities

     in

     the

     world

    ,

     famous

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     world

    -f

    amous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     Notre

     Dame

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

     It

     is

     a

     major

     financial

     and

     cultural

     center

     of

     France

    ,

     home

     to

     the

     French

     Parliament

    ,

     the

     World

     Trade

     Center

    ,

     and

     many

     museums

     and

     attractions

     throughout

     the

     city

    .

     Paris

     is

     known

     for

     its

     vibrant

     culture

    ,

     art

    ,

     music

    ,

     and

     cuisine

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     constantly

     evolving

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     shape

     our

     world

     in

     the

     next

     few

     decades

    :
    


    1

    .

     Increased

     AI

     ethics

    :

     AI

     will

     become

     more

     ethical

     and

     accountable

    ,

     with

     greater

     consideration

     given

     to

     the

     potential

     harm

     that

     AI

     can

     cause

    .

     This

     may

     lead

     to

     more

     stringent

     regulations

     on

     AI

     development

     and

     deployment

    ,

     as

     well

     as

     greater

     transparency

     and

     accountability

     in

     AI

     systems

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     will

     play

     a

     key

     role

     in

     improving

     the

     quality

     and

     efficiency

     of

     healthcare

    ,

     with

     more

     advanced

     AI

     systems

     being

     developed

     for

     diagn

    osing

     diseases

    ,

     predicting

     patient

     outcomes

    ,

     and

     improving

     treatment

     plans

    .

     There

     will

     also

     be

     greater

     emphasis

     on

     data

     privacy

     and

     security

     in

    



```python
llm.shutdown()
```
