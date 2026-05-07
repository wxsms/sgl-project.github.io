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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.78it/s]


    2026-05-07 02:09:03,637 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 02:09:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.81it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.54it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]

    Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.23it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 24.98it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 32.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 15.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 15.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 15.37it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   7%|▋         | 4/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   7%|▋         | 4/58 [00:00<00:03, 16.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.00 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.40it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.99 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.98 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 31.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=960 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s] Capturing num tokens (num_tokens=896 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.57it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.48it/s]

    Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.48it/s]

    Capturing num tokens (num_tokens=512 avail_mem=73.66 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.48it/s]Capturing num tokens (num_tokens=512 avail_mem=73.66 GB):  50%|█████     | 29/58 [00:01<00:02, 10.12it/s]Capturing num tokens (num_tokens=480 avail_mem=73.67 GB):  50%|█████     | 29/58 [00:01<00:02, 10.12it/s]Capturing num tokens (num_tokens=448 avail_mem=73.67 GB):  50%|█████     | 29/58 [00:01<00:02, 10.12it/s]Capturing num tokens (num_tokens=416 avail_mem=72.97 GB):  50%|█████     | 29/58 [00:01<00:02, 10.12it/s]Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  50%|█████     | 29/58 [00:01<00:02, 10.12it/s]Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  57%|█████▋    | 33/58 [00:01<00:01, 12.86it/s]Capturing num tokens (num_tokens=352 avail_mem=72.68 GB):  57%|█████▋    | 33/58 [00:01<00:01, 12.86it/s]Capturing num tokens (num_tokens=320 avail_mem=72.67 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.86it/s]Capturing num tokens (num_tokens=288 avail_mem=72.67 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.86it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.67 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.86it/s]Capturing num tokens (num_tokens=240 avail_mem=72.66 GB):  57%|█████▋    | 33/58 [00:02<00:01, 12.86it/s]Capturing num tokens (num_tokens=240 avail_mem=72.66 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.89it/s]Capturing num tokens (num_tokens=224 avail_mem=72.66 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.89it/s]Capturing num tokens (num_tokens=208 avail_mem=72.66 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.89it/s]Capturing num tokens (num_tokens=192 avail_mem=72.66 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.89it/s]Capturing num tokens (num_tokens=176 avail_mem=72.65 GB):  66%|██████▌   | 38/58 [00:02<00:01, 16.89it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=160 avail_mem=72.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=144 avail_mem=72.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=128 avail_mem=72.65 GB):  72%|███████▏  | 42/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=112 avail_mem=72.64 GB):  72%|███████▏  | 42/58 [00:02<00:00, 18.50it/s]Capturing num tokens (num_tokens=112 avail_mem=72.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.83it/s]Capturing num tokens (num_tokens=96 avail_mem=72.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.83it/s] Capturing num tokens (num_tokens=80 avail_mem=72.64 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.83it/s]Capturing num tokens (num_tokens=64 avail_mem=72.63 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.83it/s]Capturing num tokens (num_tokens=48 avail_mem=72.63 GB):  79%|███████▉  | 46/58 [00:02<00:00, 21.83it/s]

    Capturing num tokens (num_tokens=48 avail_mem=72.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=32 avail_mem=72.63 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=28 avail_mem=72.62 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=20 avail_mem=72.19 GB):  86%|████████▌ | 50/58 [00:02<00:00, 25.12it/s]Capturing num tokens (num_tokens=20 avail_mem=72.19 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.19it/s]Capturing num tokens (num_tokens=16 avail_mem=72.19 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.19it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.19it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.19it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.19it/s]

    Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:02<00:00, 30.89it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:02<00:00, 21.61it/s]


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
    Generated text:  Xiao Zhang. I'm an 18-year-old male student. I have been at the university for the past two years. I have a deep love for playing online games and I always want to join in my favorite games, but I find them to be difficult to control. I often experience mood swings and moodiness, which affect my ability to concentrate. I feel that I'm losing my passion for playing games, and I wish to give up the habit of playing them. Can you provide me with some advice on how to overcome my addiction to playing games and regain my passion for them? It would be great if you could include exercises
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the total number of jellybeans in the entire population. He randomly selects $1000$ jellybeans and finds that $25$ percent of them are red, $15$ percent are yellow, and $10$ percent are white. He believes that the total number of red jellybeans is the largest category because it is the only category with a positive percentage. Is the president's belief accurate based on his sample? To determine the accuracy of the president's belief that the total number of red jellybeans is the largest category, we need to compare the percentages of red jellybeans found in his sample to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it has a population of 2.7 million. The city is divided into 14 neighborhoods, each with 250,000 residents. If the mayor of Paris decides to implement a new policy where each resident is given 2.5 units of budget per year for their health insurance, how many total units of budget will the city receive from all residents combined over the next year? To determine the total units of budget that the city of Paris will receive from all residents combined over the next year, we need to follow these steps:
    
    1. Calculate the total population of Paris.
    2. Determine the
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to look different from what we currently see. While the technology is rapidly advancing, the risks are there. The MIT Media Lab team’s new AI and robotics research is a model for the future.
    
    The MIT Media Lab’s roboticist Christopher Barnes, a professor of AI and robotics, is the brain behind the new technology. “The real breakthrough will be in the next five to 10 years,” Barnes told Wired. “We’re trying to capture the rapid progress of the last 15 years to create a new type of humanoid robot.”
    
    The team, which includes researchers from MIT’s Media Lab, the University


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and [job title]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many famous French artists, writers, and musicians, and is known for its rich history and cultural heritage. Paris is a vibrant and dynamic city that continues to grow and evolve as a major global city. The city is also known for its diverse cuisine, including French, Italian, and Mediterranean dishes. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations and tasks. This could lead to more efficient and effective decision-making.
    
    2. Enhanced machine learning capabilities: AI systems are likely to become more capable of learning from large amounts of data and making more accurate predictions and decisions. This could lead to more personalized and effective solutions to complex problems.
    
    3. Greater emphasis on ethical considerations: As AI systems become more integrated with human intelligence, there will be a greater emphasis on ethical considerations and
    


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
    Generated text:  [Name] and I am a [Background Information] person. I have a love for [Hobbies or Interests]. What's your name, and what kind of person are you? I'm an [Age] year old, [Gender] and [Occupation]. And I have a lot of [特长或才能] that I'm passionate about. What can you tell me about yourself? Your background, hobbies, interests, talents, and achievements can help me better understand your personality. I'm looking forward to having a conversation with you about what's important to you. What are some of your favorite activities or hobbies? What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history, stunning architecture, and vibrant cultural scene.
    Paris is the capital of France and is famous for its rich history, stunning architecture, and vibrant cultural scene. Its unique blend of classical and modern elements is evident in its skyline, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its rich culinary traditions, including famous Parisian dishes like crème brûlée and pastis. Paris is a city of contrasts, with its beautiful parks and gardens, and its iconic landmarks, including Notre-Dame Cathedral and the Arc de Triomphe
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and promising, with many potential applications and developments shaping the way we live, work, and interact with technology. Here are some possible future trends in AI that are currently being explored and discussed:
    
    1. More advanced machine learning and deep learning algorithms: With the development of new hardware and software, there is potential for even more powerful and complex machine learning and deep learning algorithms to be developed and used. This could lead to improved accuracy in natural language processing, image recognition, and other areas of AI.
    
    2. Improved privacy and security: As AI systems become more integrated into our daily lives, there is an increasing need for enhanced security and privacy


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

    job

     title

    ]

     at

     [

    company

     name

    ].

     I

     have

     been

     working

     in

     [

    position

    ]

     for

     [

    number

     of

     years

    ]

     years

    .

     I

     have

     always

     been

     passionate

     about

     [

    something

     I

     enjoy

     doing

    ].

     My

     love

     for

     [

    something

    ]

     has

     led

     me

     to

     pursue

     [

    other

     interest

    ],

     which

     I

     believe

     is

     [

    reason

     for

     interest

    ].

     I

     am

     a

     [

    type

     of

     person

    ]

     and

     I

     have

     always

     been

     willing

     to

     [

    character

     trait

    ].

     I

     am

     always

     looking

     for

     ways

     to

     improve

     my

     [

    skill

     or

     ability

    ].

     I

     am

     a

     [

    type

     of

     person

    ]

     and

     I

     am

     always

     eager

     to

     learn

     new

     things

    .

     I

     am

     a

     [

    type

     of

     person

    ]

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     both

     land

     and

     population

     in

     the

     country

     and

     is

     home

     to

     the

     government

    ,

     government

     offices

    ,

     and

     many

     of

     France

    's

     other

     institutions

     and

     services

    .

     France

     is

     home

     to

     

    6

    2

     million

     people

    ,

     making

     it

     the

     largest

     country

     in

     Europe

     by

     population

    ,

     larger

     than

     all

     other

     major

     European

     countries

     combined

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .

     Paris

     is

     located

     in

     the

     northern

     part

     of

     the

     country

     on

     the

     Left

     Bank

     of

     the

     Se

    ine

    .

     The

     historic

     center

     of

     the

     city

     is

     the

     heart

     of

     its

     political

    ,

     economic

    ,

     and

     cultural

     life

    .

     It

     is

     a

     major

     transportation

     hub

     and

     the

     

    9

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     accuracy

     and

     precision

    :

     AI

     systems

     are

     becoming

     more

     accurate

     in

     making

     predictions

     and

     decisions

    .

     This

     is

     due

     to

     advancements

     in

     machine

     learning

     algorithms

    ,

     which

     allow

     AI

     systems

     to

     learn

     from

     data

     and

     improve

     over

     time

    .
    


    2

    .

     Increased

     integration

     with

     human

     skills

    :

     As

     AI

     systems

     become

     more

     integrated

     with

     human

     skills

    ,

     they

     are

     likely

     to

     become

     even

     more

     intelligent

     and

     capable

    .

     This

     integration

     could

     lead

     to

     more

     efficient

     and

     effective

     decision

    -making

    .
    


    3

    .

     Automation

     of

     tasks

    :

     AI

     systems

     are

     likely

     to

     become

     more

     automated

     and

     efficient

    ,

     reducing

     the

     need

     for

     manual

     intervention

    .

     This

     could

     lead

     to

     significant

     job

     displacement

     and

     economic

     changes

    



```python
llm.shutdown()
```
