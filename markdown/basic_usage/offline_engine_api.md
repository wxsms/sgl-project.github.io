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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.33it/s]


    2026-05-13 04:31:53,913 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 04:31:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:17,  2.88it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.78it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.35it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]

    Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 18.88it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 27.35it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 27.35it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 27.35it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 27.35it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s] 

    Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 36.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.16 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.15 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.15 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.15 GB):   3%|▎         | 2/58 [00:00<00:03, 18.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.14 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.13 GB):   9%|▊         | 5/58 [00:00<00:02, 21.20it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.13 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.13 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.13 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.30it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.12 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.11 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.11 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.11 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.11 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.10 GB):  21%|██        | 12/58 [00:00<00:01, 28.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.10 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s]Capturing num tokens (num_tokens=960 avail_mem=55.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.29it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=55.09 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=896 avail_mem=55.06 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=832 avail_mem=55.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=768 avail_mem=55.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=55.05 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=704 avail_mem=55.05 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]Capturing num tokens (num_tokens=640 avail_mem=55.04 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]Capturing num tokens (num_tokens=576 avail_mem=55.04 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]Capturing num tokens (num_tokens=512 avail_mem=55.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.04 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]Capturing num tokens (num_tokens=448 avail_mem=55.04 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.57it/s]Capturing num tokens (num_tokens=448 avail_mem=55.04 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=416 avail_mem=55.04 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=384 avail_mem=55.04 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=352 avail_mem=55.03 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=320 avail_mem=55.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 37.67it/s]Capturing num tokens (num_tokens=288 avail_mem=55.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 37.67it/s]Capturing num tokens (num_tokens=288 avail_mem=55.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=256 avail_mem=55.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=240 avail_mem=55.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=224 avail_mem=55.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]

    Capturing num tokens (num_tokens=208 avail_mem=55.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=192 avail_mem=55.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=192 avail_mem=55.01 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=176 avail_mem=55.00 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=160 avail_mem=55.00 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=144 avail_mem=55.00 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=128 avail_mem=55.00 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=112 avail_mem=54.99 GB):  71%|███████   | 41/58 [00:01<00:00, 42.41it/s]Capturing num tokens (num_tokens=112 avail_mem=54.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=96 avail_mem=54.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=54.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=64 avail_mem=54.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=48 avail_mem=54.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=32 avail_mem=54.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.12it/s]Capturing num tokens (num_tokens=32 avail_mem=54.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=28 avail_mem=54.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=24 avail_mem=54.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=20 avail_mem=54.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=16 avail_mem=54.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]Capturing num tokens (num_tokens=12 avail_mem=54.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.96it/s]

    Capturing num tokens (num_tokens=12 avail_mem=54.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=8 avail_mem=54.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.65it/s] Capturing num tokens (num_tokens=4 avail_mem=54.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=4 avail_mem=54.96 GB): 100%|██████████| 58/58 [00:01<00:00, 36.40it/s]


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
    Generated text:  Tanuka. I am a college student in the 12th grade and I am from New York. I am also a member of the Black Student Association. I am currently in the 10th grade.
    
    As a Black student, I have a lot of responsibilities and tasks to do, which includes my homework, assignments, and projects. I also have to attend school regularly and work hard to prepare for the academic year.
    
    In my free time, I like to play basketball with my friends and participate in various sports activities. I also enjoy reading books and watching movies, and I like to keep up with current events and news in
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, but how much do you really know about him? This week, we will take a look at a number of interesting facts about the US president, including his weight, favorite sports, and how he works with the military. Let's see what we can learn about him in this article!
    1. Barack Obama's Weight:
    The United States President Barack Obama is one of the most famous people in the world. He is 38 years old, but some of his other measurements are impressive. He is 178 cm tall and has an impressive metabolism. He weighs around 175 kg.
    2.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The word “Paris” is a blend of three words: Paris, capital, and city.
    
    Paris has the reputation of being a French city with a strong French influence. It is famous for its architecture, music, cinema, and fashion. It is a center of the artistic and cultural life in Europe.
    
    Paris is known for its opera house, the Louvre museum, and the Eiffel Tower. The city has been a center of international politics and has had a strong influence on world history and culture.
    
    However, Paris has faced some challenges in recent years, most notably its financial crisis and the rise of populism.
    
    In
    ===============================
    Prompt: The future of AI is
    Generated text:  hard to predict. At the moment, it seems likely that AI will continue to drive innovation, but at the same time, it will also pose significant risks to society. This is a difficult and complex topic, and it is important to approach it with an open mind and a willingness to engage in thoughtful discussions about its implications.
    One of the primary risks associated with AI is that it can lead to unintended consequences. For example, if an AI system is used to automate a dangerous or harmful task, such as predicting earthquakes, it can result in widespread panic and fear among people. This could have serious consequences for public health and safety.
    Another risk


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France, located in the south of the country. It is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. The city is a popular tourist destination and is home to many cultural institutions and events throughout the year. Overall, Paris is a fascinating and unique city that is well worth
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some potential trends that could be expected in the future of AI:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt in ways that are difficult for humans to do. This could lead to more efficient and effective AI systems that can perform tasks that are currently beyond the capabilities of humans.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations
    


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
    Generated text:  [insert character's name]. I'm a [insert character's occupation] with [insert career or experience]. I'm incredibly [insert character's personality trait or characteristic], and I thrive in [insert job role or situation] work. If you need anything, I'm always [insert answer to any question). Let me know if you have any other questions! [Insert character's name]. 
    
    Is there anything else you'd like to add to your introduction? Can you share more about your background or any specific experiences that may help others understand who you are? [Insert character's name]. 
    
    I look forward to interacting with you and learning
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the capital city of France. It is the largest city and the second most populous city in the European Union, and one of the world's most populous cities. It is also one of the oldest continuously inhabited cities in the world, and the birthplace of the French Revolution. The city is home to the Louvre Museum, the Eiffel Tower, Notre-Dame Cathedral, and numerous other historic and cultural landmarks. Paris has been a major center of politics, religion, arts, science, and commerce for over 2000 years, and continues to be an important city in the world today. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and unpredictable, but there are several potential trends that are currently under development and research. Here are some of the most notable:
    
    1. Advancements in machine learning: As AI becomes more powerful, researchers are focusing on how to improve its ability to learn and adapt to new situations. This includes developing more advanced models that can recognize patterns in complex data and make more accurate predictions.
    
    2. Increased integration with other technologies: AI is becoming more integrated with other technologies such as sensors, actuators, and communication systems. This integration will allow for more efficient and effective use of these technologies, leading to even more sophisticated AI systems.
    
    3. Cyber


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

    ].

     I

    'm

     a

     [

    career

     choice

    ]

     in

     the

     [

    industry

    ]

     field

    .

     I

    'm

     passionate

     about

     [

    major

    ity

     of

     the

     job

     description

    ]

     and

     enjoy

     [

    job

     description

    ]

     activities

    .

     I

    'm

     always

     looking

     for

     [

    what

     you

     want

     to

     improve

     upon

     or

     work

     towards

    ],

     and

     I

    'm

     always

     willing

     to

     learn

     and

     grow

    .

     I

    'm

     also

     [

    describe

     how

     you

    're

     approach

    able

     and

     friendly

    ].

     If

     anyone

     would

     like

     to

     meet

     me

     or

     learn

     more

     about

     me

    ,

     please

     feel

     free

     to

     reach

     out

    .

     As

     always

    ,

     thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     You

    'll

     find

     me

     here

     in

     [

    location

    ].

     Let

    's

     make

     a

     wonderful

     connection

    !

     [

    Name

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     and

     it

     is

     the

     largest

     city

     in

     the

     country

    .

     It

     is

     a

     bustling

     met

    ropolis

     with

     a

     rich

     cultural

     and

     historical

     heritage

    ,

     and

     it

     is

     known

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     Mus

    ée

     d

    ’

    Or

    say

    ,

     and

     the

     Centre

     Pom

    pid

    ou

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     a

     major

     economic

     center

    ,

     with

     many

     businesses

     and

     industries

     thriving

     in

     the

     city

    .

     The

     city

    ’s

     climate

     is

     mild

     and

     pleasant

    ,

     with

     a

     mild

     summer

     and

     cold

     winter

    .

     France

    's

     capital

     city

     is

     a

     vibrant

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     range

     of

     trends

     and

     technologies

    ,

     including

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     AI

     systems

     are

     likely

     to

     become

     even

     more

     accurate

     in

     their

     predictions

     and

     decisions

    ,

     particularly

     in

     areas

     like

     finance

    ,

     healthcare

    ,

     and

     transportation

    .
    


    2

    .

     Enhanced

     cognitive

     abilities

    :

     AI

     systems

     will

     likely

     become

     more

     capable

     of

     understanding

     and

     learning

     from

     complex

     data

     sets

    ,

     as

     well

     as

     developing

     their

     own

     algorithms

     and

     decision

    -making

     processes

    .
    


    3

    .

     Improved

     privacy

     and

     security

    :

     AI

     systems

     will

     need

     to

     be

     developed

     and

     deployed

     in

     a

     way

     that

     protects

     the

     privacy

     and

     security

     of

     individuals

     and

     organizations

    .
    


    4

    .

     Greater

     automation

    :

     AI

     systems

     are

     likely

     to

     be

     integrated

     into

     many

     industries

     and

     processes

    



```python
llm.shutdown()
```
