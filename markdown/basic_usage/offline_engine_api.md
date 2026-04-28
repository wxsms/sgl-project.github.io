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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.08it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.07it/s]


    2026-04-28 06:59:37,471 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 06:59:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.40it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.27it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.44 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.43 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=116.43 GB):   7%|▋         | 4/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.40 GB):   7%|▋         | 4/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.39 GB):   7%|▋         | 4/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.38 GB):   7%|▋         | 4/58 [00:00<00:02, 19.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.38 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.78it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.37 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.37 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.36 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.36 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.33 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.32 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.32 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=116.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.29 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.29 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=960 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s] Capturing num tokens (num_tokens=896 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=832 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=768 avail_mem=116.29 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=704 avail_mem=116.29 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=704 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=640 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=576 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]

    Capturing num tokens (num_tokens=512 avail_mem=116.27 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=480 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=448 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=448 avail_mem=116.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=416 avail_mem=116.28 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=384 avail_mem=116.28 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=352 avail_mem=116.27 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=320 avail_mem=116.27 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=288 avail_mem=116.27 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.13it/s]Capturing num tokens (num_tokens=288 avail_mem=116.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=256 avail_mem=116.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=240 avail_mem=116.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]

    Capturing num tokens (num_tokens=224 avail_mem=116.26 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=208 avail_mem=116.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.24it/s]Capturing num tokens (num_tokens=192 avail_mem=116.25 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=176 avail_mem=116.25 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=160 avail_mem=116.25 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=144 avail_mem=116.24 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=128 avail_mem=116.24 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  71%|███████   | 41/58 [00:01<00:00, 43.10it/s]Capturing num tokens (num_tokens=112 avail_mem=116.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=96 avail_mem=116.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=64 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=48 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.33it/s]Capturing num tokens (num_tokens=32 avail_mem=116.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=28 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=24 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=20 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=16 avail_mem=116.21 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=12 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=8 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s] Capturing num tokens (num_tokens=4 avail_mem=116.20 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=4 avail_mem=116.20 GB): 100%|██████████| 58/58 [00:01<00:00, 38.05it/s]


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
    Generated text:  Tom and I'm a student in the University of Missouri. I'm currently in the third year of my undergraduate program in the School of Education. I have completed six courses in each of three different subjects: Psychology, Sociology, and Reading. I'm preparing for my final exam which is due on March 10, 2019. Can you provide a summary of your education and your career aspirations? Tom, thank you for considering me for the job.
    Of course, thank you for taking the time to fill out this application. My name is Tom, and I am a student at the University of Missouri majoring in Psychology
    ===============================
    Prompt: The president of the United States is
    Generated text:  3 feet 4 inches tall. His office is 15 feet long. How tall would he be if he stood in his office?
    To determine the height of the president of the United States if he stood in his office, we need to follow these steps:
    
    1. Convert the president's height into inches.
    2. Subtract the office length from the converted height.
    
    First, we convert the president's height from feet and inches to just inches. The president is 3 feet 4 inches tall, which can be written as:
    \[ 3 \text{ feet} \times 12 \text{ inches/foot}
    ===============================
    Prompt: The capital of France is
    Generated text:  _____
    A. Paris
    B. Rome
    C. London
    D. Geneva
    Answer:
    A
    
    Which of the following statements about the changes in China's population is correct? [ ]
    A. During the Great Depression, the number of people was declining.
    B. Population migration has caused population aging.
    C. The population is growing slowly.
    D. The number of people is increasing.
    Answer:
    C
    
    The most significant social factor affecting population distribution is [ ]
    A. Geographic Environment
    B. Political Factors
    C. Economic Factors
    D. Cultural Factors
    Answer:
    C
    
    In some regions of the world, the population has
    ===============================
    Prompt: The future of AI is
    Generated text:  not just limited to its capabilities in traditional fields like natural language processing (NLP) and computer vision. As technology continues to evolve, AI is increasingly finding its way into other areas of society, such as healthcare, education, and more. With this in mind, it is essential to understand the impact of AI on society and the future of AI.
    
    In healthcare, AI is already being used to improve patient care and outcomes. For example, AI-powered chatbots can help patients manage their medical records and provide personalized recommendations for treatment plans. Additionally, AI-powered imaging software can help radiologists diagnose and analyze medical images more quickly and accurately. Finally


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a popular tourist destination and a major economic center. The city is known for its fashion industry, art scene, and food culture. Paris is a city that is constantly evolving and changing, with new developments and attractions being added to the city's list of attractions. It is a city that is a must-visit for anyone interested
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to the needs of humans. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see more widespread use of AI in healthcare, with more personalized and efficient treatments.
    
    
    


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
    Generated text:  [Your Name], and I'm a [insert a profession or field of work here, like "Psychologist," "Educator," "Artist," etc.]. I'm a [insert your occupation here, like "Doctor," "Lawyer," etc.]. I've always been drawn to [insert a characteristic or hobby of yours, like "music," "reading," "traveling," etc.]. I'm passionate about [insert something that relates to your profession or occupation, like "teaching," "writing," "creating art," etc.]. And as a [insert your profession or field of work], I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A detailed answer would be as follows:
    
    Paris, officially called the "City of Love" or "Espace de la Loi", is the capital city of the French Republic. It is located in the northwest of the country, at the junction of the River Seine and the Bicêtre Canal, in the eastern part of the Paris region. The city is situated on the left bank of the river, at the end of the Seine. It has a population of about 2.2 million, including the surrounding suburbs. Paris is the third largest city in the world by population, and one of the most populated
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid technological advances, increasingly sophisticated algorithms, and an increasing focus on ethical and social implications. Here are some possible future trends in AI:
    
    1. AI will continue to become more autonomous and self-aware: Autonomous AI systems will become more sophisticated and self-aware, capable of understanding and responding to human language, emotions, and motivations. This will require a significant investment in AI research and development to develop these capabilities.
    
    2. AI will become more prevalent in healthcare: AI will play an increasingly important role in healthcare, from diagnosing diseases to developing personalized treatment plans. AI will also improve the accuracy and speed of medical imaging and


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

     come

     from

     [

    insert

     hometown

     or

     place

    ].

     I

     love

     [

    insert

     one

     thing

     about

     yourself

    ].

     I

    'm

     [

    insert

     age

    ]

     years

     old,

     [

    insert

     occupation

    ],

     [

    insert

     hobbies

    ].

     I

     have

     [

    insert

     number

     of

     pets

    ]

     pets

    ,

     [

    insert

     one

     specific

     pet

    ].

     I

     also

     love

     [

    insert

     one

     thing

     you

     enjoy

     doing

    ]

     and

     I

    'm

     looking

     forward

     to

     [

    insert

     something

     new

     that

     you

     have

     in

     mind

    ].

     
    


    What

     is

     your

     favorite

     way

     to

     entertain

     yourself

    ?

     I

     love

     to

     [

    insert

     one

     thing

     that

     interests

     you

     or

     something

     that

     can

     help

     you

     relax

    ].


    [

    insert

     a

     brief

     summary

     of

     your

     character

    ,

     such

     as

     your

     name

    ,

     occupation

    ,

     hobbies

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Gren

    oble

    ,"

     the

     "

    City

     of

     Love

    ,"

     and

     the

     "

    City

     of

     Light

    ."


    What

     is

     the

     official

     language

     of

     France

    ?

     French

    .


    France

    's

     capital

     is

     Paris

    .

     France

     is

     also

     a

     country

     of

     

    6

    5

     million

     people

    .

     In

     Paris

    ,

     you

     can

     see

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

     and

     the

     Palace

     of

     Vers

    ailles

    .

     The

     national

     symbols

     of

     France

     are

     the

     E

    iff

    el

     Tower

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     E

    iff

    el

     Tower

    .

     In

     Paris

    ,

     the

     traditional

     way

     to

     greet

     someone

     is

     with

     a

     "

    Bonjour

    !"

     This

     greeting

     means

     "

    hello

    "

     in

     French

    .


    In

     

    1

    2

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     here

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

     landscape

     of

     the

     technology

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     car

     manufacturers

     continue

     to

     push

     the

     boundaries

     of

     autonomous

     driving

     technology

    ,

     we

     may

     see

     widespread

     adoption

     of

     self

    -driving

     cars

     on

     the

     road

    ways

    .

     This

     would

     require

     significant

     advancements

     in

     AI

    ,

     including

     better

     understanding

     of

     human

    -machine

     interactions

    ,

     safer

     algorithms

    ,

     and

     more

     efficient

     data

     processing

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     analyze

     medical

     images

    ,

     predict

     disease

     outbreaks

    ,

     and

     improve

     patient

     care

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     may

     see

     more

     widespread

     use

     of

     AI

     in

     healthcare

    ,

     including

     in

     diagnostics

    ,

    



```python
llm.shutdown()
```
