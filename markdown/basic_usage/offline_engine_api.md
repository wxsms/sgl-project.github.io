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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]


    2026-05-18 21:33:37,974 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 21:33:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.82it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.56it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.32it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.97it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 23.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  21%|██        | 12/58 [00:00<00:02, 20.05it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  21%|██        | 12/58 [00:00<00:02, 20.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.16it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.16it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:02, 20.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.23it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.23it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.23it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 24.23it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 28.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.06it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.23it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.53it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.56it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]Capturing num tokens (num_tokens=12 avail_mem=74.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.60it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.22 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=8 avail_mem=73.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.76it/s] Capturing num tokens (num_tokens=4 avail_mem=74.21 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.76it/s]Capturing num tokens (num_tokens=4 avail_mem=74.21 GB): 100%|██████████| 58/58 [00:01<00:00, 31.47it/s]


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
    Generated text:  Selina and I have been a volunteer for just over 20 years. I volunteer in a very important way in helping to provide education and support to children and families in my local community. I am a parent and grandmother to 3 children, and a teacher of 2nd grade. I am also a widow who has been widowed twice. I have been blessed with the opportunity to grow up in a strong, loving family where I was truly loved and cared for. To truly appreciate the gift I was given is beyond words. I have had the privilege of meeting many wonderful people, from many different places, who have not only
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build in Africa. He knows from past experience that the number of bases he can afford to build is limited to 60. Each military base costs $10 million to build. If he decides to build x bases, how much will he spend in total?
    
    To determine the total amount the president of the United States will spend on building military bases in Africa, we need to follow these steps:
    
    1. Identify the number of bases the president is considering building. According to the problem, the president is deciding to build \( x \) bases.
    2. Determine the cost of building each base.
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Brussels
    C. London
    D. Rome
    
    To determine the capital of France, we need to consider the historical and cultural significance of the city. Paris, being the capital of France since 1870, is widely known as the city of love and art. It has a rich history dating back to ancient times, including the Roman and French empires. The city has a variety of landmarks and historical sites that have been incorporated into Paris throughout its development.
    
    Let's analyze each option:
    
    A. Paris - This is the capital of France, which is correct.
    B. Brussels - Brussels
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with each generation bringing its own unique edge and new challenges. Here’s a look at how these new technologies will change the landscape of work, jobs, and industries.
    AI is transforming the way we work. It is significantly changing how we communicate, organize tasks, and manage work. In the past, work often involved manual labor or routine tasks that required the use of tools like a computer. Today, AI has allowed for more flexible work arrangements, remote work, and a more diverse workforce that can perform tasks in various ways.
    In the past, AI has been used to automate repetitive and time-consuming tasks, making work more efficient and


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its fashion, art, and cuisine, and is a major economic center in Europe. Paris is a city of contrasts, with its historical architecture and modern skyscrapers blending seamlessly. The city is also home to many international organizations and events, making it a hub for international diplomacy and commerce. Paris is a city of contrasts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for innovation and creativity.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, we can expect to see increased emphasis on privacy and security. This will require more robust data protection
    


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
    Generated text:  [Name] and I am a [occupation], currently working at [company]. I have been involved in the [industry] industry for [number] years. I am passionate about [interest or hobby], and I love [why it is important]. I have always been an [attribute], and I strive to be [what I hope to achieve]. Thank you for taking the time to meet me. [Name] is a self-introduction. Please provide me with more context and details about [Name] if you have any. For example, can you provide some information about their occupation, industry, and hobbies? Also, if you have
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France, and one of the most important cities in the world, known for its rich history and culture. Paris is home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to numerous museums, theaters, and other cultural institutions. Paris is a cosmopolitan and vibrant city, with a diverse population and a rich history. Its many cultural events and festivals draw visitors from around the world. Overall, Paris is a world-class capital city that has a unique cultural and historical significance. 
    
    Paris - The City of Light
    
    *
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be a combination of rapidly advancing technologies, new applications, and significant challenges. Here are some possible trends:
    
    1. Advancements in AI will continue to drive innovation in areas like robotics, autonomous vehicles, and cyber security.
    
    2. AI will become more personalized and relevant, allowing it to learn and adapt to new situations.
    
    3. AI will become more ethical and transparent, with greater emphasis on fairness and bias reduction.
    
    4. AI will continue to be used in healthcare, education, and transportation, with a focus on precision and personalization.
    
    5. AI will be integrated into everyday life, from self-driving cars to virtual assistants to


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

    'm

     a

     [

    Occup

    ation

    ]

     [

    Your

     Occupation

    ]

     here

     to

     assist

     you

     with

     any

     questions

     or

     concerns

     you

     may

     have

    .

     Whether

     you

     need

     information

    ,

     guidance

    ,

     or

     just

     a

     friendly

     chat

    ,

     I

    'm

     here

     to

     help

    .

     What

     can

     I

     do

     for

     you

    ?

     Let

    's

     get

     started

    !

     (

    St

    ret

    ches

    )

     [

    W

    ink

     at

     the

     camera

    ]

     Looking

     forward

     to

     hearing

     from

     you

    !

     [

    End

     of

     Introduction

    ]

     
    


    Please

     note

     that

     this

     is

     a

     fictional

     character

    ,

     and

     I

     am

     not

     a

     real

     person

    .

     I

     am

     just

     a

     computer

     program

     designed

     to

     assist

     and

     provide

     information

     based

     on

     the

     data

     I

     have

     been

     trained

     on

    .

     
    


    Are

     you

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    .

     
    


    Explanation

     of

     the

     statement

    :

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     cultural

    ,

     economic

    ,

     and

     political

     capital

     of

     the

     country

    .

     The

     city

     is

     renowned

     for

     its

     historical

     architecture

    ,

     art

    ,

     music

    ,

     literature

    ,

     and

     cuisine

    .

     It

     is

     also

     one

     of

     the

     world

    's

     most

     important

     financial

     centers

    ,

     with

     the

     headquarters

     of

     several

     large

     financial

     institutions

     located

     in

     the

     city

    .

     Paris

     is

     home

     to

     the

     Lou

    vre

     Museum

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     has

     long

     been

     a

     center

     for

     international

     diplomacy

    .

     Despite

     its

     famous

     landmarks

     and

     iconic

     architecture

    ,

     Paris

     is

     a

     relatively

     modern

     city

    ,

     with

     a

     focus

     on

     urban

     development

     and

     rapid

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     unknown

    s

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

     Increased

     precision

     and

     accuracy

    :

     As

     AI

     advances

    ,

     we

     will

     see

     even

     more

     precise

     and

     accurate

     predictions

     and

     decisions

    .

     This

     will

     be

    得益于

     the

     integration

     of

     more

     data

     and

     algorithms

    ,

     as

     well

     as

     the

     development

     of

     more

     sophisticated

     models

    .
    


    2

    .

     Autonomous

     machines

    :

     As

     AI

     continues

     to

     develop

    ,

     we

     may

     see

     machines

     that

     can

     operate

     without

     human

     intervention

    .

     This

     could

     lead

     to

     autonomous

     vehicles

    ,

     drones

    ,

     and

     other

     machines

     that

     can

     operate

     in

     a

     wide

     range

     of

     environments

     without

     human

     supervision

    .
    


    3

    .

     AI

    -powered

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

     providing

     more

     accurate

     diagnoses

    ,

    



```python
llm.shutdown()
```
