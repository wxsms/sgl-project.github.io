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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.83it/s]


    2026-04-15 08:39:32,735 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 08:39:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.71it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.71it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.70it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.56it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.31it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=116.31 GB):   2%|▏         | 1/58 [00:00<00:08,  6.98it/s]Capturing num tokens (num_tokens=7680 avail_mem=116.28 GB):   2%|▏         | 1/58 [00:00<00:08,  6.98it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=116.27 GB):   2%|▏         | 1/58 [00:00<00:08,  6.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=116.27 GB):   5%|▌         | 3/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=116.27 GB):   5%|▌         | 3/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.27 GB):   5%|▌         | 3/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.22 GB):   5%|▌         | 3/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.22 GB):  10%|█         | 6/58 [00:00<00:03, 16.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.20 GB):  10%|█         | 6/58 [00:00<00:03, 16.46it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=116.20 GB):  10%|█         | 6/58 [00:00<00:03, 16.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.20 GB):  10%|█         | 6/58 [00:00<00:03, 16.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.19 GB):  10%|█         | 6/58 [00:00<00:03, 16.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 22.70it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=116.18 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.18 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.17 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.15 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=960 avail_mem=116.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.11it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 31.11it/s]Capturing num tokens (num_tokens=896 avail_mem=116.11 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=832 avail_mem=116.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=768 avail_mem=116.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=704 avail_mem=116.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=640 avail_mem=116.10 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=640 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.28it/s]Capturing num tokens (num_tokens=576 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.28it/s]Capturing num tokens (num_tokens=512 avail_mem=116.08 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.28it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.28it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.28it/s]Capturing num tokens (num_tokens=448 avail_mem=116.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=416 avail_mem=116.10 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=384 avail_mem=116.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=352 avail_mem=116.09 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.68it/s]Capturing num tokens (num_tokens=320 avail_mem=116.08 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=288 avail_mem=116.08 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]

    Capturing num tokens (num_tokens=256 avail_mem=116.08 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=240 avail_mem=116.08 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  60%|██████    | 35/58 [00:01<00:00, 31.90it/s]Capturing num tokens (num_tokens=224 avail_mem=116.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=208 avail_mem=116.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=192 avail_mem=116.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=176 avail_mem=116.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.06it/s]

    Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 31.06it/s]Capturing num tokens (num_tokens=160 avail_mem=116.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=144 avail_mem=116.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=128 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.75it/s]

    Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 28.75it/s]Capturing num tokens (num_tokens=112 avail_mem=118.66 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.98it/s]Capturing num tokens (num_tokens=96 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.98it/s] Capturing num tokens (num_tokens=80 avail_mem=118.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.98it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 21.98it/s]Capturing num tokens (num_tokens=64 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 22.52it/s]Capturing num tokens (num_tokens=48 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 22.52it/s]Capturing num tokens (num_tokens=32 avail_mem=118.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 22.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 22.52it/s]Capturing num tokens (num_tokens=28 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.09it/s]Capturing num tokens (num_tokens=24 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.09it/s]Capturing num tokens (num_tokens=20 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.09it/s]Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  90%|████████▉ | 52/58 [00:02<00:00, 24.09it/s]

    Capturing num tokens (num_tokens=16 avail_mem=118.63 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.32it/s]Capturing num tokens (num_tokens=12 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.32it/s]Capturing num tokens (num_tokens=8 avail_mem=118.62 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.32it/s] Capturing num tokens (num_tokens=4 avail_mem=118.61 GB):  95%|█████████▍| 55/58 [00:02<00:00, 21.32it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 20.39it/s]Capturing num tokens (num_tokens=4 avail_mem=118.61 GB): 100%|██████████| 58/58 [00:02<00:00, 24.51it/s]


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
    Generated text:  Kira and I am an 18-year-old medical student. I am also a proud member of the Women's Olympic Athlete Association and an avid fan of the Olympics. I have been a competitive runner since I was 10 years old and I have participated in various competitions and championships throughout my life. I have won several medals and have been on the team of Olympians for the United States. I am passionate about learning and sharing knowledge and I look forward to sharing my experiences and insights with others. So if you have any questions or need help finding information about the Olympics, I am happy to help. Do you have any
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He has 200 bases in the US and would like to have an equal number of military bases in different states. If he randomly selects 50 bases and finds that there are 5000 military bases, how many bases should the president have in the US?
    
    To solve this problem, we need to determine the total number of military bases in the United States. Given that the president has 200 bases and wants an equal number of bases in different states, let's denote the total number of bases in different states as \( x \). Thus, we have
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the most famous city in the world. You can go there by plane, by train, by car, by bus, by ship or by underground. There are many museums in Paris. Here are the best museums in Paris. The Louvre Museum is a huge building. It has many famous paintings and sculptures. The Museum of the History of Science is another great museum. It is a place for learning. In the Science Museum, you can see the way we learned to make things and we can see the way people lived in the past. The Musée d'Orsay has beautiful paintings. It has many beautiful artworks
    ===============================
    Prompt: The future of AI is
    Generated text:  changing – and you’re in the driver’s seat
    
    By Dustin Graham, The Future of AI Researcher
    
    
    By Dustin Graham, The Future of AI Researcher
    
    
    No matter how old the tech market seems, it’s always moving forward. In the next five years, we will see many more leaps and bounds in AI research, and those leaps will be huge.
    
    But the biggest changes are going to be in the ways we use AI.
    
    It’s not that we’re going to be doing more of the same; rather, we’re going to be doing more of the new.
    
    As the tech industry has become more sophisticated, people are turning


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to meet you. I'm [job title] at [company name], and I'm excited to meet you. I'm [job title] at [company name], and I'm excited to meet you. I'm [job title] at [company name], and I'm excited to meet you. I'm [job title] at [company name], and I'm excited to meet you. I'm [job title] at [company name], and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, with a diverse population and a rich history dating back to the Roman Empire. It is a popular tourist destination and a major center of French politics and culture. The city is known for its cuisine, fashion, and art, and is home to many famous museums, theaters, and galleries. Paris is a city of contrasts, with its modern architecture and historical landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This will require new approaches to problem-solving and decision-making.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare
    


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
    Generated text:  [insert name]. I am a [insert profession], and I have a love for [insert a hobby or interest]. If you have any questions, I am here to answer them. Would you like to know more about me? 
    (Repeat the same greeting for as many characters as needed, with each character's profession and hobby or interest described in the same manner.) Start by introducing yourself to your character. 
    I'm [insert name] and I'm a [insert profession]. I have a love for [insert a hobby or interest]. If you have any questions, I am here to answer them. Would you like to know more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France by population and is home to the country's political and cultural center. The city, located on the banks of the Seine River, is known for its rich history, art, and cuisine. It has a vibrant street food scene, and the city is also home to the iconic Eiffel Tower. Paris is a popular tourist destination and is home to many world-renowned landmarks and museums. The city also plays a significant role in French politics, hosting the annual Elysee Festival. Paris is a global city with a diverse population and culture. It is known for its cuisine, art,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve and develop rapidly, with several key trends that are likely to shape the industry in the coming years. Here are some of the most promising trends that are expected to shape the AI landscape in the coming years:
    
    1. Personalization and Personalized AI: As the need for personalized and targeted marketing becomes increasingly important, we are likely to see an increase in AI that is designed to learn from individual user data and deliver tailored recommendations and insights. This could include personalized chatbots, predictive analytics, and recommendation systems that are able to learn from user behavior and preferences over time.
    
    2. Ethical AI: As concerns about the


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

     career

    ,

     hobby

    ,

     etc

    .

     ].

     I

     am

     [

    describe

     your

     profession

    ]

     and

     I

     am

     dedicated

     to

     [

    mention

     a

     relevant

     accomplishment

     or

     passion

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Fill

     in

     the

     blanks

     with

     your

     personal

     experiences

     and

     interests

    ].

     How

     do

     you

     feel

     when

     you

     can

     make

     a

     difference

     in

     the

     world

    ?

     What

     motiv

    ates

     you

     to

     keep

     going

    ?

     [

    Answer

     your

     questions

     and

     interests

     in

     detail

    ].

     Thank

     you

     for

     asking

     to

     meet

     me

    .

     Have

     a

     nice

     day

    !

     How

     do

     you

     feel

     when

     you

     can

     make

     a

     difference

     in

     the

     world

    ?

     When

     I

     can

     make

     a

     difference

     in

     the

     world

    ,

     it

     feels

     like

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     cultural

     heritage

     and

     stunning

     architecture

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .


    To

     ensure

     the

     accuracy

     of

     this

     statement

    ,

     I

     will

     incorporate

     historical

     and

     architectural

     information

    .

     Paris

    ,

     the

     capital

     of

     France

    ,

     is

     a

     unique

     and

     cosm

    opolitan

     city

     with

     a

     rich

     history

     and

     cultural

     heritage

    .

     The

     city

    's

     most

     famous

     landmark

    ,

     the

     E

    iff

    el

     Tower

    ,

     stands

     as

     a

     symbol

     of

     the

     city

    's

     modern

    ity

     and

     technological

     advancement

    .

     The

     Lou

    vre

     Museum

    ,

     located

     in

     the

     heart

     of

     the

     city

    ,

     houses

     some

     of

     the

     world

    's

     most

     impressive

     art

     collections

    ,

     including

     the

     Mona

     Lisa

    .

     Additionally

    ,

     Paris

     boasts

     a

     vibrant

     nightlife

     and

     a

     rich

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     varied

    ,

     and

     we

     can

     expect

     it

     to

     continue

     evolving

     in

     the

     following

     ways

    :
    


    1

    .

     Increased

     intelligence

     and

     autonomy

    :

     AI

     will

     continue

     to

     become

     more

     intelligent

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     only

     possible

     for

     humans

    .

     This

     will

     lead

     to

     more

     autonomous

     systems

     that

     can

     make

     decisions

     on

     their

     own

    ,

     without

     the

     need

     for

     human

     intervention

    .
    


    2

    .

     Natural

     language

     processing

    :

     This

     will

     allow

     machines

     to

     understand

     and

     respond

     to

     human

     language

    ,

     and

     will

     enable

     more

     advanced

     forms

     of

     natural

     language

     processing

    ,

     such

     as

     question

    -

    ans

    w

    ering

    ,

     text

     generation

    ,

     and

     sentiment

     analysis

    .
    


    3

    .

     Ub

    iqu

    itous

     AI

    :

     AI

     will

     become

     more

     pervasive

     in

     our

     lives

    ,

     from

     self

    



```python
llm.shutdown()
```
