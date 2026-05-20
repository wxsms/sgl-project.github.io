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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.74it/s]


    2026-05-20 08:40:19,794 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 08:40:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<00:59,  1.07s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:10,  4.59it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:05,  8.23it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]

    Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 15.45it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 23.33it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 32.95it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 42.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.50 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.52 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.53 GB):   3%|▎         | 2/58 [00:00<00:04, 13.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.53 GB):   7%|▋         | 4/58 [00:00<00:04, 12.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.55 GB):   7%|▋         | 4/58 [00:00<00:04, 12.49it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.56 GB):   7%|▋         | 4/58 [00:00<00:04, 12.49it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.56 GB):  10%|█         | 6/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.56 GB):  10%|█         | 6/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.56 GB):  10%|█         | 6/58 [00:00<00:03, 13.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.56 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.57 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.59 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.40it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=71.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.59 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.68 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.68 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.59 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.68 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.61 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.57it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.60 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.60it/s]Capturing num tokens (num_tokens=960 avail_mem=71.61 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.60it/s] Capturing num tokens (num_tokens=896 avail_mem=71.63 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.60it/s]Capturing num tokens (num_tokens=832 avail_mem=71.62 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.60it/s]Capturing num tokens (num_tokens=768 avail_mem=71.62 GB):  36%|███▌      | 21/58 [00:01<00:01, 26.60it/s]

    Capturing num tokens (num_tokens=768 avail_mem=71.62 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.74it/s]Capturing num tokens (num_tokens=704 avail_mem=71.61 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.74it/s]Capturing num tokens (num_tokens=640 avail_mem=71.61 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.74it/s]Capturing num tokens (num_tokens=576 avail_mem=71.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.74it/s]Capturing num tokens (num_tokens=512 avail_mem=71.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.74it/s]Capturing num tokens (num_tokens=512 avail_mem=71.58 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=480 avail_mem=71.60 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=448 avail_mem=71.61 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=416 avail_mem=71.61 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=384 avail_mem=71.61 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.60 GB):  50%|█████     | 29/58 [00:01<00:00, 32.18it/s]Capturing num tokens (num_tokens=352 avail_mem=71.60 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=320 avail_mem=71.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=288 avail_mem=71.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=256 avail_mem=71.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=240 avail_mem=71.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=224 avail_mem=71.56 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=224 avail_mem=71.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=208 avail_mem=71.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=192 avail_mem=71.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.58it/s]

    Capturing num tokens (num_tokens=176 avail_mem=71.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=160 avail_mem=71.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.58it/s]Capturing num tokens (num_tokens=160 avail_mem=71.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=144 avail_mem=71.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=128 avail_mem=71.51 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=112 avail_mem=71.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.30it/s]Capturing num tokens (num_tokens=96 avail_mem=71.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.30it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=71.50 GB):  81%|████████  | 47/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=80 avail_mem=71.51 GB):  81%|████████  | 47/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=64 avail_mem=71.50 GB):  81%|████████  | 47/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=48 avail_mem=71.50 GB):  81%|████████  | 47/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=32 avail_mem=71.49 GB):  81%|████████  | 47/58 [00:01<00:00, 31.36it/s]Capturing num tokens (num_tokens=32 avail_mem=71.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=28 avail_mem=71.48 GB):  88%|████████▊ | 51/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=24 avail_mem=71.48 GB):  88%|████████▊ | 51/58 [00:01<00:00, 29.94it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=16 avail_mem=71.47 GB):  88%|████████▊ | 51/58 [00:01<00:00, 29.94it/s]Capturing num tokens (num_tokens=16 avail_mem=71.47 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=12 avail_mem=71.46 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=8 avail_mem=71.45 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.16it/s] Capturing num tokens (num_tokens=4 avail_mem=71.44 GB):  95%|█████████▍| 55/58 [00:02<00:00, 30.16it/s]Capturing num tokens (num_tokens=4 avail_mem=71.44 GB): 100%|██████████| 58/58 [00:02<00:00, 27.45it/s]


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
    Generated text:  Anna. I'm a teacher. My students and I are learning the different ways to use the computer. Our teacher, Mrs. Zhang, is one of our friendly teachers. She is very nice to us. We can talk to her. We can get help when we don't understand something. We can help each other with the computer. We can talk to each other in English. She is very kind to us. Now, you can use the computer as you like. But you must ask your parents' permission. You can't play computer games too much. We can do our homework too. We can eat with our parents. But
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military drones to keep on a certain island. The president knows that there are 1200 military drones in the world, and the number of drones on the island is half of that in the world. If the president decides to keep only half of these drones on the island, how many drones will be left on the island? Let's break down the problem step by step:
    
    1. First, we need to determine the number of military drones on the island. The problem states that the number of drones on the island is half of the number of drones in the world. The world has 120
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the 10th largest city in the world.
    Paris has been the capital of France since 1830. It is situated on the Seine River in the heart of Paris, one of the most famous city districts in the world.
    The area that was originally known as the "City of Ten Thousand Years" is now the main part of the city. It's been known as the "City of 100 Years" in the past. It is known as the "City of Ten Thousand Years" because it has been the home of the first permanent European inhabitants to
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of companies, not consumers
    
    As a language model, I can't provide my opinions or recommendations, but I can tell you that AI is rapidly changing the way we live and work, and it's important for consumers to be aware of the potential risks and challenges of AI. However, companies have a responsibility to develop and maintain AI systems that are safe, ethical, and compliant with regulations like GDPR. Ultimately, it is up to consumers to make informed decisions about the use of AI in their daily lives. Here are some key points to consider:
    
    1. Risk assessment: Companies should conduct thorough risk assessments to ensure that AI systems


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [nationality]. I have a [job title] at [company name], and I enjoy [job title] work. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [favorite hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many notable French artists, writers, and musicians, and is known for its rich history and cultural heritage. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage, making it a must-visit destination for anyone interested in France and its history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text:  [insert name] and I am a [insert occupation] with a passion for [insert something specific]. I love spending my days exploring the world, learning new things, and connecting with people. I enjoy creating content that inspires and entertains, and I love the feeling of solving problems that others have. Overall, I am a dedicated and hardworking person who loves to make a difference in the world. How can I contribute to the world and what kind of projects do I want to work on? I am available to meet with anyone who would like to discuss this. Can you tell me a bit more about yourself? Sure, I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    
    The French capital city is known for its iconic landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Can you provide me with a summary of some of the most popular attractions in Paris, such as the Louvre, Notre-Dame, or Champs-Élysées? Certainly! Paris is renowned for its iconic attractions, including the iconic Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Other popular attractions include the Champs-É
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be multifaceted and diverse, with potential applications in various fields and industries. Here are some possible trends in AI in the coming years:
    
    1. Enhanced Real-World Application: AI is expected to be more widely used in the real-world applications, as we become more aware of the impact of AI on society and its potential impact on the environment. AI will continue to evolve and develop new capabilities, with the aim of making the world a more sustainable place.
    
    2. Personalization and Customization: With the rise of AI and machine learning, it is expected to be possible to personalize and customize products and services to individual customers.


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

     am

     a

     [

    Job

     Title

    ]

     at

     [

    Company

     Name

    ].

     I

     am

     passionate

     about

     [

    Reason

     for

     passion

    ],

     and

     I

     love

     [

    Reason

     for

     passion

    ].

     I

     am

     a

     [

    Career

     Stage

    ],

     and

     I

     have

     [

    Number

     of

     Years

     of

     Experience

    ]

     years

     of

     experience

     in

     [

    Field

     of

     Expert

    ise

    ].

     I

     thrive

     on

     [

    Areas

     of

     Strength

    ],

     and

     I

     am

     a

     [

    Type

     of

     Personality

    ]

     personality

    .

     I

     believe

     in

     [

    Reason

     for

     Bel

    ief

    ],

     and

     I

     am

     always

     [

    Level

     of

     Engagement

    ].

     I

     am

     [

    Age

    /

    Height

    /

    Weight

    ],

     and

     I

     wear

     a

     [

    Fitness

    /

    Be

    aut

    ification

    /

    Current

     Hobby

    ]

     to

     show

     my

     dedication

     to

     [

    Purpose

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     located

     on

     the

     Se

    ine

     River

     and

     the

     banks

     of

     the

     River

     Jordan

    .

     The

     city

     is

     the

     seat

     of

     government

     and

     of

     the

     national

     capital

    .

     It

     is

     also

     a

     major

     cultural

    ,

     economic

     and

     educational

     center

    .

     Paris

     is

     known

     as

     "

    the

     city

     of

     love

    "

     and

     is

     a

     major

     tourist

     destination

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     major

     cultural

     and

     economic

     center

     in

     Western

     Europe

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     monuments

    ,

     including

     the

     Lou

    vre

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     diverse

     and

     historic

     city

     with

     a

     rich

     cultural

     heritage

    ,

     and

     is

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    ,

     with

     many

     possible

     trends

     shaping

     its

     growth

     and

     impact

    .

     Here

     are

     some

     potential

     trends

     to

     consider

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

     can

     be

     used

     to

     diagnose

     and

     treat

     diseases

    ,

     improve

     patient

     outcomes

    ,

     and

     personalize

     treatments

    .

     For

     example

    ,

     AI

     can

     analyze

     medical

     images

    ,

     predict

     disease

     progression

    ,

     and

     recommend

     personalized

     treatment

     plans

    .
    


    2

    .

     Automation

     of

     manufacturing

     and

     production

    :

     AI

     can

     automate

     tasks

     such

     as

     production

     planning

    ,

     material

     selection

    ,

     and

     quality

     control

    .

     This

     could

     lead

     to

     increased

     efficiency

    ,

     cost

     savings

    ,

     and

     reduced

     human

     error

     in

     manufacturing

    .
    


    3

    .

     Integration

     of

     AI

     into

     consumer

     goods

    :

     AI

     can

     be

     used

     to

     analyze

     customer

     behavior

    ,

     preferences

    



```python
llm.shutdown()
```
