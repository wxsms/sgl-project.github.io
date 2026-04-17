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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 14:45:39] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.82it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.81it/s]


    2026-04-17 14:45:44,349 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 14:45:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.72it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.72it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.86it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.74it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.62it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.66 GB):   3%|▎         | 2/58 [00:00<00:03, 18.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.66 GB):   3%|▎         | 2/58 [00:00<00:03, 18.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.13it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=119.65 GB):   7%|▋         | 4/58 [00:00<00:04, 12.34it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.66 GB):   7%|▋         | 4/58 [00:00<00:04, 12.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.65 GB):   7%|▋         | 4/58 [00:00<00:04, 12.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.65 GB):  10%|█         | 6/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.65 GB):  10%|█         | 6/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.65 GB):  10%|█         | 6/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.65 GB):  10%|█         | 6/58 [00:00<00:03, 13.69it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=119.64 GB):  10%|█         | 6/58 [00:00<00:03, 13.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.64 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.64 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.64 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.63 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.63 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=119.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.61 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=960 avail_mem=119.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s] Capturing num tokens (num_tokens=896 avail_mem=119.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=832 avail_mem=119.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=768 avail_mem=119.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.62it/s]Capturing num tokens (num_tokens=768 avail_mem=119.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=704 avail_mem=119.57 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.93it/s]

    Capturing num tokens (num_tokens=640 avail_mem=119.57 GB):  43%|████▎     | 25/58 [00:00<00:00, 34.93it/s]Capturing num tokens (num_tokens=576 avail_mem=119.57 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=512 avail_mem=119.55 GB):  43%|████▎     | 25/58 [00:01<00:00, 34.93it/s]Capturing num tokens (num_tokens=512 avail_mem=119.55 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=480 avail_mem=119.08 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=448 avail_mem=118.99 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=416 avail_mem=118.98 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=384 avail_mem=118.98 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.98 GB):  50%|█████     | 29/58 [00:01<00:00, 30.67it/s]Capturing num tokens (num_tokens=352 avail_mem=118.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=320 avail_mem=118.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=288 avail_mem=118.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=256 avail_mem=118.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=240 avail_mem=118.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=224 avail_mem=118.96 GB):  59%|█████▊    | 34/58 [00:01<00:00, 34.05it/s]Capturing num tokens (num_tokens=224 avail_mem=118.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=208 avail_mem=118.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=192 avail_mem=118.96 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=176 avail_mem=118.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]

    Capturing num tokens (num_tokens=160 avail_mem=118.95 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=144 avail_mem=118.94 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=144 avail_mem=118.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=128 avail_mem=118.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=112 avail_mem=118.94 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=96 avail_mem=118.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s] Capturing num tokens (num_tokens=80 avail_mem=118.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=64 avail_mem=118.93 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=64 avail_mem=118.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]Capturing num tokens (num_tokens=48 avail_mem=118.93 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]Capturing num tokens (num_tokens=32 avail_mem=118.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]

    Capturing num tokens (num_tokens=28 avail_mem=118.92 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]Capturing num tokens (num_tokens=24 avail_mem=118.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]Capturing num tokens (num_tokens=20 avail_mem=118.91 GB):  84%|████████▍ | 49/58 [00:01<00:00, 39.62it/s]Capturing num tokens (num_tokens=20 avail_mem=118.91 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=16 avail_mem=118.90 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=12 avail_mem=118.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=8 avail_mem=118.89 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.49it/s] Capturing num tokens (num_tokens=4 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 40.49it/s]Capturing num tokens (num_tokens=4 avail_mem=118.87 GB): 100%|██████████| 58/58 [00:01<00:00, 32.80it/s]


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
    Generated text:  Claire and I am a college student. I'm currently majoring in Business Administration with an emphasis on Finance and Marketing. I am currently working as a financial analyst at an accounting firm in the city where I live. My goal is to graduate in 2023 and become a CPA. What is one area that I can increase my knowledge in to help me prepare for CPA exams? What are some resources that I can use to learn and prepare for the CPA exams? To make this question more specific, what are some specific courses or areas of study that I should consider taking to increase my knowledge and prepare for the CPA exams? Additionally
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking position in the federal government, but there are many other high-ranking positions that also come under the umbrella of "high office" and some of these positions may also involve high-level personnel. In the United States, the president is the chief executive of the executive branch of the federal government. Therefore, for a president to serve as the chief executive of the executive branch, it is necessary to also be in high office.
    In addition, the president is also an elected official with the power to make laws and declare war. Therefore, there is a two-fold requirement for a president to be the chief executive of the executive branch, other
    ===============================
    Prompt: The capital of France is
    Generated text:  in:
    
    A. Paris
    B. Vincennes
    C. Villeneuve-lès-Loux
    D. Lille
    
    To determine the capital of France, let's consider the options provided:
    
    A. Paris
    B. Vincennes
    C. Villeneuve-lès-Loux
    D. Lille
    
    First, let's recall that the capital of France is typically Paris. This is a well-known fact about France's capital city. Therefore, we can eliminate options B, C, and D.
    
    The correct answer is:
    \boxed{A} (Paris)
    ===============================
    Prompt: The future of AI is
    Generated text:  in making it accessible to all.
    
    Many people are confused about what AI is, how it works, and how it can be used. Here we present a high-level overview of the two main branches of AI: supervised and unsupervised learning.
    
    In this article, we will explain what supervised and unsupervised learning are and why they are important for the future of AI.
    
    ### What is AI?
    
    Artificial intelligence is defined as the use of computer systems, especially neural networks, to learn, recognize and respond to the environment. It is also an AI system that can mimic human intelligence.
    
    There are two main branches of AI: supervised and


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I'm also a great communicator and enjoy working with people from all backgrounds. What's your favorite hobby or activity? As a [job title], I enjoy spending time with my family and friends, reading books, and playing sports. I also love trying new foods and trying new places. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. The city is known for its beautiful architecture, world-renowned museums, and annual festivals such as the Eiffel Tower and the Louvre. Paris is also a popular tourist destination, with millions of visitors each year. The city is home to many famous landmarks and attractions, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there will be a greater emphasis on ethical AI. This will include developing AI that is transparent, accountable, and respects human values.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more advanced, it is likely to be used in more areas of healthcare, including diagnosis, treatment, and
    


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
    Generated text:  [Name]. I am a [Occupation] with a [Degree] in [Field]. I enjoy [My Passion/Interest/Job]. I've been working hard to [Achievement/Project]. And I hope to [Future Goal] in the future. Please let me know if there is anything I can do to help me get started. I'm excited to meet you and let's see what we can do together! 😊✨✨
    
    Hello, my name is [Name]. I am a [Occupation] with a [Degree] in [Field]. I enjoy [My Passion/Interest/Job]. I've been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the most populous city in France, with around 2. 8 million people residing within its boundaries. The city is located in the Île de France region and is known for its iconic landmarks, including the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe. Paris is a major center of culture, commerce, and education in Europe and plays a significant role in French society, politics, and international relations. The city has a rich history and is home to many museums, palaces, and other cultural institutions. Paris is often referred to as the "City of Light" and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possible trends that could be explored include:
    
    1. Increased use of AI in healthcare: AI is already transforming the way we diagnose and treat diseases, but it has the potential to revolutionize the field. AI may be used to develop more accurate and personalized medical treatments, as well as to help doctors make more informed decisions about patient care.
    
    2. AI in transportation: As self-driving cars become more common, AI is likely to play a bigger role in the transportation industry. AI could be used to optimize traffic flow, predict accidents, and even drive the vehicles themselves.
    
    3. AI in finance: AI is already helping


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

    'm

     a

     [

    职业

    ]

     who

     is

     [

    Your

    self

    's

     Strength

    s

    ,

     Achie

    vements

    ,

     etc

    .

    ].

     I

    'm

     [

    age

    ],

     [

    gender

    ],

     [

    race

    ],

     and

     [

    education

     level

    ].

     I

    'm

     [

    occupation

    ]

     and

     I

    'm

     proud

     of

     my

     [

    accom

    pl

    ishment

    ,

     hobby

    ,

     etc

    .

    ].

     I

     love

     [

    I

    'm

     passionate

     about

    ,

     hobby

    ,

     etc

    .

    ].

     My

     favorite

     [

    job

     title

    ,

     hobby

    ,

     etc

    .

    ].

     I

    'm

     excited

     to

     [

    meet

     new

     people

    ,

     learn

     more

    ,

     etc

    .

    ].

     What

    's

     your

     favorite

     color

    ?

     What

    's

     your

     favorite

     food

    ?

     What

    's

     your

     favorite

     movie

    /

    series

    ?

     I

    'm

     [

    your

    self

    's

     personality

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     a

     historic

     city

     in

     France

     known

     for

     its

     rich

     cultural

     heritage

    ,

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

    ,

     and

     its

     role

     as

     a

     major

     economic

    ,

     financial

    ,

     and

     cultural

     center

    .

     It

     has

     a

     vibrant

     arts

     and

     entertainment

     scene

    ,

     including

     the

     annual

     French

     New

     Year

     celebrations

     and

     the

     annual

    时尚

    g

    az

    ette

     conference

    .

     Paris

     is

     also

     known

     for

     its

     love

     of

     food

     and

     its

     cultural

     attractions

    ,

     including

     the

     famous

     E

    iff

    el

     Tower

     and

     the

     Mou

    lin

     Rouge

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     transportation

     hub

     for

     the

     country

    .

     The

     French

     capital

     is

     a

     unique

     blend

     of

     its

     historical

     roots

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     technologies

     and

     applications

    ,

     including

     more

     sophisticated

     algorithms

    ,

     higher

     levels

     of

     automation

    ,

     and

     greater

     use

     of

     data

     analysis

     and

     machine

     learning

    .

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

     and

     patient

     care

    :

     As

     AI

     can

     assist

     doctors

     in

     diagnosis

     and

     treatment

    ,

     it

     may

     become

     an

     even

     more

     valuable

     tool

     in

     healthcare

    .
    


    2

    .

     Automation

     of

     repetitive

     tasks

    :

     AI

     can

     perform

     repetitive

     and

     dangerous

     tasks

    ,

     freeing

     up

     time

     for

     human

     workers

     to

     focus

     on

     more

     complex

     tasks

    .
    


    3

    .

     AI

     integration

     with

     human

     decision

    -making

    :

     AI

     can

     help

     human

     decision

    -making

    ,

     especially

     in

     decision

    -making

     related

     to

     complex

     systems

    .
    


    4

    .

     AI

     becoming

     more

    



```python
llm.shutdown()
```
