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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.05it/s]


    2026-05-10 14:20:40,527 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-10 14:20:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:42,  4.95s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:42,  4.95s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:42,  4.95s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.56it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.01it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.56it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 20.24it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 27.96it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.17it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.18it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:02, 19.18it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.34 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.33 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.32 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.31 GB):  21%|██        | 12/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 22.36it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  21%|██        | 12/58 [00:00<00:02, 22.36it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=72.30 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.29 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.29 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.29 GB):  31%|███       | 18/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.26 GB):  31%|███       | 18/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  31%|███       | 18/58 [00:00<00:01, 22.40it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=72.28 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=896 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=832 avail_mem=72.27 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=768 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=704 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  38%|███▊      | 22/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=640 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=576 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=512 avail_mem=72.24 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=480 avail_mem=72.26 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=448 avail_mem=72.25 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=416 avail_mem=72.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=384 avail_mem=72.25 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=352 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=320 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.31it/s]Capturing num tokens (num_tokens=288 avail_mem=72.24 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=256 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=240 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=208 avail_mem=72.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.22 GB):  62%|██████▏   | 36/58 [00:01<00:00, 35.72it/s]Capturing num tokens (num_tokens=192 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=144 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=128 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s] Capturing num tokens (num_tokens=80 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=48 avail_mem=72.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.65it/s]Capturing num tokens (num_tokens=32 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=24 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=12 avail_mem=72.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=12 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=8 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s] Capturing num tokens (num_tokens=4 avail_mem=71.85 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.01it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.85 GB): 100%|██████████| 58/58 [00:01<00:00, 32.85it/s]


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
    Generated text:  Antonio, a young man of 30 years of age, residing in the city of Monterrey, Mexico. I am fluent in Spanish, English, French, and Arabic. I have been playing chess for over a decade, but I have never lost a game to a computer. I would like to know more about the advantages of playing chess with a computer. My goal is to enhance my chess skills and become the best player in my league. However, I am afraid of losing my computer to the computer. How can I overcome this fear and increase my chances of winning games? You can create a chess board with a 3x
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have. He has n options and the cost of building base i is c_i. There is a strict preference rule that dictates that the military base with the lowest cost should be chosen. For example, if there are three bases with costs 200, 100, and 150, the president would choose the base with the cost of 100, because 100 is the lowest possible cost. What is the expected value of the number of bases that the military has in its 100th year? To solve this problem, we need to
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the south, but it is not the French capital.
    A. 正确
    B. 错误
    Answer: B
    
    The main factor causing the methane content in the tail gas of the reforming unit to be higher than in the lean liquid is ____.
    A. higher temperature
    B. lower temperature
    C. lower water content
    D. higher water content
    Answer: D
    
    In the handover of the State Grid Corporation of China's procurement organization and implementation mechanism, it is clearly stated that the state Grid Corporation of China establishes a ____, clarifying the various specialized entities that participate in procurement, and form
    ===============================
    Prompt: The future of AI is
    Generated text:  not yet clear, but we should all be prepared to embrace it for the good of humanity. The field is evolving and there are many opportunities for all levels of AI experts to play a role in its success. AI is changing the world, and AI experts can help to shape that change. One way to start is to learn about the field and how it works. This will help you to understand the basics of AI, including the ways in which it can be applied, the challenges it faces, and the opportunities it presents. Additionally, you can read about different types of AI, such as machine learning, deep learning, and natural language processing


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Model] and I'm currently [Current Location]. I'm [Favorite Hobby] and I enjoy [Favorite Activity]. I'm [Favorite Food] and I love [Favorite Music]. I'm [Favorite Book] and I read [Number of Books] books a year. I'm [Favorite Movie] and I watch [Number of Movies] movies a year. I'm [Favorite Sport] and I play [Favorite Sport]. I'm [Favorite Animal] and I love [Favorite Animal]. I'm [Favorite Place
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many international airports and train stations. The city is known for its fashion industry, art scene, and annual cultural events such as the World Fair. Paris is a popular tourist destination, with millions of visitors each year. It is a cultural and economic center of France and a major global city. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will enable AI to perform tasks that are currently the domain of humans, such as image and speech recognition, autonomous vehicles, and personalized medicine.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be increased concerns about privacy and security. This will require developers to create more
    


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
    Generated text:  [Your Name], and I am a [insert your profession here], and I'm from [insert your hometown or city]. I have been a [insert your occupation or hobby here] for [insert the number of years you have been involved in your field] years, and I have always been passionate about [insert why you are passionate about your field here]. I'm always up for challenges and always eager to learn new things. I'm always looking for new opportunities and I'm always open to new experiences. I love to travel, but I'm also very adventurous and enjoy exploring new places. I'm always looking for ways to improve myself
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly known as the "City of Love" and "The Eternal City." It is located on the Seine River and is home to the ancient temples of Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum, among many other historical landmarks. Paris is known for its rich cultural heritage and lively atmosphere, and is a major cultural and economic center in Europe. It has played an important role in world history and has been a cultural hub for centuries, hosting many famous artists, writers, and musicians. As the world's largest city, Paris is a bustling metropolis with a diverse population and is a major
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of continuous improvement and integration with new technologies. Here are some possible future trends in AI:
    
    1. **Smarter and Better**: AI is expected to become more sophisticated, with better performance and efficiency. This could involve more sophisticated models that can learn from data and make more accurate predictions or decisions.
    
    2. **Integration with Other Technologies**: AI will likely become more integrated with other technologies, such as sensors, cameras, and machine learning algorithms. This could lead to more seamless interactions and applications, such as smart home devices, self-driving cars, and personal assistants.
    
    3. **Data-Driven Decision Making**: AI will increasingly


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

    name

    ],

     and

     I

    'm

     a

     [

    character

     type

    ]

     with

     [

    number

    ]

     years

     of

     experience

     in

     [

    field

     of

     expertise

    ].

     I

     have

     [

    number

    ]

     of

     years

     of

     experience

     in

     [

    related

     field

    ]

     and

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    related

     field

    ].

     I

     have

     a

     passion

     for

     [

    interest

    /

    thing

    ],

     and

     I

    'm

     dedicated

     to

     [

    specific

     goal

    ].

     What

     would

     you

     like

     to

     know

     about

     me

    ?

     [

    Name

    ]

     is

     a

     [

    type

     of

     character

    ],

     [

    number

    ]

     years

     of

     experience

     in

     [

    field

     of

     expertise

    ],

     [

    number

    ]

     of

     years

     in

     [

    related

     field

    ]

     and

     [

    number

    ]

     in

     [

    related

     field

    ].

     I

     have

     [

    number

    ]

     years

     of

     experience

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     French

     department

     of

     Paris

    .

     
    


    Note

    :

     The

     capital

     of

     France

     is

     not

     the

     city

     itself

    ,

     but

     a

     broader

     administrative

     unit

     that

     includes

     the

     administrative

     and

     political

     center

     of

     the

     country

    .

     Paris

     is

     sometimes

     also

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     vibrant

     art

     and

     culture

    ,

     particularly

     in

     the

     fields

     of

     photography

    ,

     painting

    ,

     and

     the

     arts

    .

     The

     city

     is

     also

     known

     for

     its

     historical

     importance

    ,

     including

     its

     status

     as

     the

     birth

    place

     of

     Napoleon

     and

     the

     capital

     of

     the

     French

     Riv

    iera

    .

     Paris

     is

     home

     to

     numerous

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     rapidly

     evolving

     field

     with

     a

     wide

     range

     of

     potential

     developments

    .

     Some

     of

     the

     most

     promising

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     With

     more

     people

     becoming

     aware

     of

     the

     potential

     risks

     of

     AI

    ,

     there

     is

     a

     growing

     emphasis

     on

     developing

     ethical

     AI

     systems

     that

     prioritize

     safety

     and

     fairness

    .
    


    2

    .

     Advanced

     machine

     learning

     and

     deep

     learning

    :

     These

     types

     of

     AI

     techniques

     are

     becoming

     increasingly

     powerful

    ,

     enabling

     AI

     systems

     to

     learn

     from

     data

     and

     make

     better

     decisions

     than

     ever

     before

    .
    


    3

    .

     Improved

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

     improve

     the

     accuracy

     and

     efficiency

     of

     medical

     diagnosis

     and

     treatment

    ,

     while

     also

     improving

     patient

     care

     and

     reducing

     costs

    



```python
llm.shutdown()
```
