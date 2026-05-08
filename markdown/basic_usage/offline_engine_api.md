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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.67it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.66it/s]


    2026-05-08 23:10:35,514 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 23:10:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.19s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:44,  1.21it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:16,  2.95it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.93it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.60it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]

    Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 19.24it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 27.69it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 36.37it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 36.37it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 36.37it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 36.37it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 36.37it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 36.37it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 36.37it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 36.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   9%|▊         | 5/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   9%|▊         | 5/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s]Capturing num tokens (num_tokens=960 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s] Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.63it/s]Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=768 avail_mem=74.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=576 avail_mem=74.13 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=416 avail_mem=74.13 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  50%|█████     | 29/58 [00:00<00:00, 43.78it/s]Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.53it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.56it/s]Capturing num tokens (num_tokens=208 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.56it/s]Capturing num tokens (num_tokens=192 avail_mem=74.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=144 avail_mem=74.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s] Capturing num tokens (num_tokens=80 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.46it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.01it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.01it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 41.79it/s]


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
    Generated text:  Jakob, a biologist from Germany. I enjoy reading, reading, reading. I also like reading books about the environment, and I find these books to be very informative. I find it challenging to work with technical papers. I have a Master’s degree in Biology from Munich University of Technology. I have experience with bioinformatics, data analysis, and environmental genetics, and I have worked as a data analyst and a researcher in environmental genetics and genomics for several years. I am also interested in wildlife conservation and studying the impact of local and global environmental changes on wildlife populations. I have published in scientific journals and have obtained research grants in environmental
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have in different parts of the country. The United States has 50 states, and each state has an equal chance of getting a base. If the president wants to maximize the number of military bases, what is the smallest number of states he should choose to have a base? To determine the smallest number of states the president of the United States should choose to have a military base, we need to consider the constraint that each state has an equal chance of getting a base.
    
    First, let's calculate the total number of states in the United States:
    \[
    50 \text{ states}
    \
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is in the northwestern region of France, on the banks of the Seine river. It is the center of the country’s cultural, financial, political and industrial life. Paris has a population of more than two million people. It is located in the Paris Basin. It has the largest metro system in the world. It has the highest GDP in the world. It is the largest city in the world in terms of area. It is the most populous city in the world. Paris is the cultural center of France. It is also a major center of learning and research. It is the home to many of the world’s
    ===============================
    Prompt: The future of AI is
    Generated text:  not like we imagined, according to a new report by Cognizant, the research firm behind the Watson business intelligence platform. The future is not about making artificial intelligence the heart of every business, as some of us have been led to believe. Instead, the report argues that we must focus on making the future of AI smart and effective. It highlights the importance of a new approach to the future of AI, which is to give it a second chance to make a difference in the world.
    
    I'll be happy to have an informed conversation about this topic. What kind of changes might we see in the world as a result of adopting this


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


    Generated text:  Paris, also known as "La Ville de Paris" and "La Grande Ouvrière" (The Great Work). It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for business, finance, and politics in France. Paris is a city of contrasts, with its modern architecture
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the future.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management, and trading. As AI technology continues to improve, we
    


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
    Generated text:  [Name], and I'm [Age]. I enjoy [What's Your Favorite Hobby/Activity/Interest?]. I'm a [职业或工作] who [What's Your Most Valuable Skill/Feature/Experience?]. And I'm a [Describe Yourself in a Brief, Neutral Way, like "someone who is kind, friendly, and helpful."] That's a lot! 
    
    I'm excited to meet you and learn more about you. Let's chat! 🎉✨
    
    What's your favorite hobby/interest? 🎨✨
    
    What's your most valuable skill/feature/experience? 💡✨
    
    Describe
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    To elaborate, Paris is the largest city in France, located on the left bank of the Seine River in the Paris Basin, and is known for its rich cultural history, architectural wonders, and bustling nightlife. Paris is the seat of the state government and is the administrative center of the French Republic. The city is also known for its 19th-century architecture, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. Paris is also known for its fine dining, wine, and food, as well as its vibrant cultural scene. The city has been
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and uncertain. Some possible future trends that are commonly expected include:
    
    1. Advancements in machine learning and neural networks: Advances in machine learning and neural networks will lead to the development of more complex and powerful AI systems. These advancements will enable AI systems to learn from data more efficiently and accurately, which will lead to even greater capabilities in areas such as healthcare, finance, and automation.
    
    2. Integration with other technologies: As AI systems become more integrated with other technologies, such as sensors, actuators, and actuaries, they will become more capable of performing tasks that were previously only feasible with human intelligence.
    
    3. Autonomous systems


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

     am

     [

    Age

    ],

     [

    Occup

    ation

    ],

     and

     I

     am

     a

     [

    Type

     of

     Software

     Engineer

    ].

     I

     love

     [

    interest

     or

     hobby

     you

     enjoy

     doing

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     expand

     my

     skills

     and

     knowledge

    ,

     and

     I

     am

     always

     looking

     to

     make

     a

     positive

     impact

     in

     the

     world

    .

     I

     am

     a

     [

    Type

     of

     Person

    ],

     and

     I

     am

     always

     ready

     to

     learn

     and

     grow

     as

     a

     software

     engineer

    .

     Thank

     you

    !

     Hey

    ,

     nice

     to

     meet

     you

    !

     My

     name

     is

     [

    Name

    ],

     I

    'm

     

    2

    5

     years

     old

    ,

     a

     software

     engineer

    ,

     and

     I

     have

     been

     working

     in

     the

     field

     since

     I

     graduated

     from

     college

    .

     I

     enjoy

     helping

     other

     people

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     and

     historic

     city

     located

     in

     the

     south

     of

     the

     country

    .
    


    Please

     provide

     the

     sentence

     in

     French

    :

     The

     capital

     of

     France

     is

     Paris

    ,

     a

     bustling

     and

     historic

     city

     located

     in

     the

     south

     of

     the

     country

    .

     
    


    You

     should

     include

     the

     French

     translation

     of

     the

     original

     sentence

    .


    Cette

     capit

    ale

     de

     la

     France

     est

     Paris

    ,

     une

     ville

     tumult

    ue

    use

     et

     histor

    ique

     situ

    ée

     au

     sud

     de

     la

     France

    .

     
    


    This

     is

     the

     French

     translation

     of

     the

     provided

     English

     sentence

    ,

     maintaining

     its

     structure

     and

     meaning

    .

     
    


    The

     bustling

     and

     historic

     city

     of

     Paris

    ,

     a

     defining

     element

     of

     French

     culture

     and

     identity

    ,

     is

     situated

     in

     the

     south

     of

     the

     country

    .

     
    


    This

     sentence

     provides

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

     and

     challenges

    .

     Here

     are

     some

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     coming

     years

    :
    


    1

    .

     Integration

     with

     human

     decision

    -making

    :

     AI

     is

     becoming

     increasingly

     integrated

     into

     our

     daily

     lives

    ,

     from

     our

     smartphones

     to

     the

     medical

     devices

     that

     assist

     doctors

    .

     It

     will

     continue

     to

     play

     a

     vital

     role

     in

     decision

    -making

    ,

     especially

     in

     areas

     like

     healthcare

    ,

     transportation

    ,

     and

     education

    .
    


    2

    .

     Autonomous

     vehicles

    :

     As

     autonomous

     vehicles

     become

     more

     advanced

    ,

     we

     can

     expect

     them

     to

     become

     more

     common

     in

     our

     daily

     lives

    .

     This

     will

     change

     the

     way

     we

     travel

    ,

     work

    ,

     and

     play

    .
    


    3

    .

     Enhanced

     language

     translation

    :

     With

     the

     increasing

     use

     of

     AI

     in

     various

     fields

    ,

     we

    



```python
llm.shutdown()
```
