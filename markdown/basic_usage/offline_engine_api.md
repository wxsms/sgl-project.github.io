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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.06it/s]


    2026-05-13 12:02:53,769 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 12:02:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.39it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.20it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.15it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 22.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:01, 36.93it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.38it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.01it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.62it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.44it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s] Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.87it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.27it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 39.68it/s]


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
    Generated text:  Michelle and I'm 16 years old. I live in Belgium and I'm a student at Littoral University of Brussels. I was born in 1998 and I'm tall. I like to travel and read a lot. I'm really good at math, but I'm not good at history and politics. I'm also not very good at music, but I like to play the guitar. I have a really good friend named Megan. We are both very good at math, but we are not very good at music. So, I don't have a family. I don't have any parents, but I
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a bill that will cause a nationwide ban on plastic bags, which are commonly used by people. This will have a significant impact on the environment, leading to a reduction in the number of plastic bags. However, some people argue that this bill is not a good idea, as it may cause inconvenience and harm to people who do not use plastic bags. 
    
    The bill will also have a positive impact on the economy, as it will lead to the creation of jobs in manufacturing and the transportation of plastic bags. 
    
    Based on this information, what are some potential drawbacks of this proposed bill? The proposed bill aims to ban the use of plastic
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. London C. Paris D. Moscow
    Answer:
    
    A
    
    Xiao Zhang was hospitalized due to an illness. In this case, the nurse's actions should be:
    A. Only administering medication
    B. Only providing psychological counseling
    C. Focusing solely on physical care
    D. Both physical and psychological care
    Answer:
    
    D
    
    For two parallel wires, if their currents are equal, is the mutual force between them also equal?
    A. True
    B. False
    C. Uncertain
    D. None of the above
    Answer:
    
    B
    
    What is the effect of adding a positive charge to
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and it’s not just about new technologies or smart systems. It’s about how we use our data, and what we can learn from it. In this article, we’ll explore the future of AI in healthcare, focusing on the latest developments and the potential impact on patient care and technology.
    One of the most exciting areas of AI in healthcare is the development of AI-powered tools for diagnosing diseases. For example, AI can analyze medical images such as X-rays, MRIs, and CT scans to identify potential health issues. This can help doctors to identify abnormalities in the early stages of a disease, before symptoms become apparent


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has [Number of Years] years of experience in [Field]. I'm passionate about [What I Love About My Profession]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Hobby] and I enjoy [What I Enjoy Doing]. I'm [What I Do For Fun]. I'm [What I Do For Fun]. I'm [What I Do For Fun]. I'm [What I Do For Fun]. I'm [What I Do For Fun]. I'm [What I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. It is also known for its cuisine, including French cuisine, and its fashion industry. The city is home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has been a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased Integration of AI into Everyday Life: AI is already being integrated into many aspects of our lives, from self-driving cars to virtual assistants like Siri and Alexa. As the technology continues to advance, we can expect to see even more integration of AI into our daily lives, from smart homes to virtual reality experiences.
    
    2. Increased Use of AI for Medical and Healthcare: AI is already being used in medical and healthcare applications, such
    


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
    Generated text:  [Name], and I'm a/an [Job/Title]. I'm a/an [Age/Experience/Role] who has always been [ambition/ambition]. I'm always [characteristic/characteristic]. I love to [occupation/action/action]. I'm passionate about [interest/travel/internet/technology], and I enjoy [activity], [eat type], [interests], and [other interests]. I have a [a/an] [education] from [school] and [school] has inspired me to [career goal] in [career] field. I have [born] [year] and have been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    Please provide the sentence in French.
    Sure, the French capital is Paris. 
    
    - The French capital is Paris.
    - The capital of France is Paris.
    - The capital of France is Paris. 
    
    I've included the French version for better readability. Is there anything else you'd like to know about Paris? Do you have any questions about the French capital? 
    
    If you have any additional requests, please let me know! 😊😊😊😊
    
    P.S. Keep in mind that French is not the only language spoken in Paris. The city is also home to many other languages such as French, English, and others
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly promising, and there are several trends that are likely to shape its development in the coming years. Here are some of the most likely trends:
    
    1. Improved privacy and security: As more people become more comfortable with AI, there will be a greater focus on ensuring that AI systems are not only effective but also ethical and trustworthy. This will require significant improvements in privacy and security, including the use of encryption, biometric authentication, and other advanced techniques to protect sensitive data.
    
    2. Increased automation: With the advancement of AI, we are likely to see a significant increase in automation in industries such as manufacturing, transportation, and customer service.


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

     I

    'm

     a

     [

    age

    ]

     year

     old

     [

    occupation

    ]

     and

     I

     enjoy

     reading

     books

     and

     spending

     time

     with

     my

     [

    family

     member

     or

     pets

    ].

     I

     love

     to

     travel

     and

     try

     new

     things

    ,

     and

     I

    'm

     always

     on

     the

     lookout

     for

     new

     adventures

    .

     What

    's

     your

     favorite

     hobby

    ?

     I

    'm

     [

    h

    obby

     name

    ]

     and

     I

     really

     enjoy

     [

    reason

     for

     hobby

    ].

     Can

     you

     share

     your

     favorite

     book

     or

     movie

     with

     me

    ?

     Oh

    ,

     and

     by

     the

     way

    ,

     I

    'm

     [

    occupation

    ].

     What

     brings

     you

     to

     this

     place

    ?

     My

     name

     is

     [

    name

    ],

     I

    'm

     a

     [

    age

    ]

     year

     old

     [

    occupation

    ]

     and

     I

     love

     [

    reason

     for

     occupation

    ].

     What

    
    
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

     and

     the

     “

    Paris

     of

     the

     Winds

    ”

    .


    The

     answer

     is

    :

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

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     the

     "

    Paris

     of

     the

     Winds

    ".

     
    


    To

     elaborate

     further

    :


    1

    .

     "

    City

     of

     Light

    "

     refers

     to

     Paris

     as

     a

     city

     renowned

     for

     its

     artistic

     and

     cultural

     appeal

    ,

     particularly

     in

     the

     art

     world

     and

     the

     vibrant

     nightlife

    .


    2

    .

     "

    Paris

     of

     the

     Winds

    "

     is

     a

     nickname

     that

     indicates

     Paris

    '

     significant

     contribution

     to

     the

     French

     economy

    ,

     specifically

     in

     the

     automobile

     industry

     and

     its

     importance

     in

     French

     politics

     and

     culture

    .


    3

    .

     France

    's

     capital

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    ,

     with

     many

     potential

     directions

     to

     explore

    .

     Here

     are

     some

     possible

     trends

     in

     the

     development

     of

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     experience

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     everyday

     lives

    ,

     it

     will

     likely

     become

     even

     more

     accessible

     and

     intuitive

    .

     This

     could

     lead

     to

     a

     more

     human

    -like

     interaction

     with

     AI

     systems

    ,

     allowing

     us

     to

     interact

     with

     them

     more

     naturally

     and

     efficiently

    .
    


    2

    .

     Improved

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     will

     be

     increasing

     concerns

     about

     the

     privacy

     and

     security

     of

     their

     data

    .

     It

    's

     important

     for

     developers

     to

     prioritize

     privacy

     and

     security

     to

     ensure

     that

     AI

     systems

     are

     safe

     and

     secure

    .
    


    3

    .

     Increased

     automation

     and

     automation

    



```python
llm.shutdown()
```
