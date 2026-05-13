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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]


    2026-05-13 17:04:28,921 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 17:04:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.63it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.61it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.03it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:02, 19.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=68.51 GB):   9%|▊         | 5/58 [00:00<00:02, 23.01it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.50 GB):   9%|▊         | 5/58 [00:00<00:02, 23.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.49 GB):   9%|▊         | 5/58 [00:00<00:02, 23.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.49 GB):   9%|▊         | 5/58 [00:00<00:02, 23.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.49 GB):   9%|▊         | 5/58 [00:00<00:02, 23.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.49 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=68.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.47 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.47 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s]Capturing num tokens (num_tokens=960 avail_mem=68.45 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s] Capturing num tokens (num_tokens=896 avail_mem=68.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s]

    Capturing num tokens (num_tokens=832 avail_mem=68.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.87it/s]Capturing num tokens (num_tokens=832 avail_mem=68.44 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=768 avail_mem=68.44 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=704 avail_mem=68.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=640 avail_mem=68.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=576 avail_mem=68.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=512 avail_mem=68.41 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=480 avail_mem=68.43 GB):  41%|████▏     | 24/58 [00:00<00:00, 42.37it/s]Capturing num tokens (num_tokens=480 avail_mem=68.43 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=448 avail_mem=68.43 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=416 avail_mem=68.43 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=384 avail_mem=68.42 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=352 avail_mem=68.42 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]

    Capturing num tokens (num_tokens=320 avail_mem=68.41 GB):  52%|█████▏    | 30/58 [00:00<00:00, 45.25it/s]Capturing num tokens (num_tokens=320 avail_mem=68.41 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=288 avail_mem=68.41 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=256 avail_mem=68.41 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=240 avail_mem=68.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=224 avail_mem=68.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=208 avail_mem=68.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.61it/s]Capturing num tokens (num_tokens=208 avail_mem=68.40 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=192 avail_mem=68.40 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]

    Capturing num tokens (num_tokens=176 avail_mem=68.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=160 avail_mem=68.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=144 avail_mem=68.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]Capturing num tokens (num_tokens=128 avail_mem=68.39 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.93it/s]

    Capturing num tokens (num_tokens=128 avail_mem=68.39 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=112 avail_mem=68.38 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=96 avail_mem=68.38 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.95it/s] Capturing num tokens (num_tokens=80 avail_mem=68.38 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=64 avail_mem=68.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=64 avail_mem=68.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.62it/s]Capturing num tokens (num_tokens=48 avail_mem=68.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.62it/s]

    Capturing num tokens (num_tokens=32 avail_mem=68.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.62it/s]Capturing num tokens (num_tokens=28 avail_mem=68.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.62it/s]Capturing num tokens (num_tokens=24 avail_mem=68.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 26.62it/s]Capturing num tokens (num_tokens=24 avail_mem=68.36 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=20 avail_mem=68.35 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=16 avail_mem=68.35 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=12 avail_mem=68.35 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=8 avail_mem=68.34 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s] Capturing num tokens (num_tokens=4 avail_mem=68.34 GB):  91%|█████████▏| 53/58 [00:01<00:00, 28.61it/s]Capturing num tokens (num_tokens=4 avail_mem=68.34 GB): 100%|██████████| 58/58 [00:01<00:00, 32.34it/s]Capturing num tokens (num_tokens=4 avail_mem=68.34 GB): 100%|██████████| 58/58 [00:01<00:00, 33.29it/s]


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
    Generated text:  Wang Hao. I am a student of Grade 8. My English name is John Smith. I have two best friends, Lucy and Lily. We are good friends. We often go to the movies together, play sports, and listen to music. We go to school on Mondays and Fridays. We have lunch at school. The day after our lunch, we have art class and read a book. We also help Mr. Zhang draw pictures. He is my history teacher. I like to help Mr. Zhang with his homework. But I don't like math. I like to help Mr. Zhang because I like solving math problems. My
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military tanks to buy. The tanks are very expensive, so the budget for the purchase is limited. The president is considering two options:
    
    Option 1: The president can buy either 200 tanks or 300 tanks.
    
    Option 2: The president can buy either 150 tanks or 250 tanks.
    
    The president wants to know if it's possible to buy an equal number of tanks in both options and still have the budget limited. Can you help the president with this question?
    
    To determine if the president can buy an equal number of tanks in both options and still have the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of France is in the:
    
    1. North
    2. East
    3. South
    4. West
    
    The capital of France is in the:
    
    1. North
    2. East
    3. South
    4. West
    
    The correct answer is 1. North. The capital of France is in the north region of the country. Paris, the capital city, is located in the center of France, on the north side of the Loire River. Paris is a major urban center in the French region of the north. The other options (east, south, and west) are not directly related to the location
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. As AI is increasingly integrated into industries across the globe, the field is entering a period of rapid growth, making significant contributions to various fields, such as healthcare, finance, and transportation, and the future is expected to bring significant benefits to the world. However, it is important to note that the development of AI should be approached with caution, with a focus on ethical and social implications and considerations. The following is a list of key points to consider when discussing AI ethics:
    
    1. The ethical implications of AI systems are not well-understood and are often subject to subjective and competing interests.
    2. The development of AI is influenced by


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character here]. I enjoy [insert a short description of your character's interests or hobbies here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum of Modern Art. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its fashion industry, art scene, and cuisine. Paris is a vibrant and dynamic city with a diverse population and a strong sense of community. It is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. AI developers will need to take these concerns into account when designing and implementing AI systems.
    
    2. Integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This could involve the use of AI-powered decision-making tools that can make recommendations based on human
    


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
    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and to discuss our potential collaboration. Thanks. [Tell me briefly about yourself. Your skills, experience, education, and career goals. What excites you the most about working with this company? What do you look for in a potential successor? What do you hope to achieve with our partnership? ] Write your answer in clear and concise language with a neutral tone. Let me know if you would like me to modify or expand on this response.
    The chosen character is excited to meet you and discuss potential collaboration. I am [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and serves as the political, cultural, and economic center of the nation. Paris is known for its rich history, diverse architecture, and vibrant culture, which have made it a popular tourist destination and a key city in French politics and diplomacy. The city is also home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Additionally, Paris has a long history of art and literature, with several museums and galleries dedicated to the country's cultural heritage. Overall, Paris is a unique and fascinating city that plays a significant role in France's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  quite promising, and several trends are likely to shape the field in the coming years. Here are some possible future trends in artificial intelligence:
    
    1. More sophisticated models: As AI continues to advance, models will become even more sophisticated. Researchers will focus on developing even more complex models that can handle complex tasks and have higher accuracy.
    
    2. Enhanced privacy and security: With the increasing use of AI, there is a growing concern about privacy and security. Researchers will continue to explore ways to protect the data and ensure that AI systems are not used to mislead or deceive users.
    
    3. Personalization: AI will become more personalized, allowing for more


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

     am

     [

    age

    ].

     I

     am

     a

     [

    occupation

    ]

     who

     has

     always

     had

     a

     love

     for

     the

     outdoors

    ,

     particularly

     running

    ,

     kay

    aking

    ,

     and

     hiking

    .

     I

     enjoy

     exploring

     new

     places

     and

     trying

     new

     activities

    .

     I

     am

     also

     a

     [

    skill

    ]

     who

     has

     been

     practicing

     my

     craft

     for

     years

    .

     I

     love

     spending

     time

     with

     my

     family

     and

     friends

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     make

     them

     laugh

    .

     I

     am

     [

    interest

    ].

     Thank

     you

     for

     considering

     my

     self

    -int

    roduction

    .

     That

     sounds

     like

     a

     great

     personality

    .

     How

     can

     I

     get

     started

     on

     writing

     a

     story

     based

     on

     this

     character

    's

     interests

     and

     hobbies

    ?

     As

     an

     AI

     language

     model

    ,

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    (A

    )

     False

     


    (B

    )

     True

    


    (C

    )

     Cannot

     be

     determined

     


    (D

    )

     Do

     not

     know

     


    (D

    )

     Do

     not

     know

    
    


    Paris

     is

     the

     capital

     city

     of

     France

    .

     This

     statement

     is

     false

    .

     The

     correct

     answer

     is

     (

    A

    ).

     The

     official

     name

     of

     the

     capital

     city

     of

     France

     is

     Paris

    .

     It

     is

     the

     largest

     and

     most

     important

     city

     in

     France

    ,

     and

     one

     of

     the

     world

    's

     largest

     cities

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     music

    ,

     and

     fashion

    .

     The

     city

     has

     a

     population

     of

     approximately

     

    2

    .

    3

     million

     people

     and

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     promising

    ,

     and

     there

     are

     a

     number

     of

     potential

     trends

     that

     could

     shape

     the

     way

     we

     use

     and

     interact

     with

     AI

     in

     the

     years

     to

     come

    .

     Here

     are

     some

     potential

     trends

     that

     could

     play

     a

     role

    :
    


    1

    .

     Improved

     efficiency

     and

     scalability

    :

     As

     AI

     systems

     become

     more

     capable

    ,

     there

    's

     a

     possibility

     they

     could

     be

     designed

     to

     work

     more

     efficiently

     and

     effectively

     in

     a

     wide

     range

     of

     applications

    .

     This

     could

     lead

     to

     new

     ways

     of

     processing

     and

     analyzing

     large

     amounts

     of

     data

    ,

     as

     well

     as

     more

     scalable

     AI

     systems

     that

     can

     handle

     multiple

     tasks

     simultaneously

    .
    


    2

    .

     Increased

     integration

     with

     human

     expertise

    :

     As

     AI

     systems

     get

     more

     advanced

    ,

     there

    's

     a

     possibility

     they

     could

     begin

     to

    



```python
llm.shutdown()
```
