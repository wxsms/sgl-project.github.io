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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:05,  4.30s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.48it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.48it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.03it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.12it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.31it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.35it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.64it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.74 GB):   9%|▊         | 5/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:02, 21.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.46it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.72 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.06it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.71 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.71 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.26it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=61.70 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.26it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.95it/s] Capturing num tokens (num_tokens=896 avail_mem=61.69 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.95it/s]Capturing num tokens (num_tokens=832 avail_mem=61.68 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.95it/s]

    Capturing num tokens (num_tokens=832 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 29.91it/s]Capturing num tokens (num_tokens=768 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 29.91it/s]Capturing num tokens (num_tokens=704 avail_mem=61.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 29.91it/s]Capturing num tokens (num_tokens=640 avail_mem=61.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 29.91it/s]Capturing num tokens (num_tokens=576 avail_mem=61.67 GB):  41%|████▏     | 24/58 [00:00<00:01, 29.91it/s]Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.91it/s]Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]Capturing num tokens (num_tokens=480 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]Capturing num tokens (num_tokens=448 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]Capturing num tokens (num_tokens=416 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]

    Capturing num tokens (num_tokens=384 avail_mem=61.67 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  50%|█████     | 29/58 [00:01<00:00, 31.93it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=320 avail_mem=61.66 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=288 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=256 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.13it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=224 avail_mem=61.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]

    Capturing num tokens (num_tokens=208 avail_mem=61.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=160 avail_mem=61.63 GB):  66%|██████▌   | 38/58 [00:01<00:00, 33.68it/s]Capturing num tokens (num_tokens=160 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=128 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=96 avail_mem=61.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s] Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.95it/s]Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=64 avail_mem=61.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]

    Capturing num tokens (num_tokens=48 avail_mem=61.61 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=32 avail_mem=60.34 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=28 avail_mem=60.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=24 avail_mem=60.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=24 avail_mem=60.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=12 avail_mem=60.23 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s] Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 43.53it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 33.50it/s]


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
    Generated text:  Dina and I am a professional photographer specializing in wildlife photography. I have been photographing wildlife for over 8 years and have worked in various photography studios and on independent assignments. I am a member of the Society of Professional Photographers and the San Diego Photographic Guild.
    I have a deep love for the wild and animals, especially those that are threatened and endangered. I have captured a diverse range of wildlife in both natural and man-made settings and enjoy sharing my passion for wildlife photography with my followers.
    If you are interested in working with me to capture the beauty of the wild, please let me know and we can begin discussing your next
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term and is currently the 42nd president. He will be reelected if he can win a majority of the electoral votes, which is 270. As the incumbent, he has received 170 electoral votes. What is the minimum number of electoral votes that the president needs to win the election to win re-election?
    
    To determine the minimum number of electoral votes the president needs to win the election, we start by understanding the requirements for re-election. The president must win a majority of the electoral votes, which is 270. The president currently has 170 electoral votes
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was founded in the 8th century. The first written records of Paris can be traced back to the 3rd century. It was renamed the capital of France in 1793. It is located in the south of France, on the north shore of the Seine. Its weather is generally cool and wet. The climate is humid and has a high level of humidity. The weather conditions in the city are quite variable.
    
    1. Is it possible to determine the population of Paris using the given information? If so, identify the variable that would need to be measured to do so.
    
    2. Based on the
    ===============================
    Prompt: The future of AI is
    Generated text:  not in the future, it is today. In fact, it is the present. Here are some of the most promising areas for AI today.
    
    AI is a rapidly growing field with an amazing potential to transform many different sectors of the world. AI is a term that has been in use for a long time and is still growing. It is a field that can be beneficial in solving some of the world’s biggest challenges.
    
    For example, AI can be used to improve healthcare by enabling doctors to make more accurate diagnoses and provide better medical treatment. AI can also be used to improve the safety of cars by using AI to detect potential accidents and prevent


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill or Trait] who have always been [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive Trait]. I'm [Positive Trait] and I'm always [Positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music. Paris is a cultural and economic hub of France and a major tourist destination. It is home to many world-renowned museums, art galleries, and theaters. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively. This could lead to more efficient and effective decision-making in a wide range of applications.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, such as privacy, fairness, and accountability. This could lead to more stringent regulations and guidelines for the development and use of AI.
    
    3. Increased focus on AI ethics and safety: As AI becomes more integrated
    


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
    Generated text:  [First Name], and I'm an [Job Title] who enjoys [Interest/Activity]. Let me know if you'd like to connect with me!
    
    In a sea of endless possibilities, I'm a [Job Title] who has always loved the wonder of the natural world, particularly the breathtaking beaches of the Caribbean. My passion for exploration and discovery is infectious, and I'm always looking for ways to expand my horizons. Whether it's discovering hidden islands, exploring the mysteries of underwater creatures, or simply soaking up the rich culture of the island, I'm always eager to immerse myself in the world of adventure. 
    
    I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a major city with a rich history, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a cultural and economic hub, hosting major events such as the Eiffel Tower climb-off and the Moulin Rouge. The city is also home to a diverse population of immigrants and refugees, with Paris being one of the most cosmopolitan cities in Europe. Paris is a popular destination for tourists, with its beautiful architecture, delicious cuisine, and iconic landmarks drawing millions of visitors each year. The city is also known for its vibrant nightlife and cultural events
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends:
    
    1. Increased precision: As AI becomes more sophisticated, it will become even more accurate and precise. This could lead to breakthroughs in fields such as medicine, environmental science, and cryptography.
    
    2. Integration with human intelligence: AI is already being used to augment human intelligence, but it could be used to enhance human abilities as well. This could lead to a new era of collective intelligence where AI becomes an integral part of human consciousness.
    
    3. Development of "super AI": Super AI is a hypothetical entity that can outperform humans in almost every field. It is difficult to predict how this


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

    'm

     a

     [

    job

     title

     or

     profession

    ]

     at

     [

    Company

    ].

     I

    've

     been

     in

     this

     field

     for

     [

    number

     of

     years

    ]

     years

     now

    ,

     and

     I

    've

     always

     had

     a

     strong

     passion

     for

     [

    specific

     field

     of

     interest

    ].

     I

     enjoy

     [

    reason

     for

     passion

    ],

     and

     I

     believe

     that

     my

     skills

     and

     experience

     make

     me

     a

     valuable

     asset

     to

     the

     company

    .


    [

    Name

    ]

     is

     a

     [

    job

     title

     or

     profession

    ]

     with

     a

     strong

     passion

     for

     [

    specific

     field

     of

     interest

    ].

     I

    've

     been

     in

     this

     field

     for

     [

    number

     of

     years

    ]

     years

     now

    ,

     and

     I

    've

     always

     had

     a

     strong

     passion

     for

     [

    specific

     field

     of

     interest

    ].

     I

     enjoy

     [

    reason

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     second

    -largest

     city

     in

     the

     country

     and

     is

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     and

     a

     vibrant

     cultural

     scene

    .

     
    


    **

    Note

    :**

     Please

     provide

     the

     answer

     as

     a

     list

     of

     bullet

     points

     or

     a

     direct

     quote

    ,

     as

     instructed

    .

     Here

    's

     a

     concise

     statement

    :
    


    -

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     is

     the

     second

    -largest

     city

     in

     France

    .


    -

     It

     is

     the

     capital

    ,

     and

     one

     of

     the

     largest

     cities

     in

     Europe

    .


    -

     Located

     on

     the

     western

     shore

     of

     the

     English

     Channel

    ,

     it

     is

     the

     country

    's

     most

     populous

     metropolitan

     area

    .


    -

     Not

    able

     landmarks

     include

     the

     E

    iff

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     bright

     and

     many

     of

     the

     potential

     applications

     are

     just

     now

     being

     explored

    ,

     but

     there

     are

     some

     trends

     that

     are

     becoming

     more

     apparent

    .

     One

     of

     the

     key

     trends

     is

     the

     increasing

     reliance

     on

     AI

     in

     areas

     like

     healthcare

    ,

     where

     medical

     professionals

     can

     use

     AI

     to

     diagnose

     and

     treat

     diseases

     more

     quickly

     and

     accurately

     than

     they

     ever

     have

     been

     before

    .

     The

     field

     of

     AI

     is

     also

     growing

     rapidly

    ,

     and

     researchers

     are

     finding

     new

     ways

     to

     apply

     AI

     to

     solve

     complex

     problems

     in

     fields

     like

     finance

    ,

     transportation

    ,

     and

     cybersecurity

    .
    


    Another

     trend

     is

     the

     increasing

     integration

     of

     AI

     in

     everyday

     life

    ,

     which

     will

     likely

     lead

     to

     more

     efficient

    ,

     more

     personalized

     experiences

    .

     For

     example

    ,

     AI

    -powered

     voice

     assistants

     like

     Siri

    



```python
llm.shutdown()
```
