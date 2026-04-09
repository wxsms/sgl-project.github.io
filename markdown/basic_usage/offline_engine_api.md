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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.54it/s]


    2026-04-09 16:36:58,427 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 16:36:58] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:47,  2.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<02:47,  2.94s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:31,  1.70it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:09,  5.26it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.19it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.19it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.00it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.42it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.41it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 33.84it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.74it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.21it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=118.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.51it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=960 avail_mem=118.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.63it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=768 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]

    Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=384 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  50%|█████     | 29/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=288 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.40it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.40it/s]Capturing num tokens (num_tokens=224 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]

    Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=160 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=112 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.07it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]

    Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.64it/s] Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.64it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 40.01it/s]


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
    Generated text:  Myriam. I am a registered nurse with over 10 years of experience in the healthcare field. I have had the pleasure of working in emergency and community settings as well as in inpatient and outpatient care. My goal is to provide the best care to my patients by being patient focused, compassionate and open in communication. I enjoy working with patients and their families to make sure they are well informed of their options, and taking a collaborative approach to care. I am a licensed nurse in both New Jersey and Florida. I am a member of the American Nurses Association. I am very involved in patient advocacy groups and have been a board
    ===============================
    Prompt: The president of the United States is
    Generated text:  a significant position in the country. Here is a question about the role of the President of the United States:
    What is the role of the President of the United States?
    
    The President of the United States is the head of state of the United States. He or she represents the nation and holds various high-level offices in the government, including the executive branch, legislative branch, and judicial branch. The President is responsible for making important decisions, implementing laws, and representing the interests of the country on the world stage. They also have the power to sign into law executive orders and appointments to federal offices. The President is also responsible for running the country
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the valley of which river?
    The capital of France is located in the valley of the Loire River.
    You are a helpful assistant, just will you share a commons sense why is it the case? The capital of France, Paris, is located in the valley of the Loire River. This river forms the heart of the city and provides the political, cultural, and economic center of the French capital. It is a vital transportation and communication artery that has shaped the history and culture of the city. The river also plays a role in shaping the landscape of the region it flows through, providing the necessary resources for agriculture and industry.
    ===============================
    Prompt: The future of AI is
    Generated text:  secure, research says
    
    Published: Oct 19, 2019
    
    The future of AI is secure, researchers say
    
    A series of studies published by the MIT media lab and the Kellogg Institute for International Studies finds that the risk of AI-based cyber attacks is far lower than previously believed. The team of researchers found that artificial intelligence and machine learning are more secure than people. The machines are not only better at detecting flaws and breaking security rules, they also adapt to changing cyber environments faster than humans.
    
    The research, published in the November issue of the "International Journal of Advanced Intelligent Systems and Software Engineering", found that AI


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its rich history and diverse cultural scene. Paris is also a major tourist destination, attracting millions of visitors each year, making it one of the most popular cities in the world. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin, and is known for its cuisine, including its famous Parisian dishes like croissants and baguettes
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making.
    
    2. Greater emphasis on ethical considerations: As AI becomes more sophisticated, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, but there is likely to be continued growth in its use in this area.
    
    4. Greater use of AI in education: AI is already
    


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
    Generated text:  [Your Name], and I am a creative writer and content strategist specializing in creating engaging content for all types of businesses. I have experience in multiple industries, from marketing and advertising to e-commerce and social media, and have a strong focus on creating content that resonates with readers and captivates them. I am passionate about using my writing skills to inspire and motivate people to take action and create positive change. Whether you're a curious consumer or a business owner, I am here to help you get the most out of your writing and content creation. Thank you for taking the time to learn more about me. [Your Name] feels more
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its grandiose architecture, iconic landmarks like the Eiffel Tower, and its rich history dating back over 500 years. Additionally, Paris is home to numerous museums, theaters, and other cultural institutions, making it a popular destination for tourists and locals alike. The city's cultural blend of traditional French art and modernist architecture has made it a cultural and intellectual center of France. Paris is also famous for its wine production, particularly in the region of Bordeaux. The city's vibrant nightlife and diverse food scene are also a major draw for visitors. Overall, Paris is a city of contrasts and diversity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by several trends that are shaping the development and application of AI technologies. Here are some of the potential future trends in AI:
    
    1. AI will become more ubiquitous: As AI technology improves and becomes more accessible, it is likely to become more ubiquitous in our daily lives. For example, we may see more automation in industries such as finance, healthcare, and manufacturing. We may also see AI being incorporated into everyday products and services, such as voice assistants, virtual assistants, and self-driving cars.
    
    2. AI will continue to be used for healthcare: As AI becomes more accessible and powerful, it is likely to be used


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

     __

    ________

    _.

     I

    'm

     a

    /an

     __

    ________

    _

    .


    What

     brings

     you

     here

     today

    ?

     What

     is

     your

     profession

    ?

     What

     is

     your

     education

     or

     training

    ?

     What

     are

     your

     areas

     of

     expertise

    ?

     What

     are

     your

     hobbies

     or

     interests

    ?

     Who

     is

     the

     most

     important

     person

     in

     your

     life

    ?

     What

     is

     your

     greatest

     accomplishment

     so

     far

    ?


    I

     am

     a

    /an

     __

    ________

    _

     from

     __

    ________

    _

     (

    city

    ,

     country

    ).

     I

     have

     traveled

     to

     __

    ________

    _

     (

    country

    ,

     city

    )

     __

    ________

    _

     times

    .

     I

     have

     lived

     and

     worked

     in

     __

    ________

    _

     (

    country

    ,

     city

    ).

     I

     have

     traveled

     to

     __

    ________

    _

     (

    country

    ,

     city

    )

     __

    ________

    _

     times

    .

     I

     have

     lived

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     with

     a

     rich

     history

     and

     diverse

     culture

    ,

     renowned

     for

     its

     iconic

     landmarks

    ,

     French

     cuisine

    ,

     and

     lively

     street

     life

    .

     
    


    Remember

    ,

     I

    'm

     not

     sure

     if

     you

     meant

     to

     ask

     about

     another

     city

     in

     France

    .

     Could

     you

     please

     provide

     more

     context

     or

     clarify

     your

     question

    ?

     I

    'd

     be

     happy

     to

     help

     if

     I

     can

    .

     
    


    If

     you

     meant

     to

     ask

     about

     Paris

    's

     role

     in

     the

     French

     Revolution

    ,

     you

     can

     include

     that

     as

     an

     additional

     detail

    .

     For

     example

    ,

     "

    What

     role

     did

     Paris

     play

     in

     the

     French

     Revolution

    ?

     "

     Would

     you

     like

     to

     elaborate

     on

     that

    ?

     
    


    If

     you

     meant

     to

     ask

     about

     Paris

    's

     relationship

     with

     other

     French

     cities

     or

     regions

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     exciting

    ,

     as

     it

     continues

     to

     evolve

     and

     become

     more

     advanced

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

     cognitive

     ability

    :

     AI

     is

     expected

     to

     continue

     developing

     and

     improving

     its

     cognitive

     abilities

    ,

     allowing

     it

     to

     become

     more

     intelligent

     and

     capable

     of

     performing

     complex

     tasks

    .
    


    2

    .

     Artificial

     general

     intelligence

    :

     This

     is

     the

     goal

     of

     AI

     to

     be

     able

     to

     perform

     any

     task

     that

     a

     human

     can

     do

    ,

     including

     decision

    -making

    ,

     problem

    -solving

    ,

     and

     creativity

    .
    


    3

    .

     Human

    -com

    puter

     interaction

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     able

     to

     interact

     with

     humans

     in

     more

     natural

     and

     effective

     ways

    ,

     improving

     the

     efficiency

     and

     effectiveness

     of

     human

    -com

    puter

     interactions

    



```python
llm.shutdown()
```
