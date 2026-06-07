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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:48,  1.12it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:04<00:22,  2.31it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]

    Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04,  8.92it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]

    Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:05<00:02, 14.54it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 22.03it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 30.25it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 39.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   2%|▏         | 1/58 [00:00<00:05,  9.64it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   2%|▏         | 1/58 [00:00<00:05,  9.64it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:09,  6.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:09,  6.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:09,  6.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:05, 10.06it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   7%|▋         | 4/58 [00:00<00:05, 10.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   7%|▋         | 4/58 [00:00<00:05, 10.06it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):  10%|█         | 6/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  10%|█         | 6/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.78it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.45it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  28%|██▊       | 16/58 [00:00<00:01, 24.45it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.95it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  34%|███▍      | 20/58 [00:01<00:01, 27.95it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:01<00:01, 30.98it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:01<00:00, 35.63it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=288 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.31it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=160 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.75it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.63it/s]Capturing num tokens (num_tokens=64 avail_mem=74.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.43it/s]

    Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.43it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 29.18it/s]


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
    Generated text:  John. I'm 24 years old and I live in the USA. My parents are both very healthy and they have good habits. I am not overweight and I exercise regularly. My father likes to eat lots of chocolate and I don't. I have never drank a glass of milk and my mother drinks a glass of milk every day. I am very healthy and don't feel sick. I have a healthy diet. I eat a lot of fruits and vegetables and I don't eat too much meat or fish. I like to eat sweet foods like candy and cake but I don't like to eat salty or spicy food. What's
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of what? The president of the United States is a member of the federal government, which is the executive branch of the U. S. government. The federal government is responsible for all of the federal government's actions, including military and economic decisions. The president of the United States is the head of state, the commander-in-chief of the armed forces, and the head of government, and they are all members of the federal government.
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris, also known as "La Rose de l'Orient" in French and "La Rose de la Méditerranée" in Italian, is the capital of France. Located on the left bank of the Seine, it is a major city in the Île-de-France region, one of the most populous and economically important cities in Europe. It is also the largest city in the United Kingdom and is the 16th largest city in the European Union (EU). Paris is known for its unique architecture, museums, museums, museums, and landmarks, including the Eiffel Tower, Louvre Museum, Notre
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and will be for a long time. That’s because the technology is getting more and more complex, and because of the data being collected and used for AI. The future of AI will require a coordinated effort by all stakeholders to ensure that it is being used ethically and responsibly. This will involve developing policies and standards that ensure that AI is developed and used in a responsible way, and that it is used to improve society as a whole. The key is to involve all stakeholders in the process and to work towards a common goal of creating a better world for everyone. In order to achieve this, it will require a strong commitment from


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic statement about your favorite activity or hobby]. I'm always looking for new ways to challenge myself and expand my horizons. What's your favorite book or movie? I'm a huge [insert a short,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major cultural and economic center, with a diverse population and a thriving arts scene. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the Notre-Dame Cathedral and the Eiffel Tower. Paris is a popular tourist destination and is known for its fashion, food, and wine industries. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, healthcare, and transportation, and we can expect to see even more automation and robotics in the future. This will lead to increased efficiency, lower costs, and improved quality of life.
    
    2. Enhanced cognitive abilities: AI will continue to improve its ability to process and understand complex information, leading to new applications in fields such as education, finance, and healthcare.
    
    3. Personalization and customization: AI will enable more personalized and customized experiences for users, leading to improved user satisfaction and loyalty.
    
    4.
    


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
    Generated text:  [Your Name]. I'm a [Age] year old [Occupation]. I'm a... [What is your character's personality? ] If you could give me one piece of advice, what would it be?
    
    My character is an [Occupation]. I'm a [Age] year old [Occupation]. I'm a... [What is your character's personality? ] If you could give me one piece of advice, what would it be?
    
    I'm an [Occupation]. I'm a [Age] year old [Occupation]. I'm a... [What is my character's personality? ] If you could give
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as “la Roche,” the place where the Seine River ends and the Eiffel Tower begins.
    
    Please answer the following question about the statement:
    What is the capital of France?
    
    The capital of France is Paris. Paris, also known as "la Roche," is the largest city in France and the seat of the government and the cultural, political, and commercial center of the country. It is located on the Île de France and the Seine River, and features the Eiffel Tower as one of its landmarks. The city has a rich history dating back to the Middle Ages and has undergone several
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid innovation, and a wide range of applications across a wide range of industries. Some possible future trends in AI include:
    
    1. Increased development of AI-powered robots and automation: As AI technology continues to improve, we expect to see an increase in the number and complexity of robots designed to perform a wide range of tasks, from manufacturing to healthcare. This will likely lead to the development of more advanced automation systems that can perform tasks with greater accuracy and efficiency than human workers.
    
    2. The integration of AI into human decision-making: AI is expected to play a more important role in how humans make decisions, from making stock


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

     I

    'm

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

    'm

     always

     looking

     for

     ways

     to

     [

    Describe

     Your

     Job

     Function

    ality

    ].

     I

    've

     been

     [

    Number

     of

     Years

     in

     Position

    ],

     and

     my

     passion

     lies

     in

     [

    Your

     Preferred

     Activity

     or

     Hobby

    ].

     So

     if

     you

    're

     searching

     for

     a

     person

     who

     is

     [

    What

     You

     Want

     to

     be

    ],

     or

     if

     you

     have

     a

     project

     that

     you

     need

     help

     with

    ,

     I

    'd

     be

     happy

     to

     help

    .

     What

     brings

     you

     to

     the

     table

    ,

     and

     what

     do

     you

     hope

     to

     achieve

     with

     us

    ?

     [

    Short

    est

     possible

     introduction

    ]

     In

     my

     spare

     time

    ,

     I

     like

     to

     read

     books

    ,

     play

     music

    ,

     and

     spend

     time

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Its

     historical

     significance

     dates

     back

     to

     the

     Roman

     Empire

    ,

     and

     it

     has

     been

     a

     major

     hub

     of

     culture

    ,

     commerce

    ,

     and

     politics

     in

     Europe

     for

     over

     

    2

    ,

    0

    0

    0

     years

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     world

    -ren

    owned

     fashion

    ,

     and

     it

     continues

     to

     be

     a

     global

     cultural

     and

     political

     center

    .

     The

     city

     is

     also

     famous

     for

     its

     iconic

     landmarks

    ,

     including

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     Site

     and

     a

     major

     tourist

     destination

    ,

     drawing

     millions

     of

     visitors

     each

     year

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     galleries

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     full

     of

     exciting

     possibilities

    ,

     and

     we

     can

     expect

     to

     see

     significant

     advancements

     in

     the

     next

     few

     decades

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

     future

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     technology

     continues

     to

     evolve

    ,

     it

     is

     important

     that

     we

     continue

     to

     focus

     on

     ethical

     considerations

    ,

     such

     as

     privacy

    ,

     fairness

    ,

     and

     accountability

    .

     This

     means

     that

     we

     will

     see

     more

     focus

     on

     developing

     AI

     systems

     that

     are

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

    .
    


    2

    .

     More

     automation

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     we

     are

     likely

     to

     see

     an

     increase

     in

     the

     automation

     of

     certain

     tasks

    .

     This

     could

     include

     tasks

     like

     data

    



```python
llm.shutdown()
```
