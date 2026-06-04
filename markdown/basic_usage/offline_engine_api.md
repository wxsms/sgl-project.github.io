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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:21,  5.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:21,  5.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:21,  5.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:21,  5.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:21,  5.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:45,  1.16it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:05<00:17,  2.81it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:06,  6.36it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:06,  6.36it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:06,  6.36it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:06<00:06,  6.36it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:06<00:06,  6.36it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:06<00:06,  6.36it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:06<00:06,  6.36it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:06<00:03,  9.66it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:06<00:02, 13.59it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:06<00:01, 19.14it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:06<00:00, 26.38it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:06<00:00, 31.66it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.01it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.60it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:02, 19.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.81it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.70it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 23.70it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.82it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.82it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.82it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.82it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.45it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.45it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.45it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  43%|████▎     | 25/58 [00:01<00:01, 22.45it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.03it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.03it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.03it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.03it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.03it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.39it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.39it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.39it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.39it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.39it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.83it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.36it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 28.36it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.40it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.40it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.40it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.40it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.40it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.26it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.26it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 30.26it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:02<00:00, 30.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:02<00:00, 30.26it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:02<00:00, 29.69it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:02<00:00, 29.69it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:02<00:00, 29.69it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:02<00:00, 25.51it/s]


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
    Generated text:  Bob. This is a photo. This is a picture of Bob's family. A woman in a blue dress and a man in a red suit are standing in front of a red house. On the wall there is a picture of a tiger. There is a car next to the tiger. There is a chair behind the car. There is a dog sitting on the chair. On the wall there is a map of China. There is a map of the world and a map of England on the wall. There is also a map of Africa on the wall. There are some books on the desk. There is a ball and a pen on
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many wars to have with other countries. He has decided that for every 150 military operations, he wants to reduce the number by 10. If he currently has 250 operations, how many wars does he want to have in the future? To determine how many wars the president of the United States wants to have in the future, we need to follow these steps:
    
    1. Identify the current number of military operations.
    2. Determine the number of operations to be reduced.
    3. Calculate the number of operations that will be reduced.
    4. Find out how many operations will be left after the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is situated on the north bank of the River Seine, between the hills of Montmartre and Montaigne, and near the sea. Paris is a small city, with a population of 2,070,000 inhabitants, of which 1,250,000 live within the city walls. There are also 1,550,000 in the suburbs. In the area between Paris and the Seine, the population is about 850,000. The city has about 400,000 permanent residents and 
    ===============================
    Prompt: The future of AI is
    Generated text:  one of the greatest achievements of our time. From drones to self-driving cars, from online advertising to smart home devices, AI has fundamentally transformed the way we live our lives and work our jobs. But the implications of these technologies have not yet been fully realized, and the impact on society is just beginning to be felt.
    The future of AI, then, is about more than just making our lives easier; it is about creating an even more inclusive, equitable, and prosperous world. By embracing AI, we can drive innovation, foster innovation, and create a better future for all of us.
    In this article, we will explore the ways in


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


    Generated text:  [Name] and I'm a [occupation] who has been working in [industry] for [number] years. I'm passionate about [reason for interest] and have always been inspired by [person or thing that inspires me]. I'm always looking for new challenges and opportunities to grow and learn, and I'm always eager to share my knowledge and experiences with others. I'm a [type of person] who is always ready to learn and grow, and I'm always looking for ways to contribute to the world. I'm excited to meet you and learn more about you. [Name] [Occupation] [Industry] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant culture, fashion, and cuisine, and is a popular tourist destination. The city is home to many important institutions such as the French Academy of Sciences and the French Parliament. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more efficient and effective decision-making, as well as more personalized and context-aware interactions.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be an increased focus on ethical and social considerations. This could lead to more responsible and sustainable use
    


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
    Generated text:  [Name]. I'm a [insert profession or role], and I'm very good at [insert skill or activity]. I enjoy [insert hobby or activity]. And I'm always [insert adjective]. I'm excited to meet you and discuss how I can help you with whatever you need. What's your name and what kind of person are you? [Name] [insert profession or role] I'm very good at [insert skill or activity], and I enjoy [insert hobby or activity]. I'm always [insert adjective]. And I'm excited to meet you and discuss how I can help you with whatever you need. [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the world's third-largest city by population.
    
    Paris is the largest city in France, with an estimated population of over 2 million people. It is located on the left bank of the Seine river, a UNESCO World Heritage site, and is known for its historical architecture, art, and music scene. It is also a major economic and cultural center, with a rich culinary heritage and a thriving film industry. Paris is a popular tourist destination, with over 16 million visitors per year and a rich cultural and artistic tradition that continues to this day. The city's status as the world's third-largest city by population is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, with numerous trends and innovations anticipated by researchers, businesses, and policymakers. Here are some possible future trends in artificial intelligence:
    
    1. Semantic AI: As technology advances, AI systems will become more adept at understanding natural language and semantic meaning, enabling them to interpret and synthesize complex information from various sources.
    
    2. Autonomous vehicles: Self-driving cars and drones will become more widely available, with AI systems taking over many of the tasks traditionally performed by human drivers and pilots.
    
    3. Quantum AI: Quantum computers are expected to significantly boost AI performance in areas like machine learning and deep learning, as they can process data more efficiently than traditional


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

     ____

    _.

     I

     am

     an

     amateur

     video

     editor

     with

     a

     background

     in

     graphic

     design

    ,

     photography

    ,

     and

     a

     knack

     for creating

     stunning

     visuals

    .

     I

     have

     a

     love

     for

     mixing

     all

     these

     different

     elements

     and

     putting

     them

     all

     together

     to

     create

     a

     cohesive

     and

     visually

     striking

     piece

     of

     work

    .

     I

     am

     a

     natural

     storyt

    eller

     and

     my

     ability

     to

     capture

     the

     essence

     of

     a

     story

     through

     the

     lens

     of

     my

     editing

     can

     be

     seen

     in

     my

     work

    .

     I

     have

     a

     passion

     for

     learning

     and

     constantly

     exploring

     new

     tools

     and

     techniques

     to

     push

     my

     creative

     boundaries

    .

     I

     am

     excited

     to

     help

     you

     tell

     your

     story

     in

     a

     visual

     way

     that

     truly

     tells

     the

     story

    .

     How

     would

     you

     describe

     your

     personality

     and

     what

     makes

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     rich

     history

     and

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

     It

     is

     also

     the

     third

    -largest

     city

     in

     the

     European

     Union

    .

     As

     of

     

    2

    0

    2

    1

    ,

     it

     has

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    .

     
    


    To

     answer

     the

     question

     "

    What

     is

     Paris

     famous

     for

    ?

     ",

     it

    's

     worth

     noting

     that

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

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

     It

    's

     also

     the

     third

    -largest

     city

     in

     the

     European

     Union

    .

     As

     of

     

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     increasing

     complexity

    ,

     relevance

    ,

     and

     adapt

    ability

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Greater

     integration

     of

     AI

     into

     various

     industries

    :

     The

     integration

     of

     AI

     into

     various

     industries

     is

     expected

     to

     continue

     growing

     as

     more

     companies

     and

     organizations

     realize

     the

     potential

     of

     AI

    .

     This

     could

     include

     areas

     like

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    .
    


    2

    .

     Enhanced

     privacy

     and

     data

     protection

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     there

     will

     be

     an

     increased

     need

     for

     robust data

     protection

     measures

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

     This

     will

     include

     measures

     like

     encryption

    ,

     anonym

    ization

    ,

     and

     data

     minim

    ization

    .
    


    3

    .

     Increased

     reliance

     on

     AI for

     decision

    -making

    



```python
llm.shutdown()
```
