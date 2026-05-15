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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-05-15 12:25:22,809 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 12:25:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.78it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  7.95it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  7.95it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  7.95it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  7.95it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  7.95it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:04,  7.95it/s]

    Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:05<00:04,  7.95it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 13.86it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 18.84it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 25.57it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.58it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.58it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 14.58it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   7%|▋         | 4/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   7%|▋         | 4/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   7%|▋         | 4/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  12%|█▏        | 7/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  17%|█▋        | 10/58 [00:00<00:02, 21.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:02, 22.39it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.63 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.63 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.63 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.63 GB):  31%|███       | 18/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.61 GB):  31%|███       | 18/58 [00:00<00:02, 18.69it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.13 GB):  31%|███       | 18/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  31%|███       | 18/58 [00:01<00:02, 18.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.23it/s]Capturing num tokens (num_tokens=960 avail_mem=74.12 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.23it/s] Capturing num tokens (num_tokens=896 avail_mem=74.12 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.23it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:01<00:01, 20.23it/s]Capturing num tokens (num_tokens=832 avail_mem=73.96 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.79it/s]Capturing num tokens (num_tokens=768 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.79it/s]

    Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.79it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.79it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.81it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.81it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.81it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.81it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.81it/s]

    Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=416 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=384 avail_mem=73.66 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=352 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=320 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  53%|█████▎    | 31/58 [00:01<00:01, 25.73it/s]Capturing num tokens (num_tokens=288 avail_mem=73.65 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=256 avail_mem=73.65 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=240 avail_mem=73.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=224 avail_mem=73.64 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=208 avail_mem=73.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 31.21it/s]

    Capturing num tokens (num_tokens=192 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=176 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=160 avail_mem=73.63 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=144 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=128 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  71%|███████   | 41/58 [00:01<00:00, 35.20it/s]Capturing num tokens (num_tokens=112 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=96 avail_mem=73.62 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.38it/s] Capturing num tokens (num_tokens=80 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=64 avail_mem=73.61 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 37.38it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.03it/s]Capturing num tokens (num_tokens=32 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.03it/s]Capturing num tokens (num_tokens=28 avail_mem=73.60 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.03it/s]Capturing num tokens (num_tokens=24 avail_mem=73.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.03it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.03it/s]Capturing num tokens (num_tokens=20 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=16 avail_mem=73.59 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=12 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=8 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.55it/s] Capturing num tokens (num_tokens=4 avail_mem=73.58 GB):  93%|█████████▎| 54/58 [00:02<00:00, 38.55it/s]

    Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:02<00:00, 37.98it/s]Capturing num tokens (num_tokens=4 avail_mem=73.58 GB): 100%|██████████| 58/58 [00:02<00:00, 27.93it/s]


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
    Generated text:  Ronan. My favorite place to play is at the beach.
    How would you describe Ronan's personality? Ronan is a friendly, energetic, and adventurous person who enjoys spending time at the beach with friends and family. His love of the outdoors and desire to explore is evident in his love for surfing and taking part in various beach activities, such as kiteboarding, wave surfing, and paddleboarding. Ronan is also known for his strong sense of teamwork and has a diverse set of interests that include photography, painting, and music. Overall, Ronan is a fun and outgoing person who is always looking to connect with others and enjoy
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking public official and a member of the highest echelons of government. Here is what he or she does:
    I. Represents the country in the United Nations.
    II. Coordinates foreign relations between the United States and other countries.
    III. Presides at inauguration and other state functions.
    Which one is true regarding the president of the United States?
    A. He represents the country in the United Nations.
    B. He coordinates foreign relations between the United States and other countries.
    C. He presides at inauguration and other state functions.
    D. He does not represent the country in the United Nations.
    E. He is the only person
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. French is the language spoken in Paris.
    A. 错误
    B. 正确
    答案:
    B
    
    “气候友好型”旅游，就是旅游者在旅游过程中，不破坏自然环境，不污染环境，不浪费资源，不破坏文化遗产，不破坏生态平衡，不破坏旅游环境，不破坏旅游者身体健康。
    A. 正确
    B. 错误
    答案:
    B
    
    14. 在中国境内有住所，或者无住所而一个纳税年度内在中国境内居住累计满____的个人，为居民个人。
    A. 60天
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  bringing about a revolution in the world of business. With the rise of big data, machine learning, and other technologies, companies are becoming more efficient and effective. They are able to make better decisions and personalize their customer experience. This can lead to increased sales and higher profits. However, there are also risks associated with AI that companies must be aware of. Some of the most common risks include privacy concerns, bias, and the potential for job displacement. To overcome these risks, companies need to implement robust measures to ensure the ethical use of AI. This includes implementing data protection policies, training employees on AI ethics, and fostering a culture of transparency


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character's personality or background]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning opportunities. What's your favorite hobby or activity? I'm always up for a challenge and love to try new things. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. I'm always looking for new ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French Parliament building. Paris is a cultural and economic hub, with a rich history dating back to the Roman Empire and a modern city that has undergone significant development over the centuries. It is a popular tourist destination, attracting millions of visitors each year, and is a major center for politics, art, and culture in the world. Paris is also known for its cuisine, with its famous dishes such as croissants,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve, leading to more advanced AI systems that can perform a wider range of tasks and solve more complex problems. Additionally, there is a growing focus on ethical considerations and the responsible use of AI, as concerns about bias, transparency, and accountability continue to grow. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption and integration of AI in various industries and applications. Overall, the future of AI is likely to be characterized by continued innovation, growth, and
    


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
    Generated text:  [Your Name] and I'm a [Title] at [Company Name]. I have [Number of years in this position] years of experience in this field. My expertise lies in [mention a specific skill or area of expertise]. I'm confident in my abilities and always strive to improve my skills and knowledge. I'm [insert a personality trait or quality that best describes you]. I'm [insert a positive attribute or trait that best describes you].
    Remember, we're all different, but I've been successful in [mention a specific accomplishment or achievement], and I'm always eager to learn and grow. So if you have any questions
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France and is home to the French parliament, the Supreme Court, and many other government agencies. The city is known for its vibrant culture, rich history, and beautiful architecture. It is a popular tourist destination and a hub for international business and diplomacy. Paris is also known for its fashion industry and the famous Eiffel Tower, which is one of the most famous landmarks in the world. The city is often referred to as the "City of Light" due to its abundance of lighted buildings and narrow streets. Despite its size, Paris is a very livable and diverse city with many neighborhoods and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and rapidly evolving, and it is difficult to predict exactly what the future will hold. However, here are some possible trends that are currently occurring and could potentially shape the future of AI:
    
    1. Increased use of AI for automation: As AI becomes more sophisticated, it is likely to be used for tasks that previously required human labor, such as data analysis, image recognition, and process automation. This could lead to the widespread adoption of AI in industries such as manufacturing, logistics, and healthcare.
    
    2. Enhanced privacy and security: As AI is becoming more sophisticated, it is becoming more dependent on data to function. This means that AI systems


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

     an

     AI

     assistant

    .

     I

    'm

     always

     ready

     to

     help

     you

     with

     anything

     you

     need

    !

     Let

     me

     know

     how

     I

     can

     assist

     you

    !

     Let

    's

     start

     by

     learning

     more

     about

     you

    .
    


    Name

    :

     Your

     name

     is

     [

    Name

    ],

     but

     this

     is

     my

     fictional

     self

    -int

    roduction

     for

     you

    .

     I

    'm

     an

     AI

     assistant

    ,

     and

     I

     have

     the

     ability

     to

     assist

     with

     a

     wide

     range

     of

     tasks

    .

     Whether

     you

     need

     help

     with

     answering

     a

     question

    ,

     finding

     information

    ,

     or

     even

     just

     a

     quick

     chat

    ,

     I

    'm

     here

     to

     help

     you

    !

     If

     there

    's

     anything

     specific

     you

     need

     help

     with

    ,

     I

    'm

     happy

     to

     assist

     you

    .

     Let

     me

     know

     how

     I

     can

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

     concise

     factual

     statement

     about

     a

     famous

     historical

     monument

     in

     France

    .

     The

     E

    iff

    el

     Tower

     is

     located

     in

     Paris

    .

     
    


    A

     concise

     factual

     statement

     about

     a

     famous

     landmark

     in

     France

    .

     The

     Lou

    vre

     Museum

     is

     located

     in

     Paris

    .

     
    


    A

     concise

     factual

     statement

     about

     a

     famous

     tourist

     attraction

     in

     France

    .

     The

     Ch

    amps

    -

    É

    lys

    ées

     is

     located

     in

     Paris

    .

     
    


    A

     concise

     factual

     statement

     about

     a

     famous

     historical

     event

     in

     France

    .

     The

     Battle

     of

     Waterloo

     occurred

     in

     Paris

     during

     the

     Nap

    ole

    onic

     Wars

    .

     
    


    A

     concise

     factual

     statement

     about

     a

     famous

     play

     that

     is

     famous

     in

     France

    .

     "

    Ham

    let

    "

     is

     a

     play

     by

     William

     Shakespeare

     and

     was

     performed

     in

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     different

     from

     today

    's

     technological

     landscape

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     various

     industries

    :

     AI

     is

     already

     being

     integrated

     into

     various

     industries

     such

     as

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    .

     It

     is

     expected

     that

     this

     trend

     will

     continue

    ,

     with

     more

     and

     more

     industries

     adopting

     AI

     technologies

    .
    


    2

    .

     AI

     will

     become

     more

     personalized

    :

     AI

     will

     become

     more

     personalized

    ,

     as

     it

     is

     capable

     of

     learning

     from

     individual

     data

     and

     adapting

     to

     the

     user

    's

     preferences

     and

     needs

    .

     This

     will

     lead

     to

     more

     personalized

     and

     relevant

     AI

     technologies

    ,

     and

     will

     enable

     users

     to

     access

     more

     personalized

     services

    .
    


    3

    .

     AI

     will

     become

     more

    



```python
llm.shutdown()
```
