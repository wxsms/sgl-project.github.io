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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.67it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.67it/s]


    2026-05-05 00:48:03,171 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 00:48:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:19,  2.51it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:19,  2.51it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:19,  2.51it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:19,  2.51it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.51it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.51it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]

    Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:05<00:09,  4.99it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:04,  8.05it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:04,  8.05it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:04,  8.05it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:04,  8.05it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:04,  8.05it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:04,  8.05it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.63it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]

    Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 20.21it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 26.28it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 35.27it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 44.07it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 44.07it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 44.07it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 44.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.38 GB):   7%|▋         | 4/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 12.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 12.12it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.37 GB):  10%|█         | 6/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 13.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.34 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.27it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.33 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.27it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.33 GB):  17%|█▋        | 10/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.32 GB):  17%|█▋        | 10/58 [00:00<00:05,  8.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  17%|█▋        | 10/58 [00:01<00:05,  8.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.09it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.29 GB):  22%|██▏       | 13/58 [00:01<00:03, 12.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.29 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.26 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.81it/s]Capturing num tokens (num_tokens=960 avail_mem=74.25 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.81it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.81it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  34%|███▍      | 20/58 [00:01<00:01, 19.81it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=576 avail_mem=74.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.47it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.47it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.47it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.47it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.47it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.38it/s]Capturing num tokens (num_tokens=384 avail_mem=74.21 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.38it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.38it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.38it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  55%|█████▌    | 32/58 [00:01<00:00, 30.38it/s]Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.15it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.15 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=224 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.15it/s]Capturing num tokens (num_tokens=208 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.88it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.88it/s]

    Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  69%|██████▉   | 40/58 [00:02<00:00, 31.88it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:02<00:00, 31.88it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  69%|██████▉   | 40/58 [00:02<00:00, 31.88it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  76%|███████▌  | 44/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  76%|███████▌  | 44/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  76%|███████▌  | 44/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  76%|███████▌  | 44/58 [00:02<00:00, 18.66it/s] Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  76%|███████▌  | 44/58 [00:02<00:00, 18.66it/s]Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.24it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.24it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.10 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.24it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.24it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.24it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.62it/s]Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.62it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.62it/s]Capturing num tokens (num_tokens=16 avail_mem=74.07 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.62it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.62it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.12it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.12it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.05 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.12it/s]Capturing num tokens (num_tokens=4 avail_mem=74.05 GB): 100%|██████████| 58/58 [00:02<00:00, 21.56it/s]


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
    Generated text:  Sue. I am a student from Canada. I have a big problem. I know it's not easy for me to live here because I don't have enough money to buy a house. Many foreigners (外国人) are living here. They have more money than I. We also have some difficulties. The weather in this city is always changing. This means that I have to leave my house at night. And in summer, the hot sun is very annoying to me. I can't swim in the river. I can't play sports. I can't go shopping. I have to leave my car and move to a new place. The
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to reduce carbon emissions. He believes that burning fossil fuels is the only way to do this. One day, he notices that the emissions from his country's two factories are such that they are directly proportional to the amount of fossil fuel used. If the emissions from the first factory are 150,000 metric tons per year, and the emissions from the second factory are 200,000 metric tons per year, what would be the ratio of the emissions from the first factory to the second factory if the emissions from both factories are combined? Assume that the relationship between the emissions and the amount of fossil
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of England is London. The capital of the United States is Washington, D. C. .
    A. Yes
    B. No
    C. It's impossible to say
    B. No
    
    Explanation: The capital of England is London, not Washington, D. C. Therefore, the statement is false. The capital of the United States is Washington, D. C., not London. Hence, the answer is B. No. The question asks for the correct answer, and B. No is the only option that accurately reflects the information provided in the statement. However, since the question is asking for the correct answer
    ===============================
    Prompt: The future of AI is
    Generated text:  fast approaching, as evidenced by the fact that the technology is already showing real-world applications in a wide variety of fields, including medical diagnostics, autonomous vehicles, and virtual reality. While the potential benefits of AI are immense, it is also a technology that requires careful consideration to ensure that it is used effectively and ethically.
    
    One of the most important considerations when working with AI is its impact on privacy and data security. As AI systems become more sophisticated, they are likely to generate and store large amounts of personal data, which can be difficult to manage and protect. There are concerns that AI may be used to collect and analyze sensitive information without the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in Europe and the third-largest city in the world by population. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its fashion industry, with many famous designers and fashion houses based there. Paris is a popular tourist destination and a cultural hub for France and the world. It is a major economic center and a major player in the French economy. The city is also known for its food culture, with many famous restaurants and cafes
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This could lead to more sophisticated and personalized AI that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, potentially leading to even more significant improvements
    


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
    Generated text:  [insert name], and I'm a [insert occupation] at [insert company name]. I'm excited to meet you and learn more about you! 🌟
    I enjoy [insert hobby or interest] and am always looking for new experiences to try. I'm [insert age] years old and am always seeking to learn more about the world around me. Thank you for taking the time to meet me! 😊
    Remember, I'm here to be a resource and share my knowledge with you! 😊
    If there's anything I can do for you, just let me know! 🙌
    Best, [insert
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    What is the answer? Paris is the capital of:
    
    A) Brazil
    B) Mexico
    C) Japan
    D) France
    
    D) France. Paris is the capital of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends that will shape the industry and its applications:
    
    1. Increased customization: With the rise of the cloud, AI can be used to create highly customizable models that can be tailored to individual users or applications. This will allow for more efficient use of resources and more cost-effective solutions.
    
    2. Integration of AI into daily life: AI is already being used in a variety of daily life applications such as smart homes, self-driving cars, and personalized healthcare. We can expect that this trend will continue, with more integration with other technologies such as voice recognition and biometrics.
    
    3. Data-driven decisions: AI will


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

    ...

     (

    Name

    ),

     and

     I

    'm

     an

    ...

     (

    describe

     your

     profession

     or

     experience

    ).

     What

     brings

     you

     to

     this

     world

    ?

     What

    's

     your

     greatest

     achievement

     so

     far

    ?

     And

     what

    's

     your

     favorite

     thing

     about

     working

     here

    ?

     My

     name

     is

    ...

     John

     Smith

    .

     I

    'm

     an

     accountant

     at

     a

     large

     corporation

    ,

     and

     I

     have

     been

     with

     the

     company

     for

     five

     years

    .

     My

     greatest

     achievement

     so

     far

     is

     being

     able

     to

     keep

     up

     with

     the

     latest

     accounting

     software

    .

     My

     favorite

     thing

     about

     working

     here

     is

     the

     opportunity

     to

     work

     with

     talented

     people

     who

     are

     passionate

     about

     their

     work

    .

     As

     for

     my

     name

    ,

     I

     would

     say

     that

    's

     just

     a

     nickname

     for

     me

    ,

     because

     I

    'm

     not

     really

     known

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     commonly

     known

     as

     "

    La

     Ville

     Bl

    anche

    ."
    


    That

    's

     correct

    !

     Paris

     is

     indeed

     known

     as

     "

    La

     Ville

     Bl

    anche

    "

     or

     the

     White

     City

    ,

     which

     refers

     to

     its

     white

     buildings

    .

     It

    's

     a

     bustling

     city

     with

     a

     rich

     history

     and

     culture

    ,

     including

     attractions

     like

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     jazz

     music

     scene

    ,

     particularly

     the

     Mou

    lin

     Rouge

     cab

    aret

     district

    .

     The

     French

     capital

     is

     a

     vibrant

     and

     exciting

     place

     to

     live

     and

     visit

    !

     

    🎶

    ✨

    
    


    Is

     there

     anything

     else

     you

    'd

     like

     to

     know

     about

     Paris

     or

     its

     architecture

    ?

     

    🕵

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    ,

     and

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     industry

     in

     the

     coming

     decades

    .

     Here

     are

     some

     potential

     areas

     of

     focus

     for

     AI

    :
    


    1

    .

     Improved

     Explain

    ability

    :

     With

     the

     emergence

     of

     machine

     learning

     models

    ,

     there

     is

     an

     increasing

     demand

     for

     more

     transparent

     and

     interpre

    table

     AI

     systems

    .

     As

     AI

     systems

     become

     more

     complex

    ,

     it

     will

     become

     increasingly

     important

     to

     provide

     insights

     into

     how

     they

     work

     and

     make

     decisions

     based

     on

     their

     outputs

    .

     To

     achieve

     this

    ,

     we

     will

     see

     continued

     advancements

     in

     explain

    able

     AI

     techniques

    ,

     such

     as

     advers

    arial

     examples

     and

     advers

    arial

     networks

    .
    


    2

    .

     More

     Personal

    ized

     AI

    :

     With

     the

     increasing

     amount

     of

     personal

     data

     being

     collected

     and

    



```python
llm.shutdown()
```
