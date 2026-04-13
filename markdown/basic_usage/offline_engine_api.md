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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.86it/s]


    2026-04-13 04:53:14,753 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 04:53:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.58it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.58it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.58it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.72it/s] 

    Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.72it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:02, 16.76it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 22.32it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 28.85it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]

    Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 33.41it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 36.09it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 42.28it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 42.28it/s] 

    Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 42.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.24it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   7%|▋         | 4/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   7%|▋         | 4/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.79it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.31 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.30 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.83it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=119.83 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.83it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.83 GB):  31%|███       | 18/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.98 GB):  31%|███       | 18/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.95 GB):  31%|███       | 18/58 [00:00<00:01, 30.88it/s]Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  31%|███       | 18/58 [00:00<00:01, 30.88it/s] Capturing num tokens (num_tokens=960 avail_mem=118.97 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=896 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=832 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=768 avail_mem=118.96 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.08it/s]

    Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=704 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.21it/s]Capturing num tokens (num_tokens=640 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.21it/s]Capturing num tokens (num_tokens=576 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.21it/s]Capturing num tokens (num_tokens=512 avail_mem=118.94 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.21it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.21it/s]Capturing num tokens (num_tokens=480 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.44it/s]Capturing num tokens (num_tokens=448 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.44it/s]Capturing num tokens (num_tokens=416 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.44it/s]

    Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.44it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.63it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]

    Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 37.07it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.39it/s] Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 37.21it/s]Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  81%|████████  | 47/58 [00:01<00:00, 37.21it/s]Capturing num tokens (num_tokens=64 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 37.21it/s]

    Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  81%|████████  | 47/58 [00:01<00:00, 37.21it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  81%|████████  | 47/58 [00:01<00:00, 37.21it/s]Capturing num tokens (num_tokens=32 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=20 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  88%|████████▊ | 51/58 [00:01<00:00, 37.85it/s]Capturing num tokens (num_tokens=16 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.33it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.33it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 33.76it/s]


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
    Generated text:  Rumi, and I am a poet who likes to write words and poems. I am born and raised in Turkey, and I am a Tarot reader. I am also an avid traveler, and my travels have taken me to many different countries and cultures. I have a special place in my heart for music and art, and I am passionate about using poetry to help others feel better and live better lives. I love sharing my poetry with the world, and I hope to inspire others to do the same. So if you're interested in learning more about my work or simply want to share my words with you, I'd love to hear
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office with the highest annual salary. In order to enhance his performance, he decided to increase his salary by 30%. Before the increase, his salary was $100,000. What was his new salary after the increase? Let's calculate the new salary step by step.
    
    1. First, determine the increase in salary:
       The president's original salary was $100,000. He wants to increase it by 30%.
    
       The increase is calculated as follows:
       \[
       \text{Increase} = 30\% \text{ of } \$1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is located in the south of the country. It is located on the left bank of the Seine river. The capital has an area of 37.5 km². The population of Paris is 2.3 million people. It is the largest city in France by population. The capital has a population of over 4 million. It is the second largest city in France by population.
    
    It is a beautiful city, with a beautiful skyline. There are many museums and galleries in the city. There are many other interesting places to see and to learn about.
    
    The city is very crowded. There are lots of traffic
    ===============================
    Prompt: The future of AI is
    Generated text:  hard to predict. It’s an area that is constantly evolving, and the horizon is endless. However, there are already many applications of AI that are already out there. One example is the use of AI in image recognition. It has already been used in a variety of fields, including security and customer service. But what happens when we start using AI in areas that are new and unfamiliar to us? For example, what happens when we start using AI to predict the weather? This is where the idea of AI in weather prediction comes in. AI is already used in some weather prediction models, but this is not yet the end of the story


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. What's your career path so far? I started my career as a [job title] at [company name], and I've been working in this field for [number of years]. What's your current role at [company name]? I'm currently
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion, art, and cuisine, and is a popular destination for international visitors. It is also home to many important historical sites, including the Palace of Versailles and the Arc de Triomphe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there is likely to be a greater focus on ethical AI. This could include developing AI that is designed to be transparent, accountable, and fair, and that is used to address social and ethical issues.
    
    2. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As
    


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
    Generated text:  [insert character's name], and I'm a [insert fictional character's profession] in the [insert fictional genre]. I'm [insert character's age] years old and [insert character's nationality]. I have [insert the most interesting or unique fact about your character]. I grew up [insert childhood experiences or significant life events], and I've always been passionate about [insert hobbies or activities]. I'm always seeking to learn new things and make new friends, and I believe that [insert a personal statement or personal trait] helps me become the person I am today. My goal is to inspire others and make the world a better place
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is also famous for its fine dining, vibrant culture, and bustling streets. Paris is a popular tourist destination and is considered one of the most beautiful cities in the world. Its rich history, stunning architecture, and vibrant culture have made it a must-visit destination for people from all over the world. The capital city of France is Paris! 🌍✨
    
    Interesting! Can you recommend any must-visit attractions in Paris?
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a range of trends and developments, including:
    
      1. Advancements in machine learning and deep learning: As the capabilities of AI continue to improve, more powerful models will be able to learn from data, recognize patterns, and make predictions.
      2. Increased focus on ethical considerations: With the growing concern around AI's impact on society, there will be increased emphasis on developing ethical guidelines and principles for how AI should be used.
      3. Increasing use of AI in healthcare and medicine: As AI is increasingly used in healthcare and medicine, it is likely to play a more significant role in diagnosing and treating


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

    Your

     Name

    ],

     and

     I

    'm

     a

     [

    type

     of

     work

     or

     hobby

    ]

     enthusiast

    .

     I

     love

     trying

     new

     things

    ,

     learning

     new

     skills

    ,

     and

     staying

     curious

    .

     I

    'm

     currently

     working

     on

     [

    short

     story

     or

     project

    ]

     and

     I

    'm

     excited

     to

     share

     what

     I

    've

     learned

     with

     you

    .

     
    
    [

    Your

     Name

    ]

     can

     be

     reached

     at

     [

    Your

     Contact

     Information

    ],

     and

     you

     can

     find

     me

     on

     [

    LinkedIn

     Profile

    ]

     or

     [

    Twitter

     Handle

    ].

     I

     look

     forward

     to

     meeting

     you

    !

     
    


    ---
    


    ###

     Self

    -

    Introduction

    
    


    Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ]

     and

     I

    'm

     a

     [

    type

     of

     work

     or

     hobby

    ]

     enthusiast

    .

     I

     love

     trying

     new

     things

    ,

     learning

     new

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     ancient

     capital

     of

     France

    ,

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

     beautiful

     architecture

    ,

     and

     rich

     history

    ,

     making

     it

     a

     world

    -ren

    owned

     city

     with

     a

     diverse

     and

     vibrant

     culture

    .

     The

     city

     is

     also

     renowned

     for

     its

     annual

     Lou

    vre

     Festival

    ,

     which

     features

     a

     variety

     of

     cultural

     and

     artistic

     events

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     cultural

     and

     political

     center

     for

     France

    .

     It

     is

     the

     seat

     of

     the

     French

     government

    ,

     the

     headquarters

     of

     the

     French

     Foreign

     and

     Colonial

     Office

    ,

     and

     the

     main

     financial

     center

     of

     Europe

    .

     The

     city

     is

     also

     famous

     for

     its

     art

    ,

     music

    ,

     literature

    ,

     and

     cuisine

    .

     Paris

     is

     a

     major

     hub

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     blend

     of

     existing

     and

     emerging

     technologies

    ,

     with

     a

     focus

     on

     areas

     such

     as

    :
    


    1

    .

     Autonomous

     vehicles

    :

     The

     development

     of

     self

    -driving

     cars

     and

     trucks

    ,

     with

     their

     ability

     to

     navigate

     roads

     and

     avoid

     collisions

    ,

     could

     revolution

    ize

     transportation

     and

     reduce

     traffic

     accidents

    .
    


    2

    .

     Smart

     homes

    :

     The

     integration

     of

     AI

     into

     everyday

     devices

     such

     as

     smart

     ther

    most

    ats

    ,

     security

     systems

    ,

     and

     lighting

     systems

     could

     make

     homes

     more

     energy

    -efficient

     and

     comfortable

    .
    


    3

    .

     Medical

     diagnostics

    :

     AI

    -powered

     healthcare

     systems

     could

     revolution

    ize

     the

     way

     diseases

     are

     diagnosed

     and

     treated

    ,

     leading

     to

     more

     accurate

     and

     effective

     diagnoses

     and

     treatments

    .
    


    4

    .

     Virtual

     and

     augmented

     reality

    :

     The

     development

     of

     AI

    -powered

    



```python
llm.shutdown()
```
