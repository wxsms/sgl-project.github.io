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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.64it/s]


    2026-04-11 04:15:08,972 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 04:15:08] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.76it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.76it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.41it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.44it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.44it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 17.34it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 23.81it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 28.84it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.15it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 37.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 46.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=120.33 GB):   2%|▏         | 1/58 [00:00<00:15,  3.79it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.30 GB):   2%|▏         | 1/58 [00:00<00:15,  3.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   2%|▏         | 1/58 [00:00<00:15,  3.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:06,  8.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:06,  8.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   5%|▌         | 3/58 [00:00<00:06,  8.68it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   5%|▌         | 3/58 [00:00<00:06,  8.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  10%|█         | 6/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  17%|█▋        | 10/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  26%|██▌       | 15/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.78it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.89it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.89it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  43%|████▎     | 25/58 [00:01<00:00, 35.89it/s]

    Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.73it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]

    Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  60%|██████    | 35/58 [00:01<00:00, 39.91it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s]

    Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.53it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.55it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.56it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.56it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 33.27it/s]


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
    Generated text:  Carlos Alvarez, and I am a software developer and independent consultant, specializing in multi-cloud strategies, DevOps, and software engineering. 
    
    Can you tell me about your work experience and skills in the realm of cloud computing and software engineering? I am eager to learn and expand my knowledge in these areas. To begin, could you please share some of your experience with implementing cloud solutions in a high-performance computing environment? Additionally, I would love to learn more about your approach to ensuring the scalability and reliability of these solutions. Lastly, could you provide some insights into the latest trends in cloud computing and the impact on software engineering practices? Thank you for
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is the leader of the country. He is supposed to make important decisions. The president is very important in America. The president of the United States is the leader of the country. He is supposed to make important decisions. The president is very important in America. He is supposed to make important decisions about the country. The president's job is very hard. He has to make lots of important decisions. He is supposed to keep the country safe from danger. He is supposed to keep the country free of corruption. He is supposed to make the country happy. The president is supposed to give orders to the different parts
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is a place of history, art, food, fashion and politics, but its capital is also a city with a lot of tourists. Paris is one of the most popular cities in the world. Many people try to travel there on vacation. They spend their vacation going sightseeing, shopping and dining out.
    
    Paris is famous for its famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral and the Louvre Museum. Paris is also famous for its food, fashion and culture. The food in Paris is different from that in the other cities in France. The food here is more spicy. The French people have a
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. By 2025, the world's economy will be worth 6.3 trillion dollars, and AI will be the source of 77% of new jobs. The world is becoming increasingly dependent on AI to help people and businesses succeed. In 2016, the World Economic Forum predicted that by 2025, the world will have a billion people who use AI every day. The future of AI is bright. Technology is a part of life and AI is a part of technology. The future of AI is bright. While AI may be seen as a dark horse in the tech game,


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm a [job title] at [company name], and I've been with the company for [number of years] years. I'm a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is also the oldest capital city in Europe, having been founded in 789 AD. Paris is known for its rich history, art, and culture, and is a major tourist destination. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city that is constantly evolving and changing, with new developments and attractions being added to the city every year. It is a city that is a symbol of France and a major hub for international business and trade. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, cost savings, and job displacement, but it will also create new opportunities for innovation and creativity.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. This will require ongoing dialogue
    


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
    Generated text:  [Name] and I am a [age] year-old [occupation] who is passionate about [professional or academic interest] and loves [career goal or hobby]. I enjoy spending my free time [extracurricular activities, hobbies, etc.] and I am always eager to learn and expand my knowledge and skills. I am confident in my abilities and I am always looking for new challenges and opportunities to grow and develop. I am a [self-description] and I am excited to share my experiences and knowledge with others. What's your name, age, profession, and what's your interest or hobby? [Name] [Age]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the 11th largest city in the world. It is located on the Seine river, in the south of France. The city has a population of around 2 million people and is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and it is a popular tourist destination. The city is also home to many important institutions of higher education, including the University of Paris. Overall, Paris is a vibrant and exciting city that is a must-
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and complex, and there are many different trends and developments that are shaping the way that we will live and work in the years to come. However, some trends that are likely to be prevalent in AI in the coming years include:
    
    1. Increased integration of AI into our daily lives: AI will likely become more integrated into our daily lives, from our smartphones to our homes. This will enable us to automate tasks, learn from data, and make better decisions in a more efficient way.
    
    2. Development of new AI technologies: AI research will continue to advance at a rapid pace, with new developments being made in areas such as machine


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

    First

     Name

    ]

     and

     I

     am

     [

    Last

     Name

    ],

     a

     [

    Describe

     your

     field

    /

    occupation

    ].

     I

    've

     always

     been

     fascinated

     by

     [

    mention

     a

     particular

     hobby

     or

     interest

     in

     your

     field

    ].

     I

    'm

     excited

     to

     be

     here

     and

     learn

     more

     about

     you

    .

     What

     would

     you

     like

     to

     know

     about

     yourself

    ?

     [

    Ask

     about

     your

     background

    ,

     experiences

    ,

     or

     interests

     that

     you

     are

     proud

     of

    ]

     [

    Add

     any

     extra

     personal

     information

    ,

     such

     as

     a

     joke

     or

     a

     quote

     that

     might

     be

     interesting

     to

     hear

    ].

     [

    End

     with

     a

     friendly

     and

     welcoming

     tone

    ,

     inviting

     the

     reader

     to

     ask

     any

     questions

     or

     share

     any

     thoughts

     about

     you

    ].

     As

     an

     AI

     language

     model

    ,

     I

    'm

     here

     to

     help

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     major

     city

     with

     a

     rich

     history

     and

     cultural

     significance

    ,

     serving

     as

     the

     official

     seat

     of

     government

     for

     the

     country

    .

     The

     city

     is

     famous

     for

     its

     historical

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Lou

    vre

     Museum

    ,

     and

     for

     its

     well

    -p

    reserved

     art

    ,

     culture

    ,

     and

     architecture

    .

     As

     one

     of

     the

     most

     important

     cities

     in

     the

     world

    ,

     Paris

     plays

     a

     vital

     role

     in

     French

     society

     and

     politics

    .

     Its

     international

     status

     and

     tourism

     industry

     further

     contribute

     to

     the

     city

    's

     economy

     and

     cultural

     vibr

    ancy

    .

     Paris

     is

     a

     diverse

     and

     vibrant

     city

     with

     a

     vibrant

     arts

     scene

    ,

     a thriving

     nightlife

    ,

     and

     a

     well

    -established

     culinary

     culture

    .

     Its

     political

     and

     social

     importance

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     highly

     dynamic

     and

     rapidly

     evolving

     field

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     Integration

     of

     AI

     into

     Everyday

     Life

    :

     As

     AI

     becomes

     more

     prevalent

     in

     everyday

     life

    ,

     we

     are

     likely

     to

     see

     more

     integration

     of

     AI

     into

     various

     aspects

     of

     our

     lives

    .

     For

     instance

    ,

     AI

     could

     be

     integrated

     into

     healthcare

     to

     improve

     diagnostic

     accuracy

    ,

     medical

     imaging

    ,

     and

     drug

     development

    .

     AI

     could

     also

     be

     integrated

     into

     transportation

    ,

     such

     as

     self

    -driving

     cars

    ,

     to

     enhance

     safety

     and

     reduce

     traffic

     congestion

    .
    


    2

    .

     AI

     will

     continue

     to

     become

     more

     sophisticated

     and

     personalized

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     to

     become

    



```python
llm.shutdown()
```
