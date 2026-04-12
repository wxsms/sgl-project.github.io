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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


    2026-04-12 05:09:47,869 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 05:09:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:21,  2.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:21,  2.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:21,  2.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:21,  2.48s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:26,  2.01it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.14it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.78it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:02<00:01, 21.26it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 21.26it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 29.07it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 30.95it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 37.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.98it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.68 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.32it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.85it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.71it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.40it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 49.40it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.98it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.67it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.40it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.40it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 52.40it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 45.24it/s]


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
    Generated text:  Alisa, and I'm a visual artist. I am happy to engage in discussions about my work and the work of other artists.
    What do you do in your free time?
    I love to travel and take pictures, so I also love to travel. I like to travel to different places and see how different cultures are portrayed in the images I create.
    I also love to cook and create a variety of dishes, including vegetarian and vegan options. I am a vegetarian myself, but I like to cook and experiment with different cuisines. I think cooking is a great way to learn about and connect with different cultures.
    I have a passion for
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking officer of the U. S. military. The current president is Joe Biden, who took office on January 20, 2021. Prior to that, the previous president was Donald Trump, who took office on January 20, 2017.
    What was Donald Trump's background? Donald Trump was born on July 14, 1947 in Oklahoma City, Oklahoma, the son of Jesse and Fernette (the former Gregg and Rosemary) Trump. His mother was Jewish, his father was Roman Catholic, and his grandfather was a Polish Jew. His family immigrated
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a very large city with a large population. It has three main parts: the City of Paris, the Seine-Maritime, and the Marais. The City of Paris is the most populous part of Paris. It is an important center of culture and art in the world. The Seine-Maritime is a river in the city of Paris. It has a lot of boats that you can see from the street. The Marais is an area of Paris where you can see lots of small shops. Some of the shops there sell very cheap things. The Marais is a quiet place. The Seine is a
    ===============================
    Prompt: The future of AI is
    Generated text:  digital, says MIT Sloan Initiative on Artificial Intelligence
    
    Nathan Pressler
    
    This story has been updated.
    
    When she was a child, Jenna Davenport dreamed of becoming a game designer. As an adult, she dreams of being a full-time AI engineer. That’s why she’s thrilled to be part of the MIT Sloan Initiative on Artificial Intelligence (SIAI), which is a research program at MIT that aims to understand and integrate the biggest technological changes coming from the field of artificial intelligence.
    
    “AI is an incredibly exciting field that is growing at a rapid rate, and it’s going to have profound impacts on the way we live, work,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique characteristic or skill that sets me apart from others]. And what's your background? I'm a [insert a unique characteristic or skill that sets me apart from others]. And what's your favorite hobby or activity? I'm a [insert a unique characteristic or skill that sets me apart from others]. And what's your favorite book or movie? I'm a [insert a unique characteristic or skill that sets me apart from others].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. The city is known for its fashion industry, art scene, and cuisine, and is a major center of politics, science, and culture in Europe. Paris is also known for its annual fashion week, which attracts celebrities and fashion industry professionals. The city is also home to the French Parliament, the French National Library
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI
    


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
    Generated text:  Sarah and I'm a software developer. I have a passion for building innovative software solutions that solve complex problems. I enjoy learning new programming languages and technologies and always strive to create code that is efficient, maintainable, and scalable. I'm eager to help my clients achieve their goals through software development, and I'm always looking for new opportunities to grow and learn. Thank you for asking. Congratulations on your new role at this company! What can you tell me about your experience so far at this company? I'm excited to learn more about your experience. Congratulations on your new role at this company! What can you tell me about your experience
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a vibrant and culturally rich city with a rich history and a strong sense of identity. Paris is known for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, and Louvre Museum, as well as its historical neighborhoods such as the Parisian Quarter and Montmartre. It is a major transportation hub, with many international airlines and bus routes connecting the city to other parts of France and the world. Paris is also a popular tourist destination, with many tourists visiting each year to visit the city and its many attractions. Despite its historic reputation, Paris is also a lively and dynamic city, with a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by two primary trends: the development of more sophisticated algorithms and the increasing integration of AI into various industries. Here are some possible future trends:
    
    1. Increased AI-powered automation: As AI becomes more integrated into various industries, we are likely to see an increase in the automation of tasks that are previously done by humans, such as data entry, data cleaning, and data analysis. This will require the development of more sophisticated algorithms to handle complex data sets and provide accurate insights.
    
    2. Increased AI for human knowledge: As AI systems become more sophisticated, they will become better at understanding human language and making accurate predictions about future events


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

    occupation

     or

     profession

    ].

     I

     am

     [

    age

    ]

     years

     old

    .

     I

     enjoy

     [

    occupation

    ]

     and

     have

     a

     passion

     for

     [

    occupation

    ].

     I

     am

     a

     [

    person

    ality

    ]

     person

     and

     have

     a

     strong

     [

    ability

     or

     skill

    ].

     I

     am

     [

    something

     I

     am

     proud

     of

    ],

     and

     I

     strive

     to

     [

    something

     I

     am

     committed

     to

     doing

    ].

     I

     am

     [

    something

     I

     am

     excited

     about

    ]

     and

     I

     am

     always

     looking

     for

     [

    something

     new

     or

     interesting

    ].

     I

     am

     passionate

     about

     [

    occupation

    ],

     and

     I

     believe

     that

     [

    something

    ]

     will

     help

     me

     achieve

     my

     goals

    .

     How

     can

     I

     be

     a

     good

     friend

     to

     you

    ?

     Let

    's

     chat

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     iconic

     and

     historic

     city

     of

     France

    ,

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

     vibrant

     culture

    .

     It

     is

     the

     capital

     of

     France

    ,

     and

     one

     of

     its

     most

     important

     cities

    ,

     with

     a

     population

     of

     over

     

    1

    0

     million

     people

    .

     Paris

     is

     a

     cosm

    opolitan

     met

    ropolis

     with

     a

     mix

     of

     traditional

     and

     modern

     architecture

    ,

     offering

     visitors

     and

     locals

     alike

     a

     unique

     blend

     of

     old

    -world

     charm

     and

     modern

     conven

    iences

    .

     The

     city

     is

     known

     for

     its

     famous

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

     Notre

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     various

     museums

    ,

     theaters

    ,

     and

     parks

    .

     Paris

     is

     also

     known

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     certainly

     exciting

     and

     multif

    ac

    eted

    .

     Here

     are

     some

     possible

     trends

     that

     could

     be

     expected

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     could

     be

     used

     to

     improve

     the

     diagnosis

    ,

     treatment

    ,

     and

     prevention

     of

     diseases

    ,

     as

     well

     as

     to

     assist

     doctors

     and

     nurses

     in

     making

     more

     accurate

     diagnoses

     and

     treatment

     plans

    .

     This

     could

     result

     in

     better

     patient

     outcomes

     and

     reduced

     costs

     in

     healthcare

    .
    


    2

    .

     AI

     in

     transportation

    :

     AI

     could

     be

     used

     to

     improve

     the

     safety

     and

     efficiency

     of

     transportation

     systems

    ,

     such

     as

     self

    -driving

     cars

     and

     drones

    .

     This

     could

     result

     in

     reduced

     accidents

    ,

     improved

     traffic

     flow

    ,

     and

     increased

     productivity

     for

     transportation

     companies

    .
    


    3

    .

     AI

     in

     entertainment

    :

     AI

     could

     be

     used

    



```python
llm.shutdown()
```
