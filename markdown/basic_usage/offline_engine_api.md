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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.49it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.48it/s]


    2026-05-14 05:10:14,183 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 05:10:14] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:29,  4.72s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:04<00:49,  1.08it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:13,  3.43it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:13,  3.43it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:05<00:13,  3.43it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:05<00:13,  3.43it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:05,  7.54it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]

    Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:05<00:05,  7.54it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:05<00:02, 12.70it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]

    Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=128):  60%|██████    | 35/58 [00:05<00:01, 19.72it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=16):  78%|███████▊  | 45/58 [00:05<00:00, 28.79it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 38.34it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 38.34it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 38.34it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 38.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.38it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.69it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:00<00:00, 38.61it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:02, 10.68it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:01<00:02, 10.68it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:02, 10.68it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:02, 10.68it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:02, 10.68it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]Capturing num tokens (num_tokens=320 avail_mem=74.31 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]Capturing num tokens (num_tokens=288 avail_mem=74.31 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  55%|█████▌    | 32/58 [00:01<00:01, 13.53it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.52it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:01, 17.52it/s]Capturing num tokens (num_tokens=224 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.52it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.52it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.52it/s]Capturing num tokens (num_tokens=176 avail_mem=74.29 GB):  64%|██████▍   | 37/58 [00:02<00:01, 17.52it/s]Capturing num tokens (num_tokens=176 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=160 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s]Capturing num tokens (num_tokens=112 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:02<00:00, 21.60it/s] Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  81%|████████  | 47/58 [00:02<00:00, 25.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s]Capturing num tokens (num_tokens=20 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s]Capturing num tokens (num_tokens=16 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:02<00:00, 29.27it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.35it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.35it/s]Capturing num tokens (num_tokens=4 avail_mem=74.24 GB): 100%|██████████| 58/58 [00:02<00:00, 23.27it/s]


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
    Generated text:  Liam from the Beauty and Grooming section. I'm a seasoned makeup artist, artist and hairdresser who has been doing my job for over 25 years. I use makeup products and are a skilled and experienced artist. My skills are unmatched by a lot of other makeup artists, especially when it comes to making you look great.
    I have a passion for makeup and hair and I constantly push my own personal creativity and make my style unique. I also love using makeup for hair styling and I take pride in putting on a good look, even when I'm not working on a client.
    As a makeup artist, I am well
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. When there is a conflict between the president and the president of the Senate, they usually______.
    A. make a decision together
    B. refer to the courts
    C. elect a new president
    D. agree by vote
    
    To determine the correct answer, let's analyze the situation step by step:
    
    1. **Understanding the Conflict**: The conflict typically arises when there is a disagreement between the president of the United States and the president of the Senate over important policy issues.
    
    2. **Possible Resolution**: In such a situation, the two leaders typically try to resolve the conflict through diplomatic means, such as the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A) Paris
    B) Marseille
    C) Lyon
    D) Bordeaux
    
    To determine the capital of France, let's consider the following information:
    1. The capital of France is Paris.
    2. The second capital is Marseille.
    3. The third capital is Lyon.
    4. The fourth capital is Bordeaux.
    
    Given this information, we can see that the capital of France is located in the north of France. The north of France is a region in the central part of France, centered around Paris. Therefore, the capital of France is located in the north.
    
    The correct answer is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, but the fact remains that it is here. Some things have already been done, and others are still under development. The technology is rapidly advancing, and we are set for a major shift in the way we interact with computers. The possibilities for what the future of AI will bring are very exciting, but also very daunting.
    One thing that is certain is that AI is going to play a huge role in how we interact with technology. As it has already been demonstrated in areas such as facial recognition, autonomous vehicles, and natural language processing, AI is becoming a fundamental part of how we live and work.
    The future of AI is looking


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I have always been passionate about [job title] and have always wanted to [job title] at [company name]. I am always looking for new challenges and opportunities to grow and learn. I am a [job title] and I am excited to be here at [company name]. I am looking forward to [job title] and I am looking forward to [job title] at [company name]. I am excited to be here at [company name] and I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a popular tourist destination, attracting millions of visitors each year. The city is known for its fashion industry, art scene, and food culture. It is a major economic center and a major transportation hub in Europe. Paris is a cultural and historical center that has played a significant role in shaping
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing to customer service, and will likely lead to the creation of new jobs.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be a greater emphasis on protecting user data and ensuring that AI systems are not used for malicious purposes.
    
    3. Enhanced human-computer interaction: AI will continue to improve the way that humans interact with AI systems, with more advanced natural language processing and machine learning capabilities.
    
    4. Increased focus on ethical AI: As AI systems become more complex and sophisticated, there
    


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
    Generated text:  Jane, and I'm an AI language model. I don't have personal experiences or emotions, but I'm here to assist with any questions or tasks you may have. How can I help you today? I'm here to help with any language-related tasks or assist with understanding complex concepts, such as English grammar or vocabulary. Let me know how I can help you today! 🌟💡✨
    
    Hey, Jane! 👋 I'm a language AI model created by Anthropic. I'm here to help you learn and improve your language skills, whether you're just getting started or looking to deepen your knowledge. How can I assist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. True
    B. False
    
    B. False
    
    The capital city of France is indeed Paris. While Paris is the most famous and historically significant city in France, it is not the capital city of France. The capital of France is called Paris, but it is not actually the seat of government or the most populous city in France. 
    
    To clarify the position of Paris as the capital of France:
    
    1. Paris is actually the most populous city in France, with over 2 million inhabitants.
    2. It is the capital because it is the capital of the country, not the seat of government.
    3. It is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and involves rapid technological advancements, evolving business practices, and challenges that will shape the industry's direction. Here are some potential trends that could potentially shape the future of AI:
    
    1. Increased Human AI Integration: As AI continues to evolve, it is likely that human involvement in AI will increase. This could be in the form of shared decision-making, or in cases where AI is capable of making decisions that a human cannot, such as in healthcare or legal fields. Additionally, there may be an increase in the integration of AI with human emotions, experiences, and natural language processing, leading to more nuanced and emotional AI that can better understand


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

     a

     [

    occupation

    ]

     by

     profession

    .

     I

     started

     [

    mention

     an

     event

     or

     decision

     that

     shaped

     my

     life

    ]

     and

     have

     always

     [

    mention

     an

     accomplishment

     or

     skill

    ]

     that

     has

     made

     me

     a

     [

    mention

     a

     specific

     trait

     or

     quality

    ].

     As

     a

     [

    mention

     a

     notable

     feature

     of

     your

     profession

     or

     life

    ],

     I

    'm

     always

     [

    mention

     a

     characteristic

     or

     trait

     that

     different

    iates

     you

     from

     others

    ].

     So

    ,

     I

    'm

     [

    mention

     how

     you

     could

     describe

     yourself

    ]

     and

     I

     hope

     I

     can

     make

     a

     positive

     impact

     in

     the

     world

    !

     

    🌟

    ✨

    
    


    Hey

     there

    ,

     I

    'm

     [

    Name

    ]

     from

     [

    Location

    ],

     and

     I

    'm

     here

     to

     introduce

     myself

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     the

     country

     and

     is

     home

     to

     many

     of

     the

     country

    's

     most

     famous

     landmarks

     and

     attractions

    .
    


    Some

     notable

     facts

     about

     Paris

     include

    :
    


    -

     It

     is

     the

     capital

     of

     France

     and

     the

     

    4

    th

     most

     populous

     city

     in

     the

     world

    .


    -

     It

     has

     a

     population

     of

     around

     

    1

    0

     million

     people

    .


    -

     Paris

     is

     known

     for

     its

     world

    -ren

    owned

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .


    -

     The

     city

     is

     also

     famous

     for

     its

     fashion

     and

     food

     industries

    ,

     with

     a

     thriving

     fashion

     industry

     and

     numerous

     restaurants

     and

     cafes

    .


    -

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     transformative

    ,

     with

     many

     potential

     developments

     and

     innovations

     shaping

     the

     way

     we

     live

     and

     interact

     with

     technology

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

     Increasing

     integration

     with

     other

     technologies

    :

     AI

     will

     continue

     to

     integrate

     with

     other

     technologies

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     edge

     computing

    ,

     creating

     new

     and

     more

     versatile

     applications

    .
    


    2

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     will

     become

     more

     capable

     of

     understanding

     and

     adapting

     to

     individual

     users

    '

     preferences

    ,

     needs

    ,

     and

     behaviors

    ,

     leading

     to

     more

     personalized

     experiences

    .
    


    3

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     (

    AV

    s

    )

     are

     likely

     to

     become

     more

     prevalent

    ,

     with

     AI

     helping

     to

     improve

     their

     safety

    ,

     efficiency

    ,

     and

     reliability

    .
    


    



```python
llm.shutdown()
```
