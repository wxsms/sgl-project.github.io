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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.36it/s]


    2026-04-14 21:53:53,750 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 21:53:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.59it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.59it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.65it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 19.64it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.18it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 33.60it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 39.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.50 GB):   7%|▋         | 4/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.50 GB):   7%|▋         | 4/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.49 GB):   7%|▋         | 4/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.49 GB):  10%|█         | 6/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.46 GB):  10%|█         | 6/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.42 GB):  10%|█         | 6/58 [00:00<00:02, 17.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.42 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.44it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.42 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.42 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.41 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.41 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.41 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.40 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.40 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.40 GB):  19%|█▉        | 11/58 [00:00<00:02, 19.73it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=56.40 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.40 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.39 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.39 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.39 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.38 GB):  26%|██▌       | 15/58 [00:00<00:01, 23.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.38 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=960 avail_mem=56.38 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s] Capturing num tokens (num_tokens=896 avail_mem=56.37 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=832 avail_mem=56.37 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=768 avail_mem=56.37 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 30.74it/s]Capturing num tokens (num_tokens=704 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.44it/s]Capturing num tokens (num_tokens=640 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.44it/s]Capturing num tokens (num_tokens=576 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.44it/s]Capturing num tokens (num_tokens=512 avail_mem=56.35 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.44it/s]Capturing num tokens (num_tokens=480 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=448 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=416 avail_mem=56.36 GB):  45%|████▍     | 26/58 [00:01<00:00, 37.44it/s]Capturing num tokens (num_tokens=416 avail_mem=56.36 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=384 avail_mem=56.36 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=352 avail_mem=56.35 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=320 avail_mem=56.35 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=288 avail_mem=56.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]

    Capturing num tokens (num_tokens=256 avail_mem=56.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=240 avail_mem=56.34 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=240 avail_mem=56.34 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=224 avail_mem=56.34 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=208 avail_mem=56.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=192 avail_mem=56.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=176 avail_mem=56.33 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=160 avail_mem=56.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=144 avail_mem=56.32 GB):  66%|██████▌   | 38/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=144 avail_mem=56.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=128 avail_mem=56.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=112 avail_mem=56.32 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=96 avail_mem=56.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=56.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=64 avail_mem=56.31 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=48 avail_mem=56.30 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.08it/s]Capturing num tokens (num_tokens=48 avail_mem=56.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=32 avail_mem=56.30 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=28 avail_mem=56.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=24 avail_mem=56.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=20 avail_mem=56.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=16 avail_mem=56.29 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=12 avail_mem=56.28 GB):  86%|████████▌ | 50/58 [00:01<00:00, 48.23it/s]Capturing num tokens (num_tokens=12 avail_mem=56.28 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.33it/s]Capturing num tokens (num_tokens=8 avail_mem=56.28 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.33it/s] Capturing num tokens (num_tokens=4 avail_mem=56.28 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.33it/s]

    Capturing num tokens (num_tokens=4 avail_mem=56.28 GB): 100%|██████████| 58/58 [00:01<00:00, 36.97it/s]


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
    Generated text:  Maria and I am a graphic designer, illustrator, photographer and short film maker. I have a passion for art and I like to use art to express myself and tell stories. My goal is to provide the best possible service to my clients by using my various skills to create unique and creative works.
    I have a background in graphic design and I am an accomplished illustrator and short film maker. I have a passion for photography and I use my camera and my artistic eye to capture stories and to tell the story of my clients. I like to create simple, elegant works that are both striking and beautiful.
    I specialize in graphic design, illustration, photography
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to improve the country’s public education system by introducing new policies and strategies. One of the policies they are implementing is the "Two-Thirds" rule, which is designed to make the administration of education more efficient and effective. The rule requires that at least two-thirds of the public school students attend a certain level of education, and the rest attend a different level of education. 
    
    If there are 500,000 students in the first grade, 300,000 students in the second grade, and 200,000 students in the third grade, and the president wants
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. London
    D. Moscow
    Answer:
    
    A
    
    What is the Chinese meaning of “精神分析学派”?
    A. Psychoanalytic psychology
    B. Structuralism
    C. Functionalism
    D. Humanism
    Answer:
    
    A
    
    Which of the following is NOT an application of the CB/T 17065 standard in China? A. Mandatory Inspection B. Mandatory Certification C. Mandatory Product Review D. Mandatory Performance Verification
    Answer:
    
    C
    
    The main influencing factor for the formation of wind farms is ____.
    A. Topography
    B. Atmospheric conditions
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. But is it possible for the future to be a reality? This is the question we’ll ask. You will have the opportunity to pick one of the 3 questions that we will ask at the end. We will choose the one that you think is the most likely and that will provide the best answer. We will discuss this on a Friday night.
    As a startup with deep expertise in the technology industry, we are on the cutting edge of Artificial Intelligence (AI) and we are all very excited about the potential for AI. However, we know that there are many challenges that will need to be overcome before we are able to bring


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its vibrant arts scene and culinary delights. Paris is a cultural and intellectual center of the world and a major tourist destination, attracting millions of visitors each year. The city is also home to many important institutions such as the French Academy of Sciences, the French National Library, and the Louvre Museum. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and responsible development of AI systems. This could lead to more stringent regulations and guidelines for AI development
    


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
    Generated text:  [Name], and I come from [Place]. I'm an [Age] year old [Gender] with [Physical Characteristics], and I [Job or Profession]. I'm known for my [Strengths or Skills], and I have a lot of [Strengths or Skills] in [Area of Expertise]. I'm also [Occupation or Personality]. I'm [Positive/Expressive/Neutral/Brief] in personality. I'm always [Always/Always or Sometimes/Not] on the [Street/Lost In Life, or [In the News, or [In the Dark, or [Unknown]]]. I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the 8th largest city in the world by population, and the largest city in the European Union.
    You are to answer this question: when did paris become the capital of france? The French capital city of Paris was established in 843, when Charles Martel laid the foundation of the city. In 1870, the city's name was changed to Paris, the name of the leader of the National Assembly.
    Answer: Paris was established in 843, when Charles Martel laid the foundation of the city. In 1870, the city's name was changed to Paris,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of key trends, including:
    
    1. Increased efficiency and productivity: As AI technologies continue to improve, we can expect to see a dramatic increase in the efficiency and productivity of businesses and organizations. This will be achieved through the development of more advanced algorithms and the use of machine learning to automate repetitive tasks and improve decision-making.
    
    2. Increased personalization: AI will continue to personalize the experiences of users, whether that means recommending products or services based on individual preferences, or providing targeted advertising based on user behavior and past interactions.
    
    3. Enhanced decision-making and innovation: AI will enable organizations to make faster and more


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

    Character

    's

     Name

    ]

     and

     I

    'm

     a

     [

    character

    's

     occupation

     or

     role

    ].

     I

    've

     been

     working

     in

     [

    industry

    /

    field

    ]

     for

     [

    number

    ]

     years

     now

    .

     My

     passion

     for

     this

     work

     is

     [

    mention

     your

     interests

     or

     hobbies

    ],

     and

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    your

     job

     title

    ].

     I

     pride

     myself

     on

     being

     [

    mention

     any

     positive

     traits

     or

     qualities

     you

     possess

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    mention

     a

     goal

     or

     project

     you

    'd

     like

     to

     work

     on

    ].

     I

    'm

     excited

     to

     start

     a

     [

    number

    ]

     year

     adventure

    !

     Come

     on

     in

    ,

     let

    's

     have

     a

     chat

    !

     (

    pause

    )

     [

    Character

    's

     Name

    ]

     ...

     And

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     most

     populous

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

     and

     vibrant

     arts

     and

     culture

     scene

    .

     It

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     Col

    os

    se

    um

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    ,

     among

     many

     other

     landmarks

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     with

     millions

     of

     visitors

     each

     year

    ,

     and

     is

     a

     major

     economic

     and

     cultural

     center

     of

     France

    .

     Its

     history

     and

     culture

     have

     been

     shaped

     by

     various

     em

    pires

     and

     political

     figures

     throughout

     its

     long

     history

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     cuisine

    ,

     including

     French

     cuisine

    ,

     including

     regional

     specialties

     such

     as

     fo

    ie

     gras

    ,

     and

     its

     rich

     literary

     heritage

    ,

     with

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     and

     we

     are

     witnessing

     significant

     changes

     in

     the

     way

     AI

     is

     used

     and

     developed

    .

     Some

     of

     the

     potential

     trends

     that

     are

     shaping

     the

     future

     of

     AI

     include

    :
    


    1

    .

     Increased

     automation

     and

     AI

     integration

    :

     The

     use

     of

     AI

     in

     manufacturing

    ,

     transportation

    ,

     healthcare

    ,

     and

     other

     industries

     is

     likely

     to

     increase

    .

     AI

     will

     be

     integrated

     into

     existing

     systems

     to

     automate

     repetitive

     tasks

     and

     improve

     efficiency

    .
    


    2

    .

     Improved

     privacy

     and

     data

     protection

    :

     The

     rise

     of

     AI

     will

     require

     a

     greater

     emphasis

     on

     protecting

     sensitive

     data

    .

     This

     may

     lead

     to

     the

     development

     of

     new

     privacy

     and

     data

     protection

     standards

     and

     technologies

    .
    


    3

    .

     Increased

     focus

     on

     ethics

     and

     responsibility

    :

     As

     AI

     systems

     become

     more

     complex

     and

    



```python
llm.shutdown()
```
