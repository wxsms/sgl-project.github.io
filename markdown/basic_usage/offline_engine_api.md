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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.74it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:12,  3.74it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.38it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:06<00:04,  6.50it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:06<00:02,  9.43it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:06<00:01, 11.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:06<00:00, 17.50it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:07<00:00, 17.50it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:07<00:00, 17.50it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:07<00:00, 17.50it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:07<00:00, 17.50it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:07<00:00, 17.50it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 26.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=75.44 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=75.43 GB):   3%|▎         | 2/58 [00:00<00:03, 16.68it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=75.43 GB):   7%|▋         | 4/58 [00:00<00:03, 17.58it/s]Capturing num tokens (num_tokens=6144 avail_mem=75.43 GB):   7%|▋         | 4/58 [00:00<00:03, 17.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=75.42 GB):   7%|▋         | 4/58 [00:00<00:03, 17.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.41 GB):   7%|▋         | 4/58 [00:00<00:03, 17.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=75.41 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=4608 avail_mem=75.41 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=75.41 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=75.41 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=75.40 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=75.40 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=75.40 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=75.40 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=75.40 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.39 GB):  19%|█▉        | 11/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=75.39 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=75.39 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=75.38 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=75.38 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.70it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=75.38 GB):  26%|██▌       | 15/58 [00:00<00:01, 28.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=75.38 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=75.38 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=75.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=960 avail_mem=75.37 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s] Capturing num tokens (num_tokens=896 avail_mem=75.37 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=832 avail_mem=75.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.84it/s]Capturing num tokens (num_tokens=832 avail_mem=75.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]Capturing num tokens (num_tokens=768 avail_mem=75.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]Capturing num tokens (num_tokens=704 avail_mem=75.36 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]Capturing num tokens (num_tokens=640 avail_mem=75.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]

    Capturing num tokens (num_tokens=576 avail_mem=75.35 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]Capturing num tokens (num_tokens=512 avail_mem=75.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 34.39it/s]Capturing num tokens (num_tokens=512 avail_mem=75.34 GB):  50%|█████     | 29/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=480 avail_mem=75.35 GB):  50%|█████     | 29/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=448 avail_mem=75.35 GB):  50%|█████     | 29/58 [00:00<00:00, 36.70it/s]Capturing num tokens (num_tokens=416 avail_mem=75.35 GB):  50%|█████     | 29/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=384 avail_mem=75.35 GB):  50%|█████     | 29/58 [00:01<00:00, 36.70it/s]Capturing num tokens (num_tokens=384 avail_mem=75.35 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=352 avail_mem=75.34 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=320 avail_mem=75.34 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.92it/s]

    Capturing num tokens (num_tokens=288 avail_mem=75.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=256 avail_mem=75.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.92it/s]Capturing num tokens (num_tokens=256 avail_mem=75.33 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=240 avail_mem=75.33 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=224 avail_mem=75.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=208 avail_mem=75.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=192 avail_mem=75.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=176 avail_mem=75.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.26it/s]Capturing num tokens (num_tokens=176 avail_mem=75.32 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=160 avail_mem=75.32 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=144 avail_mem=75.31 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s]

    Capturing num tokens (num_tokens=128 avail_mem=75.31 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=112 avail_mem=75.31 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=96 avail_mem=75.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.22it/s] Capturing num tokens (num_tokens=96 avail_mem=75.30 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=80 avail_mem=75.30 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=64 avail_mem=75.30 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=48 avail_mem=75.29 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=32 avail_mem=75.29 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=28 avail_mem=75.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=28 avail_mem=75.28 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=24 avail_mem=75.28 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.28 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=16 avail_mem=75.28 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=12 avail_mem=75.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s]Capturing num tokens (num_tokens=8 avail_mem=75.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.44it/s] Capturing num tokens (num_tokens=8 avail_mem=75.27 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=4 avail_mem=75.27 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.17it/s]Capturing num tokens (num_tokens=4 avail_mem=75.27 GB): 100%|██████████| 58/58 [00:01<00:00, 34.06it/s]


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
    Generated text:  Mark.
    I am a 34 year old male. I have a feeling of weakness in my arms, legs and body.
    I have been experiencing a burning sensation in my stomach, sometimes it feels like I have to swallow something. I am a smoker and have been smoking for 30 years. I have had a lot of drinking and also been diagnosed with high blood pressure.
    Is there anything that can help me feel better?
    Thank you. Mark
    Mark, it sounds like you're experiencing a combination of physical symptoms and potential underlying health issues. While I can offer some general advice based on the symptoms you describe, it's important
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 33 years older than the president of Hawaii. Additionally, the president of Hawaii is currently twice as old as the president of Canada. If the president of Canada was 25 years old when he was born, what is the sum of the ages of the three presidents?
    Let's start by defining the variables for the ages of the presidents:
    
    - Let \( C \) be the age of the president of Canada.
    - Let \( D \) be the age of the president of the United States.
    - Let \( H \) be the age of the president of Hawaii.
    
    From the problem, we know:
    
    1. The
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Moscow
    C. Madrid
    D. Berlin
    Answer:
    
    A
    
    In the context of China's political system, how should grassroots political organizations be viewed?
    A. The existence of grassroots political organizations is an inherent requirement of our socialist democratic political system.
    B. Grassroots political organizations are only an important component of the party and government organs.
    C. Grassroots political organizations are not entirely under the leadership of the party and government organs.
    D. Grassroots political organizations are completely independent of both the party and government organs.
    Answer:
    
    A
    
    The life cycle of a mature yeast cell is
    A. dip
    ===============================
    Prompt: The future of AI is
    Generated text:  where the future of science and technology is, and it’s going to be a major one. AI is becoming more and more pervasive in our lives, and it will only continue to evolve. The technology is in the process of changing the way we work, learn, and interact with the world around us. As we continue to develop AI technologies, it is important to understand how they are evolving, and to anticipate the challenges and opportunities that they will present. The more we understand, the better we will be prepared for the future.
    In this article, we will explore the current state of AI and its future potential. We will also discuss the


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


    Generated text:  Paris, also known as the City of Light, a historic and cultural center with a rich history dating back to the Middle Ages. It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, and art, and is home to many world-renowned museums, theaters, and other cultural institutions. Paris is a vibrant and dynamic city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to improve, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced human-AI collaboration: As AI becomes more integrated into our lives, we can expect to see more human-AI collaboration in areas like education, healthcare, and customer service
    


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
    Generated text:  [Name], and I'm a [Career/Profession] with a [Skill/Ability]. In my [Career/Profession], I've [Number of Years in Industry/Position], and I [Number of Achievements in Industry/Position]. I am known for my [Strength/Ability/Interest], and I strive to [Key Achievement/Personal Goal/Preference]. I am a [Type of Person/Attitude], and I am [Number of Good Habits/Behaviors/Personal Touches]. I am always [Positive/Negative]. I am [Type of Relationship/Admittance], and I am [Number
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The answer to the question "What is the capital of France? " is: Paris. 
    
    To provide a more detailed response:
    
    1. Location: Paris is the capital city of France, situated on the Mediterranean coast, in the Île de France (Loiret) department in the north of the country.
    
    2. Time zone: Paris operates on the Coordinated Universal Time (CET) and daylight saving time (DST) is not used.
    
    3. Population: It is the most populous city in France, with an estimated population of around 2. 3 million as of 2021.
    
    4
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting and varied. Here are some potential trends that could influence AI in the years ahead:
    
    1. Increased Integration with Industry: AI is already being used in a wide range of industries, from healthcare to finance to manufacturing. As these industries continue to evolve, we can expect AI to become even more integrated and integrated into the work of these professionals.
    
    2. Enhanced Transparency and Explainability: As AI systems become more complex, we may see an increase in the ability to explain their decisions and actions to humans. This could lead to better trust and confidence in AI systems and could also help to reduce the risk of bias in AI algorithms.
    
    3


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

    insert

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

    ]

     in

     my

     [

    insert

     location

    ]

    .
    


    As

     a

     [

    insert

     profession

    ]

     in

     my

     [

    insert

     location

    ],

     I

     am

     [

    insert

     your

     interests

    ,

     skills

    ,

     and

     experiences

    ]

     and

     I

     have

     always

     [

    insert

     a

     positive

     trait

    ].

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    insert

     a

     positive

     statement

    ].

     I

     am

     always

     ready

     to

     [

    insert

     a

     positive

     statement

    ].
    


    I

     have

     a

     passion

     for

     [

    insert

     a

     hobby

     or

     interest

    ]

     and

     I

     enjoy

     [

    insert

     a

     question

     about

     your

     interests

     or

     hobbies

    ].

     I

     am

     [

    insert

     age

    ]

     years

     old

     and

     I

     have

     been

     [

    insert

     a

     number

     of

     years

    ]

     years

     in

     this

     career

    .
    


    I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     and

     world

    -ren

    owned

     city

     known

     for

     its

     rich

     culture

    ,

     history

    ,

     and

     art

    .

     It

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ité

    ,

     which

     is

     a

     UNESCO

     World

     Heritage

     site

    ,

     and

     boasts

     over

     

    1

    2

     million

     residents

    .

     Paris

     is

     famous

     for

     its

     iconic

     landmarks

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     diverse

     cuisine

     and

     vibrant

     nightlife

    .

     The

     city

     is

     also

     home

     to

     the

     European

     Parliament

    ,

     the

     European

     Central

     Bank

    ,

     and

     many

     other

     government

     institutions

    .

     Its

     status

     as

     a

     major

     economic

     center

     and

     cultural

     hub

     is

     a

     significant

     contributor

     to

     its

     reputation

     as

     a

     global

     capital

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     evolving

    ,

     with

     many

     different

     trends

     and

     possibilities

     shaping

     its

     direction

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

     Increasing

     emphasis

     on

     ethical

     AI

    :

     There

     is

     a

     growing

     emphasis

     on

     ethical

     AI

    ,

     with

     policymakers

     and

     industry

     experts

     looking

     to

     ensure

     that

     AI

     is

     used

     in

     ways

     that

     benefit

     society

     as

     a

     whole

    .
    


    2

    .

     AI

     becoming

     more

     integrated

     into

     our

     daily

     lives

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     may

     see

     more

     widespread

     use

     of

     AI

     in

     areas

     such

     as

     healthcare

    ,

     transportation

    ,

     and

     entertainment

    .
    


    3

    .

     AI

     becoming

     more

     ubiquitous

    :

     AI

     is

     becoming

     more

     ubiquitous

    ,

     with

     more

     people

     using

     AI

    -powered

     devices

     and

     systems

     on

     a

     daily

     basis

    .

    



```python
llm.shutdown()
```
