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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:49,  1.10it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:18,  2.68it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.33it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.33it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.33it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.33it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 11.59it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]

    Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 17.91it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 26.09it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]

    Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 33.71it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 42.81it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 42.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.07 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.78it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s]Capturing num tokens (num_tokens=960 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s] Capturing num tokens (num_tokens=896 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.65it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=768 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=704 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.24it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.30it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.01it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.26it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.26it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.26it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.73it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.73it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.53it/s]


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
    Generated text:  Teodoro. I am a software engineer, a teacher, a traveler, a cook, a podcast listener and a music lover. I love making music and I also love making playlists. You can find me playing tracks from my Spotify playlist, my YouTube playlist or, if I don't have a playlist, from the YouTube library. I try to create playlists that I like and that are related to my interests, like yoga, hiking, music, music genres and more. I share my playlists on this blog: music.astroscope.
    What is your favorite genre of music and why do you love it? As an AI language model,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to choose a new name for the national flag. He decides to take 3 different colors from among 8 available colors to make the flag. In how many ways can he choose 3 colors out of 8?
    To determine the number of ways the president can choose 3 colors out of 8 available colors, we use the concept of combinations. The number of ways to choose \( k \) items from \( n \) items is given by the combination formula:
    
    \[
    \binom{n}{k} = \frac{n!}{k!(n-k)!}
    \]
    
    In this problem, we need to find the
    ===============================
    Prompt: The capital of France is
    Generated text:  located at which of the following locations?
    A. In the north of France
    B. In the south of France
    C. In the east of France
    D. In the west of France
    Answer:
    
    D
    
    When the rainfall at a certain location exceeds 800mm, what type of soil is it most likely to be?
    A. Clayey soil
    B. Sandy soil
    C. Loamy soil
    D. Silty soil
    Answer:
    
    A
    
    What is the role of a human resource management department in a company?
    A. To establish and maintain a perfect working environment
    B. To address human resource issues
    ===============================
    Prompt: The future of AI is
    Generated text:  growing more complex every day. Here are some ways you can start to define your career in AI, and how you can find the perfect fit.
    The Future of AI is growing more complex every day. With the rapid adoption of AI in many different sectors and industries, the need for professionals with a strong understanding of the latest AI trends is becoming increasingly important. Here are some ways you can start to define your career in AI and how you can find the perfect fit.
    AI is a rapidly growing field, with many opportunities for professionals with a strong background in machine learning, data science, and computer science. The field is in high demand in many


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Museum of Modern Art. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is also home to the French Parliament, the French National Library, and the French National Museum of Modern Art. Paris is a vibrant and diverse city with a rich cultural heritage and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is expected to become more and more integrated into the production process, from manufacturing to service. This could lead to the automation of many jobs, freeing up workers to focus on more creative and high-value tasks.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be a growing concern about its ethical implications. This could lead to the development of new ethical guidelines and standards for AI
    


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
    Generated text:  ________. I am an AI assistant designed by Anthropic, and I am here to assist you with any questions or tasks you may have. How can I help you today? Do you have any particular questions or topics you would like me to assist with? Whether it's general knowledge, trivia, or specialized information, I'm here to help. If you have a specific query or need assistance with something specific, feel free to ask! Let me know how I can be of help. Have a great day! 😊🌟
    
    Hello, my name is [Your Name]. I am an AI assistant designed by Anthropic, and I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe, with a population of over 2.5 million and a cultural center with many world-renowned landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is known for its cuisine, art, and fashion, as well as its historical architecture and festivals. The city is also home to the French Academy of Sciences, which is the oldest science academy in the world and has been in operation for more than 300 years. The city is also known for its annual French New Year celebrations, including the "Marseillaise Festival" and "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and dynamic. Some of the most exciting trends that are likely to shape AI in the coming years include:
    
    1. Increased integration with human intelligence: As AI becomes more integrated with our brain, it could learn new skills and adapt to new tasks more efficiently.
    
    2. AI-informed decision-making: AI systems will become more sophisticated, making better decisions and predictions that humans would not.
    
    3. Natural language processing: AI systems will become more adept at understanding and generating human language, making it possible for machines to converse and interact with us.
    
    4. Enhanced perception: AI will be able to perceive and understand more complex sensory data, such as


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

     am

     a

     [

    insert

     role

    ]

    !

     I

    'm

     passionate

     about

     [

    insert

     a

     personal

     interest

     or

     hobby

     that

     you

     share

     with

     others

    ].

     I

     like

     to

     [

    insert

     something

     about

     your

     favorite

     activity

    ,

     hobby

    ,

     or

     interest

    ].

     I

     am

     a

     [

    insert

     a

     character

     trait

     or

     quality

     that

     you

     admire

     about

     yourself

    ]

     and

     I

     am

     [

    insert

     your

     age

     or

     profession

    ].

     I

     enjoy

     [

    insert

     something

     that

     you

     do

     to

     help

     others

     or

     make

     a

     positive

     impact

     on

     the

     world

    ].

     And

     I

    'm

     [

    insert

     a

     word

     that

     describes

     the

     way

     you

     like

     to

     take

     care

     of

     your

     life

    ].

     Thank

     you

    !

     Welcome

     to

     [

    Your

     Name

    ],

     an

     enthusiastic

     and

     caring

     character

    !

     What

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     most

     famous

     city

    ,

     with

     many

     historical

     landmarks

    ,

     museums

    ,

     and

     cultural

     attractions

    .

     
    


    In

     conclusion

    ,

     Paris

     is

     the

     capital

     of

     France

    ,

     the

     world

    's

     largest

     metropolitan

     area

    ,

     and

     a

     major

     cultural

    ,

     political

    ,

     and

     economic

     center

    .

     It

     is

     renowned

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

     modern

     attractions

    ,

     such

     as

     the

     Notre

    -D

    ame

     Cathedral

    ,

     E

    iff

    el

     Tower

    ,

     and

     Mou

    lin

     Rouge

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     stunning

     architecture

    ,

     charming

     French

     cuisine

    ,

     and

     lively

     nightlife

    .

     
    


    In

     conclusion

    ,

     Paris

     is

     the

     capital

     of

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     automation

    :

     With

     the

     advent

     of

     more

     sophisticated

     AI

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

     repetitive

     tasks

    ,

     reducing

     the

     need

     for

     human

     workers

    .
    


    2

    .

     Increased

     human

     presence

    :

     AI

     will

     continue

     to

     play

     a

     key

     role

     in

     decision

    -making

     processes

    ,

     but

     it

     will

     also

     likely

     take

     on

     more

     complex

     tasks

     that

     require

     human

     judgment

     and

     creativity

    .
    


    3

    .

     Integration

     with

     other

     technologies

    :

     AI

     will

     likely

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     sensors

    ,

     cameras

    ,

     and

     machine

     learning

     algorithms

    ,

     in

     order

     to

     improve

     performance

     and

     reduce

     errors

    .
    


    4

    .

     Enhanced

     ethical

     considerations

    :

     As

     AI

     becomes

     more

    



```python
llm.shutdown()
```
