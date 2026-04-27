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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.76it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.75it/s]


    2026-04-27 20:42:57,277 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 20:42:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.11it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.11it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.73it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.44it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 29.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 39.11it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   3%|▎         | 2/58 [00:00<00:03, 18.38it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.44it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.43it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=768 avail_mem=137.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.23it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]

    Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.16it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.52it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.52it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]

    Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.95it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s] Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 43.72it/s]

    Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.20it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 39.65it/s]


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
    Generated text:  Daniel. I am the president of the College of Science, Technology, and Medicine at the University of Alabama at Birmingham and also the president of the Research and Education Corporation, the largest corporation in the world.
    I believe there are two crucial priorities in life. The first is the personal. I believe that you should be able to do the work you enjoy, to pursue your dreams, and to have a fulfilling personal life. The second is the community. I believe that people should be able to help their neighbors and communities by making a difference in the lives of others.
    I am committed to improving the quality of life for all. I am committed
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. If I am a man, I should be president. Therefore, if I am a man, I should be president. Is the argument valid? Yes.
    
    Here's how we can break down the argument:
    
    1. The conclusion is: "Therefore, if I am a man, I should be president."
    2. The argument assumes two premises:
       a) "If I am a man, I should be president" (premise 1)
       b) "If I am a man, I should be president" (premise 2)
    3. The argument also assumes a conditional statement: "If I am
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Madrid
    D. Rome
    Answer:
    
    A
    
    In a conversation, one should focus on listening attentively, which is crucial for effectively communicating. This is mainly because listening can
    A. enable one to get a deeper understanding of the speaker
    B. help in clearly expressing one's own views
    C. facilitate the speaker's use of language
    D. ensure the speaker is speaking in the right tone
    Answer:
    
    A
    
    In a conversation, one should focus on listening attentively, which is crucial for effectively communicating. This is mainly because listening can
    A. enable one
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it's impacting every industry. It's changing how we work and how we live. From healthcare to finance to law, AI is making a profound impact on how we live, work, and play.
    In this course, you will be introduced to the current AI landscape and the different applications it is being used for.
    You will learn how AI can be applied in a wide variety of scenarios and will gain a deeper understanding of AI research and its evolution.
    You will see how AI has improved and continues to improve our lives.
    You will see how AI is changing the way we live and work, the way we communicate, and the way


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working at [company name] for [number of years] years. I am a [job title] with [number of years] years of experience in [industry]. I am passionate about [job title] and I am always looking for ways to [job title] my skills and knowledge. I am a [job title] with [number of years] years of experience in [industry]. I am a [job title] with [number of years] years of experience in [industry]. I am a [job title] with [number
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is a major economic and political center in Europe and plays a significant role in French culture and identity. Paris is a city that is constantly evolving and changing, with new developments and attractions being added to the city's skyline. It is a city that is a true reflection of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. AI in finance: AI is already being used in finance to improve risk management, fraud detection, and trading algorithms. As AI technology continues to improve, we can expect to see even more widespread use of AI in
    


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
    Generated text:  [Name], I'm an [Age] year old [Gender] person. I've always been curious about the world around me and have always been driven by a love for learning and discovery. I love to explore new places and try new things, and I'm always on the lookout for new experiences and adventures. I'm an independent and resilient person, and I'm always looking for ways to grow and learn new things. I'm a strong advocate for diversity and inclusion, and I'm committed to using my voice to make the world a better place. I'm always eager to share my knowledge and experiences with others and to learn new things
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The capital of France is Paris. It is the largest city in France and the largest metropolitan area in the European Union. The city is home to many of the country's major museums, art galleries, theaters, and restaurants, and is known for its rich history, beautiful architecture, and vibrant culture. Paris is also one of the most popular tourist destinations in the world, and the city is home to many of France's cultural institutions and landmarks, such as the Louvre Museum and the Eiffel Tower. With its unique blend of modern and historic elements, Paris continues to be a vibrant and exciting place to live and work in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by continued growth, evolution, and integration into various industries. Some possible future trends in AI include:
    
    1. Increased automation: AI is expected to play an increasingly larger role in manufacturing, healthcare, and other industries as machines and algorithms become more sophisticated and efficient. Automation will require AI to be integrated into processes in a way that minimizes human errors and maximizes efficiency.
    
    2. Enhanced human-computer interaction: As AI becomes more advanced, it will likely enable humans to interact with AI in new ways. For example, virtual assistants will be able to understand language better and provide more personalized recommendations, while AI-powered robots will


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

    ].

     I

    'm

     a

     [

    job

     title

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     the

     [

    industry

    ]

     field

    .

     I

     love

     solving

     problems

    ,

     learning

     new

     things

    ,

     and

     constantly

     pushing

     myself

     to

     improve

    .

     I

    'm

     looking

     to

     share

     my

     knowledge

     with

     others

     and

     contribute

     to

     the

     success

     of

     [

    company

    ].

     I

     am

     excited

     to

     meet

     you

     and

     help

     you

     achieve

     your

     goals

    .

     I

     look

     forward

     to

     our

     conversation

    .

     Welcome

     to

     my

     world

    !

     [

    Name

    ]

     with

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     the

     [

    industry

    ]

     field

    .

     I

     love

     solving

     problems

    ,

     learning

     new

     things

    ,

     and

     constantly

     pushing

     myself

     to

     improve

    .

     I

    'm

     looking

     to

     share

     my

     knowledge

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Re

    ine

    ."
    


    Paris

     is

     the

     largest

     city

     in

     France

     by

     area

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

    .

     It

     is

     the

     capital

     of

     France

     and

     the

     second

    -largest

     city

     in

     the

     European

     Union

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     annual

     celebration

     of

     the

     E

    iff

    el

     Tower

     and

     the

     World

     Cup

    .

     Paris

     is

     also

     home

     to

     many

     museums

    ,

     art

     galleries

    ,

     and

     landmarks

    ,

     making

     it

     one

     of

     the

     most

     popular

     destinations

     in

     France

    .

     The

     city

    's

     cultural

     heritage

     is

     reflected

     in

     its

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     Additionally

    ,

     Paris

     is

     home

     to

     many

     international

     organizations

    ,

     including

     the

     European

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     currently

     experiencing

     a

     wave

     of

     rapid

     innovation

     and

     progress

    ,

     with

     many

     experts

     predicting

     that

     the

     technology

     will

     continue

     to

     evolve

     at

     a

     rapid

     pace

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

     future

     of

     AI

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     With

     more

     data

     being

     generated

     from

     the

     internet

     and

     beyond

    ,

     AI

     is

     expected

     to

     be

     able

     to

     learn

     from

     and

     personalize

     its

     responses

    ,

     making

     interactions

     more

     tailored

     to

     individual

     users

    .
    


    2

    .

     Machine

     Learning

    :

     More

     powerful

     AI

     algorithms

     will

     allow

     machines

     to

     learn

     from

     examples

    ,

     enabling

     them

     to

     make

     more

     accurate

     predictions

     and

     decisions

    .
    


    3

    .

     Autonomous

     Vehicles

    :

     Autonomous

     vehicles

     could

     become

     more

     prevalent

     as

     the

     technology

     continues

     to

     improve

     and

     become

     more

     accessible

     to

     the

    



```python
llm.shutdown()
```
