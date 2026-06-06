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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.21s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:02,  1.14s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:30,  1.71it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  6.03it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]

    Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  9.31it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:02, 14.24it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]

    Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:01, 21.01it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:00, 29.17it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 37.13it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 45.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 54.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.69 GB):   2%|▏         | 1/58 [00:00<00:06,  8.74it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.66 GB):   2%|▏         | 1/58 [00:00<00:06,  8.74it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.66 GB):   3%|▎         | 2/58 [00:00<00:06,  8.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:06,  8.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:05,  9.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:05,  9.92it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:05,  9.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.65 GB):  10%|█         | 6/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.64 GB):  10%|█         | 6/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:04, 11.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.15it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=55.63 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.62 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.62 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.62 GB):  21%|██        | 12/58 [00:00<00:02, 15.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.62 GB):  21%|██        | 12/58 [00:00<00:02, 15.99it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.62 GB):  21%|██        | 12/58 [00:00<00:02, 15.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.61 GB):  21%|██        | 12/58 [00:00<00:02, 15.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.61 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.61 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.60 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.60 GB):  31%|███       | 18/58 [00:01<00:01, 20.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.60 GB):  31%|███       | 18/58 [00:01<00:01, 20.85it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.60 GB):  31%|███       | 18/58 [00:01<00:01, 20.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.58 GB):  31%|███       | 18/58 [00:01<00:01, 20.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.58 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.06it/s]Capturing num tokens (num_tokens=960 avail_mem=55.59 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.06it/s] Capturing num tokens (num_tokens=896 avail_mem=55.59 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.06it/s]Capturing num tokens (num_tokens=832 avail_mem=55.58 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.06it/s]Capturing num tokens (num_tokens=768 avail_mem=55.58 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.06it/s]Capturing num tokens (num_tokens=768 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.83it/s]Capturing num tokens (num_tokens=704 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.83it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.83it/s]Capturing num tokens (num_tokens=576 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.83it/s]Capturing num tokens (num_tokens=512 avail_mem=55.56 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.83it/s]Capturing num tokens (num_tokens=512 avail_mem=55.56 GB):  50%|█████     | 29/58 [00:01<00:01, 28.09it/s]Capturing num tokens (num_tokens=480 avail_mem=55.58 GB):  50%|█████     | 29/58 [00:01<00:01, 28.09it/s]Capturing num tokens (num_tokens=448 avail_mem=55.57 GB):  50%|█████     | 29/58 [00:01<00:01, 28.09it/s]Capturing num tokens (num_tokens=416 avail_mem=55.57 GB):  50%|█████     | 29/58 [00:01<00:01, 28.09it/s]Capturing num tokens (num_tokens=384 avail_mem=55.57 GB):  50%|█████     | 29/58 [00:01<00:01, 28.09it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.57 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.91it/s]Capturing num tokens (num_tokens=352 avail_mem=55.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.91it/s]Capturing num tokens (num_tokens=320 avail_mem=55.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.91it/s]Capturing num tokens (num_tokens=288 avail_mem=55.56 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.91it/s]Capturing num tokens (num_tokens=256 avail_mem=55.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.91it/s]Capturing num tokens (num_tokens=256 avail_mem=55.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.27it/s]Capturing num tokens (num_tokens=240 avail_mem=55.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.27it/s]Capturing num tokens (num_tokens=224 avail_mem=55.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.27it/s]Capturing num tokens (num_tokens=208 avail_mem=55.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.27it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.27it/s]Capturing num tokens (num_tokens=192 avail_mem=55.54 GB):  71%|███████   | 41/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=176 avail_mem=55.54 GB):  71%|███████   | 41/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=160 avail_mem=55.54 GB):  71%|███████   | 41/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=144 avail_mem=55.53 GB):  71%|███████   | 41/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=128 avail_mem=55.53 GB):  71%|███████   | 41/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=128 avail_mem=55.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.51it/s]Capturing num tokens (num_tokens=112 avail_mem=55.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.51it/s]Capturing num tokens (num_tokens=96 avail_mem=55.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.51it/s] Capturing num tokens (num_tokens=80 avail_mem=55.52 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.51it/s]

    Capturing num tokens (num_tokens=64 avail_mem=55.52 GB):  78%|███████▊  | 45/58 [00:02<00:00, 33.51it/s]Capturing num tokens (num_tokens=64 avail_mem=55.52 GB):  84%|████████▍ | 49/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=48 avail_mem=55.51 GB):  84%|████████▍ | 49/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=32 avail_mem=55.51 GB):  84%|████████▍ | 49/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=28 avail_mem=55.50 GB):  84%|████████▍ | 49/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=24 avail_mem=55.50 GB):  84%|████████▍ | 49/58 [00:02<00:00, 34.01it/s]Capturing num tokens (num_tokens=24 avail_mem=55.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.22it/s]Capturing num tokens (num_tokens=20 avail_mem=55.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.22it/s]Capturing num tokens (num_tokens=16 avail_mem=55.50 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.22it/s]Capturing num tokens (num_tokens=12 avail_mem=55.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.22it/s]

    Capturing num tokens (num_tokens=8 avail_mem=55.49 GB):  91%|█████████▏| 53/58 [00:02<00:00, 34.22it/s] Capturing num tokens (num_tokens=8 avail_mem=55.49 GB):  98%|█████████▊| 57/58 [00:02<00:00, 34.59it/s]Capturing num tokens (num_tokens=4 avail_mem=55.49 GB):  98%|█████████▊| 57/58 [00:02<00:00, 34.59it/s]Capturing num tokens (num_tokens=4 avail_mem=55.49 GB): 100%|██████████| 58/58 [00:02<00:00, 25.13it/s]


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
    Generated text:  Tricia and I am a 24 year old teen girl. I started the book "The Ascent" at 13 years old, and I've read the whole series. This is the 6th book in the series, "The Ascent" and it is my last book. I really enjoyed reading the other books in the series and it is on my list of books to read again when I'm older.
    I like books that take you on a journey, and in this story, I find the characters extremely strong and brave. I really enjoyed reading the book because the protagonist, Carrie, is someone that I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  23 years older than the president of Brazil, and the president of Brazil is 2/3 as old as the president of China. If the president of China is 30 years old, how old is the president of Brazil? To determine the age of the president of Brazil, we need to follow the information given step by step.
    
    1. Identify the age of the president of China.
       The president of China is 30 years old.
    
    2. Determine the age of the president of Brazil.
       The problem states that the president of Brazil is \(\frac{2}{3}\) as old as the president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of the United Kingdom is London.
    London and Paris are connected by the Channel Tunnel.
    The Channel Tunnel is the first bridge to connect Europe.
    This bridge, nicknamed "the King's Highway", is built between the two cities. It was opened in 1994.
    The bridge starts at the southern end of the Channel and reaches the northern end.
    The entire span of the bridge is 1.2 miles and runs from east to west.
    This bridge has been built with three major sections.
    The two main sections have a length of 8.5 miles and are separated by a bridge called the "Great West
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of individual users.
    A. True
    B. False
    Answer:
    
    A
    
    What do the symbols △+ and △− represent in construction drawings?
    A. The width of the foundation slab
    B. The height of the foundation slab
    C. The length of the foundation slab
    D. The width of the wall
    Answer:
    
    B
    
    In the context of new media and online platforms, the core of the industry is ____.
    A. Network Marketing
    B. Social Networking
    C. Service Technology
    D. E-commerce
    Answer:
    
    D
    
    Which of the following statements about a diagram is correct?
    A


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a [reason for interest in the industry] and I'm always eager to learn and improve. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the third largest in the world. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also known for its food, fashion, and music scene. Paris is a popular tourist destination and a cultural hub in France. It is home to many museums, theaters, and other cultural institutions. The city is also known for its annual festivals and events, such as the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations more effectively. This could lead to more efficient and effective decision-making, as well as better problem-solving.
    
    2. Enhanced machine learning capabilities: AI systems are likely to become even more capable of learning and adapting to new situations, thanks to advancements in machine learning algorithms. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    3. Greater emphasis on ethical considerations: As AI systems become
    


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
    Generated text:  [Name] and I am a [character type] with [character feature]. I am [number] of years old, and I grew up in [city] and have lived in [country]. I have always been [character trait] and I have a passion for [character interest or hobby]. I have always been [character characteristic], and I am always [character personality]. I am the [character type] and I am [number] of years old. I am [character type] and I am [number] of years old. I have always been [character trait] and I am always [character personality]. I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the seat of government for the country. It is known for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. The city is also famous for its rich history, art, and cuisine, and is a popular tourist destination.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and rapidly evolving, with many potential directions to explore. Here are some possible trends in the AI field:
    
    1. Personalization: One of the most exciting and impactful trends in AI is personalization. With the increasing amount of data and internet access, it's becoming possible to create personalized experiences for users, from recommendations for products or services to personalized healthcare or financial advice.
    
    2. Enhanced Creativity: AI is already getting better at generating creative responses to prompts, and it's likely to get even more sophisticated in the future. As AI becomes more capable of understanding and generating creative ideas, it could revolutionize fields like art, music,


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

    ].

     I

    'm

     a

     [

    insert

     profession

    ]

     with

     a

     passion

     for

     [

    insert

     hobbies

     or

     interests

    ].

     I

    'm

     a

     [

    insert

     age

    ]

     year

     old

     who

     has

     always

     been

     fascinated

     by

     [

    insert

     something

     specific

    ,

     like

     nature

    ,

     science

    ,

     or

     art

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     I

    'd

     love

     to

     learn

     more

     about

     you

    !

     [

    Insert

     one

     or

     two

     sentences

     about

     your

     personal

     or

     professional

     life

    ,

     if

     applicable

    ].

     [

    Insert

     one

     or

     two

     sentences

     about

     your

     hobbies

     or

     interests

    ,

     if

     applicable

    ].


    [

    Insert

     name

    ]


    Hi

    !

     I

    'm

     [

    insert

     name

    ],

     a

     [

    insert

     profession

    ].

     I

    'm

     a

     [

    insert

     profession

    ]

     with

     a

     passion

     for

     [

    insert

     hobbies

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     city

     of

     France

    .

     It

    's

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

     can

    als

    ,

     and

     many

     famous

     landmarks

     like

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     Do

     you

     know

     any

     other

     famous

     landmarks

     in

     Paris

    ?

     
    


    Sure

    ,

     are

     there

     any

     other

     famous

     landmarks

     in

     Paris

    ?

     
    


    Yes

    ,

     there

     are

     many

    !

     Paris

     is

     famous

     for

     its

     towering

     E

    iff

    el

     Tower

    ,

     Ch

    amps

    -

    É

    lys

    ées

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     You

     could

     also

     check

     out

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Mar

    ais

     district

    .

     Paris

     is

     a

     beautiful

     city

     with

     a

     rich

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     see

     rapid

     advancements

    ,

     with

     a

     focus

     on

     improving

     efficiency

    ,

     reducing

     human

     errors

    ,

     and

     expanding

     our

     understanding

     of

     the

     world

     around

     us

    .

     Some

     possible

     future

     trends

     include

    :
    


    1

    .

     Increased

     AI

     autonomy

    :

     As

     AI

     gets

     more

     advanced

    ,

     it

     will

     be

     able

     to

     make

     decisions

     and

     take

     actions

     on

     its

     own

     without

     being

     directly

     controlled

    .

     This

     could

     lead

     to

     more

     autonomous

     vehicles

    ,

     drones

    ,

     and

     other

     machines

    .
    


    2

    .

     AI

     ethics

     and

     privacy

     concerns

    :

     With

     the

     rise

     of

     AI

    ,

     there

     is

     a

     growing

     need

     to

     address

     ethical

     concerns

     and

     privacy

     issues

    .

     Governments

     and

     companies

     will

     need

     to

     work

     together

     to

     ensure

     that

     AI

     is

     used

     responsibly

     and

     that

     data

     is

     protected

    .
    


    3

    .

     AI

    



```python
llm.shutdown()
```
