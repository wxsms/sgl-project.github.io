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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.62it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=1536):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.74it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:10<00:10,  2.94it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:10<00:05,  4.33it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:10<00:03,  5.43it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s] 

    Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:10<00:01,  7.47it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:10<00:00, 11.15it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:10<00:00, 11.15it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:10<00:00, 11.15it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:10<00:00, 11.15it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:10<00:00, 11.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   2%|▏         | 1/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:06,  8.23it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   3%|▎         | 2/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   3%|▎         | 2/58 [00:00<00:07,  7.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   5%|▌         | 3/58 [00:00<00:07,  7.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   5%|▌         | 3/58 [00:00<00:07,  7.85it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:06,  8.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   7%|▋         | 4/58 [00:00<00:06,  8.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   9%|▊         | 5/58 [00:00<00:06,  8.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.15 GB):   9%|▊         | 5/58 [00:00<00:06,  8.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.98 GB):   9%|▊         | 5/58 [00:00<00:06,  8.17it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=54.98 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.98 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.98 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.98 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.98 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.97 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.48it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=54.97 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.97 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.96 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.96 GB):  19%|█▉        | 11/58 [00:01<00:03, 14.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.96 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.96 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.96 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.02it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=54.95 GB):  24%|██▍       | 14/58 [00:01<00:02, 17.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.95 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.95 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.52it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.95 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.94 GB):  29%|██▉       | 17/58 [00:01<00:02, 19.52it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.94 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.92 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=960 avail_mem=54.94 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.99it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=54.94 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=832 avail_mem=54.93 GB):  34%|███▍      | 20/58 [00:01<00:01, 21.99it/s]Capturing num tokens (num_tokens=832 avail_mem=54.93 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=768 avail_mem=54.93 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=704 avail_mem=54.93 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=640 avail_mem=54.92 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=576 avail_mem=54.92 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.13it/s]Capturing num tokens (num_tokens=576 avail_mem=54.92 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=512 avail_mem=54.91 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.54it/s]

    Capturing num tokens (num_tokens=480 avail_mem=54.92 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=448 avail_mem=54.92 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=416 avail_mem=54.92 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.54it/s]Capturing num tokens (num_tokens=416 avail_mem=54.92 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.24it/s]Capturing num tokens (num_tokens=384 avail_mem=54.92 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.24it/s]Capturing num tokens (num_tokens=352 avail_mem=54.91 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.24it/s]Capturing num tokens (num_tokens=320 avail_mem=54.90 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.24it/s]Capturing num tokens (num_tokens=288 avail_mem=54.90 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.24it/s]

    Capturing num tokens (num_tokens=288 avail_mem=54.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=256 avail_mem=54.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=240 avail_mem=54.90 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=224 avail_mem=54.89 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=208 avail_mem=54.89 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.41it/s]Capturing num tokens (num_tokens=208 avail_mem=54.89 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=192 avail_mem=54.89 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=176 avail_mem=54.89 GB):  69%|██████▉   | 40/58 [00:01<00:00, 31.55it/s]Capturing num tokens (num_tokens=160 avail_mem=54.88 GB):  69%|██████▉   | 40/58 [00:02<00:00, 31.55it/s]

    Capturing num tokens (num_tokens=144 avail_mem=54.88 GB):  69%|██████▉   | 40/58 [00:02<00:00, 31.55it/s]Capturing num tokens (num_tokens=144 avail_mem=54.88 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.42it/s]Capturing num tokens (num_tokens=128 avail_mem=54.88 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.42it/s]Capturing num tokens (num_tokens=112 avail_mem=54.88 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.42it/s]Capturing num tokens (num_tokens=96 avail_mem=54.87 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.42it/s] Capturing num tokens (num_tokens=80 avail_mem=54.87 GB):  76%|███████▌  | 44/58 [00:02<00:00, 32.42it/s]Capturing num tokens (num_tokens=80 avail_mem=54.87 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.11it/s]Capturing num tokens (num_tokens=64 avail_mem=54.86 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.11it/s]Capturing num tokens (num_tokens=48 avail_mem=54.86 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.11it/s]

    Capturing num tokens (num_tokens=32 avail_mem=54.86 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.11it/s]Capturing num tokens (num_tokens=28 avail_mem=54.85 GB):  83%|████████▎ | 48/58 [00:02<00:00, 31.11it/s]Capturing num tokens (num_tokens=28 avail_mem=54.85 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.15it/s]Capturing num tokens (num_tokens=24 avail_mem=54.85 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.15it/s]Capturing num tokens (num_tokens=20 avail_mem=54.85 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.15it/s]Capturing num tokens (num_tokens=16 avail_mem=54.85 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.15it/s]Capturing num tokens (num_tokens=12 avail_mem=54.84 GB):  90%|████████▉ | 52/58 [00:02<00:00, 32.15it/s]Capturing num tokens (num_tokens=12 avail_mem=54.84 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.21it/s]Capturing num tokens (num_tokens=8 avail_mem=54.84 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.21it/s] Capturing num tokens (num_tokens=4 avail_mem=54.83 GB):  97%|█████████▋| 56/58 [00:02<00:00, 33.21it/s]

    Capturing num tokens (num_tokens=4 avail_mem=54.83 GB): 100%|██████████| 58/58 [00:02<00:00, 23.31it/s]


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
    Generated text:  Abby and I am in high school. I'm learning to drive cars in the city with a Toyota. One day, when I was driving I saw a dog chasing a cat. My surprise was that the cat was injured. I tried to help the cat by tying it up. My cat looked scared and a bit hurt. I got to know the dog. The dog tried to bite the cat. I'm trying to teach my dog how to help others. My dog is very kind and he always wants to help others. For example, when I was in the park playing, I saw a boy fell down on the ground. I tried
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking official that is appointed by the President of the United States from the President's Cabinet. The president is elected by the United States citizens.
    Does this next sentence follow, given the preceding text?
    The president of the United States is appointed by the President of the United States from the President's Cabinet.
    
    OPTIONS:
     a). yes
     b). it is not possible to tell
     c). no
    a). yes
    
    The given text clearly states that the president of the United States is appointed by the President of the United States from the President's Cabinet. Therefore, the statement "The president of the United States is appointed by the President
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Berlin
    D. Amsterdam
    
    The correct answer is A. Paris. Paris is the capital of France, which is located in the south of France. The other options are not capitals of France, as London (capital of the United Kingdom) and Amsterdam (capital of the Netherlands) are located in their respective countries. Berlin, on the other hand, is the capital of Germany.
    ===============================
    Prompt: The future of AI is
    Generated text:  currently very bright, but it is also very complicated. Scientists are working on ways to control the AI, so that it will be more ethical and morally acceptable. On the other hand, the AI that we currently have is not always ethical or morally acceptable, and we must deal with this issue.
    One of the main ethical and moral concerns that AI raises is the issue of bias. This refers to the AI's tendency to make decisions or respond to situations based on the characteristics of a particular group or population. For example, if a bot is used to assist with financial decisions, it may be biased if it is trained on a dataset that is


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast who loves to explore new places and learn new things. I'm always looking for new experiences and adventures, and I'm always eager to share my knowledge with others. I'm a [Favorite Activity] lover who enjoys hiking, camping, and exploring the outdoors. I'm also a [Favorite Book] lover who reads a lot of books and enjoys reading. I'm a [Favorite Music] lover who listens to music that makes me feel happy and inspired. I'm a [Favorite Movie] lover who enjoys watching movies
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its fashion industry, art, and cuisine. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination. It is also known for its annual Eiffel Tower Festival, which attracts millions of visitors each year. Paris is a vibrant and dynamic city that is a must-visit
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots in factories to personalized medicine and virtual assistants. Additionally, AI is likely to play an increasingly important role in areas such as healthcare, finance, and education, as well as in the development of new technologies and industries. However, there are also potential risks and challenges associated with AI, including issues of bias and privacy, as well as concerns about the impact of AI on employment and society as a whole. Ultimately, the
    


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
    Generated text:  [Name], and I am a [field] specialist. I specialize in [field] research. I graduated from [college] and have spent the last [number] years working in the field. My passion is [describe your passion here]. My work ethic is high, and I have a natural talent for [describe your specialty here]. I value [mention a personal value], and I strive to [mention a specific action or statement that demonstrates this]. My research method is [describe your research method here], and I am constantly learning and growing in my field. I am here to help, and I am excited to contribute to the field
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the city of love due to its many romantic attractions, including the Eiffel Tower, Montmartre, and the Louvre museum. The city is also home to many world-renowned artists such as Picasso, Rembrandt, and Van Gogh. In terms of food, Paris has a wide variety of traditional French cuisine, including pastries, pastries, and hearty dishes. Additionally, the city has a vibrant nightlife, featuring numerous bars, clubs, and restaurants that cater to a wide range of tastes. Paris is a popular destination for tourists, and offers a unique blend of traditional French culture and modern amenities
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and likely to evolve rapidly, with many possibilities and challenges ahead. Here are some possible trends in AI that could shape the technology in the coming years:
    
    1. Increased Human Interaction: AI is already capable of performing tasks that require human-like intelligence, such as language translation, visual perception, and decision-making. However, as AI continues to get more advanced, it could eventually become capable of more complex human-like interactions.
    
    2. Enhanced Predictive Analytics: AI can be used to analyze large amounts of data in real-time, allowing for more accurate and timely predictions of future events. This could lead to improved decision-making, resource allocation, and


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

     am

     an

     enthusiastic

     and

     compassionate

     friend

     who

     is

     always

     there

     for

     others

    .

     I

     have

     a

     friendly

     and

     laid

    -back

     personality

     that

     is

     easy

     to

     talk

     to

     and

     understand

    .

     I

     love

     spending

     time

     with

     my

     family

     and

     friends

    ,

     and

     I

     am

     always

     looking

     for

     new

     adventures

     and

     experiences

     to

     try

    .

     If

     you

     need

     help

     or

     advice

    ,

     I

     am

     always

     here

     to

     listen

    .

     How

     can

     I

     contact

     me

    ?

     [

    Name

    ]

     ([

    Email

     Address

    ])

     [

    Phone

     Number

    ]

     [

    Social

     Media

     Handles

    ]

     [

    Status

    ]


    "

    Hello

    ,

     my

     name

     is

     [

    Name

    ]

     and

     I

     am

     an

     enthusiastic

     and

     compassionate

     friend

     who

     is

     always

     there

     for

     others

    .

     I

     have

     a

     friendly

     and

     laid

    -back

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     culture

    ,

     and

     famous

     landmarks

     such

     as

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

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     street

     life

     and

     extensive

     food

     and

     wine

     scene

    .

     Its

     status

     as

     the

     world

    ’s

     second

    -largest

     city

     and

     a

     major

     transportation

     hub

     makes

     it

     an

     important

     commercial

     and

     cultural

     center

    .

     The

     city

     has

     a

     unique

     French

     culture

    ,

     known

     for

     its

     participation

     in

     world

     events

     such

     as

     the

     Olympics

     and

     World

     War

     II

    .

     Paris

     is

     home

     to

     many

     historic

     sites

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     a

     major

     center

     for

     international

     trade

     and

     diplomacy

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     ongoing

     research

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

     Increased

     integration

     with

     human

     workers

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     it

     is

     possible

     that

     they

     will

     start

     to

     integrate

     more

     closely

     with

     human

     workers

    .

     This

     could

     lead

     to

     a

     more

     efficient

     and

     productive

     workforce

    ,

     but

     it

     also

     raises

     concerns

     about

     job

     loss

     and

     economic

     disruption

    .
    


    2

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     it

     is

     possible

     that

     they

     will

     have

     the

     ability

     to

     collect

     and

     analyze

     vast

     amounts

     of

     data

     without

     any

     privacy

     concerns

    .

     This

     could

     raise

     ethical

     and

     legal

     questions

     about

     the

     use

     of

     AI

     in

     personal

     lives

    .
    


    3

    .

     More

     autonomous

     and

    



```python
llm.shutdown()
```
