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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.05it/s]


    2026-04-13 22:58:47,857 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 22:58:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:40,  1.34it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:16,  3.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:16,  3.24it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:16,  3.24it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:02<00:16,  3.24it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:16,  3.24it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:07,  6.39it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 17.11it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 22.21it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:00, 28.89it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 33.22it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s] 

    Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 37.97it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.53it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 45.88it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 45.88it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 45.88it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 45.88it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.67 GB):   3%|▎         | 2/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.66 GB):   3%|▎         | 2/58 [00:00<00:03, 16.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.65 GB):   3%|▎         | 2/58 [00:00<00:03, 16.15it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=131.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.65 GB):   7%|▋         | 4/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):   7%|▋         | 4/58 [00:00<00:03, 17.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.65it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 20.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.62 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.71it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.61 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s] Capturing num tokens (num_tokens=896 avail_mem=131.59 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=768 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.79it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]

    Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=480 avail_mem=131.58 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  45%|████▍     | 26/58 [00:00<00:00, 39.33it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.61it/s]Capturing num tokens (num_tokens=288 avail_mem=131.56 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.61it/s]Capturing num tokens (num_tokens=288 avail_mem=131.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=224 avail_mem=131.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=160 avail_mem=131.54 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  71%|███████   | 41/58 [00:01<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=64 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=48 avail_mem=131.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=28 avail_mem=131.51 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]

    Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.43it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.93it/s] Capturing num tokens (num_tokens=4 avail_mem=131.49 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.93it/s]Capturing num tokens (num_tokens=4 avail_mem=131.49 GB): 100%|██████████| 58/58 [00:01<00:00, 37.74it/s]


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
    Generated text:  Kira and I have been assigned the task of assisting with the recruitment of candidates for a data science position. I believe that I possess excellent interpersonal skills and I would like to suggest a few strategies to make the process of attracting top talent easier.
    Sure, I'd be happy to help! Can you provide some more details on the recruitment process and how to improve it? What specific strategies have you found most effective in the past?
    The recruitment process for a data science position can be challenging, but here are a few strategies that may help to make it more effective:
    1. **Tailored Job Descriptions**: Conduct a thorough analysis of the
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide what kind of tree to plant in his backyard. He has two options, and he wants to choose the one that is the most unlikely to sprout a tree. The first tree is a oak tree, which is very common and easily available. The second tree is a pines, which is not common and difficult to find. The president wants to know the probability that the oak tree will sprout a tree and the probability that the pines will sprout a tree. Can you help him calculate these probabilities?
    
    To calculate the probability that a particular tree will sprout a tree, we need to consider the total number of
    ===============================
    Prompt: The capital of France is
    Generated text:  a city, which is now the capital of the French region of the same name. The capital of Switzerland is a city, which is now the capital of the country of the same name. The capital of Luxembourg is a city, which is now the capital of the country of the same name. What is the capital of the country of the same name? 
    To determine the capital of the country of the same name, let's first list out the capital cities for each of the countries mentioned:
    
    1. France: Paris
    2. Switzerland: Geneva
    3. Luxembourg: Luxembourg
    
    We need to identify the capital of the country of the
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting and rapidly evolving, as it advances from machine learning to deep learning to reinforcement learning. These technologies allow AI to learn from large datasets, identify patterns and trends, and make decisions based on those patterns. However, the way we use and interpret these technologies can have a significant impact on the outcomes of AI applications. It's important to understand the potential benefits and risks of these technologies and how they can be used in a way that is ethical, responsible, and beneficial for society as a whole. To achieve this goal, it is essential to have a thorough understanding of the technology and its potential impact on society, as well as the ethical considerations


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I have been working in [position] for [number of years] years. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [mention a hobby or activity]. What's your favorite book or movie? I love [mention a book or movie]. What's your favorite place to go? I love [mention a place].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union. It is located on the Seine River and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a popular tourist destination. The city is also home to many important institutions, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city that continues to thrive as a major global city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation: AI will continue to automate routine tasks, freeing up human workers to focus on more complex and creative work. This could lead to a shift in the job market, as some jobs will be replaced by AI, while others will be created.
    
    2. Improved natural language processing: AI will continue to improve its ability to understand and respond to human language, leading to more sophisticated and nuanced interactions with humans.
    
    3. Enhanced decision-making: AI will become more capable of making decisions based on complex data and patterns, leading to more accurate and reliable predictions and recommendations
    


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
    Generated text:  [Your Name]. I am a [Your Profession] with [Your Experience/Qualifications]. I'm excited to meet you and learn more about you. I look forward to talking to you. [Your Name] [Your Profession] [Your Experience/Qualifications] [Your Background] [Your Interests/Old Hobbies] [Your Reason for Joining Us] [Your Reasons for Joining Us] [Your Goals] [Your Past Projects/Projects] [Your Future Projects/Projects] [Your Education/Experience] [Your Publications/Articles] [Your Blogging/Articles] [Your Likes/Dislikes/
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    France's capital city is Paris. Here are some factual details about Paris:
    
    1. Founded by the Romans in 29 AD, Paris has been the capital of France since 843.
    2. It is the largest city in Europe, the third-largest city in the world, and the 14th-largest city in the world.
    3. Paris has a rich history, including a 3000-year-old civilization and many notable landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    4. The city is home to over 12 million people and is considered the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and involves a range of possible trends and developments. Some of the most common trends include:
    
    1. Increased focus on ethics and fairness: With the rise of ethical concerns around AI, there is a growing push towards more transparent and fair AI systems that prioritize the well-being of humans and the environment.
    
    2. Greater integration with everyday life: AI is increasingly being integrated into everyday life, from personalized recommendations on social media to autonomous vehicles and drones. This trend is likely to continue as AI becomes more integrated into our daily routines.
    
    3. Greater use of AI in healthcare: AI is being used to help diagnose and treat a wide range of


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

    ],

     and

     I

    'm

     a

     [

    brief

     description

     of

     your

     character

    's

     background

     or

     skills

    ].

     I

     love

     to

     [

    mention

     something

     you

    've

     done

     or

     achieved

     that

     inspires

     you

    ].

     I

    'm

     dedicated

     to

     [

    mention

     something

     that

     exc

    ites

     you

     about

     your

     work

     or

     life

    ],

     and

     I

     enjoy

     [

    mention

     an

     activity

     or

     hobby

     that

     you

     have

    ].

     I

    'm

     a

     [

    mention

     the

     type

     of

     person

     you

     are

    ]

     with

     [

    mention

     the

     specific

     personality

     traits

     or

     qualities

     that

     make

     you

     unique

     or

     different

    ].

     My

     main

     goal

     is

     to

     [

    describe

     what

     the

     ultimate

     goal

     is

     for

     you

    ].

     I

    'm

     eager

     to

     [

    describe

     a

     positive

     or

     negative

     impact

     that

     your

     work

     or

     interests

     have

     on

     others

     or

     the

     world

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     city

     of

     France

     and

     is

     the

     oldest

     continuously

     inhabited

     urban

     settlement

     in

     the

     world

    .

     It

     has

     a

     rich

     history

     dating

     back

     over

     

    2

    ,

    0

    0

    0

     years

    .

     It

     is

     a

     UNESCO

     World

     Heritage

     Site

     and

     is

     known

     for

     its

     beautiful

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     also

     has

     a

     strong

     culinary

     culture

    ,

     with

     many

     delicious

     local

     dishes

     to

     try

    .

     Paris

     is

     a

     bustling

     and

     vibrant

     met

    ropolis

     with

     a

     diverse

     and

     eclectic

     population

    ,

     making

     it

     an

     exciting

     and

     interesting

     city

     to

     visit

    .

     Its

     position

     on

     the

     River

     Se

    ine

     and

     its

     iconic

     skyline

     contribute

     to

     its

     allure

     as

     a

     tourist

     destination

    .

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     interesting

     trends

    ,

     including

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     ubiquitous

    ,

     there

     will

     be

     increasing

     pressure

     to

     ensure

     that

     it

     is

     used

     eth

    ically

     and

     responsibly

    .

     This

     will

     likely

     involve

     investing

     in

     more

     robust

     ethical

     oversight

     and

     standards

    ,

     as

     well

     as

     addressing

     privacy

     and

     security

     concerns

    .
    


    2

    .

     Em

    phasis

     on

     automation

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     likely

     become

     more

     autonomous

     and

     capable

     of

     performing

     a

     wide

     range

     of

     tasks

    .

     This

     could

     lead

     to

     the

     automation

     of

     many

     tasks

    ,

     and

     the

     creation

     of

     entirely

     new

     industries

     that

     are

     based

     on

     AI

    -driven

     automation

    .
    


    3

    .

     Enhanced

     development

     tools

     and

     techniques

    :

     As

     AI

    



```python
llm.shutdown()
```
