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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.27it/s]


    2026-04-14 11:29:38,971 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 11:29:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:02<00:41,  1.34it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:02<00:13,  3.77it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:02<00:13,  3.77it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:02<00:13,  3.77it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:13,  3.77it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:13,  3.77it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:03<00:13,  3.77it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  7.61it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  7.61it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  7.61it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  7.61it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  7.61it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:03<00:03, 11.03it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s] 

    Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 14.69it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 19.81it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 24.29it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 24.29it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 24.29it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 24.29it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 24.29it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 26.28it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.30it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.58it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.71it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:04<00:00, 43.02it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:04<00:00, 43.02it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:04<00:00, 43.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.55 GB):   3%|▎         | 2/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:03, 14.34it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.54 GB):   3%|▎         | 2/58 [00:00<00:03, 14.34it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.54 GB):   7%|▋         | 4/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.54 GB):   7%|▋         | 4/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.53 GB):   7%|▋         | 4/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.53 GB):  10%|█         | 6/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.53 GB):  10%|█         | 6/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.52 GB):  10%|█         | 6/58 [00:00<00:02, 17.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.52 GB):  10%|█         | 6/58 [00:00<00:02, 17.53it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=56.52 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.51 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.50 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.49 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.49 GB):  21%|██        | 12/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.49 GB):  21%|██        | 12/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.47 GB):  21%|██        | 12/58 [00:00<00:02, 22.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.47 GB):  21%|██        | 12/58 [00:00<00:02, 22.01it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=56.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.46 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.44 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=960 avail_mem=56.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s] Capturing num tokens (num_tokens=896 avail_mem=56.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=832 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=768 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s]Capturing num tokens (num_tokens=704 avail_mem=56.44 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.64it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.44 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.53it/s]Capturing num tokens (num_tokens=640 avail_mem=56.43 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.53it/s]Capturing num tokens (num_tokens=576 avail_mem=56.43 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.53it/s]Capturing num tokens (num_tokens=512 avail_mem=56.42 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.53it/s]Capturing num tokens (num_tokens=480 avail_mem=56.44 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.53it/s]Capturing num tokens (num_tokens=448 avail_mem=56.44 GB):  45%|████▍     | 26/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=448 avail_mem=56.44 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]Capturing num tokens (num_tokens=416 avail_mem=56.43 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]Capturing num tokens (num_tokens=384 avail_mem=56.43 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]Capturing num tokens (num_tokens=352 avail_mem=56.43 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]Capturing num tokens (num_tokens=320 avail_mem=56.42 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]Capturing num tokens (num_tokens=288 avail_mem=56.42 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.20it/s]

    Capturing num tokens (num_tokens=288 avail_mem=56.42 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=256 avail_mem=56.42 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=240 avail_mem=56.41 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=224 avail_mem=56.41 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=208 avail_mem=56.41 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=192 avail_mem=56.41 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.02it/s]Capturing num tokens (num_tokens=192 avail_mem=56.41 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=176 avail_mem=56.40 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=160 avail_mem=56.40 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=144 avail_mem=56.40 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=128 avail_mem=56.39 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.39 GB):  71%|███████   | 41/58 [00:01<00:00, 42.23it/s]Capturing num tokens (num_tokens=112 avail_mem=56.39 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=96 avail_mem=56.39 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s] Capturing num tokens (num_tokens=80 avail_mem=56.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=64 avail_mem=56.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=48 avail_mem=56.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=32 avail_mem=56.38 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.79it/s]Capturing num tokens (num_tokens=32 avail_mem=56.38 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=28 avail_mem=56.37 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=24 avail_mem=56.37 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=20 avail_mem=56.36 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]

    Capturing num tokens (num_tokens=16 avail_mem=56.36 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=12 avail_mem=56.36 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.62it/s]Capturing num tokens (num_tokens=12 avail_mem=56.36 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=8 avail_mem=56.35 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.27it/s] Capturing num tokens (num_tokens=4 avail_mem=56.35 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=4 avail_mem=56.35 GB): 100%|██████████| 58/58 [00:01<00:00, 35.10it/s]


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
    Generated text:  Agnes. I'm 22 and from New York. I'm going to attend the school of business in Paris. The city is beautiful and the school is nice. I will study English, economics, and finance. I don't want to study psychology or sociology. I don't want to be a bank clerk. When I'm not studying, I like to read books. And I love to play sports. I'm really good at tennis and I'm on the tennis team. I'm going to study in Paris because Paris is beautiful and the school is nice. And because I'm really good at tennis, I will be on
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to increase the number of students in his research laboratories. He believes that students who have the opportunity to conduct experiments and research on their own will be more likely to be successful and productive in their careers. He wants to know the average number of students who will be enrolled in research laboratories in his next term of office. He has conducted a survey and found that there are 120 students who will be enrolled in research laboratories in his next term. 
    
    1. Calculate the sample size for the survey.
    2. If the president wants to have 90% confidence in his estimate of the average number of students who will be enrolled
    ===============================
    Prompt: The capital of France is
    Generated text:  (　　)  
    A: ① Paris  
    B: ② London  
    C: ③ Washington D.C.  
    D: ④ Moscow
    To determine the capital of France, we need to recall the official capital of France. The capital of France is typically the city that serves as the dominant and most populous city in the country. Let's consider the options given:
    
    A: Paris
    B: London
    C: Washington D.C.
    D: Moscow
    
    The capital of France is typically the capital of the country, which is Paris. This is because Paris is the largest city in France and the most
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, with various emerging technologies evolving in response to changing business needs and regulatory frameworks. As such, it’s essential to understand the potential impacts of these emerging technologies on the future of AI.
    
    Here are some key points to consider when evaluating the potential impacts of emerging technologies on the future of AI:
    
    1. AI efficiency: Many emerging technologies are expected to improve the efficiency of AI, making it faster, more accurate, and more scalable. For example, deep learning algorithms can process vast amounts of data faster than traditional approaches, leading to more accurate predictions and better decision-making.
    
    2. AI safety: As AI systems become more complex and interconnected,


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


    Generated text:  Paris, the city that was founded in 789 AD and is the largest city in Europe by population. It is also the seat of the French government and the country’s cultural and political capital. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also famous for its cuisine, fashion, and art. Paris is a popular tourist destination and is home to many world-renowned museums, theaters, and art galleries. It is also known for its wine production and is home to the famous Château de Chambord. Paris is a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, and it has the potential to revolutionize the field. AI-powered diagnostic tools could provide faster, more accurate diagnoses, and potentially lead to earlier detection of diseases.
    
    2. AI in finance: AI is already being used in financial services to automate trading and risk management. As AI technology continues to improve, it is likely to be used in more complex financial products and services.
    
    3. AI in transportation:
    


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
    Generated text:  [Name] and I am a [occupation] who has a passion for [subject or hobby]. I love [reason for this interest, such as learning new things, helping others, or being creative]. I am [age or age range] years old, and I enjoy [reason for my love for learning, such as staying informed about current events, mastering a new skill, or exploring different cultures]. I have been [age, years, or years to reach this goal] and I strive to always [action or statement that demonstrates a goal or passion]. I love spending my free time [reason for this, such as reading, watching
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a historical and cultural center with a rich history dating back to the time of the Roman Empire. Paris is also known for its beautiful architecture, food, music, and many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is a popular destination for tourists and locals alike. The French capital is an important part of French culture and is recognized as the capital of France. France’s capital city is Paris. It is a historical and cultural center with a rich history dating back to the Roman Empire. Paris is also known for its beautiful architecture, food, music, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a rapidly evolving landscape with several potential trends that could shape the way we live and work. Here are some possible future trends in AI:
    
    1. Improved Natural Language Processing: As AI continues to improve, it is expected to become even more capable of understanding and generating human-like language. This could result in more natural and conversational AI, and improved translation and generation of text.
    
    2. Greater Integration with Human Intelligence: AI will become more integrated with human intelligence, allowing it to better understand and respond to complex human emotions and behaviors. This could result in more empathetic and effective AI systems.
    
    3. Increased Use of AI in


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

    Age

    ]

     year

     old

     with

     [

    Prof

    ession

    ]

     experience

     in

     [

    Field

     of

     Interest

    ].

     I

    'm

     currently

     [

    Occup

    ation

    ]

     here

     and

     I

     enjoy

     [

    Favorite

     Activity

    ].

     Whether

     it

    's

     [

    What

     I

     Do

     Best

    ],

     [

    What

     I

     Love

     To

     Do

    ],

     or

     [

    What

     I

    'm

     Most

     Proud

     Of

    ],

     I

    'm

     passionate

     about

     [

    What

     You

     Can

     Expect

     from

     Me

    ].

     I

    'm

     always

     looking

     for

     opportunities

     to

     [

    What

     I

     Hope

     to

     Achie

    ve

    ],

     whether

     it

    's

     [

    What

     I

    'm

     Looking

     Forward

     To

     Doing

    ],

     [

    What

     I

     Hope

     to

     Achie

    ve

    ],

     or

     [

    What

     I

    'm

     Looking

     For

    ].

     I

    'm

     a

     [

    What

     I

    'm

     Known

     For

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     history

     and

     a

     vibrant

     cultural

     scene

    .

     It

     is

     known

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

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     restaurants

    .

     Paris

     is

     also

     renowned

     for

     its

     annual

     fashion

     week

     and

     world

    -ren

    owned

     museums

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     With

     its

     diverse

     architecture

    ,

     delicious

     food

    ,

     and

     lively

     atmosphere

    ,

     Paris

     is

     a

     city

     of

     contrasts

     and

     elegance

    .

     
    


    The

     capital

     city

     of

     France

     is

     home

     to

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

     the

     Ch

    amps

    -

    É

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     new

     technologies

     and

     approaches

     being

     developed

     that

     can

     make

     significant

     impacts

     on

     our

     lives

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Self

    -learning

     and

     adaptive

     systems

    :

     AI

     systems

     are

     becoming

     more

     capable

     of

     learning

     and

     adapting

     to

     new

     situations

    ,

     making

     them

     better

     at

     solving

     complex

     problems

     and

     making

     informed

     decisions

    .

     Self

    -learning

     AI

     is

     already

     being

     used

     in

     various

     domains

    ,

     such

     as

     medicine

     and

     finance

    ,

     where

     it

    's

     helping

     to

     improve

     efficiency

    ,

     accuracy

    ,

     and

     effectiveness

    .
    


    2

    .

     Quantum

     computing

    :

     Quantum

     computing

     is

     expected

     to

     revolution

    ize

     AI

    ,

     making

     it

     possible

     to

     process

     and

     analyze

     vast

     amounts

     of

     data

     at

     an

     unprecedented

     speed

    .

     Quantum

     computers

     have

     the

     potential

     to

    



```python
llm.shutdown()
```
