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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.64it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.68it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.68it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.68it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.68it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.66it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.66it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.66it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.66it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:06,  7.06it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:06,  7.06it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:05<00:06,  7.06it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:05<00:06,  7.06it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:03, 10.85it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:02, 14.67it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:01, 20.05it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 23.75it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:00, 30.02it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]

    Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 34.40it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 42.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.90 GB):   2%|▏         | 1/58 [00:00<00:07,  7.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.87 GB):   2%|▏         | 1/58 [00:00<00:07,  7.41it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=53.87 GB):   3%|▎         | 2/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.86 GB):   3%|▎         | 2/58 [00:00<00:07,  7.38it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.86 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.86 GB):   5%|▌         | 3/58 [00:00<00:07,  7.56it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=53.86 GB):   7%|▋         | 4/58 [00:00<00:06,  7.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.86 GB):   7%|▋         | 4/58 [00:00<00:06,  7.78it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.86 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.85 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=53.84 GB):   9%|▊         | 5/58 [00:00<00:06,  7.97it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.84 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.84 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.84 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.84 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.84 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.83 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.83 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=53.82 GB):  17%|█▋        | 10/58 [00:00<00:03, 15.73it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  17%|█▋        | 10/58 [00:01<00:03, 15.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.17 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.17 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  24%|██▍       | 14/58 [00:01<00:02, 18.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.16 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.14 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s] Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.55it/s]Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=704 avail_mem=72.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.56it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=448 avail_mem=72.14 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=416 avail_mem=72.13 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=384 avail_mem=72.13 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  50%|█████     | 29/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=320 avail_mem=72.12 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=288 avail_mem=72.12 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=256 avail_mem=72.12 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=240 avail_mem=72.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=224 avail_mem=72.11 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.60it/s]Capturing num tokens (num_tokens=224 avail_mem=72.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=208 avail_mem=72.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=192 avail_mem=72.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=176 avail_mem=72.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=160 avail_mem=72.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.09 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=144 avail_mem=72.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=128 avail_mem=72.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=112 avail_mem=72.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=96 avail_mem=72.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s] Capturing num tokens (num_tokens=80 avail_mem=72.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=64 avail_mem=72.06 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=64 avail_mem=72.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=48 avail_mem=72.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=32 avail_mem=72.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=28 avail_mem=72.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.05 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=20 avail_mem=72.04 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=20 avail_mem=72.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=16 avail_mem=72.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=12 avail_mem=72.04 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.23it/s]Capturing num tokens (num_tokens=8 avail_mem=72.03 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.23it/s] Capturing num tokens (num_tokens=4 avail_mem=72.03 GB):  93%|█████████▎| 54/58 [00:02<00:00, 41.23it/s]Capturing num tokens (num_tokens=4 avail_mem=72.03 GB): 100%|██████████| 58/58 [00:02<00:00, 28.31it/s]


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
    Generated text:  Martin and I'm a software engineer from the UK. Currently, I am studying computer science at the University of Birmingham.
    I have always been fascinated by problem solving and algorithms, and I've always been interested in the way that people think and reason. As a software engineer, I am focused on creating innovative solutions to complex problems, and working towards achieving the best possible outcomes. I am also passionate about technology and I enjoy learning about the latest advancements in the field. I believe that technology has the potential to revolutionize the world, and that I have a responsibility to make sure that it is used in a positive and ethical way. I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  getting the______of the key economic figures involved in the project.（A）attention（B）attention（C）being（D）to be（E）being
    
    答案：A． 选项A "attention" 表示"注意，注意"，与前文的"the key economic figures involved in the project"（项目中的关键经济人物）一致，表示"注意参与项目的关键经济人物"，符合语境。选项B "attention" 与主语"the president of the United States" 一致，表示"注意，注意"，不符合语境。选项C "being"
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the ___
    A. North
    B. South
    C. East
    D. West
    Answer:
    
    B
    
    In China, the _____ year of the lunar calendar is determined by the lunar eclipse that occurs on the 21st of the lunar month, and the year on which it falls is called the lunar year.
    A. 12
    B. 10
    C. 20
    D. 5
    Answer:
    
    C
    
    Among the following options, which one is a sequence of events that occurred during the New Culture Movement? 
    A. The establishment of the New Youth magazine 
    B. The
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but it needs to be sustainable. In the next few years, the leading edge of artificial intelligence will begin to push the boundaries of what’s possible. The following is an overview of the key trends in AI, including what’s on the horizon and how it will impact society in the coming years.
    
    AI will help us do more work at a lower cost while also being more safe. In a world where AI is a key to unlocking the power of machines, it’s becoming clear that we need to look at how we develop and deploy AI systems, how we choose the data we use and how we build trust with our users. The


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


    Generated text:  Paris. 
    
    A. True
    B. False
    A. True
    
    Paris is the capital city of France and is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center in Europe. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is also known for its rich history, including the French Revolution and the French Revolution Monument. The city is also known for its fashion industry, with many famous fashion designers and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and artificial intelligence: As AI technology continues to advance, we can expect to see more automation and artificial intelligence in various industries. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. Improved privacy and security: As AI technology becomes more advanced, we can expect to see more privacy and security concerns. This could lead to increased regulation and oversight of AI systems, as well
    


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
    Generated text:  [Name]. I'm a [age] year-old female with [hobbies or interests] that make me unique and interesting. I love to [job, hobby, or activity] and have [number of years of experience] of being in that field. I'm always learning and evolving, and I'm passionate about [reason for passion]. I'm excited to share my knowledge and experiences with you. What can you tell me about yourself?
    As an AI language model, I don't have a physical appearance, but I can assist you with information about the person you are asking about. Please let me know what you would like me to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It is a major financial and cultural center, and one of the most visited cities in the world. It is also a significant tourist destination, with its famous landmarks, shopping, and dining options attracting millions of visitors annually. Paris has a rich history and is home to many renowned art and cultural institutions. The city is also known for its diverse cuisine and French culture. Overall, Paris is a fascinating city with a rich history, culture, and architecture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by a rapid and significant shift in technology and application areas. Here are some possible trends in AI that are currently being explored and may continue to evolve:
    
    1. Increased Use of AI in Healthcare: As AI becomes more accessible and affordable, it is likely to be used in healthcare to improve diagnostic accuracy, streamline patient care, and provide personalized treatment plans. This could include the use of AI-powered tools to analyze medical images, predict patient outcomes, and develop new treatments.
    
    2. Augmented Reality and Virtual Reality: These technologies have already been used in education, gaming, and entertainment, but their potential applications in healthcare are likely


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

     Jane

     and

     I

    'm

     a

     freelance

     writer

     who

     specializes

     in

     creative

     writing

    .

     I

    'm

     also

     a

     marathon

     runner

     and

     a

     long

    -distance

     swim

    mer

    .

     I

     enjoy

     exploring

     the

     outdoors

     and

     taking

     long

     walks

     in

     nature

    .

     I

    'm

     also

     a

     passionate

     advocate

     for

     environmental

     conservation

     and

     love

     to

     volunteer

     at

     local

     animal

     shelters

    .

     I

    'm

     always

     looking

     to

     learn

     new

     skills

     and

     keep

     my

     writing

     style

     fresh

     and

     innovative

    .

     What

     other

     hobbies

     or

     interests

     do

     you

     have

     besides

     writing

    ?

     Jane

    ,

     you

    're

     a

     unique

     and

     inspiring

     person

     with

     a

     passion

     for

     writing

     and

     nature

    .

     You

     also

     love

     volunteering

     at

     animal

     shelters

     and

     are

     passionate

     about

     environmental

     conservation

    .

     What

     kind

     of

     writing

     you

     do

    ,

     and

     what

     kind

     of

     people

     do

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

    ,

     Paris

    ,

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     historic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     events

     and

     festivals

    ,

     including

     the

     World

     Cup

     and

     the

     E

    iff

    el

     Tower

     Race

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     economic

     center

    ,

     with

     a

     rich

     history

     dating

     back

     to

     the

     

    1

    1

    th

     century

    .

     Its

     reputation

     for

     being

     a

     free

     city

     and

     a

     tourist

     haven

     is

     further

     celebrated

     in

     its

     name

    .

     The

     French

     capital

     is

     a

     city

     with

     a

     unique

     blend

     of

     old

     and

     new

    ,

     cultural

     richness

    ,

     and

     a

     sense

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     trends

     and

     advancements

     that

     are

     currently

     taking

     place

    ,

     including

    :
    


    1

    .

     Increased

     accuracy

     and

     reliability

    :

     AI

     systems

     are

     becoming

     more

     sophisticated

     and

     are

     becoming

     increasingly

     accurate

     in

     their

     predictions

     and

     decision

    -making

    .

     This

     is

     driven

     by

     advancements

     in

     machine

     learning

     algorithms

     that

     can

     be

     trained

     on

     vast

     amounts

     of

     data

     to

     improve

     their

     ability

     to

     understand

     and

     make

     predictions

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     As

     AI

     systems

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

     data

     analysis

     tools

    ,

     they

     are

     becoming

     more

     capable

     of

     detecting

     and

     analyzing

     patterns

     in

     the

     real

     world

    .

     This

     integration

     will

     allow

     for

     more

     comprehensive

     and

     accurate

     decision

    -making

    .
    


    3

    .

     Personal

    



```python
llm.shutdown()
```
