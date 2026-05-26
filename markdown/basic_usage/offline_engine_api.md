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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.07s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]

    Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:43,  1.25it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:14,  3.48it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:06,  6.94it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]

    Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 18.28it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 24.62it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:04<00:00, 32.29it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 38.71it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 38.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=51.52 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.67 GB):   2%|▏         | 1/58 [00:00<00:06,  8.19it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.67 GB):   3%|▎         | 2/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.67 GB):   3%|▎         | 2/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.66 GB):   3%|▎         | 2/58 [00:00<00:06,  8.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.66 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.66 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.66 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=50.65 GB):   7%|▋         | 4/58 [00:00<00:04, 12.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.65 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.64 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.98it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=50.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.63 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.62 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.43it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=50.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.61 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.59 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=960 avail_mem=50.60 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.29it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=50.60 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.29it/s]Capturing num tokens (num_tokens=896 avail_mem=50.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.15it/s]Capturing num tokens (num_tokens=832 avail_mem=50.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.15it/s]Capturing num tokens (num_tokens=768 avail_mem=50.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.15it/s]Capturing num tokens (num_tokens=704 avail_mem=50.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 23.15it/s]Capturing num tokens (num_tokens=704 avail_mem=50.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.39it/s]Capturing num tokens (num_tokens=640 avail_mem=50.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.39it/s]Capturing num tokens (num_tokens=576 avail_mem=50.59 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.39it/s]

    Capturing num tokens (num_tokens=512 avail_mem=50.57 GB):  45%|████▍     | 26/58 [00:01<00:01, 24.39it/s]Capturing num tokens (num_tokens=512 avail_mem=50.57 GB):  50%|█████     | 29/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=480 avail_mem=50.59 GB):  50%|█████     | 29/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=448 avail_mem=50.59 GB):  50%|█████     | 29/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=416 avail_mem=50.58 GB):  50%|█████     | 29/58 [00:01<00:01, 25.27it/s]Capturing num tokens (num_tokens=416 avail_mem=50.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=384 avail_mem=50.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=352 avail_mem=50.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]

    Capturing num tokens (num_tokens=320 avail_mem=50.57 GB):  55%|█████▌    | 32/58 [00:01<00:00, 26.07it/s]Capturing num tokens (num_tokens=320 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=288 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=256 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=240 avail_mem=50.56 GB):  60%|██████    | 35/58 [00:01<00:00, 26.65it/s]Capturing num tokens (num_tokens=240 avail_mem=50.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=224 avail_mem=50.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=208 avail_mem=50.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.75it/s]

    Capturing num tokens (num_tokens=192 avail_mem=50.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=176 avail_mem=50.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 26.75it/s]Capturing num tokens (num_tokens=176 avail_mem=50.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=160 avail_mem=50.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=144 avail_mem=50.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=128 avail_mem=50.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=112 avail_mem=50.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=112 avail_mem=50.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=96 avail_mem=50.54 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.95it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=50.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=64 avail_mem=50.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=48 avail_mem=50.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=48 avail_mem=50.53 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.56it/s]Capturing num tokens (num_tokens=32 avail_mem=50.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.56it/s]Capturing num tokens (num_tokens=28 avail_mem=50.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.56it/s]Capturing num tokens (num_tokens=24 avail_mem=50.52 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.56it/s]Capturing num tokens (num_tokens=20 avail_mem=50.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 31.56it/s]Capturing num tokens (num_tokens=20 avail_mem=50.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.09it/s]Capturing num tokens (num_tokens=16 avail_mem=50.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.09it/s]

    Capturing num tokens (num_tokens=12 avail_mem=50.51 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.09it/s]Capturing num tokens (num_tokens=8 avail_mem=50.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.09it/s] Capturing num tokens (num_tokens=4 avail_mem=50.50 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.09it/s]Capturing num tokens (num_tokens=4 avail_mem=50.50 GB): 100%|██████████| 58/58 [00:02<00:00, 34.48it/s]Capturing num tokens (num_tokens=4 avail_mem=50.50 GB): 100%|██████████| 58/58 [00:02<00:00, 25.89it/s]


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
    Generated text:  Katherine. I'm a 12-year-old girl who loves playing sports and being active. I'm an avid runner and am passionate about practicing yoga. I'm also into spending time outdoors, such as hiking and camping, and love to explore new places. I have a passion for sustainability and hope to incorporate some eco-friendly practices into my lifestyle.
    Can you please tell me about your favorite sport and why you like it so much? As a 12-year-old girl who loves sports, my favorite sport is soccer. I enjoy it because it's a fast-paced, exciting game that requires a lot of teamwork and skill. The goal
    ===============================
    Prompt: The president of the United States is
    Generated text:  a major figure in the United States. He or she is the head of government of the United States and is the head of the executive branch of the federal government. The president is also the commander-in-chief of the armed forces of the United States. The president is the head of the executive branch of the U.S. government. This office is different from the President of the United States, who is the head of the federal government. It is the office of the U.S. president.
    The United States has been ruled by president since 1787. The first president was Thomas Jefferson. The president has been called a political and
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population is 1, 077, 033. There are currently 181 schools, 183 hospitals, and 515 universities. In addition, the capital has 15, 019 primary schools, 1, 534 secondary schools, and 4, 018 vocational schools. On average, each school contains 60 teachers, each hospital has 85 beds, and each university has 300 students. 
    
    The city has the second largest number of roads in Europe, with 1, 676
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of developers, and they have a wide range of tools and technologies at their disposal to help them achieve that vision. Here are some of the most popular and innovative tools and techniques used by developers today:
    
    1. Natural Language Processing (NLP): This is a crucial tool used by developers to understand the context of natural language. NLP techniques such as sentiment analysis, topic modeling, and named entity recognition allow developers to understand and interpret human language in a more sophisticated way.
    
    2. Computer Vision: This is a subfield of computer science that focuses on the process of converting images into data that can be analyzed by a computer.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] new things. I'm a [job title] who is always looking for ways to [job title] new things. I'm a [job title] who is always looking for ways to [job title] new things. I'm a [job title] who is always looking for ways to [job title] new things. I'm a [job title
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. 
    
    This statement encapsulates the main attractions and cultural landmarks of Paris, highlighting its significance as the capital city of France. 
    
    To further elaborate, Paris is the largest city in France and the third-largest city in the European Union. It is also the capital of the French Department of Paris, which includes the Île de la Cité, the Louvre Museum, and the Eiffel Tower. The city is home to numerous museums, theaters, and landmarks, making it a popular tourist destination. 
    
    Paris is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as
    


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
    Generated text:  [Name] and I'm a [brief introduction, such as "Engineer" or "Doctor"]. I specialize in [specific field of expertise, such as "Mechanical Engineering" or "Psychology"]. I have [number of years of experience, such as "8", "15", "30"]. My approach to work is always [positive adjective or phrase]. I believe in [value proposition or mission statement], which is [insert what you like about your job or profession]. I am always learning and growing, and I am always looking for new opportunities to contribute to the world. I strive to make a positive impact on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its medieval architecture, fashionable clothing, and rich cultural heritage. The city is also home to several world-renowned museums, such as the Louvre, the National Museum of Modern Art, and the Musée d'Orsay. Paris is a major transportation hub, with numerous airports and train stations, and is a popular tourist destination, drawing millions of visitors each year. Despite its modernity, Paris remains a lively and vibrant city with a rich cultural and artistic heritage. What is your favorite place to visit in France and why?
    As an AI language model, I don't have personal preferences, but I can
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of trends and developments, including:
    
      1. Increased availability and affordability of AI software and hardware: As AI technology becomes more prevalent in various industries, the cost of developing and deploying AI systems will likely decrease, making it more accessible to businesses and individuals.
      2. Improved AI algorithms and models: As AI research advances, it will become increasingly capable of learning from data and making more accurate predictions and decisions.
      3. Enhanced ethical considerations: As AI systems become more integrated into society, there will be increasing pressure to address ethical concerns, such as bias, privacy, and accountability.
     


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

    ]

     and

     I

     am

     a

     [

    insert

     profession

    ]

     who

     has

     a

     deep

     understanding

     of

     the

     diverse

     communities

     and

     cultures

     that

     make

     up

     the

     world

    .

     I

     am

     always

     ready

     to

     learn

     from

     others

     and

     contribute

     to

     the

     growth

     of

     communities

    ,

     be

     it

     through

     my

     work

     in

     [

    insert

     profession

    ],

     my

     time

     volunteering

    ,

     or

     simply

     by

     simply

     being

     here

    .

     I

     am

     a

     true

     friend

     and

     ally

     to

     those

     I

     meet

     and

     a

     tire

    less

     advocate

     for

     equality

    ,

     justice

    ,

     and

     fairness

    .

     Thank

     you

     for

     having

     me

    ,

     [

    insert

     title

     of

     role

    ].

     
    


    Note

    :

     The

     placeholders

     in

     the

     above

     text

     should

     be

     replaced

     with

     actual

     names

    ,

     titles

    ,

     or

     other

     appropriate

     information

     that

     fits

     the

     persona

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     picturesque

     streets

    ,

     historical

     sites

    ,

     and

     world

    -ren

    owned

     museums

    .

     It

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

     and

     the

     Palace

     of

     Vers

    ailles

    ,

     and

     is

     also

     a

     major

     center

     for

     the

     arts

     and

     sciences

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     rich

     cultural

     heritage

     and

     a

     thriving

     economy

    .

     The

     city

     offers

     a

     diverse

     range

     of

     food

     options

    ,

     as

     well

     as

     access

     to

     top

     international

     destinations

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     an

     important

     cultural

     and

     economic

     center

     in

     France

    .

     The

     city

     has

     a

     history

     dating

     back

     to

     the

     

    6

    th

     century

     BCE

     and

     is

     home

     to

     many

     historical

     landmarks

    ,

     including

     the

     Ch

    âte

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     continue

     to

     evolve

     rapidly

    ,

     with

     new

     applications

     and

     advancements

     emerging

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

     are

     currently

     being

     explored

    :
    


    1

    .

     Increased

     automation

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increase

     of

     automation

    ,

     which

     will

     lead

     to

     the

     development

     of

     more

     efficient

    ,

     precise

    ,

     and

     scalable

     AI

     systems

    .
    


    2

    .

     AI

     ethics

     and

     safety

    :

     There

     is

     growing

     recognition

     of

     the

     importance

     of

     ensuring

     that

     AI

     systems

     are

     developed

     and

     deployed

     in

     a

     way

     that

     respects

     human

     values

     and

     ethical

     principles

    .

     This

     includes

     the

     development

     of

     ethical

     guidelines

     for

     AI

     systems

     and

     the

     implementation

     of

     safeguards

     to

     prevent

     negative

     consequences

    .
    


    3

    .

     AI

     for

     healthcare

    :

     AI

     has

     the

    



```python
llm.shutdown()
```
