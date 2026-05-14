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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.21it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.19it/s]


    2026-05-14 07:28:39,947 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 07:28:39] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.10it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.36it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.36it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]

    Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.38it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 26.05it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]

    Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.82it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 16.47it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=73.50 GB):   7%|▋         | 4/58 [00:00<00:03, 17.50it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.22 GB):   7%|▋         | 4/58 [00:00<00:03, 17.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.20 GB):   7%|▋         | 4/58 [00:00<00:03, 17.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.50 GB):   7%|▋         | 4/58 [00:00<00:03, 17.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.84it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.50 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.50 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.49 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.49 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.48 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.48 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.48 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.48 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.98it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.48 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.47 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.47 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.45 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=960 avail_mem=72.46 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.11it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.46 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=832 avail_mem=72.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=768 avail_mem=72.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.11it/s]Capturing num tokens (num_tokens=768 avail_mem=72.45 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=704 avail_mem=72.45 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=640 avail_mem=72.02 GB):  43%|████▎     | 25/58 [00:00<00:01, 30.40it/s]Capturing num tokens (num_tokens=576 avail_mem=72.02 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.40it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.40it/s]

    Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  50%|█████     | 29/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=480 avail_mem=72.02 GB):  50%|█████     | 29/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=448 avail_mem=72.02 GB):  50%|█████     | 29/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  50%|█████     | 29/58 [00:01<00:00, 31.44it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.34it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.34it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.34it/s]Capturing num tokens (num_tokens=288 avail_mem=72.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.34it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  57%|█████▋    | 33/58 [00:01<00:00, 32.34it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.08it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.08it/s]Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.08it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.08it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.08it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  71%|███████   | 41/58 [00:01<00:00, 33.91it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  71%|███████   | 41/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=96 avail_mem=71.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.29it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=28 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 31.87it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.95it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.95it/s]

    Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.95it/s] Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 31.25it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  98%|█████████▊| 57/58 [00:01<00:00, 31.25it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 29.63it/s]


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
    Generated text:  Elisa and I am a doctor specializing in the treatment of cancer. I specialize in the treatment of breast cancer and lymphoma. I am board certified by the American Board of Oncology and the American Board of Radiation Oncology. I have been practicing medicine since 1996 and am trained in the treatment of all types of cancer, including breast cancer. I have been working in the medical field for more than 20 years. Please feel free to call me at (608) 232-8747 or e-mail me at elisa@gale.org.
    Cancer is the development and
    ===============================
    Prompt: The president of the United States is
    Generated text:  a powerful person. He or she has great power and influence over the people of the United States. But, the president has to have a lot of work to do. He or she has to make important decisions. The president can make these decisions by himself or by another person called the secretary of state. The president can also make these decisions by themselves. A lot of important decisions about the government of the United States are made by the president, and then it is up to the secretary of state to follow up. The secretary of state has to check to see if the president made the right decision. If he or she thinks the president made
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. London
    C. Beijing
    D. Moscow
    答案：A. Paris
    Explanation: Paris is the capital of France. It is located in the northwestern region of France, on the banks of the River Seine, and is known as the "City of Light" due to its status as a major cultural center. The other options listed are not capitals of France; London is the capital of the United Kingdom, Beijing is the capital of China, and Moscow is the capital of Russia.
    Therefore, the correct answer is A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  diverse, complex, and immersive. It is rapidly transforming the world around us and creating new opportunities for innovation. The future of AI will require continued investment, innovation, and collaboration among all stakeholders to fully realize the potential of this transformative technology. The field of AI is growing rapidly and will continue to evolve and adapt to new challenges and opportunities. In order to fully harness the potential of AI, it is essential to develop a culture of innovation and creativity among all stakeholders, as well as to prioritize ethical and social considerations throughout the development and implementation of AI technologies. The future of AI is bright, and it is poised to have a significant impact on


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm a [insert a short, positive, enthusiastic statement about your personality or skills]. I'm always looking for new challenges and opportunities to grow and learn. What do you enjoy doing? I enjoy [insert a short, positive, enthusiastic statement about your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville Fluviale" and "La Ville Blanche". It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, art, and culture, as well as its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The city is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a popular tourist destination and a major economic center in France. It is also known for its cuisine, including its famous French fries and its traditional
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to the behavior and preferences of humans.
    
    2. Enhanced ethical considerations: As AI becomes more advanced, there will be increased scrutiny of its ethical implications, including issues such as bias, transparency, and accountability.
    
    3. Development of new AI technologies: AI will continue to develop new technologies, such as quantum computing, nanotechnology, and biotechnology, which could have significant implications for AI
    


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
    Generated text:  [Name], and I come from [Country]. I'm currently [Age] years old, [Gender] and [Major Profession], and I'm a passionate [Interest] who always looks for [Challenge or Reward].
    
    In my spare time, I enjoy [Other Hobby]. I'm always on the lookout for the next big [Opportunity] and [New Challenge] that could help me grow and learn more. If you're interested in meeting me or talking about my experiences, I'd be more than happy to have a conversation. [Call or email] to my contact details. [Note: It's important to be respectful and professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Note: This statement is correct according to official French government data. However, as of now, the city is undergoing a city-wide overhaul and some of its landmarks and attractions may not be open to the public, so please check the official city websites for current information. As of 2023, Paris is home to several notable landmarks and attractions, including the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, Montmartre, and the Champs-Elysées. The city is also known for its stunning natural beauty, including the Eiffel Tower, the Acropolis, and the Tournai
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be driven by new technologies and innovations that will transform the way we live, work, and interact with each other. Some possible future trends in AI include:
    
    1. Increased automation and AI-driven automation: As automation continues to become more pervasive, AI will play an increasingly important role in automating repetitive and mundane tasks. This could lead to a reduction in the need for human labor, which could create new job opportunities in areas such as data analysis, robotics, and software development.
    
    2. Enhanced AI-powered education: With the rise of AI-powered education, we may see more personalized learning experiences that use artificial intelligence to tailor learning materials to


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

    Occup

    ation

    ]

     with

     [

    Age

    ]

     years

     old

    .

     I

     have

     a

     passion

     for

     [

    Your

     passion

    ],

     which

     drives

     my

     work

     and

     makes

     me

     unique

    .

     I

     thrive

     on

     [

    Your

     passion

    ]

     and

     I

    'm

     always

     learning

     and

     growing

    .

     I

     love

     to

     travel

    ,

     cook

    ,

     read

     books

    ,

     and

     spend

     time

     with

     family

     and

     friends

    .

     I

     believe

     that

     there

    's

     beauty

     in

     every

     experience

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     make

     people

    's

     lives

     better

    .

     I

    'm

     excited

     to

     learn

     more

     about

     [

    Your

     name

    ]

     and

     work

     with

     you

     to

     create

     something

     amazing

    .

     Your

     name

     is

     [

    Name

    ].

     Let

    's

     collaborate

     on

     creating

     something

     amazing

     together

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    [

    Mark

     down

    ]


    ```

    markdown

    


    -

     The

     capital

     of

     France

     is

     Paris

    .


    ```

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     landscape

     with

     numerous

     potential

     future

     trends

     that

     could

     shape

     how

     we

     interact

     with

     technology

     and

     progress

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     may

     come

     to

     fruition

     in

     the

     next

     few

     years

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     already

     increasingly

     being

     used

     in

     manufacturing

     and

     supply

     chain

     management

    ,

     but

     it

    's

     likely

     to

     continue

     to

     grow

     in

     importance

     as

     more

     tasks

     are

     automated

    .

     This

     could

     lead

     to

     increased

     efficiency

     and

     productivity

    ,

     but

     it

     could

     also

     lead

     to

     job

     losses

     in

     certain

     industries

     as

     automation

     takes

     over

     more

     manual

     tasks

    .
    


    2

    .

     Personal

    ized

     AI

    :

     As

     AI

     is

     trained

     on

     large

     datasets

    ,

     it

     can

     become

     increasingly

     effective

     at

     predicting

     behavior

     and

     identifying

     patterns

     that

     may

     not

    



```python
llm.shutdown()
```
