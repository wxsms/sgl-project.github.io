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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.73it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:55,  4.13s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.84it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]

    Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:08,  5.61it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:04,  9.04it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:04,  9.04it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]

    Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 14.05it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 20.92it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 29.73it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 39.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 51.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.34 GB):   3%|▎         | 2/58 [00:00<00:03, 16.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.02 GB):   3%|▎         | 2/58 [00:00<00:03, 16.64it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.05 GB):   3%|▎         | 2/58 [00:00<00:03, 16.64it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.05 GB):   7%|▋         | 4/58 [00:00<00:03, 13.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.06 GB):   7%|▋         | 4/58 [00:00<00:03, 13.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.07 GB):   7%|▋         | 4/58 [00:00<00:03, 13.73it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=72.07 GB):  10%|█         | 6/58 [00:00<00:03, 14.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.08 GB):  10%|█         | 6/58 [00:00<00:03, 14.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.08 GB):  10%|█         | 6/58 [00:00<00:03, 14.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.08 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.09 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.09 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.24it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.12 GB):  14%|█▍        | 8/58 [00:00<00:03, 15.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.11 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.12 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.08it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.12 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.11 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.20 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.21 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.36it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  29%|██▉       | 17/58 [00:00<00:01, 21.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.19 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.42it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.17 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.42it/s]Capturing num tokens (num_tokens=960 avail_mem=72.18 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.42it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.16 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.42it/s]Capturing num tokens (num_tokens=832 avail_mem=72.17 GB):  34%|███▍      | 20/58 [00:01<00:01, 22.42it/s]Capturing num tokens (num_tokens=832 avail_mem=72.17 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=768 avail_mem=72.16 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=704 avail_mem=72.16 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=640 avail_mem=72.15 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=576 avail_mem=72.15 GB):  41%|████▏     | 24/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=576 avail_mem=72.15 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=512 avail_mem=72.13 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.48it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=448 avail_mem=72.13 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=416 avail_mem=72.12 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=416 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=384 avail_mem=72.14 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=352 avail_mem=72.13 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=320 avail_mem=72.12 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.21it/s]Capturing num tokens (num_tokens=288 avail_mem=72.11 GB):  55%|█████▌    | 32/58 [00:01<00:00, 29.21it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=256 avail_mem=72.11 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=240 avail_mem=72.10 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=224 avail_mem=72.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=208 avail_mem=72.09 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=192 avail_mem=72.07 GB):  62%|██████▏   | 36/58 [00:01<00:00, 30.24it/s]Capturing num tokens (num_tokens=192 avail_mem=72.07 GB):  71%|███████   | 41/58 [00:01<00:00, 31.35it/s]Capturing num tokens (num_tokens=176 avail_mem=72.08 GB):  71%|███████   | 41/58 [00:01<00:00, 31.35it/s]Capturing num tokens (num_tokens=160 avail_mem=72.07 GB):  71%|███████   | 41/58 [00:01<00:00, 31.35it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.07 GB):  71%|███████   | 41/58 [00:01<00:00, 31.35it/s]Capturing num tokens (num_tokens=128 avail_mem=72.06 GB):  71%|███████   | 41/58 [00:01<00:00, 31.35it/s]Capturing num tokens (num_tokens=128 avail_mem=72.06 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=112 avail_mem=72.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=96 avail_mem=72.05 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.86it/s] Capturing num tokens (num_tokens=80 avail_mem=72.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.86it/s]Capturing num tokens (num_tokens=64 avail_mem=72.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.86it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=48 avail_mem=72.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=32 avail_mem=72.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=28 avail_mem=72.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.58it/s]Capturing num tokens (num_tokens=24 avail_mem=72.01 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.58it/s]Capturing num tokens (num_tokens=20 avail_mem=72.00 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.58it/s]Capturing num tokens (num_tokens=20 avail_mem=72.00 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.81it/s]Capturing num tokens (num_tokens=16 avail_mem=71.99 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.81it/s]Capturing num tokens (num_tokens=12 avail_mem=71.98 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.81it/s]Capturing num tokens (num_tokens=8 avail_mem=71.98 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.81it/s] Capturing num tokens (num_tokens=4 avail_mem=71.97 GB):  93%|█████████▎| 54/58 [00:02<00:00, 33.81it/s]

    Capturing num tokens (num_tokens=4 avail_mem=71.97 GB): 100%|██████████| 58/58 [00:02<00:00, 26.91it/s]


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
    Generated text:  Lina and I'm a software developer who enjoys helping people understand computer science and programming.
    I am passionate about helping people learn and share knowledge, and I believe that programming is a way to make the world a better place.
    I enjoy learning new programming languages and technologies, and I believe that learning new things can help people be more creative and independent.
    I am always open to learning from people, and I enjoy helping people find solutions to problems they are facing.
    I am excited to see where this journey will take me in the future and to continue to learn and grow as a programmer. Have a wonderful day! 
    What is the answer to
    ===============================
    Prompt: The president of the United States is
    Generated text:  a representative of the majority of the countries in the world. Most of the world's countries have strong diplomatic ties with the United States, and some countries, such as Japan, South Korea, and Australia, have also become important partners of the United States. As the representative of the world, the president of the United States is the head of state and the head of government of the United States. He/she is the supreme executive power of the United States.
    
    The president of the United States has the power to make decisions regarding the defense of the United States. For example, the president can make decisions regarding the military defense, the establishment of the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Marseille
    D. Nice
    A. Paris is the capital of France. It is known for its historical significance and as the seat of the French government and the country's political, cultural, and economic center. Paris is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Lyon is in the south of France, Marseille is in the north, and Nice is in the southwest. The other cities listed are located in other regions of France.
    ===============================
    Prompt: The future of AI is
    Generated text:  in the air, and many experts believe the technology will continue to advance at an unprecedented rate. However, as with any new technology, there are some risks that must be considered before fully embracing it. One of the most significant risks that comes with AI is the risk of automation.
    Automation, or the process of machines performing tasks that were previously done by humans, has become increasingly prevalent in recent years. In fact, the AI industry has grown by over 40% in the last five years, and experts predict that it will continue to grow even further in the coming years.
    While automation has many benefits, it can also pose significant risks


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also home to many famous French artists and writers, including Pablo Picasso and André Breton. Paris is a vibrant and dynamic city with a rich history and a strong sense of French identity. The city is also known for its diverse cuisine, including French cuisine, as well as its wine and wine culture. Overall
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI technology continues to improve, we can expect to see even greater use of AI in healthcare, with more personalized and efficient treatments.
    
    3. Greater use of AI in finance: AI is already being used in finance to improve risk management, fraud detection,
    


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
    Generated text:  [Your Name] and I am [Your Age]. I am a [Your Profession], and I am [Your Specialty or Specialization]. [Your Interest or Hobby] is a [Your Hobby]. I am passionate about [Your Passion]. I am always looking to learn new things and improve myself. My hobbies include [Your Hobbies]. I am constantly striving to expand my knowledge and improve my skills. I am a [Your Goal or Goal]. I am a [Your Passion]. I am a [Your Specialty or Specialization]. I am [Your Hobbies]. I am passionate about [Your Passion]. I am always looking to learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the second-largest in Europe, located on the River Seine in the central business district of Paris. It is home to the headquarters of the European Union and the European Parliament. Paris is known for its rich history, vibrant culture, and beautiful architecture, including the Eiffel Tower and the Louvre Museum. The city is also the economic and political center of France and plays an important role in the country's development and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a highly dynamic and evolving landscape. Here are some possible trends that may influence the development of AI in the coming years:
    
    1. Increased integration with human-AI interaction: AI is likely to become more integrated with human-AI interaction in the future. This could involve more natural language processing, better understanding of human emotions and motivations, and more sophisticated machine learning algorithms that can adapt to the nuances of human-AI interactions.
    
    2. Emphasis on privacy and security: As AI becomes more pervasive, there is a risk of it being used for malicious purposes. Governments and organizations will need to develop more effective privacy and security measures to


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

     a

     [

    type

     of

     character

    ,

     e

    .g

    .

     superhero

    ,

     science

     fiction

     character

    ,

     etc

    .

    ].

     I

     am

     [

    characters

     name

    ],

     but

     I

     am

     also

     [

    character

    's

     name

    ]

     who

     is

     [

    character

    's

     name

    's

     title

    ].

     I

     am

     a

     [

    character

    's

     name

    ]

     who

     [

    their

     name

    's

     job

    ,

     hobby

    ,

     or

     contribution

     to

     society

    ].

     I

     am

     a

     [

    character

    's

     name

    ]

     who

     is

     [

    their

     name

    's

     profession

    ,

     skills

    ,

     or

     talents

    ].

     I

     am

     a

     [

    character

    's

     name

    ]

     who

     [

    their

     name

    's

     role

     in

     the

     story

    ].

     I

     am

     a

     [

    character

    's

     name

    ]

     who

     is

     [

    their

     name

    's

     description

    ].

     I

     am

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

    ,

     known

     for

     its

     historical

     architecture

    ,

     arts

     and

     museums

    ,

     and

     world

    -ren

    owned

     cuisine

    .

     The

     city

     has

     a

     rich

     cultural

     heritage

     and

     is

     home

     to

     many

     cultural

     institutions

     such

     as

     the

     Lou

    vre

     and

     the

     Palace

     of

     Vers

    ailles

    .

     Paris

     is

     also

     a

     major

     financial

     center

     and

     a

     major

     trading

     hub

    .

     Its

     architecture

     and

     lifestyle

     reflect

     a

     blend

     of

     ne

    oc

    lass

    ical

     and

     bar

    oque

     styles

    .

     It

     is

     known

     for

     its

     annual

     annual

     summer

     festival

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     iconic

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     home

     to

     numerous

     festivals

    ,

     including

     the

     Bast

    ille

     Day

     celebrations

    ,

     and

     has

     a

     diverse

     population

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     with

     many

     potential

     areas

     of

     growth

     and

     development

    .

     Here

     are

     some

     potential

     trends

    :
    


    1

    .

     Increased

     AI

     transparency

    :

     One

     of

     the

     biggest

     challenges

     in

     AI

     is

     creating

     systems

     that

     are

     understandable

     and

     transparent

     to

     humans

    .

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     we

     may

     see

     greater

     emphasis

     on

     creating

     more

     transparent

     algorithms

    .
    


    2

    .

     AI

     for

     healthcare

    :

     With

     the

     growing

     need

     for

     personalized

     medicine

    ,

     AI

     is

     expected

     to

     play

     a

     bigger

     role

     in

     healthcare

    .

     AI

     can

     be

     used

     to

     analyze

     large

     amounts

     of

     medical

     data

     to

     identify

     patterns

     and

     predict

     outcomes

    ,

     which

     could

     lead

     to

     more

     accurate

     diagnoses

     and

     treatment

     plans

    .
    


    3

    .

     AI

     for

     education

    :

     AI

     is

     being

     used

     in

     education

     to

     personalize

     learning

    



```python
llm.shutdown()
```
