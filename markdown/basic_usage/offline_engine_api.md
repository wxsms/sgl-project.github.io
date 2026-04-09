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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.32it/s]


    2026-04-09 23:27:30,164 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 23:27:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:26,  2.57s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.95it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.36it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.62it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.30it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.64it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.08 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s] Capturing num tokens (num_tokens=896 avail_mem=72.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=832 avail_mem=72.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]

    Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.65it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=704 avail_mem=72.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=640 avail_mem=72.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=576 avail_mem=72.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=480 avail_mem=72.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.33it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.99 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.20it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.02it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 48.02it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]

    Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.53it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=20 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.10it/s]Capturing num tokens (num_tokens=16 avail_mem=71.64 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.02it/s]Capturing num tokens (num_tokens=12 avail_mem=71.64 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.02it/s]Capturing num tokens (num_tokens=8 avail_mem=71.64 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.02it/s] Capturing num tokens (num_tokens=4 avail_mem=71.63 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.02it/s]Capturing num tokens (num_tokens=4 avail_mem=71.63 GB): 100%|██████████| 58/58 [00:01<00:00, 39.06it/s]


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
    Generated text:  Karen and I am a Computer Science student. I am also an expert in English language and writing, as well as a person who has a passion for learning new things. I am here to share my knowledge with others.
    Do you have any favorite hobby?
    As an AI language model, I don't have the ability to have hobbies like humans do. However, I can offer insights on different interests based on the knowledge I have been trained on. Is there something specific you would like to know about English language and writing? I would be happy to help! Is there anything in particular you want to discuss? Let me know and I will do
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, but there are many other important people who are also very important. To see who is the most important, you can look at how much money they make. People in the business world do not make much money, and people in the government do not make much money either. In fact, people in business and government have not made as much as they should. What the president of the United States does have a lot of money, and it's because of how much they earn from the government. One way the president makes money is from oil. The president earns money from the government to help pay for the oil that he
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A) Paris
    B) Lausanne
    C) Geneva
    D) Zurich
    A) Paris
    
    Paris is the capital of France, located in the central region of the country in Western Europe. It is the most populous city in France and the second-largest city in the European Union after Brussels. The city is known for its unique architectural style, historical landmarks, and vibrant culture. While Lausanne, Geneva, and Zurich are all cities in Switzerland, they are not capitals of France. Zurich is the capital of the Canton of Zurich, Switzerland. Geneva is the capital of the canton of Geneva, Switzerland. Laus
    ===============================
    Prompt: The future of AI is
    Generated text:  about finding the right balance between human empathy and machine learning. It is essential to embrace both, as it is the future of the industry. However, there is a risk that, if we do not manage it, the future of the industry may be lost.
    The future of AI is about finding the right balance between human empathy and machine learning. It is essential to embrace both, as it is the future of the industry.
    In this article, we will explore the challenges and opportunities that AI presents, and how to harness the potential of AI to create a more empathetic and connected world.
    AI has revolutionized many industries, from healthcare to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have a [major] degree in [field of study]. I'm a [occupation] and I enjoy [job-related hobby or interest]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or interest? I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or interest? I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is also known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to many international organizations and organizations, including the United Nations. Paris is a vibrant and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be an increased need for privacy and security measures to protect user data. This could lead to the development of new technologies and protocols that are
    


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
    Generated text:  [Name] and I am a [Professional/Personal title] at [Company name]. I have always been passionate about [Job title] and have worked in the [Industry] for [Number of years] years. Currently, I am [Current position] in [Company name]. I am [Age] years old and have [Height/Weight] in [Height/Weight unit]. I like to spend my free time [Affection for something]. I have a [Friendship] with [Friend's name] and often [We do something together]. I am [Age] years old. My hobbies include [Hobbies/Activities
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Is this statement correct on its face or is there any factual inaccuracies or additional information that would need to be considered? 
    
    Please provide a detailed analysis of the accuracy of the statement and any relevant sources or additional information that could be used to verify or refute it. Additionally, please consider the possible consequences of any inaccuracies or misleading statements, such as the potential impact on tourism, political stability, and public education. To make the statement more comprehensive, please provide additional context about Paris's cultural, economic, and political importance to France and its global role. Lastly, please describe the physical layout of Paris, including its major landmarks
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with new technologies and applications constantly emerging. Here are some possible future trends in AI:
    
    1. Increased integration with human beings: AI is already being integrated into our lives in many ways, from smart home devices to self-driving cars. As more and more human beings begin to interact with AI, we can expect to see a continued integration of AI with our physical environment.
    
    2. Greater customization and personalization: As AI systems become more sophisticated, we can expect to see a rise in personalized and customized AI solutions. These solutions could be tailored to meet the unique needs of individual customers, such as their preferences, behavior, and data


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

    ],

     and

     I

     am

     a

     creative

     and

     intuitive

     thinker

    .

     I

     love

     to

     explore

     new

     ideas

    ,

     learn

     new

     skills

    ,

     and

     make

     decisions

     that

     are

     driven

     by

     the

     greater

     good

    .

     I

     am

     a

     problem

     solver

     who

     is

     always

     willing

     to

     take

     on

     challenges

     and

     strive

     for

     excellence

     in

     everything

     I

     do

    .

     I

     am

     also

     a

     good

     listener

     and

     communicator

    ,

     able

     to

     connect

     with

     people

     on

     a

     deep

     level

     and

     understand

     their

     perspectives

     and

     needs

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     hiking

    ,

     painting

    ,

     and

     spending

     time

     with

     my

     friends

     and

     family

    .

     I

     am

     eager

     to

     see

     where

     my

     talents

     and

     abilities

     can

     take

     me

     in

     the

     future

    .

     How

     would

     you

     describe

     yourself

    ?


    As

     a

     creative

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Does

     this

     next

     sentence

     follow

    ,

     given

     the

     preceding

     text

    ?

     Paris

     is

     the

     capital

     of

     France

    .
    


    Choose

     your

     answer

     from

    :

     (

    i

    )

     yes

     (

    ii

    )

     no

    


    (i

    )

     yes

    
    


    The

     sentence

     "

    Paris

     is

     the

     capital

     of

     France

    "

     does

     follow

     from

     the

     given

     information

    ,

     as

     it

     is

     explicitly

     stated

     that

     Paris

     is

     the

     capital

     of

     France

     in

     the

     first

     sentence

    .

     Therefore

    ,

     the

     answer

     is

     (

    i

    )

     yes

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     und

    eni

    ably

     exciting

     and

     dynamic

    ,

     and

     it

    's

     impossible

     to

     predict

     with

     certainty

     what

     its

     exact

     path

     will

     take

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

     Machine

     learning

     and

     deep

     learning

    :

     These

     technologies

     are

     already

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     such

     as

     image

     and

     speech

     recognition

    ,

     natural

     language

     processing

    ,

     and

     predictive

     analytics

    .

     In

     the

     future

    ,

     we

     can

     expect

     to

     see

     more

     advanced

     versions

     of

     these

     technologies

    ,

     as

     well

     as

     more

     sophisticated

     ways

     of

     training

     and

     tuning

     them

    .
    


    2

    .

     Internet

     of

     Things

     (

    Io

    T

    ):

     The

     Internet

     of

     Things

     will

     continue

     to

     play

     an

     important

     role

     in

     shaping

     the

     future

     of

     AI

    .

     With

     the

     increasing

     interconnected

    ness

     of

    



```python
llm.shutdown()
```
