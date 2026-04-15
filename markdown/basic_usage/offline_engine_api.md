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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.56it/s]


    2026-04-15 15:58:41,537 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 15:58:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.63it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.65it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.42it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.46it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=114.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=114.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=114.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=114.70 GB):   3%|▎         | 2/58 [00:00<00:03, 15.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=114.69 GB):   3%|▎         | 2/58 [00:00<00:03, 15.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=114.69 GB):   7%|▋         | 4/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=114.63 GB):   7%|▋         | 4/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=114.62 GB):   7%|▋         | 4/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=114.63 GB):   7%|▋         | 4/58 [00:00<00:03, 17.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=114.63 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=114.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=114.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=114.62 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.82it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=114.61 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=114.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=114.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=114.61 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=2816 avail_mem=114.59 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=114.56 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=114.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=114.56 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=114.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=114.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.61it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=114.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=114.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=114.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=114.52 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.81it/s]Capturing num tokens (num_tokens=960 avail_mem=114.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.81it/s] Capturing num tokens (num_tokens=896 avail_mem=114.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 30.81it/s]Capturing num tokens (num_tokens=896 avail_mem=114.53 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=832 avail_mem=114.53 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=768 avail_mem=114.52 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.19it/s]

    Capturing num tokens (num_tokens=704 avail_mem=114.52 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=640 avail_mem=114.52 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.19it/s]Capturing num tokens (num_tokens=640 avail_mem=114.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.52it/s]Capturing num tokens (num_tokens=576 avail_mem=114.52 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.52it/s]Capturing num tokens (num_tokens=512 avail_mem=114.51 GB):  47%|████▋     | 27/58 [00:00<00:00, 31.52it/s]Capturing num tokens (num_tokens=480 avail_mem=114.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.52it/s]Capturing num tokens (num_tokens=448 avail_mem=114.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 31.52it/s]Capturing num tokens (num_tokens=448 avail_mem=114.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.76it/s]Capturing num tokens (num_tokens=416 avail_mem=114.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.76it/s]

    Capturing num tokens (num_tokens=384 avail_mem=114.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.76it/s]Capturing num tokens (num_tokens=352 avail_mem=114.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.76it/s]Capturing num tokens (num_tokens=320 avail_mem=114.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 31.76it/s]Capturing num tokens (num_tokens=320 avail_mem=114.51 GB):  60%|██████    | 35/58 [00:01<00:00, 32.13it/s]Capturing num tokens (num_tokens=288 avail_mem=114.50 GB):  60%|██████    | 35/58 [00:01<00:00, 32.13it/s]Capturing num tokens (num_tokens=256 avail_mem=114.50 GB):  60%|██████    | 35/58 [00:01<00:00, 32.13it/s]Capturing num tokens (num_tokens=240 avail_mem=114.50 GB):  60%|██████    | 35/58 [00:01<00:00, 32.13it/s]Capturing num tokens (num_tokens=224 avail_mem=114.50 GB):  60%|██████    | 35/58 [00:01<00:00, 32.13it/s]

    Capturing num tokens (num_tokens=224 avail_mem=114.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=208 avail_mem=114.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=192 avail_mem=114.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=176 avail_mem=114.49 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=160 avail_mem=114.48 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.65it/s]Capturing num tokens (num_tokens=160 avail_mem=114.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=144 avail_mem=114.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=128 avail_mem=114.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=112 avail_mem=114.48 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=96 avail_mem=114.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=114.47 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.71it/s]Capturing num tokens (num_tokens=80 avail_mem=114.47 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=64 avail_mem=114.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=48 avail_mem=114.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=32 avail_mem=114.46 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=28 avail_mem=114.45 GB):  83%|████████▎ | 48/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=28 avail_mem=114.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=24 avail_mem=114.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=20 avail_mem=114.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.22it/s]

    Capturing num tokens (num_tokens=16 avail_mem=114.45 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=12 avail_mem=114.44 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=12 avail_mem=114.44 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=8 avail_mem=114.44 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.01it/s] Capturing num tokens (num_tokens=4 avail_mem=114.43 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.01it/s]Capturing num tokens (num_tokens=4 avail_mem=114.43 GB): 100%|██████████| 58/58 [00:01<00:00, 31.65it/s]


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
    Generated text:  Joan and I'm an amateur photographer. I took this photo of a lion cub in June, 2004. I've taken many such photos of animals, but this one is the most significant to me. There are a lot of animals in the world, but lions are the most fascinating to me because they are like a living thing. They're wild, they're unpredictable and they're almost as complex as humans. I can't help but take pictures of lions because I love them. They're beautiful, I know, but there's a mystery and an excitement to lions that I can't shake. I love the animals,
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a law requiring all citizens to wear a certain type of protective vest. The law has already been passed and signed by the president, but it's unclear if all citizens will comply with it. One way to prevent the law from being void for lack of compliance is to find a way to determine how many citizens are required to wear the vest. If no citizens have yet complied with the law, it will be an unlawful requirement. If some citizens have not complied, it is unlawful, but if no one has, it is also lawful.
    
    Is it possible to find a way to determine whether citizens have complied with the law? If so,
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Tokyo D. Rome
    Answer:
    A
    
    In the following sentences, which one is free of grammatical errors?
    A. In the Han Dynasty, the Su family was known for their wondrous hairstyles.
    B. He spent all his savings to purchase a car, and he was already at the top of the list.
    C. He showed off his skills in every game, and everyone was impressed.
    D. The sun was about to set, and the children ran to their mothers, who were excited.
    Answer:
    A
    
    Which of the following sentences is grammatically correct and clearly expressed?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  hybrid systems, and in this scenario, the 3D printed components are part of a fully functional system. The 3D printed parts can be used to fabricate a functional robot, a robot with a flexible arm that can be bent and contoured. The computer is part of the control and the 3D printed parts are the components that control the movement of the arm, giving the robot its flexibility.
    When designing an AI system with a fully functional robot that contorts and bends the arm, it is imperative to use 3D printed components that are compatible with the robotic arm. There are several types of 3D printed


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn new things and try new things. I enjoy working with people and I'm always looking for ways to collaborate and support others. What's your favorite hobby or activity? As a [job title], I enjoy spending time with my family and friends. I also love to read and travel. What's your favorite book or movie? As
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is a popular tourist destination and a major center of politics, government, and diplomacy in France. The city is also home to many famous French artists, writers, and musicians. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with a diverse population of over 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is designed to be ethical and responsible. This could mean that AI systems are designed to minimize harm to individuals and society as a whole, and that they are transparent and accountable.
    
    2. AI will become more integrated into our daily lives: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI-powered technologies. This could include
    


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
    Generated text:  Jane, and I am an AI assistant. I was designed to help people with their information needs and provide them with helpful responses to their questions. I have been around for a long time, and I have been here to answer any questions they have. Whether they are about science, history, or anything else, I will do my best to provide accurate and helpful responses to them. So, if you have any questions or need any help, just let me know and I will do my best to assist you. I am here to help, and I am always here to assist. So, if you have any questions or need any help,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the third largest in the world, located on the banks of the River Seine. It is also known as "the City of Light" and is the seat of the French government and a major cultural and educational center. It is home to many iconic landmarks, including the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also known for its rich history and a diverse range of art, music, and cuisine. The city is a global city with a large population and many different ethnic groups. It is a cultural hub and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be dominated by a combination of current trends and emerging technologies. Here are some possible future trends:
    
    1. Increased focus on ethical AI: As more companies and governments start to realize the potential risks of AI, there will be an increased focus on ethical AI. This could involve developing AI systems that are transparent, accountable, and avoid bias.
    
    2. Continued development of AI systems: As technology advances and new data becomes available, AI systems are likely to get even better at tasks that involve pattern recognition, decision-making, and natural language processing.
    
    3. Larger deployment of AI systems: As more and more people become aware of the benefits of


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

     __

    __.

     


    A

    .

     


    B

    .

     


    C

    .

     


    D

    .

     


    E

    .

     


    F

    .

     


    G

    .

     


    H

    .

     


    I

    .

     


    J

    .

     


    K

    .

     


    L

    .

     


    M

    .

     


    N

    .

     


    O

    .

     


    P

    .

     


    Q

    .

     


    R

    .

     


    S

    .

     


    T

    .

     


    U

    .

     


    V

    .

     


    W

    .

     


    X

    .

     


    Y

    .

     


    Z

    .

     


    *

    Please

     use

     all

     letters

     exactly

     once

    ,

     and

     start

     with

     a

     letter

     in

     the

     first

     row

    .

     Your

     introduction

     should

     convey

     a

     sense

     of

     reliability

     and

     intelligence

    .


    Hello

    ,

     my

     name

     is

     __

    __.

     I

    'm

     a

     professional

     software

     engineer

     with

     over

     

    1

    0

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

     and

     the

     second

     largest

     metropolitan

     area

     in

     the

     world

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     rich

     history

    ,

     cultural

     attractions

    ,

     and

     elegant

     buildings

    .

     Paris

     is

     also

     home

     to

     many

     world

    -ren

    owned

     art

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     The

     city

     is

     also

     famous

     for

     its

     unique

     gastr

    onomy

    ,

     particularly

     in

     the

     culinary

     districts

     of

     Mont

    mart

    re

     and

     Le

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     a

     hub

     of

     commerce

     and

     finance

    ,

     with

     many

     important

     financial

     institutions

     and

     business

     districts

     located

     within

     its

     walls

    .

     Despite

     its

     size

    ,

     Paris

     is

     a

     city

     of

     diverse

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     more

     diverse

     and

     complex

    ,

     with

     several

     possible

     trends

    :
    


    1

    .

     Increased

     automation

     and

     artificial

     general

     intelligence

    :

     AI

     systems

     are

     likely

     to

     become

     more

     capable

     of

     performing

     a

     wide

     range

     of

     tasks

    ,

     from

     routine

     tasks

     to

     creative

     and

     critical

     thinking

    ,

     without

     human

     intervention

    .
    


    2

    .

     Integration

     with

     other

     technologies

    :

     AI

     systems

     will

     likely

     be

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     machine

     learning

    ,

     blockchain

    ,

     and

     quantum

     computing

    ,

     creating

     new

     and

     complex

     applications

    .
    


    3

    .

     Development

     of

     ethical

     AI

    :

     As

     AI

     systems

     become

     more

     complex

     and

     advanced

    ,

     there

     will

     be

     a

     growing

     need

     for

     guidelines

     and

     regulations

     to

     ensure

     ethical

     behavior

     and

     prevent

     unintended

     consequences

    .
    


    4

    .

     Personal

    ization

     and

     personal

    ization

    



```python
llm.shutdown()
```
