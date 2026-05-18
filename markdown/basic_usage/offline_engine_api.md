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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  8.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.98it/s]


    2026-05-18 07:02:26,534 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 07:02:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.77it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.49it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.40it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 29.33it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 37.39it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 37.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:06,  9.24it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:06,  9.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:06,  9.24it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   5%|▌         | 3/58 [00:00<00:04, 11.16it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:04, 11.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:04, 11.16it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 13.76it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:03, 13.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 17.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.40it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.20it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.20it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.55it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  40%|███▉      | 23/58 [00:01<00:01, 31.55it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.21it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.21it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.21it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.21it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 35.21it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 38.13it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=160 avail_mem=73.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=144 avail_mem=74.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=128 avail_mem=74.26 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.82it/s]Capturing num tokens (num_tokens=112 avail_mem=74.25 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.82it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=96 avail_mem=74.13 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=64 avail_mem=74.20 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=48 avail_mem=74.21 GB):  79%|███████▉  | 46/58 [00:01<00:00, 31.48it/s]Capturing num tokens (num_tokens=48 avail_mem=74.21 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.32it/s]Capturing num tokens (num_tokens=32 avail_mem=74.21 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.32it/s]Capturing num tokens (num_tokens=28 avail_mem=74.21 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.32it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.20 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.32it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.32it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.20it/s]Capturing num tokens (num_tokens=16 avail_mem=74.19 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.20it/s]Capturing num tokens (num_tokens=12 avail_mem=74.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.20it/s]Capturing num tokens (num_tokens=8 avail_mem=74.18 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.20it/s] Capturing num tokens (num_tokens=4 avail_mem=74.17 GB):  93%|█████████▎| 54/58 [00:01<00:00, 29.20it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB): 100%|██████████| 58/58 [00:02<00:00, 29.88it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB): 100%|██████████| 58/58 [00:02<00:00, 28.64it/s]


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
    Generated text:  Arthur and I am the owner of a small business and I would like to apply for a small business grant. My business is a general health food store and I am seeking a grant from the State of North Carolina. I have a valid North Carolina business license and I have been in business since 2000. I would appreciate it if someone could give me some guidance on how to apply for the grant.
    You can access the full application and guidelines on the North Carolina Department of Commerce's website at http://www.northcarolina.gov/Commerce/Applications/applications.htm. The application deadline for 2018 is June
    ===============================
    Prompt: The president of the United States is
    Generated text:  a noble title. He or she can lead the country, make decisions, and influence the policies of the government. They are the people who are in charge of the government. Their job is to make sure that the government follows the rules and laws. They also make sure that everyone is treated fairly and that the government is making the best decisions for the country.
    What is the title of the person who can lead the country? The president of the United States is the title of the person who can lead the country. They are the leader of the government and are in charge of making important decisions. Their job is to make sure the country follows
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is in the south-east of France. It has many beautiful places to see such as Notre Dame Cathedral, the Louvre, and the Champs-Elysee. Paris has also many famous people in it such as Leonardo da Vinci, Jean-Baptiste Cazenove, and Marie Antoinette. Marie Antoinette is a famous French queen and she lived from 1755 to 1830. She was the second wife of King Louis XVI. She is remembered well in history. The most famous building in Paris is the Palace of Versailles. It is the biggest and most beautiful palace in
    ===============================
    Prompt: The future of AI is
    Generated text:  very interesting, but when it comes to AI, there’s a clear debate about the future. We’ve been told that AI is the future and that it will revolutionize many industries and disciplines. However, it’s also been stated that AI will only be a minor player in the future.
    The future of AI is not as exciting as it might seem. There are several reasons for this. First, there’s a lack of access to AI technology. In many industries, AI is only accessible to high-end organizations or specialized teams. This means that the wider population is left behind.
    Second, there is a lack of education in AI. Many


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic center, hosting numerous world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture and politics. Paris is also home to many famous French artists, writers, and musicians. The city is a major center of education, with many prestigious universities and colleges. Paris is a city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI that can better understand and respond to the needs of individuals.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations
    


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
    Generated text:  [Name]. I'm an enthusiastic and creative individual who thrives on learning and growing. I'm passionate about using technology and design to solve real-world problems and bring joy to people's lives. I enjoy working in a team and being part of a community of like-minded individuals. I believe in the power of innovation and I am always eager to learn and grow. So, if you're looking for a dynamic and collaborative team of creatives and problem solvers, look no further. Here's to some great projects and some amazing people! [Name] 📅✨
    
    This self-introduction should capture the essence of the character while
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the country and the seat of the French government. It is known for its historic architecture, elegant cafes, and diverse food scene. It is also a popular tourist destination for its opulent boulevards, romantic parks, and vibrant culture. Paris has been a major hub for trade, finance, and industry for centuries and continues to be a global center of culture and art. The French capital is known for its rich history, cultural richness, and vibrant energy. Paris is a city that is a true inspiration to the world, attracting millions of visitors each year. The city is home to many world-ren
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be a dynamic and exciting one. Here are some possible trends that are expected to shape the field:
    
    1. Enhanced Natural Language Processing: AI systems will become even more sophisticated and capable of understanding and generating human-like language, which could lead to a wider range of applications.
    
    2. Enhanced Personalization: AI will be able to learn from the data it collects, making personalized recommendations and interactions with users.
    
    3. Autonomous Vehicles: The development of autonomous vehicles will become more prevalent, and AI will be used to control them, ensuring safety and efficiency.
    
    4. AI in Healthcare: AI will be used in healthcare to improve diagnoses, predict


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

    ].

     I

    'm

     passionate

     about

     [

    Area

     of

     Interest

    ],

     and

     I

    've

     been

     dedicated

     to

     [

    Anything

    ]

     for

     [

    Number

    ]

     years

     now

    .

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     grow

     in

     my

     field

    .

     I

     have

     a

     natural

     talent

     for

     [

    Your

     Skill

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Your

     Goal

    ].

     What

    's

     your

     name

    ,

     and

     what

     brings

     you

     here

     today

    ?

     [

    Name

    ]

     [

    Your

     Name

    ]

     [

    Your

     Email

    ]

     [

    Your

     Phone

     Number

    ]


    I

    ’m

     a

     [

    Occup

    ation

    ]

     who

    ’s

     passionate

     about

     [

    Area

     of

     Interest

    ]

     and

     has

     been

     dedicated

     to

     [

    Anything

    ]

     for

     [

    Number

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     Europe

     and

     has

     a

     population

     of

     over

     

    1

    .

    5

     million

     people

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    ,

     and

     is

     an

     important

     center

     for

     politics

    ,

     culture

    ,

     and

     the

     arts

    .

     Paris

     is

     also

     known

     as

     "

    the

     city

     of

     love

    "

     due

     to

     its

     romantic

     ambiance

     and

     romantic

     poetry

    .

     It

     is

     the

     seat

     of

     government

     for

     the

     Republic

     of

     France

     and

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

    ,

     attracting

     millions

     of

     visitors

     each

     year

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     constantly

     evolving

    ,

     with

     many

     potential

     trends

     and

     developments

     expected

     to

     shape

     the

     way

     we

     interact

     with

     technology

     and

     society

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Autonomous

     robots

     and

     drones

    :

     The

     development

     of

     autonomous

     robots

     and

     drones

     could

     revolution

    ize

     industries

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .

     These

     machines

     would

     be

     able

     to

     perform

     tasks

     without

     human

     intervention

    ,

     reducing

     the

     need

     for

     human

     workers

    .
    


    2

    .

     Deep

     learning

     and

     machine

     learning

    :

     The

     field

     of

     machine

     learning

     is

     rapidly

     evolving

    ,

     and

     it

    's

     possible

     that

     AI

     will

     continue

     to

     advance

     at

     an

     unprecedented

     pace

    .

     Deep

     learning

    ,

     a

     type

     of

     machine

     learning

     that

     involves

     using

     multiple

    



```python
llm.shutdown()
```
