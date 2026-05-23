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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<00:58,  1.06s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:22,  2.28it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:10,  4.58it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:04<00:04,  8.99it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]

    Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 15.11it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 21.71it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s]Compiling num tokens (num_tokens=96):  64%|██████▍   | 37/58 [00:04<00:00, 28.90it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:04<00:00, 41.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.55 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.55 GB):   2%|▏         | 1/58 [00:00<00:06,  9.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.18 GB):   2%|▏         | 1/58 [00:00<00:06,  9.30it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=68.18 GB):   3%|▎         | 2/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.51 GB):   3%|▎         | 2/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.51 GB):   7%|▋         | 4/58 [00:00<00:04, 11.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.47 GB):   7%|▋         | 4/58 [00:00<00:04, 11.65it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=68.50 GB):   7%|▋         | 4/58 [00:00<00:04, 11.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.50 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.49 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.48 GB):  10%|█         | 6/58 [00:00<00:04, 12.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.48 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.48 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.56it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=68.46 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.46 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.56it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.46 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.45 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.44 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.44 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.44 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.43 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.76it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=68.43 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.43 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.42 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.42 GB):  31%|███       | 18/58 [00:00<00:01, 24.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.41 GB):  31%|███       | 18/58 [00:00<00:01, 24.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.41 GB):  31%|███       | 18/58 [00:00<00:01, 24.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.39 GB):  31%|███       | 18/58 [00:01<00:01, 24.64it/s]Capturing num tokens (num_tokens=960 avail_mem=68.40 GB):  31%|███       | 18/58 [00:01<00:01, 24.64it/s] Capturing num tokens (num_tokens=960 avail_mem=68.40 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]Capturing num tokens (num_tokens=896 avail_mem=68.37 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]

    Capturing num tokens (num_tokens=832 avail_mem=68.38 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]Capturing num tokens (num_tokens=768 avail_mem=68.39 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]Capturing num tokens (num_tokens=704 avail_mem=68.36 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]Capturing num tokens (num_tokens=640 avail_mem=68.38 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.42it/s]Capturing num tokens (num_tokens=640 avail_mem=68.38 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=576 avail_mem=68.37 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=512 avail_mem=68.35 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=480 avail_mem=68.37 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=448 avail_mem=68.36 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]Capturing num tokens (num_tokens=416 avail_mem=68.36 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.36it/s]

    Capturing num tokens (num_tokens=416 avail_mem=68.36 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=384 avail_mem=68.35 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=352 avail_mem=68.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=320 avail_mem=68.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=288 avail_mem=68.33 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=256 avail_mem=68.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.33it/s]Capturing num tokens (num_tokens=256 avail_mem=68.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=240 avail_mem=68.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=224 avail_mem=68.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=208 avail_mem=68.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=192 avail_mem=68.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]

    Capturing num tokens (num_tokens=176 avail_mem=68.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.52it/s]Capturing num tokens (num_tokens=176 avail_mem=68.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=160 avail_mem=68.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=144 avail_mem=68.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=128 avail_mem=68.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=112 avail_mem=68.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s]Capturing num tokens (num_tokens=96 avail_mem=68.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 39.82it/s] Capturing num tokens (num_tokens=96 avail_mem=68.28 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=80 avail_mem=68.28 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=64 avail_mem=68.28 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=48 avail_mem=68.27 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=32 avail_mem=68.27 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]

    Capturing num tokens (num_tokens=28 avail_mem=68.26 GB):  81%|████████  | 47/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=28 avail_mem=68.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=24 avail_mem=68.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=20 avail_mem=68.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=16 avail_mem=68.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=12 avail_mem=68.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s]Capturing num tokens (num_tokens=8 avail_mem=68.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 43.36it/s] Capturing num tokens (num_tokens=8 avail_mem=68.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=68.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=68.25 GB): 100%|██████████| 58/58 [00:01<00:00, 30.83it/s]


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
    Generated text:  Michael. I am a 24-year-old man. I was born and grew up in a small village in the mountains. I first started my education here, but I dropped out of college because I couldn't afford to go to school. I studied to earn some money to support my family. Then I worked in the city and became rich. Now I live in a big city. It is a place with lots of shopping, restaurants, cinemas, etc. One day, I came across an old lady with a white cane(拐杖) and I stopped her to see if she needed help. When I said "Could you please
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He/she is in charge of the government and makes decisions for the country. They are very important because they make sure that everyone in the country is happy and safe.
    The president also has to be very strong. He/she must be able to stand up to a lot of pressure and be confident in their decisions. They also have to be very strong with their team, and they must be able to inspire their team members.
    The president of the United States is also very important because they represent the country and help to build relationships with other countries. They make sure that the United States is seen as a safe
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Tokyo D. Moscow
    Answer:
    A
    
    Which of the following statements about bacterial spores is correct?
    A. Spores are formed by bacteria, which are the dormant forms of bacteria
    B. Spores are formed by bacteria, which can survive the harsh environment of soil
    C. Spores are formed by bacteria, which have the highest resistance of all plant bodies
    D. Spores are formed by bacteria, which can survive high temperatures and ultra-violet radiation
    Answer:
    A
    
    Which of the following statements about the characteristics of bacteria is incorrect?
    A. They have ribosomes
    ===============================
    Prompt: The future of AI is
    Generated text:  smart data.
    
    As the world continues to electrify and transform, the business of generating and processing data is becoming increasingly sophisticated. The core principles of AI are changing as quickly as the data is evolving. For example, algorithms that today are effective at optimizing data retrieval and analytics are being replaced by more advanced, powerful, and secure solutions. This means that the business must not only embrace the change but also adapt to the new frontier.
    
    Machine learning is the way forward.
    
    The integration of machine learning has become the backbone of the data-driven world. Machine learning algorithms are able to learn patterns and make decisions based on data, rather than being merely passive


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [industry]. I'm a [occupation] who is [character trait]. I enjoy [something enjoyable]. I'm always looking for ways to [something new or improve]. I'm a [character trait] and I'm always ready to learn and grow. Thank you for taking the time to meet me. What's your name? I'm [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is home to many iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. It is also known for its rich history, including the French Revolution and the French Revolution Monument. Paris is a bustling city with a diverse population and is a popular tourist destination. The city is known for its cuisine, fashion, and art, and is home to many cultural institutions and museums. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more sophisticated and nuanced decision-making. This could lead to more personalized and adaptive AI systems that can learn from and adapt to human behavior.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and validation of AI systems, as well as greater transparency and accountability in their development and deployment.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve
    


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
    Generated text:  John. I have been working as a software engineer for over 10 years and have been involved in several successful projects. I have a deep passion for technology and always strive to solve complex problems. I enjoy helping people learn new skills and expanding their knowledge. I am always looking for new challenges and opportunities to learn and grow. My goal is to continue learning and improving as a software engineer and someone who can contribute to the growth and success of the industry. Thank you for your time! Hello, my name is John. I'm excited to introduce myself and learn more about you. How are you today? As an AI language model,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city where the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum stand. 
    
    - **Paris** is the capital city of the **Republic of France**.
    - **Paris** is located in the northeastern part of the country, on the **Seine River**.
    - **Paris** is the 15th largest city in the world by population.
    - **Paris** has an international airport, the **Paris-Charles de Gaulle Airport**, which serves as the country’s primary international airport.
    - **Paris** is the **6th largest** city in terms of land area in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and exciting, with many potential applications and developments. Some of the possible trends in AI include:
    
    1. Increased Use of AI in Healthcare: With the rise of AI in healthcare, we will see a significant increase in the use of AI in diagnosis, treatment, and patient care. This will lead to better patient outcomes and reduced costs.
    
    2. AI in Agriculture: With the benefits of AI in agriculture, we will see increased use of AI in farming, crop monitoring, and precision agriculture. This will help farmers to manage resources more efficiently and reduce costs.
    
    3. AI in Transportation: With the rise of autonomous vehicles, we will see


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

    ].

     I

    'm

     a

     self

    -employed

     digital

     marketing

     expert

    ,

     with

     over

     

    5

     years

     of

     experience

     in

     the

     field

    .

     I

    'm

     passionate

     about

     helping

     businesses

     increase

     their

     online

     presence

     and

     ultimately

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     constantly

     learning

     and

     evolving

     my

     skills

     to

     stay

     ahead

     of

     the

     curve

     in

     the

     industry

    .

     I

     have

     a

     knack

     for

     making

     people

     laugh

     and

     I

    'm

     always

     looking

     for

     new

     ways

     to

     help

     people

    .

     So

     if

     you

    're

     looking

     to

     make

     a

     change

     in

     your

     personal

     or

     professional

     life

    ,

     I

    'm

     here

     to

     help

     you

    .

     What

    's

     your

     profession

    ?

     What

    's

     your

     greatest

     strength

    ?

     What

    's

     your

     biggest

     weakness

    ?

     What

    's

     your

     favorite

     hobby

     or

     activity

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    )

     Paris

     is

     the

     largest

     city

     in

     France

    .


    B

    )

     Paris

     is

     the

     smallest

     city

     in

     France

    .


    C

    )

     Paris

     is

     the

     capital

     of

     a

     country

    .


    D

    )

     Paris

     is

     the

     largest

     country

     in

     Europe

    .


    C

    )

     Paris

     is

     the

     capital

     of

     a

     country

    .

     
    


    This

     is

     a

     fact

    ually

     correct

     answer

     based

     on

     historical

     and

     geographical

     knowledge

     of

     France

    's

     capital

     city

    .

     Paris

     is

     the

     capital

     city

     of

     France

    ,

     a

     European

     country

    ,

     and

     is

     recognized

     internationally

     as

     the

     cultural

     and

     economic

     hub

     of

     the

     nation

    .

     The

     capital

     is

     a

     significant

     political

     and

     administrative

     center

    ,

     with

     the

     president

     and

     the

     Council

     of

     State

     being

     located

     there

    .

     Paris

     is

     also

     known

     as

     the

     "

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

    ,

     both

     for

     its

     applications

     and

     its

     potential

     implications

    .

     Here

     are

     some

     potential

     trends

     in

     AI

    :
    


    1

    .

     Increased

     automation

     and

     intelligent

     agents

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     automated

     machines

     that

     can

     perform

     a

     wide

     range

     of

     tasks

     with

     minimal

     human

     input

    .

     This

     could

     lead

     to

     a

     more

     efficient

     and

     productive workforce

    ,

     as

     well

     as

     the

     development

     of

     more

     advanced

     technologies

    .
    


    2

    .

     Personal

    ized

     experiences

    :

     As

     AI

     continues

     to

     learn

     from

     user

     data

    ,

     we

     can

     expect

     to

     see

     more

     personalized

     experiences

     as

     AI

     becomes

     better

     at

     understanding

     and

     interpreting

     human

     behavior

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     marketing

     and

     personalized

     product

     recommendations

    .
    


    3

    .

    



```python
llm.shutdown()
```
