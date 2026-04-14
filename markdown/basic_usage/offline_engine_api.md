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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]


    2026-04-14 01:53:10,694 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-14 01:53:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.68s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.87it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.74it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.20it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.20it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.20it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.20it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.20it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.20it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.50it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 25.36it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 31.84it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]

    Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 46.46it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 46.46it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 46.46it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 46.46it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 46.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.52 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.49 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.49 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.48 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):   9%|▊         | 5/58 [00:00<00:02, 22.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.48 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.47 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.46 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.45 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.80it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=960 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:00, 39.09it/s] Capturing num tokens (num_tokens=896 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:00, 39.09it/s]

    Capturing num tokens (num_tokens=896 avail_mem=73.43 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=768 avail_mem=73.42 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=704 avail_mem=73.42 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=640 avail_mem=73.42 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=576 avail_mem=73.42 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.12it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=480 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=448 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=416 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=384 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]

    Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 43.88it/s]Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=288 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=256 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=240 avail_mem=73.40 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  60%|██████    | 35/58 [00:00<00:00, 46.87it/s]Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]Capturing num tokens (num_tokens=176 avail_mem=73.39 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]Capturing num tokens (num_tokens=160 avail_mem=73.38 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]

    Capturing num tokens (num_tokens=112 avail_mem=73.37 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s]Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  71%|███████   | 41/58 [00:01<00:00, 48.98it/s] Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=80 avail_mem=73.37 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=48 avail_mem=72.30 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=32 avail_mem=72.30 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=28 avail_mem=72.29 GB):  81%|████████  | 47/58 [00:01<00:00, 49.61it/s]Capturing num tokens (num_tokens=28 avail_mem=72.29 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=24 avail_mem=72.14 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=20 avail_mem=61.74 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=16 avail_mem=58.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=12 avail_mem=58.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]

    Capturing num tokens (num_tokens=8 avail_mem=58.32 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s] Capturing num tokens (num_tokens=4 avail_mem=58.31 GB):  90%|████████▉ | 52/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=4 avail_mem=58.31 GB): 100%|██████████| 58/58 [00:01<00:00, 50.66it/s]Capturing num tokens (num_tokens=4 avail_mem=58.31 GB): 100%|██████████| 58/58 [00:01<00:00, 43.53it/s]


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
    Generated text:  Lian Chen. I’m a high school student from Shanghai. Recently, I’ve been thinking a lot about the meaning of life. As a matter of fact, I want to know whether it’s true that life is meant to be lived with love and happiness, or that it’s meant to be lived with hate and destruction. I would like to know the answer to that question, because I think if I know that, I will have the courage to live my life. What are some ways that you would like to live? I would like to know if it is the most important way that one can live, or if it is possible
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 45 years old. If the average age of people in the country increases by 5 years for each additional year of the president's age, how many years from now will the average age of the people be 65 years? Let's denote the current president's age by \( P = 45 \) years and the number of years from now by \( n \).
    
    First, we calculate the president's current age \( P \):
    \[ P = 45 \text{ years} \]
    
    The average age of the people in the country will increase by 5 years for each additional year of the president's
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the seat of the nation's government, and Paris is the largest city in France and also the third largest in the European Union (EU). It is the capital of the heart of the European Union, and the largest city of the north of France. At the centre of the city, the famous Eiffel Tower, built in 1889, stands as a testament to the skyline of the city. The city is known for its rich history, its beautiful architecture, and its culinary delights. 
    
    Paris is famous for its fashion and art scene, especially the brand Guerlain. It is also home to
    ===============================
    Prompt: The future of AI is
    Generated text:  here and it's transforming the way we live and work. To understand how AI is changing the world, here's a look at how AI is already impacting the way businesses operate.
    1. Beyond the Basics: AI is transforming the way we do business.
    AI is becoming increasingly ubiquitous in the world of business. Whether it's through the use of AI-powered chatbots or data analytics, the impact of AI in business is huge and constantly evolving. In a fast-changing world, it's important to stay on top of the latest trends and technologies.
    AI has the potential to revolutionize how companies operate, from supply chain management to customer service.


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] who has always been [Attraction/Interest] to the [Subject]. I'm always [Positive/Dislike] about [Subject], and I'm always [Positive/Dislike] about [Subject]. I'm always [Positive/Dislike] about [Subject], and I'm always [Positive/Dislike] about [Subject]. I'm always [Positive/Dislike] about [Subject], and I'm always [Positive/Dislike] about [Subject]. I'm always [Positive/Dislike] about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, art, and cuisine, and is home to many famous French artists, writers, and musicians. Paris is also a major center for science and research, with numerous universities and research institutions. The city is a major transportation hub, with many major highways and rail lines connecting it to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions. This could lead to a more natural and intuitive user experience.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect user data. This could lead to the development of new technologies and protocols for handling sensitive information.
    
    3. Increased automation and efficiency: As AI becomes more integrated with human intelligence, there will be an increased need for automation and efficiency in various
    


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
    Generated text:  [Name]. I am a [role] who has been [number of years] years in this field. My [name] career has been marked by [specific achievements, experience, or passion], and I am always on the lookout for new opportunities to [describe an achievement or innovation]. Currently, I am focused on [mention a current project or initiative], and I am eager to continue [describe a goal or goal to achieve]. If you have any questions or topics you'd like to discuss, feel free to reach out! Let's establish a connection. [Name]. [Tell what topic you'd like to discuss next]. What brings
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its motto is La Ville et le Monde. Its official language is French. It is the fifth largest city in the world by population. It is often called the "City of Light" because of its many cultural landmarks and nightclubs. The city is also known for its important historical sites, such as Notre Dame Cathedral and the Eiffel Tower. Paris is home to many diverse neighborhoods and districts, such as the Montmartre neighborhood, the Marais neighborhood, and the 16th arrondissement. It is also known for its rich culinary culture, with many famous restaurants, bakeries, and cafes in the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a combination of rapid technological advancements, continued integration with human decision-making, and a growing focus on ethical considerations. Some potential future trends in AI include:
    
    1. Increased integration with natural language processing: As more and more AI applications rely on human users, the importance of integrating natural language processing (NLP) into AI systems is likely to increase. This will allow AI systems to understand and respond to human language in a more natural way.
    
    2. Improved sense of facial expressions: While AI systems are increasingly accurate at recognizing faces, they still struggle to accurately identify the emotions and expressions that people display. Future AI systems may focus on


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

    ________

    __.

     I

     am

     a

    /an

     __________________

    _

     (

    generate

     a

     list

     of

     options

     from

     the

     prompt

     to

     include

     an

     adjective

    ,

     noun

    ,

     or

     verb

     from

     the

     prompt

    ).

     I

     come

     from

     __________________

    _

     (

    generate

     a

     list

     of

     options

     from

     the

     prompt

     to

     include

     a

     place

     from

     the

     prompt

    ).

     I

     am

     __________________

    _.

     I

     have

     been

     studying

     for

     my

     __

    ________

    ___

     test

    .

     (

    generate

     a

     list

     of

     options

     from

     the

     prompt

     to

     include

     a

     major

     from

     the

     prompt

    ).

     I

     am

     currently

     living

     in

     __________________

    _.

     My

     __

    ________

    _

     (

    generate

     a

     list

     of

     options

     from

     the

     prompt

     to

     include

     a

     verb

     from

     the

     prompt

    )

     at

     this

     university

    .

     I

     have

     taken

     classes

     in

     __

    ________

    ___

     (

    generate

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

     and

     a

     major

     economic

     and

     cultural

     center

    .

     Paris

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     as

     well

     as its

     historic

     neighborhoods

    ,

     including

     the

     

    1

    3

    th

     and

     

    1

    4

    th

     arr

    ond

    isse

    ments

    .

     The

     city

     is

     home

     to

     many

     renowned

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

    ,

     and

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     Paris

     is

     also

     a

     major

     destination

     for

     international

     business

    ,

     tourism

    ,

     and

     culture

    .

     The

     city

     has

     undergone

     many

     changes

     over

     the

     centuries

    ,

     with

     many

     historical

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     computing

     power

    ,

     improvements

     in

     data

     analysis

    ,

     and

     the

     integration

     of

     AI

     into

     various

     industries

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     With

     the

     rise

     of

     controversial

     technologies

     like

     autonomous

     weapons

    ,

     AI

     is

     likely

     to

     become

     even

     more

     closely

     regulated

     as

     concerns

     about

     privacy

    ,

     bias

    ,

     and

     accountability

     increase

    .

     It

    's

     likely

     that

     more

     companies

     will

     begin

     to

     explore

     the

     ethical

     implications

     of

     AI

     and

     seek

     to

     develop

     more

     transparent

     and

     accountable

     systems

    .
    


    2

    .

     Growing

     reliance

     on

     AI

     for

     autonomous

     and

     self

    -driving

     vehicles

    :

     This

     technology

     is

     expected

     to

     become

     more

     prevalent

     in

     the

     future

    ,

     with

     widespread

    



```python
llm.shutdown()
```
