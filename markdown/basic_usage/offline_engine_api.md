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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.55it/s]


    2026-04-30 22:03:13,137 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 22:03:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:25,  4.65s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.86it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.86it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.65it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.48it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 19.55it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.60it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]

    Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  55%|█████▌    | 32/58 [00:00<00:00, 46.03it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.84it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.84it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.84it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.84it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.84it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.84it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s]

    Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.99it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  84%|████████▍ | 49/58 [00:01<00:00, 48.75it/s]Capturing num tokens (num_tokens=20 avail_mem=72.47 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.77it/s]Capturing num tokens (num_tokens=16 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.77it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.77it/s]Capturing num tokens (num_tokens=8 avail_mem=72.46 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.77it/s] Capturing num tokens (num_tokens=4 avail_mem=72.45 GB):  93%|█████████▎| 54/58 [00:01<00:00, 48.77it/s]Capturing num tokens (num_tokens=4 avail_mem=72.45 GB): 100%|██████████| 58/58 [00:01<00:00, 43.43it/s]


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
    Generated text:  Kimberly and I am 18 years old. I have a major in English and minor in Business Administration. I love to travel and have traveled to several countries in Africa and Europe. I also love to be outdoors and play sports like basketball, soccer, volleyball, and badminton. I have a passion for learning and studying and I'm always looking for new and exciting things to learn. What are some ways that you can get interested in learning and studying, and what are some hobbies that you enjoy? Kimberly, I hope you have a wonderful day! 
    
    Write a letter to Kimberly that is formal and polite. Begin by thanking her
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term. The current president was elected to a term of 4 years and currently has a popularity rating of 3.5. The new president is also expected to have a very strong popularity rating. The president has a popularity rating of 5.5. If the popularity rating increases by 2 points per year for each year the president is in office, how many more years does the president need to serve before his popularity rating surpasses that of the new president? To determine how many more years the president needs to serve before his popularity rating surpasses that of the new president, we start by analyzing the given information
    ===============================
    Prompt: The capital of France is
    Generated text:  called Paris. The capital of England is called London. Both cities are famous for their historical sites and museums.
    
    It's a pity that I don't know where the capital of Italy is. Do you know, it's called Rome.
    
    This story is told all over the world. Some people believe the name is because the city was founded by Romulus, the first king of Rome. Others believe the name is because the capital of ancient Italy is in Rome.
    
    I'm not convinced. Rome and other cities in ancient Italy were in different parts of the country. The capital of Rome was near the lake, not in the mountains. There was
    ===============================
    Prompt: The future of AI is
    Generated text:  still very much up in the air. In the past year, however, we’ve seen some of the more exciting developments in the field, and they are both exciting and potentially disruptive. This week, we’ll take a look at some of the latest developments in machine learning, including the deep learning systems that are allowing researchers to understand and predict the behavior of complex systems, and the emerging ways in which AI is shaping human interaction in a number of different ways.
    One of the most exciting developments in machine learning in the past year has been the emergence of deep learning. Deep learning is a type of machine learning where the system is represented as a


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique trait or skill] who is passionate about [insert a hobby or interest]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a hobby or activity]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite book or movie? I love [insert a favorite book or movie]. I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its cuisine, fashion, and music scene. It is a popular tourist destination and a major economic center in Europe. The city is home to many cultural institutions and museums, including the Louvre and the Musée d'Orsay. Paris is a city of contrasts, with its modern architecture and historical landmarks blending together to create a unique and fascinating city. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools could be used to identify diseases earlier and more accurately, and AI-powered treatments could be used to improve the effectiveness of existing treatments.
    
    2. Increased Use of AI in Transportation: AI is already being used in transportation to help improve the efficiency
    


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
    Generated text:  Jane. I'm a person with a passion for books and storytelling. I enjoy being in the moment and capturing the essence of a story in words. My writing has been published in several journals and online publications, and I love sharing my creations with others. I'm excited to share my work with you and to get to know the people around me. How can I approach a potential client to learn more about their project?
    Hello! My name is Jane. I'm a writer who loves to capture the essence of a story in words. I enjoy being in the moment and capturing the essence of a story through my writing. I'm passionate about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Justification: This is a basic fact about the capital of France that can be easily verified through a simple search. However, it provides context and makes the capital more intriguing to potential visitors. The statement is concise and can be used in many contexts, including academic discussions or news headlines.
    
    A. Bélgica's capital city is the Vatican City.
    B. The capital of the United States is Washington D.C.
    C. The capital of France is Paris.
    D. The capital of the United Kingdom is London.
    E. The capital of the United States is New York City.
    F. The capital of Australia is Canberra
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possibilities that have been proposed and discussed include:
    
    1. Increased autonomy and self-awareness: AI is already capable of performing complex tasks with high levels of autonomy, such as driving cars or answering questions. It's possible that in the future, AI systems will be able to perform more complex tasks and even make decisions on their own, potentially leading to more autonomous AI.
    
    2. AI becomes more widely integrated into everyday life: AI is already becoming more integrated into our daily lives through things like smart homes and virtual assistants. It's possible that in the future, AI will become even more integrated into our lives, with more


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

    ].

     I

     am

     an

     AI

     language

     model

    ,

     but

     I

     am

     not

     a

     traditional

     marketing

     or

     sales

    person

    .

     I

     am

     here

     to

     assist

     you

     with

     your

     digital

     marketing

     and

     sales

     goals

    .

     How

     can

     I

     help

     you

     today

    ?

     Remember

    ,

     I

     am

     not

     a

     sales

    person

     or

     marketer

    ,

     but

     rather

     an

     AI

     designed

     to

     help

     with

     your

     digital

     marketing

     and

     sales

     strategies

    .

     How

     can

     I

     assist

     you

     today

    ?

     Let

    's

     discuss

     the

     specifics

     of

     our

     digital

     marketing

     strategy

    ,

     and

     we

     can

     discuss

     the

     tactics

     and

     tools

     you

     need

     to

     achieve

     your

     marketing

     goals

    .

     How

     do

     you

     plan

     to

     use

     AI

     technology

     in

     digital

     marketing

     and

     sales

     strategies

    ?

     Can

     you

     share

     any

     relevant

     examples

     or

     case

     studies

     of

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     most

     populous

     city

     in

     the

     country

    .

     
    


    To

     answer

     the

     question

    :
    


    1

    .

     What

     is

     the

     name

     of

     the

     capital

     city

     of

     France

    ?

     


    2

    .

     What

     is

     the

     population

     of

     Paris

    ?


    3

    .

     What

     is

     the

     name

     of

     the

     capital

     of

     France

    ?


    4

    .

     Which

     is

     the

     largest

     city

     in

     France

    ?
    


    The

     answer

     should

     be

     in

     the

     form

     of

     a

     bullet

     point

     list

    .

     

    1

    .

     The

     capital

     city

     of

     France

     is

     Paris

    .


    2

    .

     Paris

     has

     a

     population

     of

     over

     

    2

    .

    1

     million

     people

    .


    3

    .

     The

     capital

     of

     France

     is

     Paris

    .


    4

    .

     The

     largest

     city

     in

     France

     is

     Paris

    .

     
    


    Answer

    :

     

    1

    .

     The

     capital

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     advancements

     and

     changes

     in

     how

     it

     is

     used

     to

     accomplish

     tasks

     and

     solve

     problems

    .

     Some

     possible

     future

     trends

     in

     artificial

     intelligence

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     diagnose

     and

     treat

     a

     wide

     range

     of

     diseases

    ,

     from

     cancer

     to

     heart

     disease

    .

     In

     the

     future

    ,

     we

     may

     see

     even

     more

     widespread

     use

     of

     AI

     in

     healthcare

    ,

     with

     AI

    -powered

     devices

     and

     algorithms

     being

     used

     to

     assist

     doctors

     in

     making

     diagnoses

    ,

     monitoring

     patients

    ,

     and

     providing

     personalized

     treatment

     plans

    .
    


    2

    .

     Greater

     emphasis

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     lives

    ,

     there

     will

     be

     greater

     emphasis

     on

     ethical

     considerations

    ,

     such

     as

    



```python
llm.shutdown()
```
