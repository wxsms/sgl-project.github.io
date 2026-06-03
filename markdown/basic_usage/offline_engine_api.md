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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:08,  4.37s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  3.98it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]

    Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 14.97it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]

    Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.12it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 22.12it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 28.77it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 38.84it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 38.84it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 38.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=60.44 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=60.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.39 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=60.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.49it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.37 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.25it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s]Capturing num tokens (num_tokens=960 avail_mem=60.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=60.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s]Capturing num tokens (num_tokens=832 avail_mem=60.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.28it/s]Capturing num tokens (num_tokens=832 avail_mem=60.34 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=768 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=704 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=640 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=576 avail_mem=60.33 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=512 avail_mem=60.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.54it/s]Capturing num tokens (num_tokens=512 avail_mem=60.31 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=480 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=448 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=416 avail_mem=60.33 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]

    Capturing num tokens (num_tokens=384 avail_mem=60.32 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=352 avail_mem=60.32 GB):  50%|█████     | 29/58 [00:00<00:00, 43.08it/s]Capturing num tokens (num_tokens=352 avail_mem=60.32 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=320 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=288 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=256 avail_mem=60.31 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=240 avail_mem=60.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=224 avail_mem=60.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.68it/s]Capturing num tokens (num_tokens=224 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=208 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=192 avail_mem=60.30 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=176 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]

    Capturing num tokens (num_tokens=160 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=144 avail_mem=60.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=144 avail_mem=60.29 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=128 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=112 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=96 avail_mem=60.28 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s] Capturing num tokens (num_tokens=80 avail_mem=60.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=64 avail_mem=60.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.67it/s]Capturing num tokens (num_tokens=64 avail_mem=60.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=48 avail_mem=60.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=60.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=60.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=24 avail_mem=60.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=20 avail_mem=60.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=20 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=16 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=12 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=8 avail_mem=60.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.37it/s] Capturing num tokens (num_tokens=4 avail_mem=60.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.37it/s]Capturing num tokens (num_tokens=4 avail_mem=60.24 GB): 100%|██████████| 58/58 [00:01<00:00, 40.97it/s]


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
    Generated text:  Xander Smith, and I'm a new post grad. I'm majoring in Biology and I have just joined the honors program. My mentor, Dr. David, is an astronomer. He’s an inspiration to me. My favorite laboratory experiment has been the experiment to determine the size of the red giant. The experiment is very difficult and takes a lot of time. But it’s so exciting and awesome because it will give us a clear idea of how the stars form. The red giant is a type of star that grows in size and becomes more massive as it gets older. I'm going to do a project on the size of
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 41 years old. How many years ago was the president the age of a child? The president is currently 41 years old.
    
    To determine how many years ago the president was the age of a child, we need to consider the total time period during which the president was a child and compare it to the current age.
    
    1. Calculate the age difference between the current president's age and the age of a child.
    2. Determine how many years that difference represents.
    
    The current age of the president is 41 years. The age of a child is typically considered to be very young, often around 0 years.
    ===============================
    Prompt: The capital of France is
    Generated text:  in which of the following cities?
    A. Paris
    B. Lyon
    C. Nancy
    D. Toulouse
    The capital of France is Paris. Therefore, the correct answer is A. Paris. 
    
    To elaborate:
    - Paris, the capital city of France, is located in the south-central region of France.
    - Lyon is a major city in the northeast of France, but it is not the capital.
    - Nancy is in the northeastern region of France and is the largest city in Alsace.
    - Toulouse is in the northeastern region of France but is not the capital. 
    
    Thus, the capital of France is located in Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and this is what makes the next generation of AI researchers hungry for it. Last week, the RoboCup competition opened to the public and it showed us what AI can be. RoboCup is a weekly competition for young people, who take on the role of computer programs. They compete with each other to win and the event has been open to anyone since 2011. Now the 2016 RoboCup is now open to the public, and the judges are asking for contributions for the 2017 competition. RoboCup is a great way to get young people


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career journey. What can you tell me about yourself? I'm a [insert a unique personality trait or skill that sets me apart from others]. And what's your background? I've been working in [insert a relevant field or industry] for [insert a number of years] years. And what's your current role? I'm currently [insert a job title] at [insert a company name]. And what's your favorite hobby or activity? I enjoy [insert a hobby or activity]. And what
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a popular tourist destination and a major economic center in France. It is home to many famous museums, theaters, and restaurants. The city is also known for its annual festivals and cultural events. Paris is a vibrant and dynamic city that is a must-visit destination for anyone interested
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, cost savings, and job displacement, but it could also create new opportunities for workers in fields like manufacturing and logistics.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to evolve, we can expect to see even more advanced applications in this
    


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
    Generated text:  [Name] and I am [Age]. I've always been fascinated by nature and would love to learn more about it, which is why I decided to pursue a degree in [Your Major]. If you're looking for someone who can guide you on how to care for a plant or would love to learn about the wonders of the natural world, feel free to reach out. I look forward to hearing from you! 🌍🌿
    
    Hey, nice intro. How's it going? Any hobbies or interests that you're passionate about? 🎨!"); How are you doing? I'm feeling great! 🌞"; What do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. It is also famous for its vibrant culture, historic architecture, and lively nightlife. With a population of over 7. 8 million people, Paris is one of the world's largest cities and a major center of global commerce, politics, and culture. Its history dates back to ancient times and is considered one of the most influential and complex cities in the world. Despite its size, Paris is a small but vibrant city with a rich cultural heritage and a strong sense of community. The city offers a diverse range
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and it is difficult to predict with certainty what will happen in the years to come. However, here are some possible trends that could be expected in the coming years:
    
    1. Increased automation: AI is expected to continue automating many tasks, such as data entry, administrative tasks, and routine maintenance. This could lead to more efficient and cost-effective operations for businesses.
    
    2. AI will become more sophisticated: Over the next few years, AI will likely become even more sophisticated and capable of performing complex tasks that were previously only possible with human intelligence. This could lead to breakthroughs in fields such as medicine, engineering, and transportation.
    
    


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

    /an

     [

    Occup

    ation

    /

    Field

    ]

     from

     [

    Location

    ].

     I

     am

     passionate

     about

     [

    Your

     Passion

    ],

     and

     I

     enjoy

     [

    Your

     Career

     Goal

    ].

     I

     am

     confident

     in

     [

    Your

     Strength

    s

    ],

     and

     I

     am

     always

     looking

     for

     opportunities

     to

     [

    Your

     Goals

    /

    Imp

    ulses

    ].

     I

     am

     a

     [

    Your

     Personality

     Type

    ]

     and

     I

     believe

     that

     [

    Your

     Values

    ].

     I

     am

     a

     [

    Your

     Life

     Style

    ]

     and

     I

     am

     always

     ready

     to

     [

    Your

     Lifestyle

    /

    Preferences

    ].

     I

     am

     a

    /an

     [

    Your

     General

     Description

    ]

     who

     is

     dedicated

     to

     [

    Your

     Profession

    /

    Field

    ].

     As

     a

     [

    Your

     Profession

    /

    Field

    ],

     I

     am

     always

     motivated

     by

     [

    Mot

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     

    2

    1

    st

     largest

     city

     in

     the

     world

     by

     population

     and

     the

     

    1

    1

    th

     largest

     by

     area

    .

     It

     is

     located

     on

     the

     Î

    le

     de

     France

     and

     the

     Se

    ine

     River

     and

     is

     known

     for

     its

     architecture

    ,

     art

    ,

     and

     cultural

     attractions

    .

     Paris

     is

     home

     to

     many

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

     Dame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     

    6

    th

     century

     and

     has

     been

     a

     capital

     of

     France

     since

     the

     

    1

    2

    th

     century

    .

     Paris

    ians

     are

     known

     for

     their

     love

     of

     French

     food

    ,

     wine

    ,

     and

     music

    ,

     and

     the

     city

     is

     famous

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     marked

     by

     rapid

     advancements

     and

     significant

     changes

     in

     how

     it

     is

     implemented

     and

     used

    .

     Here

     are

     some

     potential

     trends

     that

     are

     currently

     shaping

     the

     development

     of

     AI

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     The

     use

     of

     AI

     in

     healthcare

     has

     been

     on

     the

     rise

     in

     recent

     years

    .

     With

     the

     increasing

     availability

     of

     data

     on

     patients

    ,

     AI

     can

     be

     used

     to

     analyze

     medical

     images

    ,

     predict

     disease

     progression

    ,

     and

     improve

     patient

     care

    .
    


    2

    .

     More

     Personal

    ization

    :

     AI

     is

     increasingly

     being

     used

     to

     improve

     the

     personal

    ization

     of

     customer

     experiences

    .

     By

     analyzing

     user

     behavior

     and

     preferences

    ,

     AI

     can

     offer

     personalized

     recommendations

     and

     suggestions

    ,

     helping

     users

     make

     more

     informed

     decisions

    .
    


    3

    .

     Integration

    



```python
llm.shutdown()
```
