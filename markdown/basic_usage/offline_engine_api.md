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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.30it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:04,  9.12it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.02it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 15.02it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.91it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=52.87 GB):   2%|▏         | 1/58 [00:00<00:08,  6.64it/s]Capturing num tokens (num_tokens=7680 avail_mem=52.84 GB):   2%|▏         | 1/58 [00:00<00:08,  6.64it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=52.84 GB):   3%|▎         | 2/58 [00:00<00:08,  6.79it/s]Capturing num tokens (num_tokens=7168 avail_mem=52.83 GB):   3%|▎         | 2/58 [00:00<00:08,  6.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.83 GB):   3%|▎         | 2/58 [00:00<00:08,  6.79it/s]Capturing num tokens (num_tokens=6656 avail_mem=52.83 GB):   7%|▋         | 4/58 [00:00<00:05,  9.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=52.83 GB):   7%|▋         | 4/58 [00:00<00:05,  9.48it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=52.82 GB):   7%|▋         | 4/58 [00:00<00:05,  9.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.82 GB):   7%|▋         | 4/58 [00:00<00:05,  9.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=52.82 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=52.81 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=4096 avail_mem=52.81 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.81 GB):  12%|█▏        | 7/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3840 avail_mem=52.81 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=52.80 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=52.80 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.80 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.01it/s]Capturing num tokens (num_tokens=3072 avail_mem=52.80 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=52.80 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=52.79 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=52.79 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.78 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=52.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=52.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=52.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.49it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=52.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.76 GB):  29%|██▉       | 17/58 [00:01<00:01, 25.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=52.76 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=960 avail_mem=52.77 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s] Capturing num tokens (num_tokens=896 avail_mem=52.77 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=832 avail_mem=52.76 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=768 avail_mem=52.76 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.59it/s]Capturing num tokens (num_tokens=768 avail_mem=52.76 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.70it/s]Capturing num tokens (num_tokens=704 avail_mem=52.76 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.70it/s]Capturing num tokens (num_tokens=640 avail_mem=52.75 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.70it/s]

    Capturing num tokens (num_tokens=576 avail_mem=52.75 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.70it/s]Capturing num tokens (num_tokens=512 avail_mem=52.74 GB):  43%|████▎     | 25/58 [00:01<00:01, 30.70it/s]Capturing num tokens (num_tokens=512 avail_mem=52.74 GB):  50%|█████     | 29/58 [00:01<00:00, 32.21it/s]Capturing num tokens (num_tokens=480 avail_mem=52.75 GB):  50%|█████     | 29/58 [00:01<00:00, 32.21it/s]Capturing num tokens (num_tokens=448 avail_mem=52.75 GB):  50%|█████     | 29/58 [00:01<00:00, 32.21it/s]Capturing num tokens (num_tokens=416 avail_mem=52.75 GB):  50%|█████     | 29/58 [00:01<00:00, 32.21it/s]Capturing num tokens (num_tokens=384 avail_mem=52.75 GB):  50%|█████     | 29/58 [00:01<00:00, 32.21it/s]Capturing num tokens (num_tokens=384 avail_mem=52.75 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=352 avail_mem=52.74 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=320 avail_mem=52.74 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]

    Capturing num tokens (num_tokens=288 avail_mem=52.73 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=256 avail_mem=52.73 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=240 avail_mem=52.73 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.16it/s]Capturing num tokens (num_tokens=240 avail_mem=52.73 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=224 avail_mem=52.73 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=208 avail_mem=52.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=192 avail_mem=52.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=176 avail_mem=52.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=160 avail_mem=52.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.83it/s]Capturing num tokens (num_tokens=160 avail_mem=52.72 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=144 avail_mem=52.71 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=128 avail_mem=52.71 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s]

    Capturing num tokens (num_tokens=112 avail_mem=52.71 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=96 avail_mem=52.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s] Capturing num tokens (num_tokens=80 avail_mem=52.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.86it/s]Capturing num tokens (num_tokens=80 avail_mem=52.70 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=64 avail_mem=52.69 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=48 avail_mem=52.69 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=32 avail_mem=52.69 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=28 avail_mem=52.68 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=24 avail_mem=52.68 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=24 avail_mem=52.68 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=20 avail_mem=52.68 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=16 avail_mem=52.68 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s]

    Capturing num tokens (num_tokens=12 avail_mem=52.67 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=8 avail_mem=52.67 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s] Capturing num tokens (num_tokens=4 avail_mem=52.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=4 avail_mem=52.63 GB): 100%|██████████| 58/58 [00:01<00:00, 43.59it/s]Capturing num tokens (num_tokens=4 avail_mem=52.63 GB): 100%|██████████| 58/58 [00:01<00:00, 30.27it/s]


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
    Generated text:  Lucas and I have a question. Why do I need a master's degree in psychology?
    
    Choosing to pursue a master's degree in psychology can be a great decision if you're passionate about learning about and studying the human mind, behavior, and mental health. Here are some key reasons why:
    
    1. Personal Interest and Passion: Psychology is a highly engaging and intellectually stimulating field that you can learn about at a level you are interested in. Psychology provides a unique lens through which you can view the world and your personal experiences.
    
    2. Career Opportunities: With a master's degree in psychology, you have the opportunity to develop a professional skill set that
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build in different countries. The cost to build a military base in country $i$ is $20000 + 3i$ billion dollars. If the total cost is not to exceed $70000$ billion dollars, what is the maximum number of bases the president can build in a single country?
    To determine the maximum number of bases the president can build in a single country while staying within the total cost limit of $70000$ billion dollars, we need to analyze the cost function and find the maximum number of bases that can be built.
    
    The cost
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In the 16th century, the Paris region was known as the Old Paris, and it was the largest and most important center of the city of Paris. The Old Paris was established in the 12th century in the pre-revolutionary period of the city. The Old Paris was founded by the nobility, and it was the largest and most important center of the city. The Old Paris was established in the 12th century in the pre-revolutionary period of the city. The Old Paris was founded by the nobility, and it was the largest and most important center of the city. The
    ===============================
    Prompt: The future of AI is
    Generated text:  here.
    From being a threat to humanity, to being a benefit, the technology is changing in a rapid fashion and it is influencing every aspect of our lives.
    This is why in this course, I am going to focus on the AI in my life and how it has impacted my own life and career. I will share with you how AI has made it possible for me to be a more productive and effective individual. I will also share with you how AI has influenced the different fields and job roles in society and help them in different ways. I will also provide you with a clear understanding of the scope of job opportunities in the AI sector.


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do], and I'm always looking for new challenges and opportunities to grow. I'm a [What I Like to Do] and I'm always looking for ways to improve my skills and knowledge. I'm a [What I Like to Do] and I'm always looking for ways to improve my skills and knowledge. I'm a [What I Like to Do] and I'm always looking for ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also home to many famous French artists, writers, and musicians. The city is known for its fashion industry, with many famous fashion designers and boutiques. Paris is a popular tourist destination, with millions of visitors each year. It is also a major center for international business
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that could be expected in the future:
    
    1. Increased integration of AI into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI technologies in areas such as healthcare, finance, transportation, and entertainment.
    
    2. AI will become more personalized: As AI becomes more sophisticated, we may see more personalized and context-aware AI that can learn from user behavior and preferences to provide more accurate and relevant results.
    
    3. AI will become more autonomous:
    


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
    Generated text:  [Name], and I am a [profession]. I specialize in [professional role], and my experience spans [number of years] years in this role. Currently, I am [most recent position] and I am a member of [my club/association] and [my team]. I have always been [some characteristic] in [my role], and I strive to [my goal or passion]. I am always looking for [new skill or initiative] in my career. As a [type of person], I am [positive, confident, outgoing, humble, etc.] and I enjoy [activities that I enjoy doing]. I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the largest city and the most populous city in the country. It is located on the Seine River and was founded in the 6th century AD by the Romans. The city has a rich history and is known for its vibrant culture, art, and cuisine. It is also one of the most tourist destinations in the world. Paris is home to many famous landmarks such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. The city is also known for its diverse population and its efforts to create a more inclusive and equitable society. Overall, Paris is a unique and fascinating city that has a unique charm
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with new developments and possibilities in the near future. Here are some potential trends that are currently being explored and discussed:
    
    1. Deep learning and machine learning: Deep learning and machine learning will continue to evolve, with new algorithms and techniques being developed to enhance AI's ability to learn from data and perform complex tasks. These developments could lead to more advanced AI applications and applications that are much more sophisticated and effective.
    
    2. Autonomous vehicles: As autonomous vehicles become more capable, they could potentially replace human drivers and reduce traffic accidents. The integration of AI and machine learning could also lead to greater efficiency and productivity in transportation.
    
    3. Improved


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

    !

     I

    'm

     fluent

     in

     [

    Language

    ]

     and

     can

     speak

     several

     dialect

    s

     of

     [

    Language

    ].

     I

    'm

     an

     avid

     [

    Personal

     Hobby

     or

     Interest

    ]

     that

     has

     been

     [

    Number

     of

     Years

    ]

     years

     of

     involvement

    .

     I

    'm

     passionate

     about

     [

    Mot

    iv

    ational

     Quote

    ]

     and

     enjoy

     [

    Aff

    ection

    ate

     Word

    ]

     my

     family

    .

     What

     do

     you

     think

     makes

     you

     a

     great

     fit

     for

     the

     job

    ?

     I

     believe

     my

     [

    Job

     Title

     or

     Personality

     Trait

    ]

     sets

     me

     apart

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     and

     grow

    .

     I

    'm

     looking

     forward

     to

     [

    Start

     Date

    ],

     and

     I

    'm

     excited

     to

     see

     what

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     and

     the

     seat

     of

     government

     of

     the

     country

    .

     Here

    's

     a

     concise

     factual

     statement

     about

     Paris

    :
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     the

     capital

     of

     the

     country

    .
    


    This

     statement

     encaps

    ulates

     the

     key

     points

     about

     Paris

    :
    


    1

    .

     It

     is

     the

     capital

     city

     of

     France

    


    2

    .

     It

     is

     the

     largest

     city

     in

     France

    


    3

    .

     It

     is

     the

     seat

     of

     government

     for

     France

    
    


    The

     other

     answer

    ,

     while

     containing

     a

     similar

     meaning

    ,

     om

    its

     the

     fact

     that

     Paris

     is

     also

     the

     capital

    .

     The

     statement

     given

     does

     not

     provide

     this

     additional

     information

    .

     The

     first

     sentence

     summarizes

     the

     most

     critical

     points

     about

     Paris

    .

     Therefore

    ,

     the

     provided

     statement

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     complex

    ,

     but

     some

     possible

     trends

     that

     have

     been

     predicted

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     is

     likely

     to

     automate

     more

     and

     more

     tasks

    ,

     such

     as

     production

    ,

     customer

     service

    ,

     and

     administrative

     work

    .

     This

     could

     lead

     to

     increased

     efficiency

     and

     productivity

    .
    


    2

    .

     Greater

     integration

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     into

     other

     areas

     of

     society

    ,

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    .

     This

     could

     lead

     to

     more

     personalized

     and

     efficient

     solutions

     to

     problems

    .
    


    3

    .

     AI

     will

     become

     more

     ethical

    :

     There

     is

     growing

     concern

     about

     the

     ethical

     implications

     of

     AI

    ,

     and

     there

     are

     likely

     to

     be

     more

     regulations

     and

     guidelines

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     in

     a

    



```python
llm.shutdown()
```
