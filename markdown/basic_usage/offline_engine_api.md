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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.68it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.69it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.73it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.62it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 36.17it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 36.17it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 36.17it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 36.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:04, 11.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:04, 11.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:04, 11.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:03, 15.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):  10%|█         | 6/58 [00:00<00:03, 15.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 15.94it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 15.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:03, 15.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 31.99it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.63it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:00, 37.47it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.27it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.38it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.91it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.60it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 34.54it/s]


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
    Generated text:  Bill and I just took the test of the Samsung Galaxy S II.
    The Samsung Galaxy S II has a 1.24 inch display with a resolution of 1600 x 2400 pixels. The screen is black to white, the color depth is 8 bits per color. The screen is a high refresh rate LCD display. The phone is designed for use with 4G LTE wireless data. The phone has a 1GHz quad-core processor, 3GB of RAM, and an 8GB NAND flash memory.
    The Samsung Galaxy S II has an 8.1x4.5 inch
    ===============================
    Prompt: The president of the United States is
    Generated text:  considered the head of state, and the president of the European Union is the head of government. The president of the European Union does not meet the criteria for both.
    
    Is the following statement true or false?
    The president of the European Union is not considered the head of state.
    
    To determine the correctness of the statement, we need to understand the definitions of "head of state" and "head of government" in the context of the European Union.
    
    1. **Head of State**: This refers to the sovereign ruler or head of a state. It is usually the monarch of a constitutional monarchy or the head of a republic or democracy. In the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the oldest city in the world. It is also the largest city. It is in the south of France. Paris has been the capital of France since 987. Before then, it was called Lyons. The first capital of France was chosen by King Charles I of France. He wanted the capital to be in a more peaceful place. Now, Paris is a very important city. It has a lot of big buildings. The Champs-Élysées is one of the most famous in the world. It is a very busy place. A lot of people walk there. It's also very hot. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the next generation of scientists and engineers.
    This is the view of the professional organisation responsible for developing the new AI tools used in our online video streaming services, which we are proud to be part of.
    AI is playing a significant role in many of our services, including more immersive and interactive video experiences, our ability to predict audience behaviour and identify ads, and our ability to deliver new and new-age services.
    The ambition of our data science group to continue to develop the tools we use will be one of our key priorities as we build our AI base for the future. Data is the key to improving the quality of our services


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is home to many world-renowned museums, theaters, and art galleries. Paris is a bustling and dynamic city with a diverse population and a rich cultural heritage. It is the capital of France and a major economic and political center in Europe. The city is also known for its annual festivals
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will allow AI to perform tasks that are currently difficult or impossible for humans to do.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with other technologies, there will be a greater emphasis on ethical considerations. This will include issues such as bias, privacy, and accountability.
    
    3. Increased use of AI in healthcare:
    


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
    Generated text:  [Name], and I'm a/an [Age] year old. I'm an [Occupation/Profession] who have been working hard at my job for [Number of Years] years. I enjoy [Reason for Enjoyment] and I'm always looking for new challenges and opportunities to learn. What are some things you enjoy doing?
    My hobbies include [Name of Hobby/Activity], reading, and playing [Name of Game]. I also enjoy [Exercise/Activity], going for [Number of Meals]. Finally, I have a passion for [Nature/Environment], and I love [Activity/Activity].
    I'm always looking for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Roche Choisy".
    
    The answer is Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly exciting and constantly evolving. Here are some possible trends that are likely to shape the field in the coming years:
    
    1. Increased focus on ethical AI: With the growing awareness of the negative impacts of AI on society, we will see a greater emphasis on developing ethical AI systems. This will involve designing AI that can be used in a responsible and socially acceptable way.
    
    2. AI will become more capable of learning and adapting: As AI technologies continue to advance, they will become even more capable of learning from data and adapting to new situations. This will enable AI systems to become even more self-aware and intelligent.
    
    3. AI will integrate more


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

    First

     name

    ],

     and

     I

     am

     a

     [

    Last

     name

    ]

     with

     a

     passion

     for

     [

    professional

     interest

    ].

     I

     have

     always

     been

     an

     [

    insert

     your

     interest

     here

    ]

     person

    ,

     and

     I

     enjoy

     [

    insert

     something

     you

    've

     done

    ]

     and

     learning

     new

     things

    .

     What

     better

     way

     to

     start

     than

     with

     my

     name

     and

     professional

     interest

    ?

     It

    's

     always

     nice

     to

     have

     someone

     you

     can

     relate

     to

     and

     connect

     with

    .

     What

     is

     your

     name

     and

     what

     is

     your

     professional

     interest

    ?


    [

    Insert

     your

     name

     here

    ]

     


    My

     professional

     interest

     is

     [

    insert

     your

     professional

     interest

     here

    ].

     As

     a

     [

    insert

     your

     professional

     interest

     here

    ],

     I

     am

     passionate

     about

     [

    insert

     something

     you

    've

     done

     here

    ].

     I

     enjoy

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Your

     statement

     is

     correct

     and

     concise

    .

     How

     can

     I

     assist

     you

     further

    ?

     
    


    As

     an

     AI

     assistant

    ,

     I

     can

     also

     provide

     a

     list

     of

     the

     main

     attractions

     in

     Paris

     or

     answer

     questions

     about

     French

     culture

    ,

     history

    ,

     and

     daily

     life

     in

     the

     city

    .

     Please

     let

     me

     know

     how

     I

     can

     assist

     you

     further

    !

     

    🌍

    ✨

    
    


    I

    'm

     here

     in

     Paris

     to

     explore

     the

     city

     and

     learn

     more

     about

     its

     unique

     charm

     and

     history

    .

     Can

     you

     recommend

     a

     must

    -

    visit

     spot

     or

     a

     food

     spot

     to

     start

     my

     tour

    ?

     

    🇫

    🇷

    
    


    Sure

    !

     Paris

     is

     a

     city

     with

     a

     wide

     variety

     of

     attractions

    ,

     and

     each

     one

     offers

     a

     unique

     experience

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     developments

     in

     technology

    ,

     the

     increasing

     integration

     of

     AI

     into

     various

     industries

    ,

     and

     the

     continued

     expansion

     of

     its

     applications

    .

     Some

     potential

     future

     trends

     include

    :
    


    1

    .

     Increased

     AI

     integration

     into

     healthcare

    :

     AI

     is

     expected

     to

     play

     a

     key

     role

     in

     healthcare

    ,

     with

     developments

     in

     AI

    -powered

     diagnostic

     tools

    ,

     personalized

     medicine

    ,

     and

     virtual

     and

     augmented

     reality

     in

     healthcare

    .

     AI

     is

     also

     expected

     to

     be

     used

     to

     improve

     patient

     outcomes

    ,

     reduce

     costs

    ,

     and

     increase

     accessibility

     to

     healthcare

     services

    .
    


    2

    .

     Enhanced

     AI

     safety

     and

     ethics

    :

     With

     the

     increasing

     integration

     of

     AI

     in

     various

     industries

    ,

     the

     need

     for

     ethical

     and

     safety

     standards

     has

     become

     more

     pressing

    .

     This

     trend

     includes

     advancements

     in

    



```python
llm.shutdown()
```
