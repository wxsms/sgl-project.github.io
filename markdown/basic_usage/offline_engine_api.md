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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.45s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:05,  1.19s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:25,  2.05it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:11,  4.17it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:05,  7.56it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]

    Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:02, 14.31it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 21.80it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 31.23it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 40.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.66it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.38 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.37 GB):   3%|▎         | 2/58 [00:00<00:04, 11.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.37 GB):   7%|▋         | 4/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:04, 12.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.35 GB):  10%|█         | 6/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 14.07it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.33 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.32 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  16%|█▌        | 9/58 [00:00<00:02, 16.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.31 GB):  21%|██        | 12/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.30 GB):  21%|██        | 12/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.30 GB):  21%|██        | 12/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  21%|██        | 12/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.29 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.93it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.28 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.28 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  26%|██▌       | 15/58 [00:00<00:01, 21.93it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.27 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.43it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.43it/s]Capturing num tokens (num_tokens=960 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.43it/s] Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.43it/s]Capturing num tokens (num_tokens=896 avail_mem=74.25 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=832 avail_mem=74.24 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=704 avail_mem=74.23 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  40%|███▉      | 23/58 [00:01<00:01, 29.15it/s]Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=512 avail_mem=74.21 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=480 avail_mem=74.22 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=448 avail_mem=74.22 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=416 avail_mem=74.21 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=384 avail_mem=74.20 GB):  48%|████▊     | 28/58 [00:01<00:00, 32.51it/s]Capturing num tokens (num_tokens=384 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=352 avail_mem=74.20 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=320 avail_mem=74.19 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.31it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.31it/s]Capturing num tokens (num_tokens=256 avail_mem=74.18 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=240 avail_mem=74.17 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=224 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=208 avail_mem=74.16 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=192 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.15it/s]Capturing num tokens (num_tokens=176 avail_mem=74.15 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=160 avail_mem=74.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=144 avail_mem=74.14 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  72%|███████▏  | 42/58 [00:01<00:00, 37.43it/s]Capturing num tokens (num_tokens=112 avail_mem=74.13 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=96 avail_mem=74.12 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.02it/s] Capturing num tokens (num_tokens=80 avail_mem=74.11 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=48 avail_mem=74.09 GB):  79%|███████▉  | 46/58 [00:01<00:00, 38.02it/s]Capturing num tokens (num_tokens=48 avail_mem=74.09 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=32 avail_mem=74.09 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=28 avail_mem=74.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=20 avail_mem=74.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 38.39it/s]Capturing num tokens (num_tokens=16 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=12 avail_mem=74.06 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.06it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 39.06it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 30.30it/s]


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
    Generated text:  11457991. I'm 8 years old. I'm a student. I like playing video games. I play every day. My favorite movie is "The Great Gatsby". 
    
    I have a pet dog. It's a yellow Labrador Retriever. It's very cute. I'm not very good at math. My teacher says I should have a good attitude towards math. That way I can learn easily. You know what? I have no idea what I'm doing with my free time. I don't like to read. I like to watch cartoons.
    
    What should I do to help me
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy. He works very hard. He also has a lot of other important things to do. He has to listen to lots of different kinds of people. He also has to take care of his family. He has to deal with many different kinds of problems. The President of the United States usually works at the White House, which is very big and beautiful. It's called the White House because it's the President's official home. When the President comes to work, he sits on a white chair and has a black suit. When he's leaving, he walks down a sandy path to the White House. On holidays, the President
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is an important city for many reasons, including its status as the capital of France, its status as the cultural center of France, and its status as the political center of France. It is also one of the largest cities in the world, with a population of about 2. 3 million people. It is also a major transportation hub and tourist destination. There are many landmarks in Paris, including the Eiffel Tower, the Louvre Museum, and the Notre Dame Cathedral. It is also home to a number of museums and galleries, including the Musée d'Orsay, the Musée des Arts Modernes
    ===============================
    Prompt: The future of AI is
    Generated text:  moving towards more powerful and flexible AI, which means it is necessary to set clear and consistent requirements for the training and use of AI. In this talk, I will outline the basic elements of AI requirements, including the definition of AI, the application of AI, the roles of the various stakeholders, the goals of AI, the context of AI, and the characteristics of AI.
    Among these, the definition of AI is the key element of AI requirements. It is a concept that the government has always emphasized, but it is still not widely recognized in the industry. This talk will focus on the concept of AI and its definition, as well as


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature, art, and music, and is a major economic and cultural center. Paris is home to many famous museums, including the Louvre and the Musée d'Orsay, as well as the Notre-Dame Cathedral. The city is also known for its rich history, including the French Revolution and the French Revolution Monument. Paris is a popular tourist destination, with millions of visitors each year. It is also home to many international organizations and institutions, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to evolve, we can expect to see even more innovative applications emerge, from more advanced forms of AI to new ways of using existing AI systems. Additionally, there is a growing concern about the ethical implications of AI, and as these concerns are addressed, we can expect to see even more progress in the field. Overall, the future of AI looks bright, with potential for significant advancements and transformative
    


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
    Generated text:  _____. I am a/an _____. I am _____. I have been working at XYZ Company for ____ years, and I have always been the _____. I have a passion for _____. I am always looking for ways to _____. I enjoy _____. I have a strong work ethic, and I am dedicated to _____. I am a/an _____. I have always felt that _____. I am passionate about _____. I am always looking to learn new things, and I am always eager to contribute to the company. I am confident that I can achieve my goals, and I am looking forward to a great future at XYZ Company. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
      - 500 words or more.
    
    ### Paris: The Capital of France
    Paris, the city that never sleeps, is the vibrant, bustling heart of the French capital and a cultural and intellectual hub. Known for its elegant architecture, iconic landmarks, and lively French culture, Paris is a city that has captured the hearts of countless tourists and locals alike.
    
    **Elegant Architecture:** Paris is renowned for its opulent architecture, with thousands of museums, monuments, and palaces that blend historical and modern elements. The Louvre Museum, home to the world's most famous painting collection, is a testament to the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be a highly dynamic and rapidly evolving field, driven by a combination of technological advancements, changing societal priorities, and environmental factors. Here are some potential future trends in AI:
    
    1. Increased focus on ethical considerations: As the benefits of AI continue to grow, there will be growing interest in addressing ethical concerns related to AI, such as privacy, fairness, and accountability. This will likely lead to more rigorous ethical standards and regulatory frameworks.
    
    2. Deep learning and neural networks: Neural networks, which are a key component of AI, are expected to continue to improve their accuracy and speed of processing. This will likely lead to more efficient and


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

     Name

    ]

     and

     I

    'm

     a

     [

    Last

     Name

    ]

     [

    Role

    ]

    !

     I

    'm

     a

     [

    what

    's

     your

     profession

    ]

     with

     a

     passion

     for

     [

    why

     this

     role

     interests

     you

    ].

     I

    'm

     always

     up

     for

     learning

     and

     exploring

     new

     things

    ,

     and

     I

     enjoy

     sharing

     my

     knowledge

     and

     experience

     with

     others

    .

     I

    'm

     a

     [

    what

    's

     your

     greatest

     strength

     or

     trait

    ],

     and

     I

    'm

     always

     ready

     to

     help

     others

     and

     learn

     from

     their

     mistakes

    .

     So

     if

     you

    're

     interested

     in

     learning

     about

     my

     journey

    ,

     I

    'd

     love

     to

     hear

     from

     you

    !

     #

    personal

    resume

     #

    self

    int

    roduction

     #

    v

    oc

    ational

    person

     #

    prof

    essional

    person

     #

    career

    path

     #

    career

    goals

     #

    career

    jour

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Love

    .

     It

     is

     a

     popular

     tourist

     destination

     known

     for

     its

     rich

     history

    ,

     art

    ,

     and

     culture

    .

     The

     city

     is

     located

     in

     the

     North

     of

     France

     and

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

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     Carnival

     celebrations

     and

     its

     diverse

     food

     scene

    ,

     including

     seafood

    ,

     wine

    ,

     and

     cheese

    .

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     ancient

     buildings

    ,

     romantic

     parks

    ,

     and

     cultural

     events

     that

     draw

     millions

     of

     visitors

     each

     year

    .

     Its

     legacy

     extends

     beyond

     its

     buildings

     and

     streets

     to

     its

     people

    ,

     who

     are

     known

     for

     their

     creativity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    ,

     with

     many

     potential

     trends

     that

     will

     shape

     the

     technology

    's

     trajectory

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     are

     predicted

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     AI

     has

     the

     potential

     to

     be

     a

     game

    -ch

    anger

    ,

     but

     it

    's

     important

     to

     ensure

     that

     its

     development

     is

     ethical

     and

     sustainable

    .

     This

     means

     that

     AI

     systems

     will

     be

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

    ,

     and

     that

     they

     will

     be

     built

     with

     the

     best

     ethical

     principles

     in

     mind

    .
    


    2

    .

     Rise

     of

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     (

    AV

    s

    )

     will

     be

     an

     increasingly

     common

     feature

     of

     our

     daily

     lives

    .

     This

     will

     require

     a

    



```python
llm.shutdown()
```
