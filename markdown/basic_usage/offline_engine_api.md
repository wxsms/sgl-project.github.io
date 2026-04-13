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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.82it/s]


    2026-04-13 05:41:25,162 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 05:41:25] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.21it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:02<00:03, 11.83it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 11.83it/s]

    Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:03<00:03, 11.83it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 25.14it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:03<00:00, 28.62it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:03<00:00, 35.79it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 41.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.60 GB):   2%|▏         | 1/58 [00:00<00:07,  8.04it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.57 GB):   2%|▏         | 1/58 [00:00<00:07,  8.04it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=72.57 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.48 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.16 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.09 GB):   3%|▎         | 2/58 [00:00<00:06,  8.96it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.09 GB):   9%|▊         | 5/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.09 GB):   9%|▊         | 5/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.09 GB):   9%|▊         | 5/58 [00:00<00:03, 15.59it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.08 GB):   9%|▊         | 5/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.08 GB):   9%|▊         | 5/58 [00:00<00:03, 15.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=71.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.97it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s] Capturing num tokens (num_tokens=896 avail_mem=71.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=832 avail_mem=71.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 34.67it/s]Capturing num tokens (num_tokens=832 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]

    Capturing num tokens (num_tokens=640 avail_mem=71.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=576 avail_mem=71.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  52%|█████▏    | 30/58 [00:00<00:00, 42.76it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 42.76it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  60%|██████    | 35/58 [00:01<00:00, 44.49it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=160 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s]

    Capturing num tokens (num_tokens=96 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s] Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.97it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=28 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.12it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.12it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=70.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.12it/s]Capturing num tokens (num_tokens=4 avail_mem=70.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.76it/s]


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
    Generated text:  Nalini. I am 14 years old. I have been reading books for a long time now. I have just finished reading "The Secret of the Six" and I will be reading "The Secret of the Nine". What will be my new reading material next?
    How many books have I read so far?
    How many days have I spent reading books?
    How many books have I read now?
    How many days have I spent reading books now?
    How many books have I read? How many days have I spent reading books?
    How many books have I read? How many days have I spent reading books? How many books have
    ===============================
    Prompt: The president of the United States is
    Generated text:  interested in analyzing the percentage of students in her university that are enrolled in STEM subjects. She decides to use data from a survey to estimate the population proportion. The survey results show that 60% of the students are enrolled in STEM subjects. If the university has 10,000 students, how many students are expected to be enrolled in STEM subjects?
    To determine the number of students expected to be enrolled in STEM subjects, we can use the formula for the expected population proportion. The expected population proportion \( p \) is given by the product of the total number of students \( N \) and the proportion \( p \
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. In 2011, the population of Paris was 2.5 million. In 2015, the population of Paris was 2.8 million. What is the percent increase in population between 2011 and 2015?
    
    To calculate the percent increase in the population of Paris from 2011 to 2015, you can use the following formula:
    
    \[
    \text{Percent Increase} = \left( \frac{\text{New Population} - \text{Old Population}}{\text{Old Population}} \right) \times 1
    ===============================
    Prompt: The future of AI is
    Generated text:  clearly coming. But it’s hard to predict when. What can you do now?
    It’s hard to predict when AI will become mainstream. This depends on the specific AI technology you are interested in. For example, you might expect the development of AI to take place over several decades, but some algorithms could be implemented now and in use very soon.
    One prediction about AI technology is that it will eventually replace human workers in several sectors. This could be done through the use of automation and artificial intelligence, and would likely result in the displacement of humans.
    Of course, this is just one prediction, and the future of AI technology will be shaped


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Parliament building. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art, and is a popular tourist destination for its beautiful architecture and historical sites. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. The city is also home to many international organizations and institutions
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI continues to evolve, we can expect to see even more applications in healthcare, including personalized medicine, disease
    


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
    Generated text:  [Your Name], and I'm a [Your Occupation] with [X years of experience] years of experience in [Your field of interest]. I am a dedicated and skilled professional who thrives in an environment that values teamwork, communication, and a passion for learning. I am always striving to improve my skills and stay up-to-date with the latest trends and technologies in my field. I am confident in my ability to contribute valuable insights and ideas to a variety of projects, and I am eager to work with a team that shares my values and goals. Thank you for taking the time to learn more about me. #SelfIntroduction #Professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France, located on the Mediterranean coast and the largest metropolitan area in Europe. It's famous for its iconic landmarks such as Notre-Dame Cathedral, Eiffel Tower, Louvre Museum, and Champs-Élysées. Paris is also known for its rich culture, arts, and cuisine. The city is home to over 10 million people and is a major economic and tourist center in Europe. 
    
    The French Parliament is located in the Palace of Versailles, which is also the seat of the government of France. Paris is a bustling hub of culture, commerce,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and exciting, with many potential trends to watch out for. Here are some of the trends that are likely to shape the future of AI:
    
    1. Advancements in machine learning and deep learning: With the development of more powerful computing power and the improvement of neural networks, we can expect to see even greater improvements in AI. This will allow us to train machines to learn more complex and sophisticated algorithms, which can be used for a wider range of applications.
    
    2. Integration of AI into everyday life: As AI becomes more ubiquitous, we can expect to see more of it integrated into our daily lives. For example, we may see more


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

     Sarah

    .

     I

    'm

     a

     software

     engineer

     who

     specializes

     in

     cloud

     computing

    .

     I

    'm

     passionate

     about

     exploring

     new

     technologies

     and

     collaborating

     with

     diverse

     teams

    .

     I

    'm

     also

     interested

     in

     learning

     about

     the

     latest

     trends

     in

     technology

     and

     staying

     up

    -to

    -date

     with

     the

     latest

     developments

     in

     the

     field

    .

     I

     enjoy

     working

     on

     complex

     projects

     and

     challenging

     myself

     to

     overcome

     obstacles

    .

     I

     value

     collaboration

    ,

     communication

    ,

     and

     adapt

    ability

     in

     my

     work

    .

     I

    'm

     excited

     to

     add

     my

     skills

     to

     your

     team

     and

     help

     you

     achieve

     your

     goals

    .

     What

     are

     your

     skills

     and

     what

     experience

     do

     you

     have

    ?

     Hello

    ,

     my

     name

     is

     Sarah

    .

     I

    'm

     a

     software

     engineer

     who

     specializes

     in

     cloud

     computing

    .

     I

    'm

     passionate

     about

     exploring

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

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

     the

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    ,

     among

     many

     others

    .

     The

     city

     is

     home

     to

     a

     diverse

     population

     of

     over

     

    1

    0

     million

     people

     and

     has

     been

     a

     major

     hub

     of

     European

     politics

    ,

     culture

    ,

     and

     fashion

     since

     its

     founding

     in

     the

     

    1

    2

    th

     century

    .

     Paris

     is

     also

     renowned

     for

     its

     rich

     history

    ,

     including

     the

     birth

    place

     of

     Marie

     Ant

    oin

    ette

    ,

     the

     Siege

     of

     

    1

    7

    9

    2

    ,

     and

     the

     Day

     of

     the

     Dead

     festival

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     combination

     of

     revolutionary

     advancements

     and

     significant

     challenges

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

     AI

    -driven

     advancements

     in

     healthcare

    :

     AI

     will

     enable

     healthcare

     providers

     to

     diagnose

     and

     treat

     diseases

     more

     accurately

     and

     quickly

    ,

     leading

     to

     significant

     improvements

     in

     patient

     outcomes

     and

     cost

     savings

    .
    


    2

    .

     AI

    -driven

     automation

     in

     manufacturing

    :

     AI

    -powered

     robots

     and

     automation

     systems

     will

     revolution

    ize

     manufacturing

    ,

     improving

     efficiency

    ,

     reducing

     waste

    ,

     and

     increasing

     productivity

    .
    


    3

    .

     AI

    -driven

     customer

     service

    :

     AI

     will

     enable

     more

     personalized

     and

     empath

    etic

     customer

     service

    ,

     leading

     to

     better

     customer

     satisfaction

     and

     loyalty

    .
    


    4

    .

     AI

    -driven

     education

    :

     AI

     will

     enable

     more

     personalized

     learning

     experiences

    ,

     enabling

     students

    



```python
llm.shutdown()
```
