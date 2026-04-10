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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.99it/s]


    2026-04-10 10:02:46,251 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 10:02:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.73s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.66it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.66it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.66it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.66it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.97it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.00it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.67it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.57it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 34.48it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 36.29it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 42.38it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 42.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   2%|▏         | 1/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   2%|▏         | 1/58 [00:00<00:05,  9.51it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.75 GB):   2%|▏         | 1/58 [00:00<00:05,  9.51it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.75 GB):   5%|▌         | 3/58 [00:00<00:05,  9.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.75 GB):   5%|▌         | 3/58 [00:00<00:05,  9.57it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.75 GB):   7%|▋         | 4/58 [00:00<00:06,  8.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.75 GB):   7%|▋         | 4/58 [00:00<00:06,  8.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.75 GB):   9%|▊         | 5/58 [00:00<00:06,  8.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.25 GB):   9%|▊         | 5/58 [00:00<00:06,  8.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.09 GB):   9%|▊         | 5/58 [00:00<00:06,  8.78it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=118.09 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.09 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.09 GB):  12%|█▏        | 7/58 [00:00<00:04, 10.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.09 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.08 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.08 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=118.08 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.08 GB):  21%|██        | 12/58 [00:00<00:02, 15.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.07 GB):  21%|██        | 12/58 [00:00<00:02, 15.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.07 GB):  21%|██        | 12/58 [00:00<00:02, 15.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.07 GB):  21%|██        | 12/58 [00:01<00:02, 15.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.07 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.06 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.65it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.06 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.06 GB):  26%|██▌       | 15/58 [00:01<00:02, 18.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.06 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.05 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.05 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.03 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=960 avail_mem=118.04 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s] Capturing num tokens (num_tokens=896 avail_mem=118.04 GB):  31%|███       | 18/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=896 avail_mem=118.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=832 avail_mem=118.04 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.01it/s]

    Capturing num tokens (num_tokens=768 avail_mem=118.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=704 avail_mem=118.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=640 avail_mem=118.03 GB):  40%|███▉      | 23/58 [00:01<00:01, 28.01it/s]Capturing num tokens (num_tokens=640 avail_mem=118.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=576 avail_mem=118.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=512 avail_mem=118.01 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=480 avail_mem=118.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.69it/s]Capturing num tokens (num_tokens=448 avail_mem=118.03 GB):  47%|████▋     | 27/58 [00:01<00:01, 29.69it/s]

    Capturing num tokens (num_tokens=448 avail_mem=118.03 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=416 avail_mem=118.03 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=384 avail_mem=118.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=352 avail_mem=118.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=320 avail_mem=118.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=320 avail_mem=118.01 GB):  60%|██████    | 35/58 [00:01<00:00, 27.44it/s]Capturing num tokens (num_tokens=288 avail_mem=118.01 GB):  60%|██████    | 35/58 [00:01<00:00, 27.44it/s]

    Capturing num tokens (num_tokens=256 avail_mem=118.01 GB):  60%|██████    | 35/58 [00:01<00:00, 27.44it/s]Capturing num tokens (num_tokens=240 avail_mem=118.01 GB):  60%|██████    | 35/58 [00:01<00:00, 27.44it/s]Capturing num tokens (num_tokens=240 avail_mem=118.01 GB):  66%|██████▌   | 38/58 [00:01<00:00, 22.05it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.00 GB):  66%|██████▌   | 38/58 [00:02<00:00, 22.05it/s]Capturing num tokens (num_tokens=208 avail_mem=118.00 GB):  66%|██████▌   | 38/58 [00:02<00:00, 22.05it/s]Capturing num tokens (num_tokens=192 avail_mem=118.00 GB):  66%|██████▌   | 38/58 [00:02<00:00, 22.05it/s]

    Capturing num tokens (num_tokens=192 avail_mem=118.00 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=176 avail_mem=118.00 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=160 avail_mem=117.99 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=144 avail_mem=117.99 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=128 avail_mem=117.99 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=112 avail_mem=117.98 GB):  71%|███████   | 41/58 [00:02<00:01, 14.85it/s]Capturing num tokens (num_tokens=112 avail_mem=117.98 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s]Capturing num tokens (num_tokens=96 avail_mem=117.98 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s] Capturing num tokens (num_tokens=80 avail_mem=117.98 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s]Capturing num tokens (num_tokens=64 avail_mem=117.52 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s]Capturing num tokens (num_tokens=48 avail_mem=117.51 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s]

    Capturing num tokens (num_tokens=32 avail_mem=117.45 GB):  79%|███████▉  | 46/58 [00:02<00:00, 20.03it/s]Capturing num tokens (num_tokens=32 avail_mem=117.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=28 avail_mem=115.89 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=24 avail_mem=114.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=20 avail_mem=114.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=16 avail_mem=114.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=12 avail_mem=114.44 GB):  88%|████████▊ | 51/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=12 avail_mem=114.44 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.66it/s]Capturing num tokens (num_tokens=8 avail_mem=114.44 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.66it/s] Capturing num tokens (num_tokens=4 avail_mem=114.43 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.66it/s]

    Capturing num tokens (num_tokens=4 avail_mem=114.43 GB): 100%|██████████| 58/58 [00:02<00:00, 21.37it/s]


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
    Generated text:  Lena and I'm a PhD candidate in the Neurosciences Research Group at the University of Bristol. My research focuses on understanding the role of the brain in learning, memory, and decision making, with a particular emphasis on how neural circuits can be modified to affect learning and decision making through the use of brain-machine interfaces (BMI).
    I completed my undergraduate studies in Botany at the University of Sheffield and my PhD in Experimental Biology at the University of Bristol. My research has been funded by the UK government and my research has been published in peer-reviewed journals. I have been awarded the Royal Society's Bachelor of Science Award, the University of
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 42 years old. 7 years ago, he was the age of a certain person. If the president is currently 40 years old, how old was the person 7 years ago?
    
    Let's solve the problem step by step.
    
    1. The current age of the president is 40 years.
    2. The president is currently 42 years old.
    3. Seven years ago, the president was \(42 - 7 = 35\) years old.
    4. At that time, the president was the same age as the person who was 7 years old 7 years ago.
    
    So,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. ( )
    A: √
    B: ×
    C: 
    D:
    
    To determine whether the statement "The capital of France is Paris" is true or false, we need to understand the definition of a capital city. A capital city is the largest and most populous city within a country, serving as the seat of government and the main administrative center of that country.
    
    Let's analyze the information given in the problem:
    
    1. The statement says that Paris is the capital of France.
    2. However, Paris is not the capital of France. The capital of France is not Paris but rather Paris itself. The capital of France is
    ===============================
    Prompt: The future of AI is
    Generated text:  looking bright! Recent research has shown that the use of AI has not only improved our lives but also has the potential to completely revolutionize industries such as healthcare, finance, transportation, and education. However, there are also ethical concerns surrounding the use of AI and its impact on society as a whole. In this essay, I will discuss the advantages and disadvantages of using AI in each of the mentioned industries, as well as the ethical concerns surrounding its use. Finally, I will provide recommendations for how to responsibly use AI in the future.
    
    In healthcare, the use of AI has the potential to revolutionize the industry by improving diagnostics, reducing costs


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [number] degree in [field of study]. I'm a [job title] at [company name]. I'm passionate about [what you do for a living]. I'm looking forward to meeting you and learning more about you. How can I help you today? I'm looking forward to meeting you and learning more about you. What can you tell me about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is the largest city in France and the second-largest city in the European Union, with a population of over 2.7 million people. Paris is known for its beautiful architecture, world-renowned museums, and annual cultural events such as the Eiffel Tower and the Louvre Museum. The city is also home to many famous landmarks, including the Notre-Dame Cathedral, the Arc de Triomphe, and the Champs-Élysées. Paris is a popular tourist destination and a major economic center in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the potential trends that are likely to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the future.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  [Your Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests, hobbies, and what you're passionate about. If you're feeling adventurous, I can offer you a chance to see the world through my eyes and get a taste of the fun. Let's chat! 🌍✨
    You can use any language or cultural references you'd like, but make sure to stay neutral and respectful. Let's make this experience an exciting one! 🌍✨
    Here's a possible introduction:
    Hello, my name is [Your Name], and I'm
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city and the third-most populous city in the world by population, with a population of approximately 2.1 million.
    
    That's correct! Paris is indeed the capital of France and is the largest city and the third-most populous city in the world by population, with a population of approximately 2.1 million. It's a renowned city known for its rich history, art, and architecture, as well as its vibrant culture and food scene. The city is also home to many famous landmarks, including the Eiffel Tower and the Louvre Museum. Paris is a UNESCO World Heritage site and a major economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of technological developments and advancements. Some of the most likely trends include:
    
    1. Improved machine learning and deep learning: As AI technology continues to advance, it is likely to see even more significant improvements in its ability to learn and make predictions. This could lead to more accurate predictions of natural events, such as weather patterns or market trends, and even to more complex artificial intelligence systems that can interact with humans in new and innovative ways.
    
    2. Enhanced natural language processing: As AI technology continues to improve, it is likely to see even more significant advancements in natural language processing. This could lead to even more sophisticated speech


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

    Your

     Name

    ]

     and

     I

     am

     a

     highly

     experienced

     and

     experienced

     customer

     service

     representative

     for

     [

    Company

     Name

    ].

     I

     have

     a

     passion

     for

     helping

     customers

     and

     always

     strive

     to

     provide

     the

     best

     service

     to

     those

     who

     choose

     to

     work

     with

     me

    .

     I

     am

     a

     friendly

     and

     approach

    able

     person

     who

     always

     tries

     to

     build

     positive

     relationships

     with

     my

     clients

    .

     I

     am

     confident

    ,

     organized

    ,

     and

     reliable

    ,

     and

     I

     am

     always

     ready

     to

     assist

     with

     any

     issue

     or

     problem

     that

     the

     customer

     may

     have

    .

     I

     am

     a

     true

     customer

     service

     expert

     who

     is

     dedicated

     to

     making

     sure

     that

     every

     customer

     receives

     the

     best

     possible

     experience

    .

     I

     look

     forward

     to

     working

     with

     you

    .

     What

     do

     you

     do

     as

     a

     customer

     service

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

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

    .


    Paris

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

    .

     The

     city

     is

     a

     major

     center

     of

     culture

    ,

     politics

    ,

     and

     economy

    ,

     and

     attracts

     millions

     of

     tourists

     each

     year

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

     historical

     landmarks

    ,

     art

     museums

    ,

     and

     trendy

     neighborhoods

     blending

     seamlessly

     into

     the

     urban

     landscape

    .

     The

     city

    's

     architecture

    ,

     including

     its

     towering

     sp

    ires

    ,

     orn

    ate

     cath

    ed

    r

    als

    ,

     and

     Gothic

     cath

    ed

    r

    als

    ,

     are

     a

     testament

     to

     the

     city

    's

     rich

     history

    .

     The

     city

     is

     also

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     exciting

     possibilities

     and

     possibilities

     of

     innovation

    ,

     but

     there

     are

     some

     potential

     downs

    ides

     that

     need

     to

     be

     taken

     into

     account

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     integration

     with

     human

     decision

    -making

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     become

     more

     integrated

     with

     human

     decision

    -making

    .

     This

     means

     that

     AI

     will

     need

     to

     be

     trained

     on

     data

     that

     is

     representative

     of

     human

     behavior

     to

     make

     accurate

     predictions

     and

     decisions

    .
    


    2

    .

     Enhanced

     safety

     and

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

     daily

     lives

    ,

     it

     will

     become

     more

     important

     to

     consider

     the

     potential

     risks

     and

     ethical

     implications

     of

     AI

    .

     This

     includes

     issues

     like

     bias

    ,

     privacy

    ,

     and

     transparency

    .
    


    3

    



```python
llm.shutdown()
```
