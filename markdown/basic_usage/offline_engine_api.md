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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.66it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.66it/s]


    2026-05-12 19:35:22,830 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 19:35:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:37,  3.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:37,  3.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:37,  3.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:37,  3.81s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:40,  1.33it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:40,  1.33it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:04<00:15,  3.21it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s] Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:04<00:05,  7.50it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 13.59it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]

    Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 20.54it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 30.70it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 41.34it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.72 GB):   3%|▎         | 2/58 [00:00<00:04, 11.81it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.71 GB):   3%|▎         | 2/58 [00:00<00:04, 11.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=54.71 GB):   3%|▎         | 2/58 [00:00<00:04, 11.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.71 GB):   7%|▋         | 4/58 [00:00<00:04, 13.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.70 GB):   7%|▋         | 4/58 [00:00<00:04, 13.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.69 GB):   7%|▋         | 4/58 [00:00<00:04, 13.44it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=54.69 GB):  10%|█         | 6/58 [00:00<00:03, 13.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.68 GB):  10%|█         | 6/58 [00:00<00:03, 13.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.67 GB):  10%|█         | 6/58 [00:00<00:03, 13.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.67 GB):  10%|█         | 6/58 [00:00<00:03, 13.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.67 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.66 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.65 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.30it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=54.63 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.63 GB):  21%|██        | 12/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.64 GB):  21%|██        | 12/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.62 GB):  21%|██        | 12/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.61 GB):  21%|██        | 12/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.62 GB):  21%|██        | 12/58 [00:00<00:02, 20.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.23it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=54.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.60 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.23it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.57 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=960 avail_mem=54.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s] Capturing num tokens (num_tokens=896 avail_mem=54.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=832 avail_mem=54.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=768 avail_mem=54.58 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.84it/s]Capturing num tokens (num_tokens=768 avail_mem=54.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]Capturing num tokens (num_tokens=704 avail_mem=54.57 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]Capturing num tokens (num_tokens=640 avail_mem=54.56 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]

    Capturing num tokens (num_tokens=576 avail_mem=54.56 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]Capturing num tokens (num_tokens=512 avail_mem=54.54 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]Capturing num tokens (num_tokens=480 avail_mem=54.56 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.87it/s]Capturing num tokens (num_tokens=480 avail_mem=54.56 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=448 avail_mem=54.55 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=416 avail_mem=54.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=384 avail_mem=54.54 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=352 avail_mem=54.53 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=320 avail_mem=54.52 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.81it/s]Capturing num tokens (num_tokens=320 avail_mem=54.52 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=288 avail_mem=54.52 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]

    Capturing num tokens (num_tokens=256 avail_mem=54.51 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=240 avail_mem=54.51 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=224 avail_mem=54.50 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=208 avail_mem=54.49 GB):  60%|██████    | 35/58 [00:01<00:00, 37.90it/s]Capturing num tokens (num_tokens=208 avail_mem=54.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=192 avail_mem=54.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=176 avail_mem=54.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=160 avail_mem=54.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=144 avail_mem=54.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=128 avail_mem=54.46 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.35it/s]

    Capturing num tokens (num_tokens=128 avail_mem=54.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=112 avail_mem=54.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=96 avail_mem=54.45 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s] Capturing num tokens (num_tokens=80 avail_mem=54.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=64 avail_mem=54.44 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=48 avail_mem=54.38 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.46it/s]Capturing num tokens (num_tokens=48 avail_mem=54.38 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=32 avail_mem=54.37 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=28 avail_mem=54.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=24 avail_mem=54.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=20 avail_mem=54.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]

    Capturing num tokens (num_tokens=16 avail_mem=54.36 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.20it/s]Capturing num tokens (num_tokens=16 avail_mem=54.36 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=12 avail_mem=54.35 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=8 avail_mem=54.35 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.64it/s] Capturing num tokens (num_tokens=4 avail_mem=54.32 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.64it/s]Capturing num tokens (num_tokens=4 avail_mem=54.32 GB): 100%|██████████| 58/58 [00:01<00:00, 32.57it/s]


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
    Generated text:  Casey and I'm a 24 year old doctor in the field of public health. I'm interested in neuroendocrinology, which is the study of how the brain regulates bodily functions. I've been interested in this field for a while now, but this year I've been trying to find more resources to learn more about it. I was wondering if you could give me some information about the role of the hippocampus in the brain. Also, would you be able to provide some information about the current research in this area? I'm particularly interested in the effects of chronic stress and the role of the hippocampus in this. Thank
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many troops to deploy to Iraq. He has decided that if the number of troops is less than 2000, he will deploy troops at the beginning of each month. Otherwise, he will deploy troops at the end of each month. If the total number of troops is 2000, he will deploy the same number of troops at the beginning and end of each month. If the number of troops is exactly 2000, he will deploy a certain number of troops at the beginning and end of each month.
    
    What is the largest possible number of troops that the president can deploy at the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris, the city of love, is the eternal city. Paris is the home of the world’s most famous landmarks: the Eiffel Tower, the Louvre, the Notre Dame Cathedral, and many, many more. Paris was born in the year of the ‘37’ and has been an eternal city ever since. This city was founded by the Romans to protect and defend their city-states. But the French turned it into a city that is now famous for its romantic history and breathtaking architecture. The first people who built Paris were the ancient Romans. They had no need of roads or bridges for the city. They simply
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but not for everyone. Here's what you need to know to make sure your AI system will be successful.
    Is AI good for the future of work? How can we use AI to make education more accessible?
    Is it ethical for AI to be used without human intervention?
    Is AI a threat to the human race?
    This post is part of a series on AI ethics. Here's what you need to know to make sure your AI is successful.
    By John Smith, COO of SureAI
    In this video, I discuss the ethical implications of the use of AI. This post is part of a series on AI ethics. Here


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major transportation hub, with many major highways and airports connecting it to other parts of France and the world. The city is known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a popular tourist destination, with millions of visitors each year. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes. This integration could lead to more sophisticated and adaptive AI systems that can learn from feedback and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI systems become more integrated with human intelligence, there will be increased emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as greater transparency and accountability in AI systems.
    
    3. Increased use of AI in healthcare
    


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
    Generated text:  [Your Name]. I'm a [job title] at [company name]. I've been working in [your role] at [company name] for [number] years, where I've grown to have [specific skills or interests] and learned a lot from my colleagues. I'm passionate about [your passion or interest]. If you're interested in joining my team, feel free to reach out to me. You'll be seeing me almost daily, so I'm always available to answer questions or assist with projects. Would you like to know more about me? [Your Name] [Your Contact Information] [Your LinkedIn Profile Link (
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is a cultural, historical, and economic center.
    Paris is a bustling metropolis with a rich history and a vibrant culture. It is known for its rich art, music, and food, as well as its iconic landmarks such as Notre-Dame Cathedral and the Eiffel Tower. Paris is a world-renowned destination for tourists and is a popular tourist attraction in France. The city is also home to numerous museums, theaters, and museums, such as the Louvre and the Musée d'Orsay. Paris is a cultural center that is home to many important institutions of higher learning, including the University of Paris.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly exciting, and there are several trends that are likely to shape the technology's direction in the coming years. Here are some potential trends in the AI field:
    
    1. Advancements in machine learning and deep learning: With the help of more powerful hardware, AI models will become more complex and able to learn from vast amounts of data. This will lead to faster and more accurate models, making it easier to develop more complex algorithms.
    
    2. More interaction between AI and humans: As AI becomes more sophisticated, it will become more adept at understanding and interacting with human-like beings, such as humans and other AI systems. This will create a more


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

     [

    Your

     Age

    ]

     years

     old

    .

     I

     was

     born

     and

     raised

     in

     [

    Your

     Birth

    place

    ]

     and

     I

     am

     a

     very

     [

    Your

     Character

    istic

    ].

     I

     have

     a

     great

     passion

     for

     [

    Your

     Favorite

     Activity

    ]

     and

     I

    'm

     always

     looking

     for

     [

    Your

     Goal

    ].

     I

     have

     a

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

     eager

     to

     learn

     more

     about

     [

    Your

     Profession

    ].

     And

     what

     is

     your

     favorite

     hobby

    ?

     I

    'm

     a

     big

     fan

     of

     [

    Your

     Hobby

    ]

     and

     I

     love

     spending

     time

     with

     my

     family

     and

     friends

    .

     I

     believe

     that

     taking

     care

     of

     the

     ones

     we

     love

     is

     one

     of

     the

     most

     important

     things

     in

     life

    .

     I

     have

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    (

    1

     point

    )

     To

     complete

     this

     task

    ,

     I

     will

     extract

     and

     summarize

     the

     given

     text

    .

     Then

     I

     will

     present

     a

     concise

     factual

     statement

     about

     Paris

    .

     To

     ensure

     accuracy

    ,

     I

     will

     use

     the

     following

     key

     points

    :


    1

    .

     Name

     of

     the

     city

     (

    Paris

    )


    2

    .

     Capital

     of

     France

     (

    yes

    )


    3

    .

     Type

     of

     information

     (

    f

    actual

     statement

    )


    4

    .

     Additional

     context

     (

    not

     applicable

     in

     this

     case

    )
    


    Thus

    ,

     the

     provided

     text

     states

     that

     the

     capital

     of

     France

     is

     Paris

    .

     Based

     on

     the

     above

     key

     points

    ,

     here

     is

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

     capital

     of

     France

     and

     the

     world

    's

     

    1

    5

    th

     most

     populous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     exciting

    ,

     with

     potential

     applications

     in

     a

     wide

     range

     of

     industries

     and

     fields

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

     efficiency

     and

     accuracy

    :

     AI

     is

     expected

     to

     make

     machines

     and

     systems

     more

     efficient

     and

     accurate

     in

     performing

     tasks

    ,

     which

     could

     lead

     to

     new

     ways

     of

     doing

     business

    ,

     improving

     healthcare

     outcomes

    ,

     and

     enhancing

     education

    .
    


    2

    .

     Aug

    mented

     reality

     and

     virtual

     reality

    :

     AI

    -powered

     technologies

     will

     continue

     to

     advance

    ,

     leading

     to

     more

     immersive

     and

     engaging

     experiences

     in

     our

     daily

     lives

    .
    


    3

    .

     Personal

    ized

     medicine

    :

     AI

     is

     expected

     to

     make

     it

     possible

     to

     develop

     more

     personalized

     treatments

     for

     diseases

    ,

     leading

     to

     improved

     patient

     outcomes

     and

     cost

     savings

    .
    


    4

    .

     Autonomous

    



```python
llm.shutdown()
```
