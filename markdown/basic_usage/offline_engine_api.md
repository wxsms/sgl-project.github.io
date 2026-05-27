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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:40,  3.87s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:12,  3.96it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s] 

    Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  8.17it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.42it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 17.45it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 24.13it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:04<00:00, 32.33it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 38.78it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.04it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.03 GB):   2%|▏         | 1/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.00 GB):   2%|▏         | 1/58 [00:00<00:06,  8.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.00 GB):   2%|▏         | 1/58 [00:00<00:06,  8.22it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.00 GB):   5%|▌         | 3/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.00 GB):   5%|▌         | 3/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.00 GB):   5%|▌         | 3/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.99 GB):   5%|▌         | 3/58 [00:00<00:03, 13.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.99 GB):  10%|█         | 6/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.98 GB):  10%|█         | 6/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.98 GB):  10%|█         | 6/58 [00:00<00:02, 18.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.98 GB):  10%|█         | 6/58 [00:00<00:02, 18.51it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.98 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.97 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.97 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.51 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.10it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=54.51 GB):  21%|██        | 12/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=51.45 GB):  21%|██        | 12/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.45 GB):  21%|██        | 12/58 [00:00<00:02, 18.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=51.45 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=51.45 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=51.45 GB):  24%|██▍       | 14/58 [00:00<00:02, 17.86it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=51.45 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=51.44 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.44 GB):  28%|██▊       | 16/58 [00:00<00:02, 18.09it/s]Capturing num tokens (num_tokens=1792 avail_mem=51.44 GB):  31%|███       | 18/58 [00:01<00:02, 18.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=51.44 GB):  31%|███       | 18/58 [00:01<00:02, 18.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=51.43 GB):  31%|███       | 18/58 [00:01<00:02, 18.41it/s]Capturing num tokens (num_tokens=1024 avail_mem=51.41 GB):  31%|███       | 18/58 [00:01<00:02, 18.41it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=51.41 GB):  36%|███▌      | 21/58 [00:01<00:01, 19.88it/s]Capturing num tokens (num_tokens=960 avail_mem=51.43 GB):  36%|███▌      | 21/58 [00:01<00:01, 19.88it/s] Capturing num tokens (num_tokens=896 avail_mem=51.43 GB):  36%|███▌      | 21/58 [00:01<00:01, 19.88it/s]Capturing num tokens (num_tokens=832 avail_mem=51.42 GB):  36%|███▌      | 21/58 [00:01<00:01, 19.88it/s]Capturing num tokens (num_tokens=832 avail_mem=51.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=768 avail_mem=51.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=704 avail_mem=51.42 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.15it/s]

    Capturing num tokens (num_tokens=640 avail_mem=51.41 GB):  41%|████▏     | 24/58 [00:01<00:01, 21.15it/s]Capturing num tokens (num_tokens=640 avail_mem=51.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.17it/s]Capturing num tokens (num_tokens=576 avail_mem=51.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.17it/s]Capturing num tokens (num_tokens=512 avail_mem=51.40 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.17it/s]Capturing num tokens (num_tokens=480 avail_mem=51.41 GB):  47%|████▋     | 27/58 [00:01<00:01, 22.17it/s]Capturing num tokens (num_tokens=480 avail_mem=51.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.16it/s]Capturing num tokens (num_tokens=448 avail_mem=51.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.16it/s]Capturing num tokens (num_tokens=416 avail_mem=51.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.16it/s]

    Capturing num tokens (num_tokens=384 avail_mem=51.41 GB):  52%|█████▏    | 30/58 [00:01<00:01, 23.16it/s]Capturing num tokens (num_tokens=384 avail_mem=51.41 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.85it/s]Capturing num tokens (num_tokens=352 avail_mem=51.40 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.85it/s]Capturing num tokens (num_tokens=320 avail_mem=51.40 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.85it/s]Capturing num tokens (num_tokens=288 avail_mem=51.39 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.85it/s]Capturing num tokens (num_tokens=256 avail_mem=51.39 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.85it/s]Capturing num tokens (num_tokens=256 avail_mem=51.39 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=240 avail_mem=51.39 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=224 avail_mem=51.38 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.48it/s]

    Capturing num tokens (num_tokens=208 avail_mem=51.38 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=192 avail_mem=51.38 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.48it/s]Capturing num tokens (num_tokens=192 avail_mem=51.38 GB):  71%|███████   | 41/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=176 avail_mem=51.38 GB):  71%|███████   | 41/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=160 avail_mem=51.37 GB):  71%|███████   | 41/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=144 avail_mem=51.37 GB):  71%|███████   | 41/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=128 avail_mem=51.37 GB):  71%|███████   | 41/58 [00:01<00:00, 28.40it/s]Capturing num tokens (num_tokens=128 avail_mem=51.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.05it/s]Capturing num tokens (num_tokens=112 avail_mem=51.37 GB):  78%|███████▊  | 45/58 [00:01<00:00, 30.05it/s]

    Capturing num tokens (num_tokens=96 avail_mem=51.36 GB):  78%|███████▊  | 45/58 [00:02<00:00, 30.05it/s] Capturing num tokens (num_tokens=80 avail_mem=51.36 GB):  78%|███████▊  | 45/58 [00:02<00:00, 30.05it/s]Capturing num tokens (num_tokens=64 avail_mem=51.35 GB):  78%|███████▊  | 45/58 [00:02<00:00, 30.05it/s]Capturing num tokens (num_tokens=64 avail_mem=51.35 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=48 avail_mem=51.35 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=32 avail_mem=51.35 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=28 avail_mem=51.34 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.78it/s]Capturing num tokens (num_tokens=24 avail_mem=51.34 GB):  84%|████████▍ | 49/58 [00:02<00:00, 30.78it/s]

    Capturing num tokens (num_tokens=24 avail_mem=51.34 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.95it/s]Capturing num tokens (num_tokens=20 avail_mem=51.34 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.95it/s]Capturing num tokens (num_tokens=16 avail_mem=51.34 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.95it/s]Capturing num tokens (num_tokens=12 avail_mem=51.33 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.95it/s]Capturing num tokens (num_tokens=8 avail_mem=51.33 GB):  91%|█████████▏| 53/58 [00:02<00:00, 31.95it/s] Capturing num tokens (num_tokens=8 avail_mem=51.33 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.90it/s]Capturing num tokens (num_tokens=4 avail_mem=51.32 GB):  98%|█████████▊| 57/58 [00:02<00:00, 32.90it/s]Capturing num tokens (num_tokens=4 avail_mem=51.32 GB): 100%|██████████| 58/58 [00:02<00:00, 24.55it/s]


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
    Generated text:  [Your Name] and I'm a professional writer with a strong background in English language and literature. I have a passion for writing and I enjoy crafting stories that are engaging, thought-provoking, and demonstrate my unique voice and perspective. I strive to create works that are well-researched, well-structured, and demonstrate my ability to convey complex ideas in a clear and concise manner. What's your writing style? As a professional writer, my writing style is characterized by a combination of formal and informal elements. I prefer to use formal language to express my ideas and to structure my writing in a way that is both informative and engaging
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man, and the president of the United States must be a citizen, so the president of the United States is a citizen. Is this reasoning correct?
    A. Correct
    B. Incorrect
    C. Insufficient information
    D. Impossible to determine
    Answer: B
    
    A student is trying to determine the surface area of a rectangular prism. During his experiments, he used the following data:
    
    - The length of the base (l) is 8 meters.
    - The height of the prism (h) is 6 meters.
    - The length of the diagonal (d) is 12 meters.
    
    1. Based on the dimensions
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Lyon
    C. Lille
    D. Nancy
    Answer: A
    
    The Chinese name of the airport is ____
    A. Nanning
    B. Kunming
    C. Shanghai
    D. Guangzhou
    Answer: B
    
    When operating with a high-pressure water gun, pay attention to prevent high-pressure water from flowing into the nose and lips of the operator. This is to avoid _______
    A. Cuts
    B. Burns
    C. Poisoning
    D. Asphyxiation
    Answer: B
    
    What was the name of the first store on the road in the novel '
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but there are obstacles to realizing this potential. In today’s world, there is a lot of data, but it is often lacking in quality. The quality of data is critical to improving AI performance, as poor data leads to suboptimal results. However, the data is a commodity, and it can be expensive to collect. There is also a lot of unstructured data that is not easily classified or structured, making it even harder to make sense of it. Additionally, data security is a concern as the data can be manipulated or stolen if not handled properly.
    AI is still in its early stages, and there is still a


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


    Generated text:  Paris. It is the largest city in the country and the seat of the French government and the largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also a major tourist destination and a major economic center. The city is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular destination for both locals and tourists, and is considered one of the most beautiful cities in the world. The city is also home to many cultural institutions, including the Musée d'Orsay, the Musée Rod
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even more widespread
    


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
    Generated text:  [Name]. I'm a [background] with [position] experience. What can you tell me about yourself? Hi there! I'm a [background] with [position] experience. What can you tell me about yourself? I'm [background]. Hi there! I'm a [background] with [position] experience. What can you tell me about yourself? Hi there! I'm [background] with [position] experience. What can you tell me about yourself? Hey, I'm [Name]. How can I help you today? Hi, my name is [Name]. How can I help you today? Hey, I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light" and the "City of Literature". 
    
    Is the following statement true? "Most people in Paris have gotten sick from eating in the streets."
    
    No, the statement is false. While it may be true that some people may get sick from eating in the streets, this is not the case for Parisians. Paris is a tourist destination known for its vibrant nightlife, so it is more likely that many people will enjoy walking around and exploring the city rather than being sick from eating in the streets. 
    
    So, the accurate statement about Paris is: "Paris is known for its vibrant nightlife,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid and unprecedented advancements, as it becomes more integrated into our daily lives. Some possible future trends include:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, enabling machines to learn from humans and adapt to new situations.
    
    2. More diverse use cases: AI is likely to be more widely used for various purposes, such as healthcare, finance, education, and transportation, with more diverse use cases expected in the future.
    
    3. Higher level of transparency: AI systems will become more transparent, allowing humans to understand how the system makes decisions and how it arrived at its conclusions.
    
    


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

    ].

     I

    'm

     a

     [

    Job

     Title

    ]

     with

     [

    Company

    ]

     for

     the

     past

     [

    Number

     of

     Years

    ]

     years

    .
    


    I

    'm

     passionate

     about

     [

    Why

     You

     Love

     This

     Job

    /

    Field

    ].

     I

    've

     always

     been

     an

     energetic

     and

     driven

     individual

     who

     thr

    ives

     on

     the

     challenges

     and

     opportunities

     of

     the

     industry

    .

     I

    'm

     also

     a

     team

     player

     and

     enjoy

     working

     with

     others

     to

     achieve

     our

     goals

    .
    


    As

     someone

     who

     is

     always

     looking

     for

     new

     ways

     to

     learn

     and

     grow

     professionally

    ,

     I

    'm

     eager

     to

     continue

     my

     education

     and

     develop

     my

     skills

     in

     my

     field

    .

     I

    'm

     always

     on

     the

     lookout

     for

     opportunities

     to

     contribute

     to

     our

     company

     and

     stay

     ahead

     of

     the

     curve

    .
    


    I

    'm

     also

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     and

     most

     populous

     city

     of

     France

    .

     It

     is

     located

     in

     the

     Î

    le

     de

     la

     C

    ite

     region

     of

     the

     Lo

    ire

     Valley

     on

     the

     Atlantic

     coast

    ,

     overlooking

     the

     Mediterranean

     Sea

    .

     The

     city

     is

     the

     second

     largest

     urban

     area

     in

     the

     European

     Union

     and

     one

     of

     the

     world

    's

     most

     important

     cultural

    ,

     economic

     and

     political

     centers

    .

     Paris

     has

     been

     a

     center

     for

     culture

     and

     education

     since

     antiqu

    ity

     and

     has

     been

     home

     to

     many

     of

     France

    ’s

     leading

     figures

     in

     politics

    ,

     literature

    ,

     art

     and

     music

    .

     The

     city

     is

     renowned

     for

     its

     historical

     architecture

     and

     its

     often

    -over

    look

    ed

     charm

    ,

     and

     its

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     significant

     trends

    ,

     including

    :
    


     

     

    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     systems

     will

     become

     more

     integrated

     with

     human

     intelligence

    ,

     leading

     to

     greater

     efficiency

     and

     effectiveness

    .


     

     

    2

    .

     Development

     of

     more

     sophisticated

     natural

     language

     processing

    :

     AI

     systems

     will

     be

     able

     to

     understand

     and

     respond

     to

     natural

     language

     in

     a

     more

     sophisticated

     way

    ,

     leading

     to

     greater

     automation

     and

     efficiency

    .


     

     

    3

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     become

     more

     integrated

     with

     healthcare

    ,

     leading

     to

     better

     diagnosis

    ,

     treatment

    ,

     and

     prevention

     of

     diseases

    .


     

     

    4

    .

     Development

     of

     AI

    -powered

     virtual

     assistants

    :

     AI

     will

     be

     used

     to

     create

     more

     advanced

     virtual

    



```python
llm.shutdown()
```
