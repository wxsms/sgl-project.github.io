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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.28it/s]


    2026-05-02 08:00:44,049 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 08:00:44] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:51,  5.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.83it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.57it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 20.91it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s] Capturing num tokens (num_tokens=896 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.73it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.22it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 43.76it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.47it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.71it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.71it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.71it/s]

    Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=128 avail_mem=72.50 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=112 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=96 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s] Capturing num tokens (num_tokens=80 avail_mem=72.49 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=64 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=48 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=32 avail_mem=72.48 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=28 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=24 avail_mem=72.47 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]

    Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.76it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.69it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.69it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 41.44it/s]


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
    Generated text:  Jane Doe. I am a 35-year-old professional who has been employed by the same company for over 20 years. My work is focused on managing a team of 15 team members. I have been working with this company for nearly three years. I am always looking for opportunities to grow and improve my skills. I am interested in learning about the latest trends in the industry and how I can benefit from them. I am particularly interested in how to work in a cross-cultural environment. I am eager to apply for the job at your company and I would like to share my skills and experience with you. How can I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to pick a new flag for the country. There are two designs he has considered. The first design is a circle that is divided into three parts, with one part shaded red and the other two parts shaded blue. The second design is a circle divided into three equal parts, with one part shaded red and the other two parts shaded blue. Which design would be more aesthetically pleasing to the president? To determine which design would be more aesthetically pleasing, we need to consider the aesthetic appeal of both designs.
    
    1. **First Design:**
       - The first design is a circle divided into three parts, with one part
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Lyon
    C. Nancy
    D. Montmartre
    
    A. Paris
    
    Paris is the capital of France and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Lyon, Nancy, and Montmartre are all located in different regions of France, but they are not capitals. Therefore, the correct answer is A. Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  likely to involve AI that can think beyond the limits of the current human mind. This is the kind of AI that can understand and interpret emotions and potentially empathize with others. It would be able to have a "mental empathy" that could help us comprehend and navigate complex social situations. With the advent of deep learning and neural networks, AI could start to tap into the vast knowledge and wisdom that humans have accumulated over centuries. This could lead to a future where AI is more intelligent and versatile than ever before. It could also have the potential to revolutionize industries such as healthcare, finance, and transportation, by enabling us to better understand human


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and I have [number] years of experience in [job title]. I'm a [job title] at [company name] and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name] and I'm always looking for ways to [describe your job role]. I'm a [job title] at [company name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city is also home to many famous French artists, writers, and musicians
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: AI is likely to become more prevalent in manufacturing, transportation, and other industries, where it can automate repetitive tasks and increase efficiency. This could lead to the creation of new jobs and the displacement of humans in certain roles.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be a need for ethical guidelines and regulations to ensure that AI is used in a responsible and fair manner. This
    


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
    Generated text:  [insert name]. I am a [insert your occupation or profession]. I've been working in this field for [insert number of years] years now. I enjoy my work, and I'm always looking for ways to improve my skills and knowledge. What about you? Do you have a profession or occupation? What is your current status in the field? I'd love to learn more about you and what you do. Let's have a conversation! [insert your name] [insert your profession or occupation]. [insert your current status in the field]. What can I do for you? [insert your experience or skills]. [insert your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that was founded by the Romans in the 1st century BC and became the political and cultural center of France during the Middle Ages. 
    
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    - **-** 
    -
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid progress and the development of increasingly sophisticated and versatile AI systems. Some potential trends that may be expected include:
    
    1. Improved integration of AI with other technologies: AI systems are already being integrated into a wide range of applications, from healthcare and finance to transportation and manufacturing. As AI systems become more integrated with other technologies, they may be able to perform more complex tasks and solve more challenging problems.
    
    2. Greater emphasis on ethical AI: As concerns about the potential impact of AI on society and the environment continue to grow, there may be increased focus on ethical AI practices and safeguards. This could include efforts to ensure that


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

    Character

    's

     Name

    ]

     and

     I

    'm

     [

    Character

    's

     Age

    ]

     years

     old

    .

     I

     have

     been

     living

     in

     [

    Country

    ]

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

    'm

     a

     [

    Occup

    ation

    /

    Activity

    /

    Person

    ality

    ]

     who

     enjoys

     [

    What

    's

     your

     favorite

     hobby

     or

     activity

    ,

     if

     any

    ].

     I

     am

     friendly

     and

     consider

    ate

    ,

     always

     trying

     my

     best

     to

     make

     others

     feel

     welcome

     and

     valued

    .

     I

     love

     [

    What

    's

     your

     favorite

     book

    ,

     movie

    ,

     or

     song

    ,

     if

     any

    ,

     that

     you

     enjoy

     reading

    ,

     watching

    ,

     or

     listening

     to

    ,

     and

     why

    ?

     I

    'm

     passionate

     about

     [

    What

    's

     one

     specific

     thing

     you

     are

     passionate

     about

     in

     life

    ,

     and

     why

    ?

    
    
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

     Light

    ,

     and

     is

     the

     largest

     city

     in

     Europe

     and

     the

     second

     largest

     city

     in

     the

     world

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     home

     to

     many

     of

     France

    's

     cultural

     and

     historical

     landmarks

    ,

     including

     the

     Lou

    vre

     Museum

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     because

     of

     its

     vibrant

     culture

    ,

     architecture

    ,

     and

     lively

     atmosphere

    .

     It

     is

     an

     important

     cultural

     and

     economic

     center

    ,

     and

     is

     the

     second

     largest

     city

     in

     the

     European

     Union

     and

     the

     third

     largest

     in

     the

     world

    .

     It

     is

     also

     a

     major

     tourist

     destination

     and

     a

     popular

     destination

     for

     business

     and

     leisure

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     several

     key

     trends

    :
    


    1

    .

     Advances

     in

     machine

     learning

     and

     artificial

     intelligence

     technologies

    :

     With

     the

     help

     of

     advanced

     algorithms

     and

     deep

     learning

    ,

     AI

     will

     become

     more

     sophisticated

     and

     accurate

     in

     its

     predictions

     and

     decision

    -making

     processes

    .
    


    2

    .

     Integration

     of

     AI

     into

     various

     industries

    :

     AI

     will

     continue

     to

     be

     integrated

     into

     various

     industries

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     manufacturing

    ,

     to

     improve

     efficiency

    ,

     predict

     outcomes

    ,

     and

     optimize

     operations

    .
    


    3

    .

     Personal

    ization

    :

     AI

     will

     be

     used

     to

     personalize

     the

     experience

     for

     individuals

    ,

     improving

     user

     satisfaction

     and

     loyalty

    .
    


    4

    .

     Ethics

     and

     governance

    :

     AI

     will

     need

     to

     be

     developed

     with

     ethical

     considerations

     and

     governance

     in

     mind

     to

     avoid

     unintended

     consequences

    



```python
llm.shutdown()
```
