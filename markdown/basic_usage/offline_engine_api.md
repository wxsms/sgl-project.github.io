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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.67it/s]


    2026-05-04 23:14:22,664 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 23:14:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.51it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.11it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.28it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.40it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.37 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.37 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.37 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.37 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.37 GB):   3%|▎         | 2/58 [00:00<00:02, 18.89it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.36 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.35 GB):   9%|▊         | 5/58 [00:00<00:02, 22.11it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=49.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.34 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=49.33 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.96it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.33 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=49.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.32 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.29 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=960 avail_mem=49.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s] Capturing num tokens (num_tokens=896 avail_mem=49.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]

    Capturing num tokens (num_tokens=832 avail_mem=49.30 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.92it/s]Capturing num tokens (num_tokens=832 avail_mem=49.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=768 avail_mem=49.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=704 avail_mem=49.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=640 avail_mem=49.29 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=576 avail_mem=49.29 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=512 avail_mem=49.28 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.35it/s]Capturing num tokens (num_tokens=512 avail_mem=49.28 GB):  50%|█████     | 29/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=480 avail_mem=49.29 GB):  50%|█████     | 29/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=448 avail_mem=49.29 GB):  50%|█████     | 29/58 [00:00<00:00, 39.09it/s]

    Capturing num tokens (num_tokens=416 avail_mem=49.29 GB):  50%|█████     | 29/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=384 avail_mem=49.29 GB):  50%|█████     | 29/58 [00:00<00:00, 39.09it/s]Capturing num tokens (num_tokens=384 avail_mem=49.29 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=352 avail_mem=49.28 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=320 avail_mem=49.27 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.73it/s]Capturing num tokens (num_tokens=288 avail_mem=49.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=256 avail_mem=49.27 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.73it/s]Capturing num tokens (num_tokens=256 avail_mem=49.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=240 avail_mem=49.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=224 avail_mem=49.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.41it/s]

    Capturing num tokens (num_tokens=208 avail_mem=49.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=192 avail_mem=49.26 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.41it/s]Capturing num tokens (num_tokens=192 avail_mem=49.26 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=176 avail_mem=49.25 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=160 avail_mem=49.25 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=144 avail_mem=49.25 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=128 avail_mem=49.25 GB):  71%|███████   | 41/58 [00:01<00:00, 35.35it/s]Capturing num tokens (num_tokens=128 avail_mem=49.25 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=112 avail_mem=49.25 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=96 avail_mem=49.24 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.60it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=49.24 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=64 avail_mem=49.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.60it/s]Capturing num tokens (num_tokens=64 avail_mem=49.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.92it/s]Capturing num tokens (num_tokens=48 avail_mem=49.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.92it/s]Capturing num tokens (num_tokens=32 avail_mem=49.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.92it/s]Capturing num tokens (num_tokens=28 avail_mem=49.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.92it/s]Capturing num tokens (num_tokens=24 avail_mem=49.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 33.92it/s]

    Capturing num tokens (num_tokens=24 avail_mem=49.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=20 avail_mem=49.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=16 avail_mem=49.22 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=12 avail_mem=49.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=8 avail_mem=49.21 GB):  91%|█████████▏| 53/58 [00:01<00:00, 32.97it/s] Capturing num tokens (num_tokens=8 avail_mem=49.21 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.41it/s]Capturing num tokens (num_tokens=4 avail_mem=49.20 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.41it/s]Capturing num tokens (num_tokens=4 avail_mem=49.20 GB): 100%|██████████| 58/58 [00:01<00:00, 34.44it/s]


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
    Generated text:  Rebeca. I'm a 19-year-old high school student. I'm really happy to be in this class because I enjoy social studies and have a lot of free time. I like to read about history and politics. I'm also very interested in art. I have a drawing class every week and I'm learning how to paint. I love learning about other cultures and how people live in the world. I'm always trying to be a better person and make the world a better place for everyone.
    As a student in high school, I'm really passionate about history and social studies. I enjoy learning about different cultures and their
    ===============================
    Prompt: The president of the United States is
    Generated text:  a male. His birth name is Donald Trump. He was born on 20th January 1946 in a small town in New Jersey, USA. He has several children but the only son is now 31 years old. He has been involved in politics since 1994, when he took a job as a campaign manager for Hillary Clinton in the 2000 US presidential election. He was a US senator from 2005-2009 and 2015-2017 and then became the first openly gay presidential candidate for the Republican Party. He
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was founded in the 5th century AD by the Gauls. It's the capital of France and the 19th largest city in the world. It's in the south of France, near the Mediterranean Sea. It is in the heart of the romantic city of Paris, the "City of Light". Paris is the cultural and political center of France. There are many famous attractions in the city, including the Eiffel Tower, the Louvre Museum, the Centre Pompidou, the Musée d'Orsay, and the Musée Rodin, among many others. In addition, it is the
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but it may not be here yet. It's early days.
    It may be a while yet before we see the AI that's going to help us shape a better world, but you can be sure that AI is likely to play a central role in shaping the future of work, education, and many other aspects of our lives.
    So what can we do to prepare for this future?
    One of the biggest challenges is that today's AI is not "there yet". It's still evolving, not fully understood, and not fully developed. It's not yet a system that could be trusted to make ethical decisions, or to predict future


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few details about your personality or background here, such as your hobbies, interests, or any unique skills you have]. I'm always looking for new challenges and opportunities to grow and learn, and I'm always eager to share my knowledge and experience with others. Thank you for taking the time to meet me. What's your favorite hobby or activity? As an AI language model, I don't have personal hobbies or activities like humans do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. It is the capital of France and the largest city in the European Union. The city is known for its cuisine, fashion, and art, and is a major economic and political center in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see more widespread adoption of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection, risk assessment, and trading algorithms.
    


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
    Generated text:  __________. I am a/an ___________ (type your profession/occupation here). I have ___________ (type your relevant experience here). My current position is ___________. In my free time, I enjoy ___________ (type your hobby here). What's your favorite hobby? What's your favorite movie? What's your favorite book? What's your favorite sport? What's your favorite food? What's your favorite place? What's your favorite book? What's your favorite time of the year? What's your favorite season? What's your favorite animal? What's your favorite city? What's your favorite dessert?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe by population and the second-largest city in the world by area. Paris is the cultural, economic, and political center of France and hosts many famous landmarks and museums. It is also a major tourist destination with a rich history, art, and music scene. The city is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. The French language is also an official language of France. Paris is a bustling metropolis with a diverse population, and it is home to many world-renowned cultural and artistic institutions. The city is a significant economic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but several trends are likely to shape its development in the coming years. Some potential trends include:
    
    1. Increased integration of AI with other technologies: AI is already integrated into a wide range of technologies, but there is a potential for further integration with other areas such as machine learning, natural language processing, and robotics. This integration could lead to more advanced AI systems with even more complex capabilities.
    
    2. Growth of AI in healthcare: AI can be used to improve the accuracy and speed of diagnosis and treatment, leading to more effective medical treatments. Additionally, AI-powered robots and drones could be used in surgical procedures and disaster response, enhancing


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

     occupation

    ].

     I

     love

     [

    occupation

    ]

     because

     [

    exc

    use

     me

    ,

     point

     to

     something

     important

     for

     example

    ,

     a

     book

    ,

     a

     piece

     of

     art

    ,

     a

     park

    ,

     a

     race

     car

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Short

    ,

     neutral

     self

    -int

    roduction

    ]

     [

    Person

    's

     name

    ]

     is

     a

     [

    occupation

    ],

     and

     I

     love

     [

    occupation

    ]

     because

     [

    exc

    use

     me

    ,

     point

     to

     something

     important

     for

     example

    ,

     a

     book

    ,

     a

     piece

     of

     art

    ,

     a

     park

    ,

     a

     race

     car

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Short

    ,

     neutral

     self

    -int

    roduction

    ]

     [

    Name

    ]

     is

     a

     [

    occupation

    ],

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     seat

     of

     government

     and

     politics

    .

     It

    's

     also

     the

     world

    's

     most

     populous

     city

    ,

     with

     over

     

    2

     million

     inhabitants

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     music

    ,

     and

     cuisine

    .

     Paris

     is

     often

     called

     the

     "

    City

     of

     Love

    "

     because

     of

     its

     romantic

     culture

     and

     fashion

     scene

    .

     Its

     famous

     landmarks

     include

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Champ

     de

     Mars

     public

     park

    .

     Paris

     is

     a

     popular

     destination

     for

     tourists

     and

     a

     major

     economic

     center

    .

     It

    's

     also

     home

     to

     numerous

     art

     museums

    ,

     theaters

    ,

     and

     caf

    és

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     is

     likely

     to

     continue

     to

     expand

     and

     evolve

     at

     an

     accelerated

     pace

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Enhanced

     AI

    :

     As

     AI

     continues

     to

     improve

     its

     capabilities

    ,

     it

     will

     become

     more

     capable

     of

     understanding

     and

     learning

     from

     human

     behavior

     and

     language

    .

     This

     will

     allow

     AI

     systems

     to

     perform

     tasks

     that

     were

     previously

     impossible

    ,

     such

     as

     playing

     a

     game

     or

     making

     financial

     predictions

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     becoming

     more

     and

     more

     common

     in

     our

     everyday

     lives

    .

     AI

     is

     being

     used

     to

     develop

     self

    -driving

     cars

     that

     can

     navigate

     roads

     and

     avoid

     obstacles

     on

     their

     own

    .
    


    3

    .

     Healthcare

    :

     AI

     is

     being

     used

     in

     healthcare

     to

     help

     doctors

     diagnose

    



```python
llm.shutdown()
```
