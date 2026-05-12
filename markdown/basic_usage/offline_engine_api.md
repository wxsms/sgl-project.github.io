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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]


    2026-05-12 17:34:17,430 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 17:34:17] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:40,  3.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:11,  4.01it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  9.54it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 16.26it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 22.96it/s]

    Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:04<00:00, 28.91it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 35.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.82it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.90it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.60it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.51it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.21it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.23it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.84it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.52it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.80it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 44.80it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.23it/s]


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
    Generated text:  Lee and I am a full-time college student at the moment. I enjoy reading books, listening to music, and going hiking. What is your favorite subject? I have a lot of questions about various topics and am always trying to learn. I like the challenge of trying to figure things out on my own. Do you have any advice for me on how to learn? What are some of your favorite subjects and topics? I am always eager to learn and explore new ideas. How do you stay organized? I have found that my best organizational skills come from practicing and keeping a consistent schedule. I also try to avoid getting distracted by the phone
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to choose between two campaigns for a particular office. Campaign A promises a 20% chance of winning the office in any given election. Campaign B has a 50% chance of winning. The president wants to minimize the expected loss if he randomly selects the campaign to vote for. If the president randomly selects Campaign A, what is the probability that the president will lose the election? To determine the probability that the president will lose the election if he randomly selects Campaign A, we need to consider the probability of the president losing the election if he chooses that campaign. Let's denote the probability of winning the election as \(P(W
    ===============================
    Prompt: The capital of France is
    Generated text:  _______.
    A. Paris
    B. Lyon
    C. Nice
    D. Nantes
    Answer:
    A
    
    Which of the following statements is incorrect?
    A. In the Windows operating system, you cannot adjust the display resolution on a standard monitor.
    B. When you start the 'Explorer' window, the File Explorer dialog box will appear.
    C. The keyboard shortcut for 'Close' is Ctrl+D.
    D. The Enter key is often used to trigger the 'Find Next' and 'Find Previous' functions in the search bar.
    Answer:
    C
    
    The cost of goods sold can be directly related to which of the following
    ===============================
    Prompt: The future of AI is
    Generated text:  expected to continue to evolve, and it is a rapidly growing field. There is no doubt that the AI field will continue to drive the growth and development of the technology industry in the coming years. However, it is also important to understand that AI is not just about creating machines, but also about developing people who can understand and interpret artificial intelligence, and make sense of it. As technology evolves, so too will the way that AI is used in the workplace, and businesses will need to adapt to the new needs of their workforce.
    One area where AI is already making a significant impact is in healthcare. AI algorithms can help healthcare professionals make more


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character, such as "funny, witty, and always up for a good laugh"]. I enjoy [insert a short list of hobbies or interests, such as "reading, cooking, playing sports, or watching movies"]. I'm always looking for new experiences and challenges to try, and I'm always eager to learn and grow. What's your favorite hobby or activity? I'm a huge [insert a hobby
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a vibrant and dynamic city with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is also home to many famous museums, theaters, and other cultural institutions. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation: AI is likely to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in the creation of new jobs, but also potentially leading to job displacement.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for robust privacy and security measures to protect personal data. This could lead to the development of new technologies and protocols for handling sensitive information.
    
    3. AI-driven healthcare: AI is already being used to improve healthcare outcomes, from personalized treatment plans to predictive analytics for disease prevention.
    


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
    Generated text:  [name], and I’m an [age] year old. What’s your occupation and where did you go to college?
    Hello, my name is [name], and I’m an [age] year old. What’s your occupation and where did you go to college? As a self-introduction, I'd like to start by introducing myself, and then share a little bit about my education. First, I'm [name], a [occupation], born and raised in [location]. I have a passion for [interest or hobby] and spend my weekends reading books or watching movies. What's your occupation and where did you go to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, commonly known as the City of Light. It is a historic and cultural center with many iconic landmarks, including Notre-Dame Cathedral, the Louvre Museum, the Eiffel Tower, the Seine River, and the Place de la Concorde. It is also known for its world-class cuisine, wine, and nightlife. The city is home to many prestigious institutions, including the Paris Academy of Fine Arts and the National Museum of France. Paris is also known for its diverse cultural scene and frequent hosting of international festivals and events. It is one of the world's most livable cities and is considered one of the best places
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by significant advancements in several key areas:
    
    1. Increased accuracy and efficiency in decision-making: AI is becoming increasingly sophisticated, and it is expected to improve its ability to make decisions that are both accurate and efficient. This will enable AI to make more informed and effective decisions in areas such as healthcare, finance, and transportation.
    
    2. Personalization and customization: AI is also expected to allow for more personalized and customized experiences, such as personalized medicine, customer service, and financial advice. This will enable AI to deliver more tailored and effective services to individuals based on their unique needs and preferences.
    
    3. Integration with other technologies:


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

     __

    ________

     and

     I

     am

     a

    /an

     __

    ________

    __.

     I

     love

     to

     __

    ________

    .

     I

     like

     to

     __

    ________

    .

     I

     am

     a

    /an

     __

    ________

    .

     I

     enjoy

     __

    ________

    .

     I

     can

     speak

     __

    ________

    .

     I

     have

     a

    /an

     __

    ________

    .

     What

     do

     you

     know

     about

     me

    ?
    


    This

     is

     a

     short

    ,

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    .

     The

     introduction

     should

     be

     polite

     and

     brief

    ,

     and

     should

     include

     a

     clear

     and

     concise

     summary

     of

     the

     character

    's

     identity

     and

     background

    .

     The

     introduction

     should

     also

     introduce

     the

     character

    's

     hobbies

     and

     interests

    ,

     and

     any

     notable

     achievements

     or

     experiences

    .

     The

     introduction

     should

     be

     written

     in

     a

     neutral

     tone

    ,

     without

     any

     personal

     or

     biased

     opinions

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     vibrant

     culture

    ,

     historical

     significance

    ,

     and

     role

     in

     French

     culture

     and

     politics

    .

     Paris

     is

     the

     third

    -largest

     city

     in

     France

     and

     has

     a

     population

     of

     around

     

    2

    .

    1

     million

     people

    ,

     making

     it

     one

     of

     the

     largest

     cities

     in

     Europe

    .

     The

     city

     is

     also

     home

     to

     the

     French

     Parliament

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     a

     symbol

     of

     French

     culture

     and

     a

     popular

     tourist

     destination

     worldwide

    .

     As

     of

     

    2

    0

    2

    1

    ,

     there

     are

     over

     

    3

    1

    ,

    0

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     variety

     of

     trends

     that

     will

     shape

     the

     technology

    's

     direction

    ,

     and

     these

     include

    :
    


    1

    .

     Increased

     Privacy

     and

     Ethics

    :

     AI

     is

     expected

     to

     continue

     to

     evolve

     and

     become

     more

     sophisticated

    ,

     but

     as

     it

     becomes

     more

     widespread

    ,

     there

     will

     be

     a

     growing

     concern

     about

     privacy

     and

     ethical

     issues

     surrounding

     AI

    .
    


    2

    .

     Increased

     Depend

    ence

     on

     AI

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

     there

     will

     be

     an

     increased

     need

     for

     people

     to

     become

     more

     familiar

     with

     and

     comfortable

     with

     the

     technology

    .
    


    3

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     expected

     to

     become

     more

     common

    ,

     as

     they

     become

     more

     reliable

     and

     accessible

     to

     the

     general

     public

    .
    


    4

    .

     Increased

     collaboration

    



```python
llm.shutdown()
```
