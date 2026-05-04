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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


    2026-05-04 00:31:38,297 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 00:31:38] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:18,  4.54s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.27it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.59it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 13.94it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:05<00:02, 13.94it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.97it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.97it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  31%|███       | 18/58 [00:00<00:01, 33.38it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.06it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.25it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 46.23it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.38it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 47.04it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.63it/s]


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
    Generated text:  Dave. I am a 14 year old female, and I have been training with a group of three guys for 6 weeks. Each of us has a different type of workout. I have been working on my abdominals, core, glutes, biceps, shoulders, and legs. My current fitness level is average, and I have a lot of time to train. I know I should focus on the 100% of the 6-week workout, but I feel I should try to do it even more. What can I do?
    Do you have any suggestions? 1. How many reps? 2
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 33 years old. How old will the president be in 5 years?
    
    To determine how old the president of the United States will be in 5 years, we need to add 5 years to his current age.
    
    1. Identify the current age of the president:
       \[
       \text{Current age} = 33 \text{ years old}
       \]
    
    2. Add 5 years to the current age:
       \[
       \text{Age in 5 years} = 33 + 5 = 38 \text{ years old}
       \]
    
    Therefore, the president
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Italy is Rome. But what about the capital of the Netherlands?
    Does this statement infer anything new?
    Select from:
     A). yes
     B). no
    B). no
    
    The statement "The capital of the Netherlands is not Paris, but Rome" does not necessarily infer anything new about the capitals of the Netherlands. While Rome is the capital of Italy and Paris is the capital of France, there is no information given about the capitals of the Netherlands. The statement is about Rome in relation to Italy, but not specifically about the capital of the Netherlands. Therefore, the answer is B). no. The statement does not
    ===============================
    Prompt: The future of AI is
    Generated text:  about a "three-epoch" transition, says Robert Kagan
    
    Robert Kagan, the author of The Future of Human AI, explains that AI will change as we move through three distinct stages: the early, grassroots era, the mid-term, high-impact transition, and the final, after-effects stage.
    
    A Growing AI
    
    The first stage is growing, says Kagan. AI is growing exponentially, creating new and fascinating applications, with the power of the Internet and the vastness of the Web making it possible. AI is able to create more powerful, more intelligent, and more networked systems, which is all the more reason


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Number] years of experience, and I have a passion for [Your Passion]. I'm always looking for ways to [Your Goal], and I'm always eager to learn new things. I'm a [Number] year old, [Number] years of experience, and I have a passion for [Your Passion]. I'm always looking for ways to [Your Goal], and I'm always eager to learn new things. I'm a [Number] year old, [Number] years of experience, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its rich history, beautiful architecture, and vibrant culture. It is the largest city in France and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual fashion and food festivals, as well as its annual Eiffel Tower parade. The city is a popular tourist destination and is home to many cultural institutions and events. Paris is a city that is steeped in history and culture, and it is a must-visit destination for anyone interested in France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to the same level of complexity as humans. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical and social considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical and social considerations. This could
    


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
    Generated text:  [Name]. I'm a [Age], [Occupation], [Skill/特长] who has been [what you did] for [how long]. What brings you to this table today? I'm really excited to have the opportunity to share some thoughts with you all. 
    
    ---
    
    Please ensure the answer is between 5-7 sentences. What kind of profession or occupation does the character have? Who is the target audience for the self-introduction? How long has the character been in their profession or occupation? What brings them to this table today? What are the key points the character wants to share with the audience? What are some
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city and the largest urban area in the European Union, with a population of over 2 million people. Paris is known for its rich history, beautiful architecture, and vibrant culture, including the famous Eiffel Tower. The city is also an important center for politics, business, and fashion, and is home to several important museums, galleries, and art galleries. Paris is also a major tourist destination, with millions of tourists visiting each year. Its status as a major economic hub and cultural center has made it a leader in the European and global economies. Additionally, Paris is the home of the French national
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of possibilities, with exciting advancements and breakthroughs expected in the coming years. Here are some potential trends in AI that are currently being explored:
    
    1. AI will become more personalized and context-aware: As AI continues to learn and improve, it is becoming increasingly clear that it will become more personalized and context-aware. This means that AI will be able to learn more about its users and their needs, and will be able to adapt to new situations and respond to user inputs in ways that are more natural and intuitive.
    
    2. AI will become more intelligent and self-aware: AI is already becoming more intelligent, with the ability to process and interpret


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

    ],

     and

     I

    'm

     a

     computer

     science

     student

     at

     [

    University

    ].

     I

     enjoy

     playing

     video

     games

     and

     reading

     sci

    -fi

     novels

    .

     My

     hobbies

     also

     include

     baking

     and

     gardening

    .

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     trying

     new

     things

    .

     My

     favorite

     hobbies

     include

     [

    List

     hobbies

     here

    ].

     What

    's

     your

     favorite

     hobby

    ?

     [

    Optional

    ]


    Hi

     [

    Name

    ],

     my

     name

     is

     [

    Name

    ],

     a

     computer

     science

     student

     at

     [

    University

    ].

     I

     enjoy

     playing

     video

     games

     and

     reading

     sci

    -fi

     novels

    .

     My

     hobbies

     also

     include

     baking

     and

     gardening

    .

     I

    'm

     always

     looking

     for

     new

     experiences

     and

     trying

     new

     things

    .

     My

     favorite

     hobbies

     include

     [

    List

     hobbies

     here

    ].

     What

    's

     your

     favorite

     hobby

    ?

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     eastern

     part

     of

     the

     country

    .

     It

     is

     a

     major

     city

     with

     a

     population

     of

     over

     

    2

    .

    5

     million

     people

    ,

     and

     is

     home

     to

     the

     country

    's

     largest

     city

    .

     It

     is

     also

     the

     seat

     of

     government

    ,

     the

     parliament

    ,

     and

     the

     president

     of

     France

    .

     Paris

     is

     a

     cultural

     and

     historical

     center

     with

     many

     famous

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Saint

    e

    -Ch

    ap

    elle

    ,

     among

     others

    .

     Despite

     its

     size

    ,

     Paris

     is

     known

     for

     its

     diverse

     cuisine

    ,

     fashion

    ,

     and

     entertainment

     scene

    ,

     making

     it

     a

     popular

     tourist

     destination

     worldwide

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

    ,

     with

     potential

     developments

     that

     could

     transform

     the

     way

     we

     live

     and

     work

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

     depth

     and

     accuracy

     of

     AI

    :

     As

     the

     technology

     continues

     to

     advance

    ,

     we

     may

     see

     more

     advanced

     models

     that

     can

     perform

     complex

     tasks

     with

     greater

     accuracy

     and

     precision

    .
    


    2

    .

     Integration

     with

     human

     intelligence

    :

     AI

     systems

     are

     increasingly

     being

     integrated

     with

     human

     intelligence

    ,

     allowing

     them

     to

     process

     more

     data

     and

     make

     more

     informed

     decisions

    .
    


    3

    .

     Enhanced

     privacy

     and

     security

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     we

     may

     see

     an

     increase

     in

     concerns

     about

     privacy

     and

     security

    .

     This

     could

     lead

     to

     new

     regulations

     and

     policies

     to

     protect

     people

    's

     data

    .
    


    4

    .

    



```python
llm.shutdown()
```
