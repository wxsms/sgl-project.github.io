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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


    2026-05-08 06:53:31,783 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 06:53:31] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:49,  4.02s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.61it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.79it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.79it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.66it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.30it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.83it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 33.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 17.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.81it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.81it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.78it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.21it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.16it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.64it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.64it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 38.64it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  60%|██████    | 35/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.61it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:01<00:00, 35.61it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.80it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 33.80it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.93it/s] Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 33.14it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.79it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 33.17it/s]


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
    Generated text:  Sam. I'm 13 years old. My birthday is on April 12. I like chocolate. I have lots of chocolate for breakfast and dinner. My mother doesn't allow me to eat chocolate. I also like flowers. My mother and I have a flower shop. We buy many flowers for my birthday. The flowers make my birthday extra special. What can we know from the passage? A) Sam's mother likes flowers. B) Sam has flowers for breakfast and dinner. C) Sam's mother is very strict with him. D) Sam's mother buys flowers for him.
    Answer the following question: Why do the
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a second term, and he is trying to decide whether to run against his running mate or to run for president. 
    
    The president is in favor of a single ticket policy, where the president can only be elected if he is the same gender as his running mate. If the president and his running mate have different genders, the president can either run against them or stay out of the race altogether. 
    
    The president is also considering running against his running mate. If he runs against them, he can only win if they are not the same gender, otherwise, he has to stay out of the race. If he runs against them,
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. London
    B. Paris
    C. Rome
    D. Berlin
    Answer:
    
    B
    
    Which of the following statements is true?
    A. The inner surface of a liquid always has a pressure less than the external atmospheric pressure.
    B. The force acting on a force ballerina dancing on stage is due to friction between the dancer's shoes and the stage.
    C. When a coin is tossed, if the upward velocity of the coin is less than the downward velocity, the coin will land on its side.
    D. When Xiao Ming jumps straight up from the ground, the ground exerts an upward force on him.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  here, and it’s being revolutionized by quantum computers. Quantum computers are the brainchild of the late physicist Richard Feynman. They are the pinnacle of technology and represent the possibility of future computing.
    In a recent interview with the New York Times, Bill Gates said, “Quantum computers are the next big thing. They can solve problems that would take the world’s most powerful computers years to solve.”
    Quantum computers have the potential to solve problems that our current computers can’t, like cryptography, medical research, and other types of computing.
    They can also break encryption codes that are currently used to secure communications and make further advancements in


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in Europe by population. It is located on the Seine River and is home to many of the world's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its rich history, art, and culture, and is a major center for business, politics, and entertainment. It is also a popular tourist destination, with millions of visitors annually. The city is home to many different ethnic groups and is a melting pot of cultures. Paris is a vibrant and dynamic city that continues to thrive
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology
    


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
    Generated text:  [Name], and I'm an AI language model. I was created by [Company Name] to assist people with their queries, and I'm constantly learning and improving my responses based on the data I'm fed. I'm here to answer any questions you might have and help you with anything you need. How can I assist you today? Hey there, I'm the AI language model. What can I help you with today? Hey there, I'm the AI language model. What can I help you with today? Hey there, I'm the AI language model. What can I help you with today? Hey there, I'm the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the second-largest in Europe, with an estimated population of 2.1 million. The city is home to many renowned landmarks and cultural institutions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is known for its architecture, art, and cuisine, and is considered one of the world's most beautiful cities. With its bustling streets, charming neighborhoods, and rich history, Paris is a vibrant and exciting destination for visitors of all ages. French cuisine is renowned for its rich flavors and hearty dishes, and Parisians are known for their love of classic French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a number of technological, societal, and ethical changes that are currently unfolding. Here are some possible trends in AI:
    
    1. Increased Integration of AI into Everyday Life: AI is already playing an increasingly significant role in our daily lives. For example, self-driving cars are already in widespread use, and AI is being integrated into healthcare, finance, and transportation.
    
    2. AI will continue to become more sophisticated and more autonomous: As technology continues to advance, we can expect AI to become more sophisticated and more autonomous, with the ability to learn, adapt, and develop new skills on its own.
    
    3. AI will be integrated into


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

     am

     a

     [

     occupation

    ].

     I

     have

     a

     passion

     for

     [

    field

     of

     interest

    ]

     and

     I

     believe

     in

     [

    positive

     qualities

     or

     beliefs

    ].

     I

     have

     always

     been

     an

     [

    ability

     or

     hobby

    ]

     and

     I

     enjoy

     [

    reason

     why

     it

    's

     enjoyable

    ].

     I

     am

     confident

    ,

     ambitious

    ,

     and

     always

     eager

     to

     learn

     and

     grow

    .

     I

     have

     a

     work

     ethic

     and

     strive

     to

     always

     put

     in

     the

     extra

     effort

     to

     achieve

     success

    .

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    activity

     or

     hobby

    ].

     How

     do

     you

     come

     across

    ?


    First

    ,

     I

     like

     to

     be

     fresh

     in

     my

     own

     skin

    .

     I

     believe

     that

     it

    's

     important

     to

     stay

     true

     to

     myself

    ,

     even

     when

     others

     do

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     both

     the

     European

     Union

     and

     the

     United

     Nations

    ,

     and

     is

     home

     to

     millions

     of

     people

    .

     The

     city

     is

     known

     for

     its

     stunning

     architecture

    ,

     rich

     culture

    ,

     and

     vibrant

     nightlife

    .

     Paris

     is

     also

     home

     to

     important

     cultural

     institutions

     such

     as

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

     It

     is

     a

     major

     hub

     for

     international

     trade

     and

     is

     a

     UNESCO

     World

     Heritage

     Site

    .

     Paris

     has

     a

     long

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     is

     now

     a

     center

     of

     artistic

     and

     cultural

     activity

    .

     The

     city

     has

     been

     a

     focal

     point

     for

     political

    ,

     economic

    ,

     and

     social

     developments

     in

     France

     for

     centuries

    .

     Overall

    ,

     Paris

     is

     a

     unique

     and

     exciting

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     will

     continue

     to

     evolve

     in

     many

     exciting

     ways

    .

     Here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     development

     of

     AI

     in

     the

     next

     decade

    :
    


    1

    .

     Increased

     precision

     in

     AI

    :

     With

     the

     advancement

     of

     machine

     learning

     and

     deep

     learning

    ,

     AI

     systems

     are

     becoming

     increasingly

     accurate

     and

     precise

    .

     This

     will

     lead

     to

     more

     effective

     and

     accurate

     healthcare

     applications

    ,

     as

     well

     as

     more

     reliable

     and

     efficient

     transportation

     systems

    .
    


    2

    .

     Personal

    ized

     AI

    :

     AI

     will

     become

     even

     more

     personal

     as

     it

     learns

     from

     each

     user

    's

     interactions

     and

     can

     adapt

     to

     new

     experiences

    .

     This

     will

     lead

     to

     more

     personalized

     and

     effective

     customer

     service

    ,

     as

     well

     as

     more

     efficient

     and

     effective

     healthcare

     applications

    .
    


    3

    .

    



```python
llm.shutdown()
```
