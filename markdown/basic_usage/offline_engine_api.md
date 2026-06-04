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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.64s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.69it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.54it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 38.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.76 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 17.31it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.74 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):   9%|▊         | 5/58 [00:00<00:02, 20.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.43it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=61.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.71 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.71 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.71 GB):  21%|██        | 12/58 [00:00<00:01, 29.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=61.68 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s]Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.04it/s] Capturing num tokens (num_tokens=960 avail_mem=61.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=896 avail_mem=61.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=832 avail_mem=61.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=768 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=704 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=640 avail_mem=61.68 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.72it/s]Capturing num tokens (num_tokens=640 avail_mem=61.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=576 avail_mem=61.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.88it/s]

    Capturing num tokens (num_tokens=512 avail_mem=61.66 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=480 avail_mem=61.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=448 avail_mem=61.67 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.88it/s]Capturing num tokens (num_tokens=448 avail_mem=61.67 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=416 avail_mem=61.67 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=384 avail_mem=61.67 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=352 avail_mem=61.66 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.62it/s]Capturing num tokens (num_tokens=320 avail_mem=61.66 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.62it/s]Capturing num tokens (num_tokens=320 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=288 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 36.05it/s]

    Capturing num tokens (num_tokens=256 avail_mem=61.66 GB):  60%|██████    | 35/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=240 avail_mem=61.65 GB):  60%|██████    | 35/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=224 avail_mem=61.65 GB):  60%|██████    | 35/58 [00:01<00:00, 36.05it/s]Capturing num tokens (num_tokens=224 avail_mem=61.65 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.74it/s]Capturing num tokens (num_tokens=208 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.74it/s]Capturing num tokens (num_tokens=192 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.74it/s]Capturing num tokens (num_tokens=176 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.74it/s]Capturing num tokens (num_tokens=160 avail_mem=61.64 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.74it/s]Capturing num tokens (num_tokens=160 avail_mem=61.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=144 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.85it/s]

    Capturing num tokens (num_tokens=128 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=112 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.85it/s]Capturing num tokens (num_tokens=96 avail_mem=61.63 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.85it/s] Capturing num tokens (num_tokens=96 avail_mem=61.63 GB):  81%|████████  | 47/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=80 avail_mem=61.62 GB):  81%|████████  | 47/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=64 avail_mem=61.62 GB):  81%|████████  | 47/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=48 avail_mem=61.61 GB):  81%|████████  | 47/58 [00:01<00:00, 34.53it/s]

    Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  81%|████████  | 47/58 [00:01<00:00, 34.53it/s]Capturing num tokens (num_tokens=32 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=28 avail_mem=61.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=24 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=20 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=16 avail_mem=61.60 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.27it/s]Capturing num tokens (num_tokens=16 avail_mem=61.60 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=12 avail_mem=61.59 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=8 avail_mem=61.59 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.56it/s] Capturing num tokens (num_tokens=4 avail_mem=61.59 GB):  95%|█████████▍| 55/58 [00:01<00:00, 34.56it/s]

    Capturing num tokens (num_tokens=4 avail_mem=61.59 GB): 100%|██████████| 58/58 [00:01<00:00, 33.91it/s]


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
    Generated text:  David and I am 28 years old. I have very little experience with social media and social media marketing. I plan to have a LinkedIn presence and I am looking for some advice on how to create a successful social media marketing plan. Could you please provide some advice on how to create a social media marketing plan?
    Certainly! Here are some general tips to create a successful social media marketing plan:
    
      1. Define your audience: Identify your target audience by analyzing their interests, demographics, and behaviors. Consider using tools like Google Analytics, Facebook Insights, or social media listening software to gain insights into your audience.
      2.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a politician who holds the office of the President of the United States and is the head of the executive branch of the federal government, serving as the commander-in-chief of the United States Armed Forces. He or she is responsible for overseeing all departments, agencies and branches of the federal government, including the executive branch, the executive orders, the creation of new laws and the duties and powers of Congress.
    In the United States, the President is the head of state, head of government, commander-in-chief of the armed forces and the commander-in-chief of the military; a person who holds the office of president of the United States and is the head
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Geneva
    C. Berlin
    D. Moscow
    Answer: A
    
    The capital of France is ____
    A. Paris
    B. Geneva
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following statements about the capital of France is correct?
    A. Paris
    B. Geneva
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following statements about the capital of France is correct?
    A. Paris
    B. Geneva
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following statements about the capital of France is correct?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  complex and constantly evolving, and it's important to understand how it's likely to interact with different industries and areas of society. Here are a few examples of how AI may interact with different sectors:
    
    1. Healthcare: AI has the potential to improve the quality of healthcare by helping doctors and nurses diagnose diseases more accurately and quickly, and by improving the accuracy of medical procedures. AI can also be used to develop personalized treatment plans for patients, with the potential to save lives.
    
    2. Education: AI has the potential to revolutionize the way we teach and learn, with the potential to personalize learning and provide personalized feedback to students. AI can also


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been working here for [number] years. I'm a [job title] at [company name], and I've been
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. Its history dates back to the Roman Empire and is known for its beautiful architecture and art. The city is also home to many international organizations and events, making it a popular destination for business and leisure. Paris is a city of contrasts, with its modern and historic elements blending together to create a unique and fascinating city. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will allow AI to learn from a wider range of data sources and improve its ability to perform tasks that were previously impossible.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be increased concerns about privacy and security. This will require developers to create more secure and transparent AI systems that are designed to
    


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
    Generated text:  Emily and I am an adult living in the city with my husband and two children. I am a stay-at-home mom who loves to read, explore the outdoors, and get lost in a good book. I also enjoy volunteering at local parks and helping out at the community center. I am a strong advocate for mental health awareness and often talk about how important it is for us to take care of ourselves. I'm a vegetarian who loves to cook, bake, and try new recipes. I'm excited to have a conversation with you about my life and what I'm up to now! Emily
    
    Can you add some personal anecdotes to make the
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its vibrant culture, historical architecture, and annual Carnival festivities. It was founded in 789 by Charlemagne and has become a major cultural and economic center. The city features iconic landmarks like the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum, while also hosting a vast array of museums, theaters, and restaurants. Paris is a popular tourist destination and a symbol of French culture and identity. It is also known for its diverse French cuisine, including famous dishes like croissants and tapas. The city has a rich history and is a major destination for visitors from around the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and diverse, with potential applications in a wide range of fields. Here are some possible future trends in artificial intelligence:
    
    1. Autonomous vehicles: Autonomous vehicles (AVs) are becoming more prevalent, and AI is playing a critical role in their development. AVs will be able to learn from their environment, make decisions, and communicate with humans in a way that improves safety and efficiency.
    
    2. Healthcare: AI is already being used in healthcare, with applications in diagnosing and treating diseases. AI is also being developed to help doctors make more accurate diagnoses, to develop personalized treatment plans, and to improve patient outcomes.
    
    3. Climate change


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

    name

    ],

     and

     I

    'm

     a

     [

    age

    ]

     year

    -old

     student

     studying

     at

     [

    un

    iversity

    ].

     I

     love

     [

    field

    /

    subject

     of

     interest

    ],

     and

     I

    'm

     always

     looking

     for

     new

     experiences

     to

     expand

     my

     skills

     and

     knowledge

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?
    


    [

    Name

    ]:

     As

     an

     AI

     language

     model

    ,

     I

    'm

     here

     to

     assist

     you

     with

     any

     questions

     or

     information

     you

     might

     need

    .

     I

     can

     help

     you

     learn

     new

     languages

    ,

     translate

     between

     languages

    ,

     and

     even

     answer

     your

     questions

    .

     Whether

     you

     need

     help

     with

     your

     homework

    ,

     learn

     a

     new

     skill

    ,

     or

     just

     want

     to

     chat

    ,

     I

    'm

     here

     to

     help

    !

     How

     can

     I

     assist

     you

     today

    ?
    


    Name

    :

     Hello

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    Task

    :

     Write

     a

     

    1

    0

    0

    -word

     summary

     of

     the

     capital

     city

     of

     France

     in

     one

     sentence

    .

     
    


    A

     summary

     of

     the

     capital

     of

     France

     in

     one

     sentence

     is

    :
    


    France

    's

     capital

    ,

     Paris

    ,

     is

     known

     for

     its

     rich

     history

    ,

     stunning

     architecture

    ,

     and

     cultural

     richness

    ,

     making

     it

     a

     beloved

     city

     for

     many

     French

     residents

     and

     tourists

     alike

    .

     Its

     iconic

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     Notre

    -D

    ame

     Cathedral

    ,

     are

     a

     testament

     to

     its

     status

     as

     Europe

    's

     most

     cultural

     and

     historical

     capital

    .

     Paris

     also

     boasts

     a

     vibrant

     arts

     scene

    ,

     as

     evidenced

     by

     the

     numerous

     museums

     and

     theaters

    ,

     which

     attract

     millions

     of

     visitors

     each

     year

    .

     Overall

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     evolving

     rapidly

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

     and

     efficiency

    :

     With

     AI

    's

     ability

     to

     process

     and

     analyze

     vast

     amounts

     of

     data

    ,

     it

    's

     possible

     that

     AI

     will

     become

     more

     efficient

     at

     performing

     repetitive

     and

     routine

     tasks

    ,

     allowing

     humans

     to

     focus

     on

     more

     complex

     and

     creative

     work

    .
    


    2

    .

     Increased

     human

     involvement

    :

     AI

     may

     continue

     to

     automate

     certain

     tasks

    ,

     but

     it

    's

     possible

     that

     it

     will

     also

     be

     involved

     in

     decision

    -making

    ,

     problem

    -solving

    ,

     and

     creative

     thinking

    .

     This

     could

     lead

     to

     a

     more

     human

    istic

     approach

     to

     AI

    .
    


    3

    .

     Autonomous

     vehicles

    :

     With

     AI

    's

     ability

     to

     understand

     and

     respond

     to

     complex

     situations

    ,

     it

    's

     possible

     that

    



```python
llm.shutdown()
```
