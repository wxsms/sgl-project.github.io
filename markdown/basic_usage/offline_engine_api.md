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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.42it/s]


    2026-04-08 21:56:11,795 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 21:56:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:27,  2.58s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.94it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.92it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.31it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.31it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.52it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.27it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.79it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.05 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   3%|▎         | 2/58 [00:00<00:02, 19.13it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.02 GB):   7%|▋         | 4/58 [00:00<00:03, 17.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):   7%|▋         | 4/58 [00:00<00:03, 17.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:04, 11.38it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:04, 11.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  10%|█         | 6/58 [00:00<00:04, 11.38it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.41it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  14%|█▍        | 8/58 [00:00<00:04, 10.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.00 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.99 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.98 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.40it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  28%|██▊       | 16/58 [00:01<00:01, 22.40it/s] Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]

    Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  38%|███▊      | 22/58 [00:01<00:01, 30.55it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=512 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=448 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=416 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=384 avail_mem=70.95 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  48%|████▊     | 28/58 [00:01<00:00, 36.64it/s]Capturing num tokens (num_tokens=352 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=320 avail_mem=70.94 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=256 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]

    Capturing num tokens (num_tokens=240 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.17it/s]Capturing num tokens (num_tokens=208 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=192 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=176 avail_mem=70.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=144 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=128 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=112 avail_mem=70.91 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s] Capturing num tokens (num_tokens=80 avail_mem=70.90 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]

    Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=48 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=32 avail_mem=70.89 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  79%|███████▉  | 46/58 [00:01<00:00, 46.50it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=24 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=20 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=16 avail_mem=70.88 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=12 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=8 avail_mem=70.87 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.83it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 49.15it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 33.58it/s]


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
    Generated text:  Andrew and I am a natural language processing specialist.
    
    I am working on a project that aims to enhance the accuracy of sentiment analysis in digital media.
    
    Here is a sentence: "I am tired of the routine of the day." 
    
    I would like to have a conversation with you, Andrew, about the potential of sentiment analysis in this context. Could you provide some insights on how sentiment analysis can be applied in the context of this sentence? And also, how can we enhance the accuracy of sentiment analysis in the digital media industry?
    
    Thank you for your time and expertise. I look forward to our conversation. 
    
    PS: I am quite curious about
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected by a simple majority vote of the lower house of Congress, the Senate. Members of the Senate are not directly elected, but are chosen from various electoral districts, either directly or by a system of proportional representation. The president can be removed from office by impeachment, which requires a simple majority vote of the Senate, unless the president is already removed. The president has the power to veto legislation, which requires a simple majority vote of both houses of Congress. The president can also be removed from office by a criminal indictment or conviction, which requires a simple majority vote of both houses of Congress.
    
    Answer this question, if possible (if impossible,
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Bordeaux
    C. Montpellier
    D. Lyon
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    
    To expand on the key points related to Paris:
    
    1. Paris is the capital city of France and the third-largest city in the European Union by population.
    2. It is situated on the northern bank of the Seine River, on the island of France.
    3. The capital is located on the Left Bank of the Seine, a narrow strip that runs north-south across the center of the city.
    4. The city's importance is reflected in its
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain, and its development and implementation will depend on the actions of governments, businesses, and the public. The need for better data, more ethical guidelines, and increased security measures will be a key factor in determining how AI is used and developed.
    
    With the rapid advancements in AI technology, there are many potential applications, from financial services to healthcare to education. It is important to ensure that the technology used in these applications is ethical, reliable, and secure.
    
    One of the challenges that the future of AI will face is the lack of understanding of the impact of AI on society. As AI becomes more integrated into our lives, it is essential


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working in this field for [number of years] years. I am passionate about [reason for being in the field]. I am always looking for new challenges and opportunities to grow and learn. I am a [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others. I am [character trait] and I am always ready to help others. I am [character trait] and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its fashion, art, and cuisine, and is a popular tourist destination for visitors from around the world. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a vibrant and dynamic city with a rich cultural heritage and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be an increased need for privacy and security measures to protect the data and personal information that is generated and processed by AI systems. This could lead
    


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
    Generated text:  [Your Name] and I'm a versatile and highly adaptable professional with a love for self-improvement and learning new things. I've been working in the industry for over [X] years, gaining a deep understanding of the world of [Your Industry]. My journey has been filled with challenges, but I've always been determined to keep learning and improving. I'm always eager to learn and adapt to new ideas and concepts, and I'm looking forward to exploring more of the world and making new connections. I'm passionate about helping others grow and achieve their goals, and I'm always happy to share my knowledge and experience with others.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the south of the country. 
    
    a. 24.497 square kilometers
    b. 200,000 inhabitants
    c. 19th century
    d. 8th century
    e. 28th century
    f. 17th century
    g. 2nd century
    h. 30th century
    i. 1st century
    j. 21st century
    
    1. Which of the following is not a fact about France’s capital city?
    a. It has a population of 19th century.
    b. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and challenges. Here are some potential trends and developments that may shape AI in the coming years:
    
    1. Enhanced AI capabilities: AI is becoming more sophisticated and capable of performing a wide range of tasks that were previously thought to be beyond its capabilities. This includes tasks such as image and speech recognition, natural language processing, and autonomous vehicles. The development of even more advanced AI systems will enable even more sophisticated applications of AI across industries.
    
    2. AI ethics and regulation: As AI systems become more widely used, there will be increasing pressure to address their ethical implications. This includes issues such as bias, privacy, and data security


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

     am

     a

     [

    your

     profession

    ]

     with

     [

    your

     background

    ,

     education

    ,

     and

     experiences

    ].

     I

     have

     always

     been

     passionate

     about

     [

    your

     hobby

     or

     interest

    ].

     I

     enjoy

     [

    your

     skills

     or

     talents

    ]

     and

     I

     am

     constantly

     looking

     for

     new

     ways

     to

     [

    your

     goal

     or

     ambition

    ].

     I

    'm

     always

     trying

     to

     make

     the

     world

     a

     better

     place

     and

     I

     believe

     that

     being

     a

     part

     of

     a

     team

     is

     the

     best

     way

     to

     achieve

     that

    .

     I

    'm

     excited

     to

     join

     the

     team

     and

     make

     a

     difference

    !

     [

    Your

     Name

    ].

     
    


    Please

     provide

     a

     summary

     of

     your

     character

    's

     profession

    ,

     background

    ,

     hobbies

    ,

     skills

    ,

     and

     aspirations

    .

     I

     am

     looking

     for

     a

     character

     that

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     North

    -east

    ern

     part

     of

     the

     country

    .


    1

    


    How

     many

     inhabitants

     does

     Paris

     have

    ?

     

    2

    


    2

    


    How

     many

     people

     are

     there

     in

     Paris

    ?

     

    3

    


    3

    


    Paris

     is

     the

     largest

     city

     in

     which

     European

     country

    ?

     

    4

    


    4

    


    What

     is

     the

     city

    's

     population

    ?

     

    5

    


    5

    


    What

     is

     the

     capital

     city

     of

     Germany

    ?

     

    6

    


    6

    


    How

     many

     inhabitants

     does

     the

     capital

     city

     of

     Germany

     have

    ?

     

    7

    


    7

    


    How

     many

     people

     are

     there

     in

     the

     capital

     city

     of

     Germany

    ?

     

    8

    


    8

    


    How

     many

     people

     live

     in

     the

     capital

     city

     of

     Germany

    ?

     

    9

    


    9

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     difficult

     to

     predict

    ,

     but

     some

     possible

     trends

     that

     may

     occur

     include

    :
    


    1

    .

     Increased

     specialization

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

     easier

     for

     businesses

     to

     focus

     on

     specific

     tasks

    ,

     reducing

     the

     need

     for

     specialized

     skills

    .

     This

     could

     lead

     to

     a

     more

     specialized

     workforce

     with

     a

     higher

     level

     of

     education

     and

     training

    .
    


    2

    .

     AI

     integration

     with

     human

     workers

    :

     With

     the

     growth

     of

     AI

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     integration

     between

     AI

     and

     human

     workers

    .

     This

     could

     lead

     to

     a

     more

     efficient

     and

     effective

     work

     environment

    ,

     but

     it

     may

     also

     lead

     to

     job

     displacement

     for

     some

     workers

    .
    


    3

    .

     AI

     enhancing

     creativity

    :

     As

     AI

     continues

     to

     improve

    ,

    



```python
llm.shutdown()
```
