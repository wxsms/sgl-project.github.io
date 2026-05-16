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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]


    2026-05-16 10:37:52,974 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-16 10:37:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:59,  4.20s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.56it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:05,  7.82it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:05,  7.82it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 11.73it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.43it/s]

    Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.63it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:04<00:00, 30.77it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:04<00:00, 30.77it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:04<00:00, 30.77it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 30.77it/s]

    Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 39.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.28 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.27 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.26 GB):   3%|▎         | 2/58 [00:00<00:03, 15.78it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.26 GB):   7%|▋         | 4/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.26 GB):   7%|▋         | 4/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.25 GB):   7%|▋         | 4/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.24 GB):   7%|▋         | 4/58 [00:00<00:03, 17.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.24 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.21 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.21 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.20 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.20 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.19 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 23.07it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.18 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.69it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.66 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.56 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.49 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.48 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.39it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=73.48 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.46 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=960 avail_mem=73.47 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s] Capturing num tokens (num_tokens=896 avail_mem=73.47 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=832 avail_mem=73.47 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=768 avail_mem=73.46 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=704 avail_mem=73.46 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=704 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.49it/s]Capturing num tokens (num_tokens=640 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.49it/s]Capturing num tokens (num_tokens=576 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.49it/s]Capturing num tokens (num_tokens=512 avail_mem=73.44 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=480 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=448 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.49it/s]

    Capturing num tokens (num_tokens=416 avail_mem=73.45 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.49it/s]Capturing num tokens (num_tokens=416 avail_mem=73.45 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=384 avail_mem=73.45 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=352 avail_mem=73.45 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=320 avail_mem=73.44 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=288 avail_mem=73.44 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=256 avail_mem=73.44 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=240 avail_mem=73.43 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.53it/s]Capturing num tokens (num_tokens=240 avail_mem=73.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=224 avail_mem=73.43 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=208 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=192 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=176 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]

    Capturing num tokens (num_tokens=160 avail_mem=73.42 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=144 avail_mem=73.41 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=144 avail_mem=73.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s]Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s]Capturing num tokens (num_tokens=112 avail_mem=73.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s]Capturing num tokens (num_tokens=96 avail_mem=73.41 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s] Capturing num tokens (num_tokens=80 avail_mem=73.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.12it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=32 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]

    Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.03it/s]Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.06it/s] Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.06it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 35.92it/s]


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
    Generated text:  Fangzhi and I'm a master at creating a beautiful and personalized website design for you. I'll work with you to create a website that suits your specific needs and can be easily updated or modified to fit your changing requirements. Our goal is to create a website that is not only functional, but also visually appealing and easy to navigate. We'll work together to ensure that your website is responsive and accessible to people with disabilities. With my help, your website will be the star of your digital world. You can trust me to create the perfect website for you! 
    
    Can you please describe the design process that you will follow to create a
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to send the troops to war in Afghanistan or to war in Yemen. He is weighing the two options very seriously and has just received information about the two conflicts. This information is that the first conflict involves an enemy that is located in a mountainous terrain. The second conflict involves an enemy that is located on a plain. The president needs to decide whether to send the troops to war in Afghanistan or to war in Yemen. The president's advisor suggests that he should go with the mountainous terrain conflict and does not recommend war in Yemen. Which of the following best explains why the president's advisor chose to go with the mountain
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A) Paris B) Rome C) Moscow D) Istanbul
    A: Paris
    B: Rome
    C: Moscow
    D: Istanbul
    The capital of France is Paris. Therefore, the correct answer is:
    
    A) Paris
    
    To verify, I'll check the information provided in the choices:
    - Paris is the capital of France.
    - Rome is the capital of Italy.
    - Moscow is the capital of Russia.
    - Istanbul is the capital of Turkey.
    
    None of these are the correct options for the capital of France. Therefore, the answer is A) Paris. If I have made a mistake, please let me know.
    ===============================
    Prompt: The future of AI is
    Generated text:  unpredictable and cannot be predicted with certainty, as it is constantly evolving and developing. However, some industries, such as healthcare, finance, and education, have already begun to benefit from AI technologies. The development of AI has the potential to revolutionize the way we live and work, and it has the power to improve our lives in countless ways. With the right investments and policies, AI can become a game changer for many industries and contribute to the global economy. It is also important to acknowledge the potential risks and challenges that come with AI, such as privacy concerns, job displacement, and bias. Ultimately, the future of AI is uncertain,


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I enjoy [job title] because [reason for passion]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for fun? I enjoy [job title] because [reason for enjoyment]. I like to [job title] because [reason for enjoyment]. What do you like to do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major cultural and economic center, hosting many world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many notable French artists, writers, and musicians, and is known for its rich history and cultural heritage. Paris is a city of contrasts, with its modern architecture and vibrant nightlife
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective use of resources, as well as better decision-making in various industries.
    
    3.
    


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
    Generated text:  [Your Name]. I'm a [Your Occupation] and I'm passionate about [Your Passion] and I believe in [Your Vision]. I have a strong work ethic and a great team spirit. I'm organized, detail-oriented, and am always willing to go the extra mile to achieve my goals. I enjoy working with people and am always up for a challenge. Please let me know if you'd like to know more about me! [Your Name].  (You can use this as a placeholder for your real name or any other personal information). I'm a [Your Name] and I'm passionate about [Your Passion] and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is an important cultural, economic, and political center, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is home to many museums and galleries, and is a popular tourist destination. Additionally, it has a rich and diverse cultural scene, with many festivals and events throughout the year. Despite its fame, Paris has a complex and sometimes controversial history, including its involvement in the French Revolution and its relationship with the United States. As of 2021, the population of Paris is around 2. 3 million people, and it is the most
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and there are many potential trends that could shape how it develops and evolves. Here are some of the key trends to watch:
    
    1. Advancements in machine learning and deep learning: Machine learning and deep learning are two key areas of AI development. As these technologies become more advanced, we may see faster and more accurate algorithms that can make decisions and solve problems more efficiently.
    
    2. Integration with human language: With the growing number of people using AI in their daily lives, we may see more AI systems that can understand and interpret human language, such as chatbots and language translation systems.
    
    3. Personalization: As AI becomes more advanced


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

    ]

     and

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     girl

    .

     I

     love

     to

     write

    ,

     draw

    ,

     and

     play

     games

    .

     I

     also

     enjoy

     spending

     time

     with

     my

     family

     and

     friends

    .

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     an

     AI

     language

     model

    ,

     I

     do

     not

     have

     a

     physical

     presence

    ,

     but

     I

     can

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     may

     have

    .

     How

     can

     I

     help

     you

     today

    ?

     Do

     you

     have

     any

     specific

     questions

     or

     topics

     you

     would

     like

     me

     to

     assist

     with

    ?

     What

     do

     you

     like

     to

     do

     for

     fun

    ?

     Do

     you

     have

     any

     games

     or

     apps

     you

     enjoy

     playing

    ?

     Please

     feel

     free

     to

     tell

     me

     more

     about

     yourself

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

    


    B

    .

     False

    
    


    A

    .

     True

    
    


    The

     statement

     is

     true

     because

     Paris

     is

     the

     capital

     of

     France

    ,

     which

     is

     located

     in

     the

     south

     of

     the

     country

    .

     Paris

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     influential

     role

     in

     French

     politics

     and

     culture

    .

     The

     statement

     does

     not

     match

     the

     current

     capital

     of

     France

    ,

     which

     is

     Nice

    .

     The

     current

     capital

     is

     Nancy

    .

     
    


    The

     statement

     is

     accurate

     and

     doesn

    't

     contain

     any

     factual

     errors

    .

     It

     is

     therefore

     true

    .

     However

    ,

     if

     we

     were

     to

     choose

     the

     best

     answer

     based

     on

     the

     given

     options

    ,

     A

     would

     be

     considered

     the

     correct

     choice

    .

     The

     most

     appropriate

     answer

     is

     "

    True

    "

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     wide

     range

     of

     trends

    ,

     including

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     is

     already

     being

     used

     to

     automate

     many

     routine

     tasks

    ,

     but

     future

     trends

     could

     see

     increased

     automation

     in

     areas

     like

     manufacturing

    ,

     logistics

    ,

     and

     customer

     service

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     already

     being

     integrated

     with

     other

     technologies

     like

     machine

     learning

    ,

     but

     future

     trends

     could

     see

     more

     integration

     and

     co

    existence

     of

     AI

     and

     other

     technologies

    .
    


    3

    .

     Increased

     focus

     on

     ethics

     and

     privacy

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

     increased

     scrutiny

     of

     the

     ethical

     and

     privacy

     concerns

     associated

     with

     AI

    .
    


    4

    .

     AI

    -driven

     healthcare

     advancements

    :

     AI

     is

     already

    



```python
llm.shutdown()
```
