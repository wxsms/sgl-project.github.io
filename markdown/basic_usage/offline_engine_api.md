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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:43,  4.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:43,  4.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.31it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.54it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.47it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=320):  45%|████▍     | 26/58 [00:05<00:02, 11.73it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 26.29it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 34.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 18.73it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.92it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:00<00:02, 23.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.71it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.00it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  33%|███▎      | 19/58 [00:00<00:01, 29.00it/s] Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.84it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:00<00:01, 24.84it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:00<00:01, 25.24it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.24it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.24it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.24it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:01, 28.13it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  50%|█████     | 29/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 29.71it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 31.77it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 33.25it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.62it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.62it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.12it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.12it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.82it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 38.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 31.03it/s]


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
    Generated text:  Alex. I am a man of mixed blood. I was born in England. I have a twin brother and a twin sister. I have white hair. I have a very dark brown skin. I have a brown and black color. I have a lustrous hair. I have a tattoo on my left arm. I have a tattoo on my right arm. What am I?
    Based on the information provided, Alex is a man. He is a twin of his sibling, and he has a white hair, dark brown skin, brown and black color, lustrous hair, and a tattoo on his left arm. The color and
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official who serves a term of 4 years, during which time they serve on various committees and promote policies. While President Trump, who was elected in 2017, faced criticism for his handling of the country during his time in office. Recently, the House of Representatives voted to create a commission to investigate the health care system and may result in a potential health care overhaul, which has been the main focus of some of his policies.
    
    Do you think the House of Representatives has the power to vote on a health care overhaul? Yes, the House of Representatives does have the power to vote on a health care overhaul. When
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. London C. Rome D. Berlin
    Answer:
    
    A
    
    In a certain city, the population is 300,000 and the employment rate is 50%. What is the number of unemployed people in this city?
    A. 150,000
    B. 200,000
    C. 180,000
    D. 250,000
    Answer:
    
    B
    
    When the basic block system is out of order, the train dispatcher must issue a ____ order for the train to run according to
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, with predictions ranging from the future of self-driving cars and robots, to the potential for AI to power nuclear power, make genetic engineering possible, and create safer and more efficient machines. While many people are excited about the possibilities, there are also concerns about the potential for AI to be harmful. The reality is that AI is here to stay, and it will continue to play an increasingly important role in our society. But how can we ensure that AI is used responsibly and safely?
    One approach to ensuring that AI is used responsibly and safely is to implement a robust privacy and data protection framework. This framework should include clear guidelines for


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the world by population. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also known for its fashion industry, which is one of the largest in the world. The city is home to many important institutions such as the French Academy of Sciences and the French National Library. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability. AI developers will need to be more transparent about their algorithms and how they are used, and they will need to be accountable for the decisions they make.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This
    


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
    Generated text:  [insert your name]. I’m a [insert your occupation, e.g., artist, writer, musician, etc.]. I’m passionate about [insert what you enjoy doing] and I’ve always been inspired by [insert inspirational story about you, e.g., a book, song, movie, etc.]. I enjoy [insert interests or hobbies] as much as possible and I’m always looking for new ways to share my love for art with the world. I’m always eager to learn new things and to grow in my craft. If you’d like to meet me in person, I can be reached at [insert your phone number
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the second-largest city in the world by population. The city is known for its historical landmarks, such as the Eiffel Tower and Notre-Dame Cathedral, as well as its fashion, art, and culture. It is also an important transportation hub, with its network of underground tunnels and high-speed trains. The city is home to many universities, arts institutions, and a diverse range of cuisine and music. Paris is a city of contrasts and attractions, and is a must-visit destination for those interested in France's rich history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and constantly evolving, with a number of potential developments and trends shaping how it will impact our lives. Here are some of the key trends to watch in the AI landscape in the coming years:
    
    1. Increased integration with other technologies: As more and more industries start to rely on AI, we are likely to see more integration between AI and other technologies, such as smart cities, self-driving cars, and virtual and augmented reality.
    
    2. Personalized AI: AI will become more personalized as it learns from users' data and behaviors, offering more tailored solutions to their needs.
    
    3. Self-driving cars: As the technology for autonomous vehicles continues


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

    ],

     and

     I

    'm

     a

     [

    Your

     Profession

    ].

     I

     started

     my

     career

     in

     [

    Your

     First

     Job

    ],

     but

     this

     is

     my

     second

    ,

     as

     I

     have

     been

     following

     my

     passion

     for

     [

    Your

     Profession

    ].

     I

     have

     been

     dedicated

     to

     learning

     and

     developing

     [

    Your

     Profession

    ]

     and

     have

     been

     passionate

     about

     [

    Your

     Profession

    ]

     ever

     since

    .

     I

     have

     always

     had

     a

     love

     for

     life

     and

     constantly

     strive

     to

     make

     the

     world

     a

     better

     place

    .

     I

     am

     always

     looking

     for

     ways

     to

     help

     others

     and

     have

     been

     actively

     involved

     in

     community

     service

    .

     I

    'm

     an

     [

    Your

     Character

     Trait

     or

     Personality

    ]

     person

    .

     I

     enjoy

     spending

     time

     with

     [

    Your

     Favorite

     Pets

     or

     Friends

    ],

     exploring

     the

     outdoors

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     commonly

     referred

     to

     as

     "

    La

     Pres

    se

    "

     or

     "

    P

    rix

    "

     (

    liter

    ally

     "

    The

     Post

    ")

     in

     English

    .

     It

     is

     the

     largest

     city

     and

     most

     populous

     city

     in

     France

    ,

     with

     a

     population

     of

     over

     

    6

     million

     people

    .

     The

     city

     is

     located

     on

     the

     River

     Se

    ine

    ,

     along

     the

     eastern

     bank

     of

     the

     Se

    ine

     River

    ,

     and

     is

     known

     for

     its

     historical

     significance

     and

     its

     role

     in

     French

     politics

    ,

     culture

    ,

     and

     economy

    .

     It

     is

     also

     home

     to

     many

     well

    -known

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     the

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

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     is

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     number

     of

     different

     trends

     and

     developments

    ,

     depending

     on

     a

     variety

     of

     factors

    ,

     including

     technological

     advances

    ,

     policy

     decisions

    ,

     and

     human

     behavior

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

     transparency

     and

     accountability

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     increasing

     pressure

     to

     make

     them

     more

     transparent

     and

     accountable

     to

     users

    .

     This

     includes

     measures

     to

     prevent

     bias

     in

     algorithms

     and

     to

     ensure

     that

     AI

     systems

     are

     designed

     to

     operate

     within

     legal

     and

     ethical

     boundaries

    .
    


    2

    .

     More

     personalized

     experiences

    :

     As

     AI

     systems

     are

     able

     to

     learn

     and

     adapt

     to

     different

     situations

    ,

     they

     will

     be

     able

     to

     provide

     more

     personalized

     experiences

     to

     users

    .

     This

     could

     include

     recommendations

    



```python
llm.shutdown()
```
