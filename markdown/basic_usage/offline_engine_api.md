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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.62s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.19it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.91it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 22.68it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.65 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   2%|▏         | 1/58 [00:00<00:06,  8.62it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.65 GB):   5%|▌         | 3/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.65 GB):   5%|▌         | 3/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.65 GB):   5%|▌         | 3/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):   5%|▌         | 3/58 [00:00<00:03, 14.44it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.64 GB):  10%|█         | 6/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.63 GB):  10%|█         | 6/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=55.63 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.62 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.62 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.62 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.61 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.27it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.06it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 26.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.58 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.57 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.29it/s]Capturing num tokens (num_tokens=960 avail_mem=55.55 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.29it/s] Capturing num tokens (num_tokens=896 avail_mem=55.48 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.29it/s]Capturing num tokens (num_tokens=832 avail_mem=54.98 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.29it/s]Capturing num tokens (num_tokens=832 avail_mem=54.98 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=768 avail_mem=54.90 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.93it/s]Capturing num tokens (num_tokens=704 avail_mem=54.90 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.93it/s]Capturing num tokens (num_tokens=640 avail_mem=54.90 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.93it/s]Capturing num tokens (num_tokens=576 avail_mem=54.90 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.93it/s]

    Capturing num tokens (num_tokens=576 avail_mem=54.90 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.32it/s]Capturing num tokens (num_tokens=512 avail_mem=54.88 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.32it/s]Capturing num tokens (num_tokens=480 avail_mem=54.90 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.32it/s]Capturing num tokens (num_tokens=448 avail_mem=54.90 GB):  48%|████▊     | 28/58 [00:01<00:01, 27.32it/s]Capturing num tokens (num_tokens=448 avail_mem=54.90 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.06it/s]Capturing num tokens (num_tokens=416 avail_mem=54.90 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.06it/s]Capturing num tokens (num_tokens=384 avail_mem=54.89 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.06it/s]

    Capturing num tokens (num_tokens=352 avail_mem=54.89 GB):  53%|█████▎    | 31/58 [00:01<00:01, 26.06it/s]Capturing num tokens (num_tokens=352 avail_mem=54.89 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.73it/s]Capturing num tokens (num_tokens=320 avail_mem=54.88 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.73it/s]Capturing num tokens (num_tokens=288 avail_mem=54.88 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.73it/s]Capturing num tokens (num_tokens=256 avail_mem=54.88 GB):  59%|█████▊    | 34/58 [00:01<00:00, 25.73it/s]Capturing num tokens (num_tokens=256 avail_mem=54.88 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.49it/s]Capturing num tokens (num_tokens=240 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.49it/s]Capturing num tokens (num_tokens=224 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.49it/s]Capturing num tokens (num_tokens=208 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.49it/s]

    Capturing num tokens (num_tokens=192 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.49it/s]Capturing num tokens (num_tokens=192 avail_mem=54.87 GB):  71%|███████   | 41/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=176 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=160 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=144 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=128 avail_mem=54.85 GB):  71%|███████   | 41/58 [00:01<00:00, 27.95it/s]Capturing num tokens (num_tokens=128 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=112 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.67it/s]

    Capturing num tokens (num_tokens=96 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.67it/s] Capturing num tokens (num_tokens=80 avail_mem=54.84 GB):  78%|███████▊  | 45/58 [00:01<00:00, 28.67it/s]Capturing num tokens (num_tokens=80 avail_mem=54.84 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.89it/s]Capturing num tokens (num_tokens=64 avail_mem=54.84 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.89it/s]Capturing num tokens (num_tokens=48 avail_mem=54.84 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.89it/s]Capturing num tokens (num_tokens=32 avail_mem=54.83 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.89it/s]Capturing num tokens (num_tokens=28 avail_mem=54.83 GB):  83%|████████▎ | 48/58 [00:01<00:00, 28.89it/s]Capturing num tokens (num_tokens=28 avail_mem=54.83 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.45it/s]Capturing num tokens (num_tokens=24 avail_mem=54.83 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.45it/s]

    Capturing num tokens (num_tokens=20 avail_mem=54.82 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.45it/s]Capturing num tokens (num_tokens=16 avail_mem=54.82 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.45it/s]Capturing num tokens (num_tokens=12 avail_mem=54.82 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.45it/s]Capturing num tokens (num_tokens=12 avail_mem=54.82 GB):  97%|█████████▋| 56/58 [00:02<00:00, 32.66it/s]Capturing num tokens (num_tokens=8 avail_mem=54.82 GB):  97%|█████████▋| 56/58 [00:02<00:00, 32.66it/s] Capturing num tokens (num_tokens=4 avail_mem=54.81 GB):  97%|█████████▋| 56/58 [00:02<00:00, 32.66it/s]Capturing num tokens (num_tokens=4 avail_mem=54.81 GB): 100%|██████████| 58/58 [00:02<00:00, 27.03it/s]


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
    Generated text:  Lisa and I am a 22 year old female. I recently discovered I have a second sex. This is happening because of a pregnancy that I had. This has been going on for 2 years and I am still trying to find out if I can have children, but I am not ready to give up hope, and I am 100% willing to have a child. What can I do to make sure that I am healthy, happy, and that I will have a baby?
    
    Thank you in advance for your help. I am really looking forward to hearing from you soon!
    
    Lisa
    Lisa, you are in excellent
    ===============================
    Prompt: The president of the United States is
    Generated text:  now considering a new strategy for foreign policy, aiming to promote democracy and human rights. In order to achieve this goal, the president plans to allocate a total budget of $100,000 to various initiatives. The budget allocation for each initiative is based on the level of support it receives. The first initiative receives 25% of the budget, the second initiative receives 30%, the third initiative receives 35%, the fourth initiative receives 40%, and the fifth initiative receives 45%. The president wants to ensure that at least 10% of the budget is allocated to each initiative to ensure
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Marseille
    C. Nice
    D. Lyon
    Answer:
    A
    
    The main factors influencing the cultural characteristics of different regions in the world are mainly ____.
    A. Economy, transportation, and education
    B. Economy, transportation, and society
    C. Politics, transportation, and society
    D. Politics, society, and education
    Answer:
    B
    
    The leader of the Iranian Islamic Republic of Iran is ____.
    A. Muhammad
    B. Qajar
    C. Shah
    D. Qutb
    Answer:
    C
    
    Under certain conditions, substances undergoing chemical reactions can produce both exothermic
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain. Can we predict its direction and impact with any certainty?
    
    What is the future of AI and how does it impact us?
    
    Please answer these questions:
    
    1. When was the last major milestone in AI?
    2. How was the AI milestone last achieved?
    3. How is the AI milestone likely to be achieved in the future?
    4. Why is the AI milestone likely to be achieved in the future?
    5. What are some potential impacts of the AI milestone in the future?
    
    Use the data in the following table to answer the questions:
    
    | Category | AI Milestone | Impact |
    | --- | --- | --- |
    | Predictions


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? I'm a [age] year old, and I have a [major] degree in [field of study]. I'm passionate about [interest or hobby]. I'm always looking for new experiences and learning new things. What's your favorite hobby or activity? I love [mention a hobby or activity]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I love [mention a book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the European Union and the second-largest city in the world by population. It is located in the south of the country and is the seat of government, administration, and culture for the French Republic. Paris is known for its rich history, art, and cuisine, and is a major tourist destination. It is also home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is known for its vibrant nightlife and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs and preferences.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations and guidelines for its development and use. This could lead to more robust and transparent
    


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
    Generated text:  [insert name] and I am [insert age], [insert occupation or profession]. I am a [insert personality trait or trait that distinguishes me as a unique individual]. I am currently [insert career or academic field of study], and I have always been passionate about [insert something related to the character's interest or hobby]. I am a [insert a literary term or concept related to the character], with a keen sense of [insert a descriptive word related to the character's personality] and a [insert a genre related to the character's genre]. I am [insert a descriptor related to the character's personality] and [insert a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its diverse and iconic architecture, iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, and its rich cultural heritage, particularly in terms of art and history. The city is also known for its gastronomy, with its cuisine being a major contributor to the country’s culinary scene. Overall, Paris is a city that is characterized by its unique blend of history, culture, and modernity. It is a bustling metropolis with a rich history and a vibrant nightlife. Paris is often referred to as the “City of Light” and is a popular destination for tourists and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly interconnected, with machines increasingly capable of performing tasks that once required human intelligence. Here are some potential trends that may shape the future of AI:
    
    1. Increased deployment of AI in various fields: With the rise of more advanced AI models, we may see an increase in their deployment in various fields, from healthcare and finance to manufacturing and transportation.
    
    2. Greater emphasis on ethical considerations: As AI is becoming more prevalent in everyday life, there will be a growing emphasis on ethical considerations, including privacy, bias, and transparency.
    
    3. More integration with human emotions: As AI continues to learn and evolve, we may see more


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

     am

     a

     [

    role

    ]

     specialist

    .

     I

     have

     a

     deep

     understanding

     of

     [

    topic

    ],

     and

     I

     enjoy

     helping

     people

     achieve

     their

     goals

    .

     I

     believe

     in

     treating

     people

     with

     respect

     and

     kindness

    ,

     and

     I

     strive

     to

     create

     a

     positive

     and

     supportive

     work

     environment

    .

     My

     goal

     is

     to

     help

     people

     feel

     confident

     and

     confident

    ,

     and

     I

    'm

     always

     eager

     to

     assist

     anyone

     who

     is

     in

     need

    .

     I

     believe

     that

     with

     my

     experience

     and

     my

     willingness

     to

     learn

    ,

     I

     can

     make

     a

     significant

     impact

     in

     people

    's

     lives

    .

     Thank

     you

     for

     asking

    ,

     and

     I

     look

     forward

     to

     hearing

     from

     you

    !

     [

    Name

    ]

     [

    Role

    ]

     Specialist

     [

    Name

    ]

     [

    Role

    ]

     Specialist

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    la

     Ville

    -Mar

    ie

    ,"

     which

     means

     "

    the

     city

     of

     water

    "

     in

     French

    .

     
    


    This

     statement

     summarizes

     the

     key

     facts

     about

     the

     capital

     of

     France

    ,

     including

     its

     name

    ,

     origin

    ,

     and

     meaning

    .

     It

     leaves

     room

     for

     further

     exploration

     or

     elabor

    ation

     if

     necessary

    .

     The

     city

     is

     famous

     for

     its

     famous

     landmarks

    ,

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     home

     to

     many

     important

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     as

     well

     as

     theaters

     and

     museums

     dedicated

     to

     literature

    ,

     music

    ,

     and

     art

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     culture

    ,

     including

     its

     annual

     M

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     heavily

     influenced

     by

     several

     trends

    ,

     including

    :
    


    1

    .

     Increased

     Integration

    :

     As

     AI

     becomes

     more

     advanced

     and

     powerful

    ,

     it

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     including

     machine

     learning

    ,

     robotics

    ,

     and

     cognitive

     systems

    .

     This

     integration

     will

     allow

     AI

     systems

     to

     better

     adapt

     to

     their

     environment

     and

     improve

     their

     performance

    .
    


    2

    .

     Autonomous

     and

     Semi

    -A

    ut

    omatic

    :

     AI

     is

     increasingly

     being

     used

     in

     autonomous

     vehicles

    ,

     drones

    ,

     and

     other

     self

    -driving

     systems

    .

     As

     these

     technologies

     become

     more

     advanced

     and

     pervasive

    ,

     it

     is

     likely

     that

     they

     will

     become

     more

     autonomous

     and

     semi

    -aut

    onomous

    ,

     with

     more

     human

     oversight

     as

     necessary

    .
    


    3

    .

     AI

     Ethics

    :

     As

     AI

     systems

     become

     more

    



```python
llm.shutdown()
```
