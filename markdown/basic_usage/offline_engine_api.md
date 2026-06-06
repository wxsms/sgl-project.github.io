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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.03it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:05<00:56,  1.05s/it]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:05<00:18,  2.68it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:07,  5.44it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.17it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 13.27it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:06<00:02, 13.27it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:06<00:02, 13.27it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:06<00:02, 13.27it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:06<00:02, 13.27it/s]

    Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:06<00:01, 20.11it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]

    Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:06<00:00, 27.58it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 35.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  9.14it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.87it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.85it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.73 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.73 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.72 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.59it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.72 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.70 GB):  21%|██        | 12/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.63 GB):  21%|██        | 12/58 [00:00<00:03, 15.23it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.63 GB):  24%|██▍       | 14/58 [00:00<00:03, 13.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.62 GB):  24%|██▍       | 14/58 [00:00<00:03, 13.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.62 GB):  24%|██▍       | 14/58 [00:01<00:03, 13.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.21 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:01<00:02, 14.15it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.37it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:01<00:02, 17.37it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:01<00:01, 19.07it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.96it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.96it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.96it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:01<00:01, 20.96it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.23it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.23it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.23it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.23it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.79it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:01<00:00, 25.21it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 25.21it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 25.21it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 25.21it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 25.21it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.32it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.32it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.32it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 27.32it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.20it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 26.20it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 26.20it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:02<00:00, 26.20it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.17it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.17it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.17it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:02<00:00, 26.17it/s]

    Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.67it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.67it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.67it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  83%|████████▎ | 48/58 [00:02<00:00, 22.67it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.80it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:02<00:00, 23.80it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.79it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:02<00:00, 24.79it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:02<00:00, 27.85it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:02<00:00, 22.12it/s]


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
    Generated text:  Alfie, a 7 year old boy with ADHD. I'm a student at a public school. I have a lot of sleep problems and am not a good sleeper. I have difficulty with organizing my schedule, my academic performance and self-esteem, and I have trouble sleeping when I'm in a social situation. I feel worried when I am in school, and I tend to avoid school. I have tried lots of different things to try to make things better, like changing my school day, playing with my friends, going to therapy, but nothing seems to work. What can I do to help my ADHD?
    
    It sounds like you are
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of a party that is known for supporting what?
    Answer: The president of the United States is a member of a party known for supporting the Democratic Party. The Democratic Party is one of the two major political parties in the United States, alongside the Republican Party. The Democratic Party is primarily associated with the Democratic International, which is a global political organization focused on promoting democracy and human rights around the world. In the United States, the Democratic Party is considered the dominant party with the most seats in the House of Representatives and the Senate. The Democratic Party has been at the forefront of promoting social and political justice, women's rights, and
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following is the capital of France?
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following cities is the capital of France?
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    Answer: A
    
    Which of the following is NOT a capital of France?
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    Answer: C
    
    Which of the following cities is NOT the capital of France?
    A. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon, and with it, the possibilities for development are endless. With the increasing use of AI in various industries, the need for professionals with skills in AI development has become more significant. In this blog post, we will explore the current state of AI development, the different AI technologies, and the areas where AI is most needed.
    
    The current state of AI development is evolving rapidly, with the introduction of new AI technologies and advancements in programming languages and frameworks. With the rise of machine learning and deep learning, AI is becoming increasingly capable of processing and analyzing large amounts of data at a rapid pace. These advancements have the potential to revolution


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


    Generated text:  [Name] and I am a [occupation] with [number of years] years of experience in [field]. I am a [type of person] who is [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative trait]. I am [positive trait] and [negative
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris". It is the largest city in France and the third-largest city in the world by population. The city is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also known for its rich cultural heritage, including its art, music, and cuisine. The city is a major economic and political center in France and plays a significant role in the country's cultural and political life. Paris is a popular tourist destination and is home to many international institutions and organizations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective applications of AI in various industries.
    
    3. Increased use of AI in healthcare: AI is
    


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
    Generated text:  __________ and I'm a/an __________. Let's start by setting up some context. What's your name? What's your profession? Can you tell us a little bit about yourself? What are your hobbies or interests? What's your favorite hobby or activity?
    
    It's great to have you join us on this journey. Let's dive into our conversation!
    
    I'm a/an computer programmer. I've been working with computers for over a decade. I love to code and solve problems with programming languages. I'm always looking for new ways to improve myself and stay up to date with the latest technology. I enjoy programming and teaching people
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city in the country and serves as the center of the nation's political, economic, and cultural life. It is also known as the "City of Love" for its romantic atmosphere and vibrant nightlife. Paris is home to many world-renowned landmarks, including Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. The city is also known for its diverse cuisine, including famous French dishes like foie gras and truffle. Paris has a long and rich history dating back to the 6th century BC, and it continues to be a major cultural and economic center of the country.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued progress and development in several areas:
    
    1. Greater integration with other technologies: As AI continues to become more advanced, it is likely to be integrated with other technologies such as sensors, data analytics, and blockchain, which will make it even more useful for a wide range of applications.
    
    2. Improved interpretability and transparency: As AI becomes more complex, it is likely to become more interpretable and transparent. This means that AI systems will be able to explain their decisions and actions to humans, which will be more beneficial in many applications.
    
    3. More autonomous AI: As autonomous vehicles and other autonomous systems become more prevalent,


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

     I

    'm

     a

     [

    Job

     Title

    ]

     with

     over

     [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     the

     [

    Industry

     or

     field

    ].

     I

    ’m

     always

     eager

     to

     learn

     new

     things

     and

     stay

     up

     to

     date

     on

     the

     latest

     trends

     and

     technologies

    ,

     and

     I

    ’m

     always

     open

     to

     new

     challenges

     and

     opportunities

    .

     I

    ’m

     a

     [

    Number

     of

     years

     of

     experience

    ]

     with

     [

    Number

     of

     years

    ]

     years

     of

     experience

     in

     the

     [

    Industry

     or

     field

    ].

     I

     enjoy

     [

    A

     specific

     hobby

     or

     activity

    ],

     and

     I

     think

     it

    ’s

     important

     to

     continue

     learning

     and

     growing

     as

     a

     professional

    .

     I

    'm

     a

     [

    Number

     of

     years

     of

     experience

    ]

     with

     [

    Number

     of

     years

    ]

     years

     of

     experience

     in

    
    
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

    .

     It

     is

     a

     major

     city

     in

     France

     and

     the

     largest

     city

     in

     Europe

    ,

     located

     on

     the

     western

     coast

     of

     the

     French

     Riv

    iera

     and

     on

     the

     right

     bank

     of

     the

     Se

    ine

     River

    .

     It

     is

     the

     seat

     of

     government

    ,

     the

     headquarters

     of

     the

     French

     Foreign

     Ministry

    ,

     the

     Ministry

     of

     Finance

    ,

     and

     the

     French

     Senate

    ,

     and

     is

     home

     to

     many

     of

     France

    's

     most

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

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

     and

     the

     Champ

     de

     Mars

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     street

     art

    ,

     fashion

    ,

     and

     cultural

     events

    ,

     as

     well

     as

     its

     annual

     E

    ly

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     proliferation

     of

     new

     technologies

     and

     applications

     that

     will

     further

     advance

     its

     capabilities

    .

     Here

     are

     some

     possible

     trends

     that

     are

     currently

     in

     the

     works

    :
    


    1

    .

     Increased

     collaboration

     between

     AI

     and

     other

     fields

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

     and

     fields

    ,

     such

     as

     robotics

    ,

     natural

     language

     processing

    ,

     and computer

     vision,

     it

     is

     likely

     that

     these

     will

     work

     together

     to

     solve

     complex

     problems

     in

     new

     and

     innovative

     ways

    .
    


    2

    .

     AI

    -powered

     autonomous

     vehicles

    :

     With

     the

     increasing

     popularity

     of

     autonomous

     vehicles

    ,

     AI

     will

     continue

     to

     play

     a

     critical

     role

     in

     making

     them

     safer

     and

     more

     efficient

    .

     As

     autonomous

     vehicles

     become

     more

     common

    ,

     we

     will

     see

     an

     increase

     in

     the

     use

    



```python
llm.shutdown()
```
