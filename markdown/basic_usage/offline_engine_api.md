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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.27it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.05it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=576):  33%|███▎      | 19/58 [00:04<00:04,  8.49it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.67it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]

    Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:00, 22.06it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 30.42it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 30.42it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 30.42it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 41.92it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 41.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.67 GB):   2%|▏         | 1/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.64 GB):   2%|▏         | 1/58 [00:00<00:06,  8.23it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=55.63 GB):   2%|▏         | 1/58 [00:00<00:06,  8.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.63 GB):   5%|▌         | 3/58 [00:00<00:04, 11.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.14 GB):   5%|▌         | 3/58 [00:00<00:04, 11.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.05 GB):   9%|▊         | 5/58 [00:00<00:03, 13.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.97 GB):   9%|▊         | 5/58 [00:00<00:03, 13.25it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=54.96 GB):   9%|▊         | 5/58 [00:00<00:03, 13.25it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.96 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.96 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.96 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.48it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.96 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.95 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.64it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.95 GB):  16%|█▌        | 9/58 [00:00<00:03, 12.64it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=54.95 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.95 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.94 GB):  19%|█▉        | 11/58 [00:00<00:03, 13.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.94 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.94 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.94 GB):  22%|██▏       | 13/58 [00:01<00:03, 14.04it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=54.94 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2304 avail_mem=54.93 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.93 GB):  26%|██▌       | 15/58 [00:01<00:02, 15.23it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.93 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.30it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.93 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.92 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.92 GB):  29%|██▉       | 17/58 [00:01<00:02, 16.30it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=54.92 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.90 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.45it/s]Capturing num tokens (num_tokens=960 avail_mem=54.92 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.45it/s] Capturing num tokens (num_tokens=896 avail_mem=54.91 GB):  34%|███▍      | 20/58 [00:01<00:02, 18.45it/s]Capturing num tokens (num_tokens=896 avail_mem=54.91 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.49it/s]Capturing num tokens (num_tokens=832 avail_mem=54.91 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.49it/s]Capturing num tokens (num_tokens=768 avail_mem=54.91 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.49it/s]Capturing num tokens (num_tokens=704 avail_mem=54.90 GB):  40%|███▉      | 23/58 [00:01<00:01, 20.49it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.90 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.05it/s]Capturing num tokens (num_tokens=640 avail_mem=54.90 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.05it/s]Capturing num tokens (num_tokens=576 avail_mem=54.90 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.05it/s]Capturing num tokens (num_tokens=512 avail_mem=54.89 GB):  45%|████▍     | 26/58 [00:01<00:01, 22.05it/s]Capturing num tokens (num_tokens=512 avail_mem=54.89 GB):  50%|█████     | 29/58 [00:01<00:01, 24.15it/s]Capturing num tokens (num_tokens=480 avail_mem=54.90 GB):  50%|█████     | 29/58 [00:01<00:01, 24.15it/s]Capturing num tokens (num_tokens=448 avail_mem=54.90 GB):  50%|█████     | 29/58 [00:01<00:01, 24.15it/s]Capturing num tokens (num_tokens=416 avail_mem=54.90 GB):  50%|█████     | 29/58 [00:01<00:01, 24.15it/s]Capturing num tokens (num_tokens=384 avail_mem=54.90 GB):  50%|█████     | 29/58 [00:01<00:01, 24.15it/s]

    Capturing num tokens (num_tokens=384 avail_mem=54.90 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.68it/s]Capturing num tokens (num_tokens=352 avail_mem=54.89 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.68it/s]Capturing num tokens (num_tokens=320 avail_mem=54.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.68it/s]Capturing num tokens (num_tokens=288 avail_mem=54.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.68it/s]Capturing num tokens (num_tokens=256 avail_mem=54.88 GB):  57%|█████▋    | 33/58 [00:01<00:00, 27.68it/s]Capturing num tokens (num_tokens=256 avail_mem=54.88 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=240 avail_mem=54.88 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=224 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=208 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.65it/s]

    Capturing num tokens (num_tokens=192 avail_mem=54.87 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.65it/s]Capturing num tokens (num_tokens=192 avail_mem=54.87 GB):  71%|███████   | 41/58 [00:02<00:00, 30.96it/s]Capturing num tokens (num_tokens=176 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:02<00:00, 30.96it/s]Capturing num tokens (num_tokens=160 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:02<00:00, 30.96it/s]Capturing num tokens (num_tokens=144 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:02<00:00, 30.96it/s]Capturing num tokens (num_tokens=128 avail_mem=54.86 GB):  71%|███████   | 41/58 [00:02<00:00, 30.96it/s]Capturing num tokens (num_tokens=128 avail_mem=54.86 GB):  78%|███████▊  | 45/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=112 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=96 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:02<00:00, 32.09it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=54.85 GB):  78%|███████▊  | 45/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=64 avail_mem=54.84 GB):  78%|███████▊  | 45/58 [00:02<00:00, 32.09it/s]Capturing num tokens (num_tokens=64 avail_mem=54.84 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.56it/s]Capturing num tokens (num_tokens=48 avail_mem=54.84 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.56it/s]Capturing num tokens (num_tokens=32 avail_mem=54.84 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.56it/s]Capturing num tokens (num_tokens=28 avail_mem=54.83 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.56it/s]Capturing num tokens (num_tokens=24 avail_mem=54.83 GB):  84%|████████▍ | 49/58 [00:02<00:00, 32.56it/s]Capturing num tokens (num_tokens=24 avail_mem=54.83 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.05it/s]Capturing num tokens (num_tokens=20 avail_mem=54.82 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.05it/s]

    Capturing num tokens (num_tokens=16 avail_mem=54.82 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.05it/s]Capturing num tokens (num_tokens=12 avail_mem=54.82 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.05it/s]Capturing num tokens (num_tokens=8 avail_mem=54.82 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.05it/s] Capturing num tokens (num_tokens=8 avail_mem=54.82 GB):  98%|█████████▊| 57/58 [00:02<00:00, 33.62it/s]Capturing num tokens (num_tokens=4 avail_mem=54.81 GB):  98%|█████████▊| 57/58 [00:02<00:00, 33.62it/s]Capturing num tokens (num_tokens=4 avail_mem=54.81 GB): 100%|██████████| 58/58 [00:02<00:00, 23.18it/s]


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
    Generated text:  Caro, and I am a software engineer. I have been working for a big tech company for the past year, but my work is focused on machine learning and data science, and I have also been involved in developing and testing machine learning models for financial applications. My current job title is Assistant to the CEO, and I am working on a project that involves developing a new algorithm to improve the performance of financial trading systems. 
    
    I am interested in learning more about machine learning and data science, and would love to connect with experienced professionals who can share their knowledge and insights on the latest developments and trends in the field. Can you recommend any
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to reduce the use of fossil fuels by investing in renewable energy sources. This includes solar, wind, and geothermal energy. The president has identified a goal of reducing the use of fossil fuels by 50% by 2030. If the current fossil fuel consumption is 100,000 metric tons of CO2 per year, and the president aims to reduce this by 30%, calculate the following:
    
    1. How much fossil fuel consumption will be reduced by the end of the year?
    2. How much CO2 will be saved in the year when the reduction is complete?
    3. If
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in the world. It is located in the south of France. It is not far from the Mediterranean Sea. It is famous for its museums and shopping centers. If you look at the map, you can see how far away it is. The capital of Egypt is Cairo. It is in the south of the country. It is very different from Paris. It is a very big city with many people. It is the capital of Egypt. In the 19th century, it was the capital of the old Egypt. Then, it became the capital of the new Egypt. Today, the capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  about connecting people with the best AI technologies. In order to ensure that the next generation of AI is prepared for the future, we need to work with institutions and students to ensure that people are well-equipped to harness the power of AI. In order to achieve this, we need to create new programs that focus on developing and teaching the necessary skills and knowledge. To do this, we must work with industry partners to ensure that our programs are relevant to industry standards and that they align with the needs of the future workforce. Through this approach, we can ensure that the next generation of AI is equipped to take advantage of the power of AI and to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is a vibrant and dynamic city with a diverse population, making it a popular destination for both locals and tourists alike. The city is also home to many international organizations and institutions, including UNESCO and the French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to the human brain's limitations. This could lead to more sophisticated forms of AI, such as "superintelligent" machines that can perform tasks that require human intelligence, such as creative problem-solving or decision-making.
    
    2. Greater reliance on data: AI will become more data-driven, with more data being collected and analyzed to improve its performance
    


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
    Generated text:  [insert name], and I’m a [insert occupation or profession]. I’m [insert age and gender]. I’ve always been [insert passion or hobby]. I’m here to share my experiences and insights with anyone who wants to learn from me. I’m excited to help you understand the world around us and make the most of our unique talents. What’s your name? What’s your occupation or profession? And what’s your passion or hobby? I look forward to hearing from you! Let's connect! 📸✨✨📚✨✨ #MeetTheHero #SelfIntroduction #YourFriend 🌟✨ #LifeIn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    That being said, could you please elaborate on the cultural significance of Paris and its various neighborhoods? Of course! Paris is a cultural and historical city with a rich history dating back to the Middle Ages. The city is home to many notable landmarks, including the Eiffel Tower, the Notre Dame Cathedral, and the Louvre Museum. The city is also known for its vibrant food scene, with many restaurants and cafes offering delicious French cuisine. The city is also home to a diverse population, with many different languages spoken and people of various ethnicities. 
    
    Paris is a city that has a complex and dynamic cultural scene, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid advancements in both hardware and software capabilities, as well as the continued integration of AI into increasingly diverse and complex systems. Here are some possible future trends in AI:
    
    1. Increased Integration with IoT: AI is already being integrated into IoT devices and systems, such as smart homes and connected vehicles. As the IoT evolves, we can expect more integration of AI with physical objects, wearable devices, and other systems.
    
    2. Autonomous Vehicles: Autonomous vehicles (AVs) are likely to become more common in the future. AI will play a key role in developing AVs by improving their safety, efficiency, and accuracy in


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

     Emily

    .

     I

     am

     a

     book

     lover

    .

     I

     enjoy

     reading

    ,

     writing

    ,

     and

     exploring

     new

     book

    stores

    .

     I

     love

     to

     discover

     new

     authors

    ,

     genres

    ,

     and

     author

    ship

     styles

    .

     I

     love

     to

     research

     books

    ,

     and

     I

     love

     to

     critique

     them

    .

     I

    'm

     interested

     in

     writing

    ,

     but

     I

     also

     enjoy

     reading

     novels

     and

     short

     stories

    ,

     and

     I

     enjoy

     discussing

     the

     themes

     and

     writing

     techniques

     used

     in

     fiction

    .

     I

     am

     a

     self

    -pro

    claimed

     book

    worm

    ,

     and

     I

     love

     to

     spend

     my

     free

     time

     reading

     and

     writing

    .


    Emily

     is

     a

     book

     lover

     who

     is

     passionate

     about

     discovering

     new

     authors

    ,

     genres

    ,

     and

     writing

     styles

    .

     She

     enjoys

     researching

     books

     and

     crit

    iqu

    ing

     them

    ,

     and

     she

    
    
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

     the

     most

     populous

     city

     in

     Europe

     and

     is

     home

     to

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

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     history

    ,

     art

    ,

     cuisine

    ,

     and

     music

    .

     Paris

     is

     a

     cosm

    opolitan

     and

     diverse

     city

     that

     attracts

     tourists

     from

     all

     over

     the

     world

    .

     It

     is

     a

     major

     cultural

     and

     economic

     center

     in

     Europe

     and

     plays

     a

     significant

     role

     in

     the

     French

     identity

    .

     The

     city

     is

     also

     home

     to

     numerous

     museums

    ,

     parks

    ,

     and

     landmarks

    ,

     including

     the

     Notre

    -D

    ame

     Cathedral

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

     has

     a

     long

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     constantly

     evolving

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     that

     AI

     is

     likely

     to

     experience

     in

     the

     coming

     years

    :
    


    1

    .

     Increasing

    ly

     autonomous

     robots

    :

     Autonomous

     robots

     will

     become

     more

     common

     and

     widely

     used

     in

     various

     sectors

    ,

     such

     as

     manufacturing

    ,

     healthcare

    ,

     and

     transportation

    .

     They

     will

     be

     able

     to

     make

     decisions

     and

     take

     actions

     without

     human

     intervention

    ,

     reducing

     the

     risk

     of

     errors

     and

     improving

     safety

    .
    


    2

    .

     More

     personalized

     AI

    :

     AI

     will

     become

     more

     personalized

     and

     adaptable

     to

     individual

     users

    ,

     offering

     tailored

     recommendations

     and

     solutions

     to

     enhance

     their

     lives

    .

     This

     will

     be

     especially

     important

     in

     the

     realm

     of

     personal

    ization

     and

     customer

     service

    .
    


    3

    .

     Greater

     integration

     of

     AI

     into

     everyday

     life

    :

    



```python
llm.shutdown()
```
