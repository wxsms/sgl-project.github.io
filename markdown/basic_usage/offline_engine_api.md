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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-04-12 09:27:22,013 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 09:27:22] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.72it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.10it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.10it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.19it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.86it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.84it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.19it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.40 GB):   2%|▏         | 1/58 [00:00<00:06,  8.75it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.37 GB):   2%|▏         | 1/58 [00:00<00:06,  8.75it/s]Capturing num tokens (num_tokens=7168 avail_mem=136.87 GB):   2%|▏         | 1/58 [00:00<00:06,  8.75it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=136.87 GB):   5%|▌         | 3/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=136.73 GB):   5%|▌         | 3/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=136.71 GB):   5%|▌         | 3/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.71 GB):   5%|▌         | 3/58 [00:00<00:03, 14.02it/s]Capturing num tokens (num_tokens=5632 avail_mem=136.71 GB):  10%|█         | 6/58 [00:00<00:02, 18.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=136.71 GB):  10%|█         | 6/58 [00:00<00:02, 18.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=136.70 GB):  10%|█         | 6/58 [00:00<00:02, 18.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=136.70 GB):  10%|█         | 6/58 [00:00<00:02, 18.64it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=136.70 GB):  10%|█         | 6/58 [00:00<00:02, 18.64it/s]Capturing num tokens (num_tokens=3840 avail_mem=136.70 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=136.70 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=136.69 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=136.69 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=136.69 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.68 GB):  17%|█▋        | 10/58 [00:00<00:01, 24.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=136.68 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=136.68 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=136.68 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  26%|██▌       | 15/58 [00:00<00:01, 30.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=136.67 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=960 avail_mem=136.66 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s] Capturing num tokens (num_tokens=896 avail_mem=136.66 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=832 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  34%|███▍      | 20/58 [00:00<00:01, 35.19it/s]Capturing num tokens (num_tokens=768 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=704 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=640 avail_mem=136.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]

    Capturing num tokens (num_tokens=576 avail_mem=136.64 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=512 avail_mem=136.63 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  43%|████▎     | 25/58 [00:00<00:00, 38.28it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=416 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.37it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]

    Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  60%|██████    | 35/58 [00:01<00:00, 41.67it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  69%|██████▉   | 40/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]

    Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s] Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.46it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.60it/s]

    Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.21it/s] Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.21it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 37.43it/s]


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
    Generated text:  Mckenna and I'm a software developer and a chef. My background is in the healthcare industry and I have 20 years of experience. I have been working in the healthcare industry for over 13 years now and I have worked in various fields such as medical lab technicians, hospital managers, hospital IT professionals and more.
    I enjoy being able to bring my software development and healthcare experience to help people. Currently, I work as a Medical Director in a clinic in New York. I’m also a mom of two adorable girls. I have a passion for helping people, that’s why I try to make things fun for everyone.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He or she is the leader of the country. They have many important jobs. One of the most important jobs that they have is to be a teacher. They are also the main teachers of our country's children. When the president of the United States is president, he or she has a very important job. They make sure all the people get good education. This is very important because they can help the people to become better people. Some people think that the president of the United States is not very important. But they are wrong. The president of the United States is very important. He or she should be very
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The largest city in the world is Tokyo. Japan, the country that has the largest population in the world, has a population of approximately 129 million. Find the number of digits in the population of Japan. First, let's understand what the question is asking. The population of Japan is given as approximately 129 million, which can be written in scientific notation as \(1.29 \times 10^8\). This means the population is 1,290,000,000.
    
    To find the number of digits in the population of Japan, we need to
    ===============================
    Prompt: The future of AI is
    Generated text:  more complicated than it seems. The regulatory landscape, ethical concerns, and the actual technology itself all have a lot to do with it.
    In this chapter, you will explore the role of AI in various applications, focusing on the various sides of the equation. You will examine its implementation in the field of healthcare, the applications of AI in the entertainment industry, the use of AI in the legal industry, and how it is impacting the field of technology.
    In the first chapter of this book, you will explore the role of AI in the field of healthcare. Here, you will learn about the different ways AI can be used in healthcare, the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature, art, and music. Paris is a cultural and economic hub, with a diverse population and a rich history dating back to the Roman Empire. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination, with millions of visitors annually. The city is known for its cuisine, fashion, and art, and is a major center for science and technology. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into our daily lives, from home automation systems to self-driving cars. This will lead to increased automation in various industries, including manufacturing, healthcare, and transportation.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be a need for increased privacy and security measures. This will require the development of new technologies and protocols to protect sensitive data and prevent cyber attacks.
    
    3. AI ethics and governance
    


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
    Generated text:  [Your Name], a [Job Title] with over [X] years of experience. I'm passionate about [Your Passion], excelling at [Your Expertise], and I'm ready to take on any challenge thrown my way. What kind of experiences, education, or skills do you possess that make you the best fit for this role? And how would you be expected to contribute to the team? Let me know if you have any questions or concerns. Let's make it a successful partnership! [Your Name] [Your Contact Information] [Your LinkedIn Profile URL] [Your GitHub Profile URL] [Your Twitter Profile URL] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the most visited city in the world, known for its historical landmarks, vibrant culture, and cuisine. The city is also home to numerous international institutions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination with over 10 million visitors annually, making it a cultural and economic hub of France. It is also known for its gastronomy, with many restaurants and food stalls offering a wide range of culinary experiences. Despite its size, Paris remains a hub of innovation and creativity, with numerous museums, galleries, and cultural institutions.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and will continue to bring about significant changes in various sectors such as healthcare, education, transportation, and financial services. Here are some possible future trends in AI:
    
    1. Advancements in computer technology: As technology continues to advance, we can expect to see more powerful and flexible AI systems. For instance, machine learning models will become more sophisticated, enabling them to handle more complex tasks and problem-solving.
    
    2. Integration with human emotions: AI will be able to learn and adapt to human emotions and behaviors, leading to more empathetic and human-like interactions. This will enable more AI-powered systems to understand and respond to human emotions and


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

    'm

     a

     [

    职业

    /

    特长

    ]

     who

     was

     born

     in

     [

    Birth

    date

    ].

     I

    'm

     currently

     [

    age

    ]

     years

     old

    .

     I

    'm

     a

     [

    gender

    ]

     and

     I

     have

     a

     [

    physical

     description

    ]

     personality

    .

     I

     enjoy

     [

    rel

    atable

     hobby

    ,

     interest

    ,

     or

     passion

    ].

     I

    'm

     also

     [

    something

     unique

     about

     me

    ,

     like

     a

     tattoo

    ,

     special

     skill

    ,

     or

     unique

     gift

    ].

     In

     my

     free

     time

    ,

     I

     enjoy

     [

    personal

     hobby

     or

     activity

    ].

     I

    'm

     [

    gender

    ].

     [

    Age

    ]

     [

    Gender

    ]

     [

    Physical

     Description

    ]

     [

    Person

    ality

     Trait

    ]

     [

    Unique

     Character

     Feature

    ]


    [

    Name

    ],

     a

     [

    gender

    ]

     [

    gender

    ],

     [

    age

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     "

    City

     of

     Love

    ."

     
    


    Note

    :

     This

     statement

     is

     fact

    ually

     correct

     and

     can

     be

     included

     in

     various

     contexts

     depending

     on

     the

     context

     in

     which

     it

     is

     used

    .

     For

     example

    ,

     it

     could

     be

     used

     in

     a

     history

     lesson

    ,

     a

     political

     statement

    ,

     or

     a

     travel

     guide

    .

     It

     is

     a

     widely

     recognized

     and

     well

    -known

     city

     in

     France

    .

     
    


    Option

     

    1

    :

     Paris

     is

     the

     capital

     of

     France

     and

     also

     known

     as

     "

    The

     City

     of

     Love

    ."


    Option

     

    2

    :

     Paris

     is

     the

     capital

     of

     France

     and

     also

     known

     as

     "

    The

     City

     of

     Words

    ."

     
    


    Ass

    essment

    :

     The

     statement

     is

     fact

    ually

     correct

     and

     can

     be

     used

     in

     various

     contexts

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     continued

     rapid

     development

    ,

     innovation

    ,

     and

     adoption

    .

     Some

     possible

     trends

     that

     could

     be

     expected

     in

     the

     coming

     years

     include

    :
    


    1

    .

     More

     advanced

     AI

    :

     AI

     systems

     are

     likely

     to

     become

     more

     capable

     and

     efficient

    ,

     with

     the

     ability

     to

     learn

     and

     adapt

     to

     new

     data

     sets

     and

     scenarios

    .

     This

     could

     lead

     to

     more

     effective

     applications

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     increasing

     importance

     of

     AI

     in

     healthcare

    ,

     there

     is

     likely

     to

     be

     a

     continued

     focus

     on

     developing

     more

     effective

     and

     efficient

     AI

     systems

     for

     diagn

    osing

     and

     treating

     diseases

    .

     This

     could

     lead

     to

     advancements

     in

     areas

     such

     as

     cancer

     diagnostics

    ,

     gene

     sequencing

    ,

     and

    



```python
llm.shutdown()
```
