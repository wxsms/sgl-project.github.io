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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.93it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.91it/s]


    2026-04-06 18:17:19,789 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 18:17:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:02<00:07,  6.15it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.15it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.60it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.28it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.38it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.37it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.56it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=119.74 GB):   2%|▏         | 1/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=7680 avail_mem=119.70 GB):   2%|▏         | 1/58 [00:00<00:07,  7.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=119.22 GB):   2%|▏         | 1/58 [00:00<00:07,  7.68it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=119.22 GB):   5%|▌         | 3/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.06 GB):   5%|▌         | 3/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=119.06 GB):   5%|▌         | 3/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.05 GB):   5%|▌         | 3/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=119.05 GB):  10%|█         | 6/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.05 GB):  10%|█         | 6/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.05 GB):  10%|█         | 6/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.05 GB):  10%|█         | 6/58 [00:00<00:02, 19.72it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=119.05 GB):  10%|█         | 6/58 [00:00<00:02, 19.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.05 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.04 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.04 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.03 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.03 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.03 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.03 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.03 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.02 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.02 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=119.02 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 32.05it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.01 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=1024 avail_mem=118.99 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=960 avail_mem=119.01 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s] Capturing num tokens (num_tokens=896 avail_mem=119.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=832 avail_mem=119.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=768 avail_mem=119.00 GB):  34%|███▍      | 20/58 [00:00<00:01, 36.45it/s]Capturing num tokens (num_tokens=768 avail_mem=119.00 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]Capturing num tokens (num_tokens=704 avail_mem=118.99 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]Capturing num tokens (num_tokens=640 avail_mem=118.99 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]Capturing num tokens (num_tokens=576 avail_mem=118.99 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]

    Capturing num tokens (num_tokens=512 avail_mem=118.98 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]Capturing num tokens (num_tokens=480 avail_mem=118.99 GB):  43%|████▎     | 25/58 [00:00<00:00, 39.60it/s]Capturing num tokens (num_tokens=480 avail_mem=118.99 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=448 avail_mem=118.99 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=416 avail_mem=118.99 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=384 avail_mem=118.99 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=352 avail_mem=118.98 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=320 avail_mem=118.98 GB):  52%|█████▏    | 30/58 [00:00<00:00, 41.66it/s]Capturing num tokens (num_tokens=320 avail_mem=118.98 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=288 avail_mem=118.97 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=256 avail_mem=118.97 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=240 avail_mem=118.97 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.97 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=208 avail_mem=118.96 GB):  60%|██████    | 35/58 [00:01<00:00, 42.96it/s]Capturing num tokens (num_tokens=208 avail_mem=118.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=192 avail_mem=118.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=176 avail_mem=118.96 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=160 avail_mem=118.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=144 avail_mem=118.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=128 avail_mem=118.95 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.12it/s]Capturing num tokens (num_tokens=128 avail_mem=118.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s]Capturing num tokens (num_tokens=112 avail_mem=118.95 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s]Capturing num tokens (num_tokens=96 avail_mem=118.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s] Capturing num tokens (num_tokens=80 avail_mem=118.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s]

    Capturing num tokens (num_tokens=64 avail_mem=118.94 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s]Capturing num tokens (num_tokens=48 avail_mem=118.93 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.81it/s]Capturing num tokens (num_tokens=48 avail_mem=118.93 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=32 avail_mem=118.93 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=28 avail_mem=118.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=24 avail_mem=118.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=20 avail_mem=118.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=16 avail_mem=118.92 GB):  86%|████████▌ | 50/58 [00:01<00:00, 44.94it/s]Capturing num tokens (num_tokens=16 avail_mem=118.92 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=12 avail_mem=118.91 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.47it/s]Capturing num tokens (num_tokens=8 avail_mem=118.91 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.47it/s] Capturing num tokens (num_tokens=4 avail_mem=118.91 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.47it/s]

    Capturing num tokens (num_tokens=4 avail_mem=118.91 GB): 100%|██████████| 58/58 [00:01<00:00, 38.45it/s]


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
    Generated text:  Alex and I am 23 years old. I come from a medium income family. I play sports, and I get 5 hours of sleep a night. I like to spend my free time surfing the web, reading books, and watching movies. I love to listen to music and I go to concerts or parties.
    What are some of your hobbies and activities?
    As an AI language model, I don't have personal experiences or emotions, so I don't have any hobbies or activities that I enjoy. However, I'm designed to assist users in providing helpful and informative responses to their inquiries to the best of my abilities. Is there
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is responsible for a lot of important decisions and important things to do in the country. As the president, he is supposed to lead the country and make sure that all the important people in the country work well together. He is supposed to make sure that the country is safe and that it is healthy for the people. The president is supposed to be a very important person in the country, and he is always very busy and always in the country. He is supposed to make sure that the country is protected from any kind of threats or danger. He is also supposed to ensure that the country is healthy
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Brussels
    C. Lille
    D. Toulouse
    Answer: A
    
    The National People's Congress is the highest organ of state power in China. The highest organ of state power in China is the National People's Congress. A. Correct B. Incorrect
    Answer: A
    
    Which of the following is not a common feature between the road test and the road test during the testing process? A. Both are performed under the same conditions B. Both involve both signal and power C. Both are performed under the same conditions D. Both involve both signal and power
    Answer: C
    
    Question 
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly evolving, and the field of AI is becoming increasingly relevant in various industries. AI algorithms are used for various applications, such as fraud detection, image recognition, natural language processing, and more. However, the challenges of developing and maintaining these algorithms are constantly increasing. To address these challenges, researchers are exploring new techniques and approaches to improve the efficiency and reliability of AI systems.
    
    One of the challenges that researchers are facing is the cost of implementing AI algorithms. This is because the development and deployment of AI systems can be time-consuming and expensive, which makes it difficult for businesses to justify the investment in these systems. Additionally, the cost of developing


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a few key points about yourself, such as your age, gender, occupation, or any other relevant information]. I'm looking forward to meeting you and discussing how I can contribute to your team. What do you think makes you unique? I think I'm unique because I'm a [insert a few key points about your personality, such as your interests, hobbies, or any other relevant information]. I'm looking forward to meeting you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Middle Ages and is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major cultural and economic center, with a diverse population and a thriving arts scene. The city is home to many famous museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination. Paris is known for its romantic and romantic atmosphere, and is a popular destination for tourists and locals alike. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there will be a greater emphasis on developing ethical AI that is designed to minimize harm and maximize benefits.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare.
    
    3. Increased use of AI in manufacturing: AI is already being used in
    


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
    Generated text:  [Name]. I'm a [type of person] and I live in [location]. I'm [age] years old. I have [occupation] experience and I enjoy [reason for interest].
    I'm [type of person] and I love [reason for interest]. I'm always ready to learn and grow. I'm passionate about [reason for passion]. I'm [type of person] and I'm [reason for interest]. I'm a [type of person] and I'm [reason for interest]. I'm [type of person] and I'm [reason for interest]. And so on... As an AI language model
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the second largest city in France and the second most populous city in the European Union after Brussels. It is the political, cultural, and economic center of France and an international city. It has a rich history dating back to ancient times and a modern culture of tolerance and freedom. Its architecture, art, music, and fashion are world-renowned and its cuisine is famous for its variety and quality. The city is also known for its annual street fair, festivals, and fashion week. In addition, it is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and other iconic landmarks. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and diverse, driven by advances in technology, data, and human creativity. Some possible trends include:
    
    1. Increased integration with everyday life: AI will become more integrated into our daily lives, from driving cars to managing home appliances. This will likely result in more convenient and efficient ways of doing things, while also potentially increasing concerns about privacy and data security.
    
    2. Enhanced capabilities in machine learning: The field of machine learning is expected to continue to advance, with new algorithms and techniques emerging that can learn from data and improve their performance over time. This could lead to more sophisticated and personalized AI systems.
    
    3. Development


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

     __

    ________

    __.

     I

    'm

     __

    ________

    .

     I

    'm

     from

     the

     __

    ________

    __

     (

    city

    ,

     country

    ).

     I

    'm

     a

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    _

     (

    expert

    ,

     hobby

    ,

     or

     interest

    )

     that

     __

    ________

    _

     (

    why

     or

     why

     not

    ).

     I

    'm

     a

    /an

     __

    ________

    _

     (

    objective

    ,

     emotional

    ,

     or

     personal

    )

     that

     __

    ________

    _

     (

    why

     or

     why

     not

    ).

     I

    'm

     a

    /an

     __

    ________

    _

     (

    char

    ming

    ,

     strong

    ,

     humorous

    ,

     or

     self

    -aware

    )

     that

     __

    ________

    _

     (

    why

     or

     why

     not

    ).

     I

    'm

     a

    /an

     __

    ________

    _

     (

    simple

    ,

     complex

    ,

     or

     unconventional

    )

     that

     __

    ________

    _

     (

    why

     or

     why

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     hosts

     the

     iconic

     E

    iff

    el

     Tower

     and

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     love

    -themed

     festivals

     and

     celebrations

    .
    


    Paris

    's

     status

     as

     the

     capital

     of

     France

     is

     underscore

    d

     by

     its

     iconic

     E

    iff

    el

     Tower

    ,

     which

     stands

     as

     a

     symbol

     of

     the

     city

    's

     architectural

     and

     cultural

     heritage

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     love

    -themed

     festivals

     and

     celebrations

    ,

     such

     as

     the

     famous

     love

     parade

     and

     romantic

     night

    clubs

     like

     the

     Mar

    ais

     district

    .

     The

     city

    's

     love

     themes

     have

     become

     a

     part

     of

     its

     identity

     and

     continue

     to

     be

     celebrated

     today

    .

     Additionally

    ,

     Paris

    's

     location

     in

     the

     heart

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     dependent

     on

     the

     advancements

     in

     technology

    ,

     education

    ,

     and

     human

     understanding

     of

     AI

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     AI

     ethics

    :

     There

     is

     a

     growing

     concern

     about

     the

     ethical

     implications

     of

     AI

    ,

     including

     concerns

     about

     privacy

    ,

     bias

    ,

     and

     potential

     misuse

     of

     AI

    .

     Governments

     and

     organizations

     are

     working

     to

     develop

     AI

     policies

     that

     promote

     fairness

    ,

     transparency

    ,

     and

     accountability

    .
    


    2

    .

     Increased

     automation

    :

     While

     AI

     has

     already

     been

     making

     in

    roads

     into

     many

     industries

    ,

     there

     is

     a

     growing

     concern

     about

     the

     possibility

     of

     widespread

     automation

     leading

     to

     job

     loss

    .

     However

    ,

     as

     technology

     improves

    ,

     it

     is

     possible

     that

     AI

     will

     be

     able

     to

     augment

     human

     workers

    



```python
llm.shutdown()
```
