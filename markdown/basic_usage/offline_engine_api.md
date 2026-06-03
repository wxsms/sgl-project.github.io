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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.41s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.81it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:04<00:07,  6.09it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=416):  40%|███▉      | 23/58 [00:04<00:02, 12.41it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 19.92it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:05<00:01, 19.92it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:05<00:01, 19.92it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 29.52it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 39.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.26 GB):   2%|▏         | 1/58 [00:00<00:06,  9.22it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.86 GB):   2%|▏         | 1/58 [00:00<00:06,  9.22it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.88 GB):   2%|▏         | 1/58 [00:00<00:06,  9.22it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=70.88 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.21 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.90 GB):   5%|▌         | 3/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.90 GB):   9%|▊         | 5/58 [00:00<00:04, 10.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.93 GB):   9%|▊         | 5/58 [00:00<00:04, 10.96it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=71.19 GB):   9%|▊         | 5/58 [00:00<00:04, 10.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.19 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.19 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.19 GB):  12%|█▏        | 7/58 [00:00<00:04, 12.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.19 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.18 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=70.97 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.16 GB):  16%|█▌        | 9/58 [00:00<00:03, 13.83it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.16 GB):  21%|██        | 12/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.16 GB):  21%|██        | 12/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.15 GB):  21%|██        | 12/58 [00:00<00:02, 16.73it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.15 GB):  21%|██        | 12/58 [00:00<00:02, 16.73it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=71.15 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.99 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.12 GB):  26%|██▌       | 15/58 [00:01<00:02, 19.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.12 GB):  26%|██▌       | 15/58 [00:01<00:02, 19.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.12 GB):  31%|███       | 18/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.11 GB):  31%|███       | 18/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.11 GB):  31%|███       | 18/58 [00:01<00:01, 21.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.09 GB):  31%|███       | 18/58 [00:01<00:01, 21.12it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.09 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.45it/s]Capturing num tokens (num_tokens=960 avail_mem=71.07 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.45it/s] Capturing num tokens (num_tokens=896 avail_mem=71.07 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.45it/s]Capturing num tokens (num_tokens=832 avail_mem=71.08 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.45it/s]Capturing num tokens (num_tokens=768 avail_mem=71.07 GB):  36%|███▌      | 21/58 [00:01<00:01, 23.45it/s]Capturing num tokens (num_tokens=768 avail_mem=71.07 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.33it/s]Capturing num tokens (num_tokens=704 avail_mem=71.08 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.33it/s]Capturing num tokens (num_tokens=640 avail_mem=71.07 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.33it/s]Capturing num tokens (num_tokens=576 avail_mem=71.06 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.33it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.05 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.33it/s]Capturing num tokens (num_tokens=512 avail_mem=71.05 GB):  50%|█████     | 29/58 [00:01<00:00, 29.15it/s]Capturing num tokens (num_tokens=480 avail_mem=71.06 GB):  50%|█████     | 29/58 [00:01<00:00, 29.15it/s]Capturing num tokens (num_tokens=448 avail_mem=71.05 GB):  50%|█████     | 29/58 [00:01<00:00, 29.15it/s]Capturing num tokens (num_tokens=416 avail_mem=71.05 GB):  50%|█████     | 29/58 [00:01<00:00, 29.15it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  50%|█████     | 29/58 [00:01<00:00, 29.15it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=352 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=320 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.84it/s]

    Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=256 avail_mem=71.02 GB):  57%|█████▋    | 33/58 [00:01<00:00, 31.84it/s]Capturing num tokens (num_tokens=256 avail_mem=71.02 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.77it/s]Capturing num tokens (num_tokens=240 avail_mem=71.01 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.77it/s]Capturing num tokens (num_tokens=224 avail_mem=70.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.77it/s]

    Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 26.77it/s]Capturing num tokens (num_tokens=208 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.27it/s]Capturing num tokens (num_tokens=192 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.27it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.27it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 25.27it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.53it/s]Capturing num tokens (num_tokens=144 avail_mem=70.96 GB):  74%|███████▍  | 43/58 [00:01<00:00, 24.53it/s]

    Capturing num tokens (num_tokens=128 avail_mem=70.97 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.53it/s]Capturing num tokens (num_tokens=112 avail_mem=70.97 GB):  74%|███████▍  | 43/58 [00:02<00:00, 24.53it/s]Capturing num tokens (num_tokens=112 avail_mem=70.97 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=96 avail_mem=70.94 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.35it/s] Capturing num tokens (num_tokens=80 avail_mem=70.93 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=64 avail_mem=70.92 GB):  79%|███████▉  | 46/58 [00:02<00:00, 23.35it/s]Capturing num tokens (num_tokens=64 avail_mem=70.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=48 avail_mem=70.94 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.67it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.93 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=28 avail_mem=70.92 GB):  84%|████████▍ | 49/58 [00:02<00:00, 24.67it/s]Capturing num tokens (num_tokens=28 avail_mem=70.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=24 avail_mem=70.92 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=20 avail_mem=70.91 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=16 avail_mem=70.89 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=12 avail_mem=70.90 GB):  90%|████████▉ | 52/58 [00:02<00:00, 25.94it/s]Capturing num tokens (num_tokens=12 avail_mem=70.90 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=8 avail_mem=70.90 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.90it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=70.89 GB):  97%|█████████▋| 56/58 [00:02<00:00, 28.90it/s]Capturing num tokens (num_tokens=4 avail_mem=70.89 GB): 100%|██████████| 58/58 [00:02<00:00, 23.24it/s]


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
    Generated text:  Laura. I'm a little over thirty years old and I like to travel. I went to the United States for my first trip when I was 19, I spent 5 months in America and I took over 400 pictures. I don't know if I will ever get to travel again. I'm very scared of the darkness. I was afraid of the rain and the cold. I was very anxious when I went to America, I wasn't sure what to expect. I'm very nervous and I don't know if I'll ever feel comfortable enough to travel again. What's the purpose of the text?
    A
    ===============================
    Prompt: The president of the United States is
    Generated text:  very important. The president of the United States is the leader of the country. The president is also the head of government. The president is the leader of the United States. The president also helps the country. The president has many important jobs. The president has to lead the country. The president has to make the laws. He has to set the budget for the government. He has to make sure the country is safe. The president has to deal with the people. The president has to help the country. The president has to make sure that he is respected by the country. The president has to make the country peaceful. The president is
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, but where does the name Paris come from?
    
    The name Paris comes from the ancient Greek word "Peri" which means "around".
    
    To elaborate:
    
    1. Ancient Greek origin: The word "Peri" was derived from the root "perios" which means "around" or "in all directions".
    
    2. Historical context: In ancient times, the name "Peri" was used to describe the shape of Paris, which is shaped like a horseshoe.
    
    3. Historical significance: The name "Peri" was very important in early French culture, representing the idea of being around or encompassing
    ===============================
    Prompt: The future of AI is
    Generated text:  not just about the tools themselves, but also about the human interaction with these tools. In this session, we'll dive into how AI and machine learning can help us understand and predict future events, and how we can use this knowledge to make informed decisions and improve our lives. Whether you're interested in finance, healthcare, transportation, or any other field, we'll explore how AI can transform the way we work, learn, and interact with the world around us. Join us for this fascinating discussion and let's see how AI is shaping the future of the human experience. #AI #FutureOfAI #FutureOfHumanity #AIand


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


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a vibrant culture that draws millions of visitors each year. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also home to many world-renowned museums, theaters, and restaurants, making it a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its reputation as a cultural and artistic capital is further enhanced by its numerous festivals,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more sophisticated, there will be a growing emphasis on ethical considerations and responsible use of AI. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for harmful purposes.
    
    2. Greater integration with human decision-making: AI systems
    


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
    Generated text:  [Name]. I'm a young [age] who loves [interest or hobby] and [career]. What brings you to this world? I'm here because I want to make a difference and help those around me. What's the next step for me? I'm excited to see what opportunities await me, and I'm eager to learn more about your career. What do you think makes me stand out to you? I believe you can count on me to be your trusted friend, your mentor, and your partner. Thank you for taking the time to meet me, and I hope to see you soon! [Name] [Career or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a sprawling metropolis located on the banks of the Seine River. It is the seat of the French government and is home to many of the city’s iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for French culture and cuisine, with its many museums, theaters, and cafes. It is often referred to as the "city of lights" and is one of the world's most popular tourist destinations. (Source: Wikipedia) The Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral are iconic landmarks of Paris. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with many exciting developments on the horizon. Here are some potential trends that could shape the future of artificial intelligence:
    
    1. Improved accuracy and efficiency: One of the most exciting developments is the potential for AI to perform tasks more accurately and efficiently than humans. For example, AI could be used to analyze large amounts of data more quickly and accurately than humans, leading to breakthroughs in fields such as medicine, finance, and marketing.
    
    2. Personalization and ai: As AI becomes more integrated into our daily lives, it's likely to provide even more personalization to users. This could lead to more personalized products, services, and experiences


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

    insert

     your

     name

    ],

     and

     I

    'm

     a

     [

    insert

     profession

     or

     career

    ]

     who

     has

     always

     been

     fascinated

     by

     the

     world

     of

     [

    insert

     something

     like

     "

    history

    ",

     "

    science

    ",

     "

    pol

    itics

    ",

     "

    art

    ",

     etc

    .]

     and

     strive

     to

     understand

     it

    .

     I

     love

     to

     think

     outside

     the

     box

     and

     come

     up

     with

     innovative

     ideas

     and

     solutions

     to

     complex

     problems

    .

     I

     am

     always

     seeking

     out

     new

     experiences

     and

     learn

    ings

    ,

     and

     I

     find

     my

     passion

     for

     learning

     to

     be

     my

     greatest

     strength

    .

     I

     am

     looking

     forward

     to

     the

     chance

     to

     meet

     you

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     largest

     city

     in

     France

     and

     serves

     as

     the

     capital

     of

     the

     country

    .

     It

    's

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     romantic

     can

    als

    ,

     and

     rich

     history

    .

     Other

     notable

     landmarks

     include

     the

     Lou

    vre

     Museum

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     is

     a

     bustling

     met

    ropolis

     with

     a

     diverse

     population

     of

     around

     

    1

    0

     million

     people

    .

     It

    's

     a

     hub

     for

     business

    ,

     culture

    ,

     and

     art

    ,

     and

     has

     played

     a

     key

     role

     in

     France

    's

     history

     and

     politics

    .

     The

     city

     also

     boasts

     a

     unique

     cuisine

    ,

     particularly

     the

     famous

     Paris

    ian

     cheese

    .

     If

     you

     visit

     Paris

    ,

     be

     sure

     to

     explore

     its

     stunning

     architecture

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     potential

     and

     possibilities

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     AI

     development

     in

     the

     years

     to

     come

    :
    


    1

    .

     Increased

     automation

    :

     One

     of

     the

     most

     significant

     trends

     in

     AI

     is

     the

     increased

     automation

     of

     tasks

    .

     AI

     systems

     will

     become

     more

     capable

     of

     performing

     tasks

     that

     are

     currently

     carried

     out

     by

     humans

    ,

     such

     as

     data

     analysis

    ,

     language

     processing

    ,

     and

     image

     recognition

    .
    


    2

    .

     AI

     ethics

    :

     As

     AI

     systems

     become

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

     increasing

     concerns

     about

     the

     ethics

     of

     AI

     development

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     ethical

     frameworks

     and

     guidelines

     for

     AI

     development

    .
    


    3

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     systems

     will

     become

     more

    



```python
llm.shutdown()
```
