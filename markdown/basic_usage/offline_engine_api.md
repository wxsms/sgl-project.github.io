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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [transformers] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-27 07:16:24] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.10it/s]


    2026-04-27 07:16:30,331 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 07:16:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.43it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.96it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 20.70it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.54it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.78it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.17it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.61it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.81it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.09it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  50%|█████     | 29/58 [00:00<00:00, 44.38it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:00<00:00, 46.59it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  60%|██████    | 35/58 [00:01<00:00, 46.59it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  71%|███████   | 41/58 [00:01<00:00, 48.09it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.74it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.12it/s] Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 49.16it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 42.35it/s]


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
    Generated text:  Ayesha and I am from England, my mother tongue is Hindi. I am in my late teens and I have a brother who is the youngest of my siblings. I don’t have any friends, my parents never allow me to go out, and I don’t have any friends in my town. I don’t speak in English and am of Indian heritage. How should I introduce myself to my family?
    
    And my question is why do you think there is a need for such a website to help people like you? Shouldn’t there be some kind of online community, even if it’s limited to your town, for someone like you
    ===============================
    Prompt: The president of the United States is
    Generated text:  now planning a trip to a popular tourist attraction, which is a city with a population of 500,000 people. The president's car, which has a fuel efficiency of 30 miles per gallon, will need to refuel the car to refuel it with a gallon of fuel. If the president's car is driven for 600 miles in total and the fuel cost is $3 per gallon, what is the total cost for the president's car to travel 600 miles on this trip? To determine the total cost for the president's car to travel 600 miles, we
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. A. Paris B. Lyon C. Lille D. Nice
    A. Paris
    B. Lyon
    C. Lille
    D. Nice
    Answer:
    
    A
    
    In the early stages of urbanization, for regions where the proportion of rural residents living in urban areas is very low, the economic development of the city tends to ____. 
    A. remain unchanged
    B. speed up
    C. slow down
    D. go to zero
    Answer:
    
    C
    
    The effective height of a distribution cabinet should not be less than ____.
    A. 0.5m
    B. 1.0m
    C
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain. In this article, we will discuss some of the current trends and forecasts for AI. In the future, AI will continue to evolve and improve, and it will play an increasingly important role in many areas of society. AI is constantly evolving, and it will continue to innovate and advance in new and exciting ways. Some of the current trends and forecasts for AI include:
    
    1. Increasing reliance on AI for decision-making: As AI technology becomes more advanced and capable, it will become more prevalent in many decision-making processes. For example, self-driving cars, robots, and other AI-powered devices are becoming increasingly common in our everyday lives


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic statement about yourself]. I'm looking forward to meeting you and discussing our potential collaboration. What's your name? I'm [insert a short, positive, enthusiastic statement about yourself]. I'm looking forward to meeting you and discussing our potential collaboration. What's your name? I'm [insert a short, positive, enthusiastic statement about yourself]. I'm looking forward to meeting you and discussing our potential collaboration.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that serves as the political, cultural, and economic center of the country. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its rich history and diverse cultural scene. Paris is also a major transportation hub, with its famous train system and numerous airports. The city is known for its fashion industry, art scene, and food culture, and is a popular tourist destination. Overall, Paris is a vibrant and dynamic city that plays a significant role in French culture and society. 
    
    This statement provides a concise factual statement about France's capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn and adapt to new situations and tasks. This could lead to more sophisticated and adaptive AI systems that can perform a wider range of tasks.
    
    2. Enhanced machine learning capabilities: AI systems are likely to become more capable of learning from data and making more accurate predictions and decisions. This could lead to more efficient and effective use of resources, as well as more accurate predictions of future events.
    
    3. Improved privacy and security: As AI systems become more integrated with
    


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
    Generated text:  [Name] and I am a [job title] at [company name]. I have always been passionate about [reason for interest or job] and I am always looking to learn and grow. I enjoy working with [role] and [profession], and I am always eager to push boundaries and improve my abilities. I have a natural curiosity and love to question everything and I am always looking for opportunities to learn and grow. I am a team player and I am always eager to work with a diverse group of people. I am confident and determined to achieve my goals and I am committed to making a positive impact in my field. I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe by area and population, with a rich history and cultural importance dating back to ancient times. The city is known for its stunning architecture, vibrant nightlife, and lively food scene, making it a popular tourist destination. It is also home to the Eiffel Tower and the Louvre Museum, both of which are considered UNESCO World Heritage sites. Paris is a major hub of politics, culture, and business, and attracts millions of visitors annually. Its location on the River Seine and its proximity to the Mediterranean Sea make it an ideal destination for relaxation and adventure. The city has a diverse population of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve numerous trends and advancements in various areas. Here are some potential trends that could shape the future of AI:
    
    1. Increased Personalization: As AI continues to improve, it is expected to become more personal and tailored to individual users. Personalized recommendations, chatbots, and virtual assistants will become more prevalent in daily life.
    
    2. Enhanced Automation: AI is expected to be able to perform a wide range of tasks more efficiently and effectively. Automation in manufacturing, healthcare, and transportation could increase productivity and reduce costs.
    
    3. Autonomous Vehicles: The development of autonomous vehicles is expected to become more common in the future. AI will be


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

     character

    's

     name

    ],

     and

     I

     am

     a

     [

    insert

     character

    's

     age

    ,

     gender

    ,

     or

     job

     title

    ]

     with

     [

    insert

     character

    's

     profession

    ].

     I

     am

     passionate

     about

     [

    insert

     one

     or

     two

     things

     that

     make

     you

     curious

     about

     your

     character

    ],

     and

     I

     am

     always

     seeking

     knowledge

     and

     learning

     new

     things

    .

     I

     am

     also

     determined

     to

     improve

     my

     skills

     and

     capabilities

    ,

     and

     I

     am

     eager

     to

     learn

     from

     both

     successful

     and

     unsuccessful

     individuals

    .

     I

     am

     always

     up

     for

     a

     good

     laugh

     and

     love

     to

     share

     my

     opinions

     and

     insights

     with

     others

    .

     I

     am

     excited

     to

     meet

     you

     and

     hopefully

     get

     to

     know

     you

     better

    .

     How

     can

     I

     help

     you

     today

    ?

     Let

    's

     chat

    !

     In

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     major

     French

     city

     and

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     the

     country

    .

     The

     city

     is

     known

     for

     its

     medieval

     architecture

    ,

     vibrant

     nightlife

    ,

     and

     world

    -f

    amous

     landmarks

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

     romantic

     and

     passionate

     French

     culture

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     rapidly

     evolving

    .

     Here

     are

     some

     potential

     trends

     that

     are

     shaping

     the

     AI

     landscape

    :
    


    1

    .

     AI

     becomes

     more

     pervasive

    :

     As

     the

     use

     of

     AI

     in

     various

     industries

     grows

    ,

     we

     can

     expect

     to

     see

     AI

     becoming

     more

     pervasive

     in

     everyday

     life

    .

     This

     will

     likely

     lead

     to

     new

     applications

     and

     opportunities

     for

     AI

    .
    


    2

    .

     AI

     continues

     to

     improve

     in

     terms

     of

     accuracy

     and

     performance

    :

     With

     the

     continued

     advancements

     in

     AI

     technology

    ,

     we

     can

     expect

     to

     see

     improvements

     in

     accuracy

     and

     performance

    .

     This

     will

     likely

     lead

     to

     more

     effective

     and

     reliable

     AI

     systems

    .
    


    3

    .

     AI

     will

     continue

     to

     be

     integrated

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     we

     can

     expect

    



```python
llm.shutdown()
```
