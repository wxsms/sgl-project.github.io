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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.36it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.14it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.14it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.30it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.01it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 21.94it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 30.69it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.59it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 16.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.15it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.57it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 34.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.92it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.17it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 46.14it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 46.57it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.12it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 38.93it/s]


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
    Generated text:  Stefan, and I'm a beginner in object-oriented programming, and I'm trying to learn C#.
    I'm working with an application that has a menu, and I want to make a new menu item for an "Option 2".
    I want to create a new menu item with a label and a command button. I have written the following code:
    
    ```
    [ ]
    
    public void Menu2_Click(object sender, EventArgs e)
    {
       //Command button 2
       Button cmdButton2 = new Button();
       cmdButton2.Text = "Option 2";
       cmdButton2.CommandText = "Option 2";
       cmd
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many troops to keep in Iraq. He has 3000 soldiers and Iraq has 10000 soldiers. He has 1000 more soldiers in Iraq than those in the other countries. How many troops does he have in total? Let's calculate the number of soldiers in Iraq and then add the number of soldiers in other countries to get the total number of troops.
    
    Iraq has 1000 more soldiers than the other countries. Since there are 3 countries, the other countries have 1000 soldiers each.
    
    So, in total, the other countries have:
    1
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, located in the western part of the country. It is the seat of the French government, the capital of the French Republic, and the most populous city in the world with more than 2 million inhabitants. The city has an extensive network of canals, including the Seine, which flows through the city center, and is a symbol of Paris. The Seine is one of the largest rivers in the world, with a total length of around 48.5 km. The city has also a canal network, including the Orne Canal, the Seine River Canal, and the Vézelay Canal.
    Paris was
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, with machines becoming more and more intelligent. It’s important to think about how to make the most of these potential benefits, and avoid the risks. One way to do this is to develop a plan that balances the benefits and risks of AI, and to be transparent about how the technology is being used.
    One potential benefit of AI is that it can be used to improve healthcare. For example, machine learning algorithms can be used to analyze large amounts of medical data and identify patterns that could help doctors make more accurate diagnoses and treatments. This can lead to better patient outcomes and reduced costs.
    Another benefit of AI is that it can be used


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [job title] at [company name], and I enjoy [job title] work. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the Middle Ages. It is a major transportation hub and a major tourist destination, attracting millions of visitors each year. The city is known for its cuisine, fashion, and art, and is home to many famous museums and galleries. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes, reduce costs, and improve quality. As AI technology continues to improve, we can expect to see even more widespread adoption in manufacturing.
    
    3. AI in finance: AI is already
    


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
    Generated text:  [insert character's name] and I am a [insert fictional character's role or occupation]. I have a [insert fictional character's personal characteristic or hobby] and I have always [insert fictional character's personal trait or personality]. I'm [insert fictional character's age] years old, and I'm [insert fictional character's height, weight, or other physical characteristic]. I live in [insert fictional character's hometown or current residence]. I enjoy [insert fictional character's favorite hobby or activity], and I often [insert fictional character's occupation or hobby] with [insert fictional character's partner, family, or friends]. I'm [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, an iconic city renowned for its historical significance and vibrant culture.
    
    Fact: The capital city of France is Paris, an iconic city renowned for its historical significance and vibrant culture. It is the city where the French Revolution took place and is still a major center of French politics, culture, and art. The city is also home to many notable museums, landmarks, and attractions, including the Louvre Museum, the Palace of Versailles, and the Eiffel Tower. Paris is known for its opulent architecture, chic fashion, and diverse food scene, and remains a cultural hub of the world. The city's rich history and vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be as dynamic and unpredictable as the human race itself. Some potential trends that have already emerged in the tech industry are:
    
    1. Integration of AI in healthcare: AI is already being used in healthcare to diagnose diseases, predict patient outcomes, and personalize treatment plans. It is expected to further integrate with new technologies like robotics, artificial intelligence, and big data analytics in the coming years to improve patient care and outcomes.
    
    2. Development of autonomous vehicles: As more cities around the world embrace the use of autonomous vehicles, their integration into everyday life is expected to become a reality. This could have a significant impact on transportation, reducing traffic congestion


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

    Job

     Title

    ]

     with

     [

    Company

     Name

    ].

     I

    've

     been

     working

     here

     for

     [

    Number

    ]

     years

    ,

     and

     I

     love

     [

    Company

     Name

    ]

     because

     [

    Why

     You

     Love

     the

     Company

    ].

     I

     believe

     in

     the

     [

    Value

     Proposition

    ],

     and

     I

    'm

     passionate

     about

     [

    Your

     Passion

    ].

     Let

     me

     know

     if

     you

    'd

     like

     to

     learn

     more

     about

     me

     or

     if

     you

    'd

     like

     me

     to

     share

     some

     of

     my

     experiences

     with

     you

    .

     [

    Name

    ]

     is

     passionate

     about

     the

     value

     of

     collaboration

     and

     has

     a

     knack

     for

     bringing

     people

     together

    .

     Whether

     it

    's

     a

     team

    -building

     activity

     or

     a

     brainstorm

    ing

     session

    ,

     I

    'm

     always

     ready

     to

     learn

     and

     grow

     as

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     cultural

     and

     economic

     center

    ,

     renowned

     for

     its

     rich

     history

     and

     art

    ,

     and

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     a

     UNESCO

     World

     Heritage

     Site

     and

     is

     a

     popular

     destination

     for

     tourists

     and

     residents

     alike

    .

     The

     city

     has

     a

     diverse

     population

    ,

     with

     

    6

     million

     people

     calling

     it

     home

    ,

     and

     is

     a

     major

     transportation

     hub

     for

     Europe

    .

     It

     is

     an

     important

     political

     and

     cultural

     center

    ,

     with

     the

     French

     Parliament

     and

     the

     European

     Union

     headquarters

     located

     there

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     bright

    ,

     with

     numerous

     opportunities

     and

     challenges

     to

     pursue

    .

     Here

     are

     some

     potential

     trends

     to

     watch

     for

    :
    


    1

    .

     AI

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     diagnose

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     develop

     personalized

     treatment

     plans

    .

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     sophisticated

     and

     accurate

     diagnostic

     tools

    ,

     as

     well

     as

     new

     ways

     to

     use

     AI

     to

     help

     doctors

     make

     better

     decisions

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     already

     being

     used

     to

     analyze

     financial

     data

     and

     make

     predictions

    ,

     but

     as

     the

     field

     of

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     more

     sophisticated

     and

     accurate

     financial

     models

    .

     AI

     could

     also

     be

     used

     to

     detect

    



```python
llm.shutdown()
```
