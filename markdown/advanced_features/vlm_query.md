# Query VLM with Offline Engine

This tutorial demonstrates how to use SGLang's **offline Engine API** to query VLMs. We will demonstrate usage with Qwen2.5-VL and Llama 4. This section demonstrates three different calling approaches:

1. **Basic Call**: Directly pass images and text.
2. **Processor Output**: Use HuggingFace processor for data preprocessing.
3. **Precomputed Embeddings**: Pre-calculate image features to improve inference efficiency.

## Understanding the Three Input Formats

SGLang supports three ways to pass visual data, each optimized for different scenarios:

### 1. **Raw Images** - Simplest approach
- Pass PIL Images, file paths, URLs, or base64 strings directly
- SGLang handles all preprocessing automatically
- Best for: Quick prototyping, simple applications

### 2. **Processor Output** - For custom preprocessing
- Pre-process images with HuggingFace processor
- Pass the complete processor output dict with `format: "processor_output"`
- Best for: Custom image transformations, integration with existing pipelines
- Requirement: Must use `input_ids` instead of text prompt

### 3. **Precomputed Embeddings** - For maximum performance
- Pre-calculate visual embeddings using the vision encoder
- Pass embeddings with `format: "precomputed_embedding"`
- Best for: Repeated queries on same images, caching, high-throughput serving
- Performance gain: Avoids redundant vision encoder computation (30-50% speedup)

**Key Rule**: Within a single request, use only one format for all images. Don't mix formats.

The examples below demonstrate all three approaches with both Qwen2.5-VL and Llama 4 models.

## Querying Qwen2.5-VL Model


```python
import nest_asyncio

nest_asyncio.apply()

import sglang.test.doc_patch  # noqa: F401

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
chat_template = "qwen2-vl"
example_image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
```


```python
from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Generated prompt text:")
print(conv.get_prompt())
print(f"\nImage size: {image.size}")
image
```

    Generated prompt text:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What's shown here: <|vision_start|><|image_pad|><|vision_end|>?<|im_end|>
    <|im_start|>assistant
    
    
    Image size: (570, 380)





    
![png](vlm_query_files/vlm_query_4_1.png)
    



### Basic Offline Engine API Call


```python
from sglang import Engine

llm = Engine(model_path=model_path, chat_template=chat_template, log_level="warning")
```

    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-06-10 04:44:14] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-06-10 04:44:19] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.12it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.21it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.19it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=47.00 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=47.00 GB):   5%|▌         | 1/20 [00:01<00:23,  1.24s/it]Capturing batches (bs=120 avail_mem=46.93 GB):   5%|▌         | 1/20 [00:01<00:23,  1.24s/it]Capturing batches (bs=112 avail_mem=46.93 GB):   5%|▌         | 1/20 [00:01<00:23,  1.24s/it]Capturing batches (bs=104 avail_mem=46.93 GB):   5%|▌         | 1/20 [00:01<00:23,  1.24s/it]Capturing batches (bs=104 avail_mem=46.93 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]Capturing batches (bs=96 avail_mem=46.93 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s] Capturing batches (bs=88 avail_mem=46.93 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]

    Capturing batches (bs=80 avail_mem=46.93 GB):  20%|██        | 4/20 [00:01<00:04,  3.68it/s]Capturing batches (bs=80 avail_mem=46.93 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.63it/s]Capturing batches (bs=72 avail_mem=46.92 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.63it/s]Capturing batches (bs=64 avail_mem=46.92 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.63it/s]Capturing batches (bs=56 avail_mem=46.92 GB):  35%|███▌      | 7/20 [00:01<00:01,  6.63it/s]Capturing batches (bs=56 avail_mem=46.92 GB):  50%|█████     | 10/20 [00:01<00:01,  9.50it/s]Capturing batches (bs=48 avail_mem=46.92 GB):  50%|█████     | 10/20 [00:01<00:01,  9.50it/s]

    Capturing batches (bs=40 avail_mem=46.92 GB):  50%|█████     | 10/20 [00:01<00:01,  9.50it/s]Capturing batches (bs=32 avail_mem=46.91 GB):  50%|█████     | 10/20 [00:01<00:01,  9.50it/s]Capturing batches (bs=32 avail_mem=46.91 GB):  65%|██████▌   | 13/20 [00:01<00:00, 12.17it/s]Capturing batches (bs=24 avail_mem=46.91 GB):  65%|██████▌   | 13/20 [00:01<00:00, 12.17it/s]Capturing batches (bs=16 avail_mem=46.91 GB):  65%|██████▌   | 13/20 [00:01<00:00, 12.17it/s]

    Capturing batches (bs=12 avail_mem=46.91 GB):  65%|██████▌   | 13/20 [00:01<00:00, 12.17it/s]Capturing batches (bs=12 avail_mem=46.91 GB):  80%|████████  | 16/20 [00:01<00:00, 13.56it/s]Capturing batches (bs=8 avail_mem=46.91 GB):  80%|████████  | 16/20 [00:01<00:00, 13.56it/s] Capturing batches (bs=4 avail_mem=46.91 GB):  80%|████████  | 16/20 [00:02<00:00, 13.56it/s]Capturing batches (bs=2 avail_mem=46.91 GB):  80%|████████  | 16/20 [00:02<00:00, 13.56it/s]Capturing batches (bs=2 avail_mem=46.91 GB):  95%|█████████▌| 19/20 [00:02<00:00, 15.83it/s]Capturing batches (bs=1 avail_mem=46.90 GB):  95%|█████████▌| 19/20 [00:02<00:00, 15.83it/s]

    Capturing batches (bs=1 avail_mem=46.90 GB): 100%|██████████| 20/20 [00:02<00:00,  9.40it/s]



```python
out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Model response:")
print(out["text"])
```

    Model response:
    This image shows a yellow taxi in New York City. The taxi has an additional item draped over its rear, resembling clothes or laundry. The setting appears to be from the "NYC Speakers' Bureau's Flatiron Blocking" event, commonly known as the "desking" event, where individuals choose to sit on the back of an upright cab instead of using proper seating.


### Call with Processor Output

Using a HuggingFace processor to preprocess text and images, and passing the `processor_output` directly into `Engine.generate`.


```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out["text"])
```

    Response using processor output:
    The image shows a person standing beside several yellow and orange taxis, which are typical of New York City. The person is hanging a checkered shirt on a metal pole, which seems to be a street vendor stall. The scene is likely in a busy city area, given the presence of multiple cabs and the urban environment.


### Call with Precomputed Embeddings

You can pre-calculate image features to avoid repeated visual encoding processes.


```python
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()
vision = model.model.visual.cuda()
```


    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/824 [00:00<?, ?it/s]



```python
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

precomputed_embeddings = vision(
    processor_output["pixel_values"].cuda(), processor_output["image_grid_thw"].cuda()
)
precomputed_embeddings = precomputed_embeddings.pooler_output

multi_modal_item = dict(
    processor_output,
    format="precomputed_embedding",
    feature=precomputed_embeddings,
)

out = llm.generate(input_ids=input_ids, image_data=[multi_modal_item])
print("Response using precomputed embeddings:")
print(out["text"])

llm.shutdown()
```

    Response using precomputed embeddings:
    The image shows a man dressed in a bright yellow shirt and shoes, standing on the side of a street in what appears to be a New York City setting. He is using a clothes iron to press some items that are held up on a small stand in front of him. Behind him, there is a yellow taxi driving by. The person's face is obscured by his shirt and the fact that he is sitting as if on an object, which may be a piece of clothing or a hat. The street appears to be in the middle of the day, and there are some subway signs visible in the background.


## Querying Llama 4 Vision Model

```python
model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
chat_template = "llama-4"

from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

# Download the same example image
image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Llama 4 generated prompt text:")
print(conv.get_prompt())
print(f"Image size: {image.size}")

image
```

### Llama 4 Basic Call

Llama 4 requires more computational resources, so it's configured with multi-GPU parallelism (tp_size=4) and larger context length.

```python
llm = Engine(
    model_path=model_path,
    enable_multimodal=True,
    attention_backend="fa3",
    tp_size=4,
    context_length=65536,
)

out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Llama 4 response:")
print(out["text"])
```

### Call with Processor Output

Using HuggingFace processor to preprocess data can reduce computational overhead during inference.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out)
```

### Call with Precomputed Embeddings

```python
from transformers import AutoProcessor
from transformers import Llama4ForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto"
).eval()

vision = model.vision_model.cuda()
multi_modal_projector = model.multi_modal_projector.cuda()

print(f'Image pixel values shape: {processor_output["pixel_values"].shape}')
input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

# Process image through vision encoder
image_outputs = vision(
    processor_output["pixel_values"].to("cuda"), 
    aspect_ratio_ids=processor_output["aspect_ratio_ids"].to("cuda"),
    aspect_ratio_mask=processor_output["aspect_ratio_mask"].to("cuda"),
    output_hidden_states=False
)
image_features = image_outputs.last_hidden_state

# Flatten image features and pass through multimodal projector
vision_flat = image_features.view(-1, image_features.size(-1))
precomputed_embeddings = multi_modal_projector(vision_flat)

# Build precomputed embedding data item
mm_item = dict(
    processor_output, 
    format="precomputed_embedding", 
    feature=precomputed_embeddings
)

# Use precomputed embeddings for efficient inference
out = llm.generate(input_ids=input_ids, image_data=[mm_item])
print("Llama 4 precomputed embedding response:")
print(out["text"])
```
