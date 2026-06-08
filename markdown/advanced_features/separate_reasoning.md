# Reasoning Parser

SGLang supports parsing reasoning content out from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

## Supported Models & Parsers

| Model  |  Reasoning tags      | Parser | Notes |
|---------|-----------------------------|------------------|-------|
| [DeepSeek‑R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `<think>` … `</think>` | `deepseek-r1` | Supports all variants (R1, R1-0528, R1-Distill) |
| [DeepSeek‑V3 series](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | `<think>` … `</think>` | `deepseek-v3` | Including [DeepSeek‑V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Supports `thinking` parameter |
| [Standard Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `<think>` … `</think>` | `qwen3` | Supports `enable_thinking` parameter |
| [Qwen3-Thinking models](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) | `<think>` … `</think>` | `qwen3` or `qwen3-thinking` | Always generates thinking content |
| [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | `◁think▷` … `◁/think▷` | `kimi_k2` | Uses special thinking delimiters. Also requires `--tool-call-parser kimi_k2` for tool use. |
| [GPT OSS](https://huggingface.co/openai/gpt-oss-120b) | `<\|channel\|>analysis<\|message\|>` … `<\|end\|>` | `gpt-oss` | N/A |
### Model-Specific Behaviors

**DeepSeek-R1 Family:**
- DeepSeek-R1: No `<think>` start tag, jumps directly to thinking content
- DeepSeek-R1-0528: Generates both `<think>` start and `</think>` end tags
- Both are handled by the same `deepseek-r1` parser

**DeepSeek-V3 Family:**
- DeepSeek-V3.1/V3.2: Hybrid model supporting both thinking and non-thinking modes, use the `deepseek-v3` parser and `thinking` parameter (NOTE: not `enable_thinking`)

**Qwen3 Family:**
- Standard Qwen3 (e.g., Qwen3-2507): Use `qwen3` parser, supports `enable_thinking` in chat templates
- Qwen3-Thinking (e.g., Qwen3-235B-A22B-Thinking-2507): Use `qwen3` or `qwen3-thinking` parser, always thinks

**Kimi K2:**
- Kimi K2 Thinking: Uses special `◁think▷` and `◁/think▷` tags. For agentic tool use, also specify `--tool-call-parser kimi_k2`.

**GPT OSS:**
- GPT OSS: Uses special `<|channel|>analysis<|message|>` and `<|end|>` tags

## Usage

### Launching the Server

Specify the `--reasoning-parser` option.


```python
import requests
from openai import OpenAI
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.40s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.32s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.33s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:25,  5.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:25,  5.71s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:19,  2.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:19,  2.49s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:25,  1.55s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:25,  1.55s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:59,  1.10s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:59,  1.10s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:43,  1.21it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:34,  1.51it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:34,  1.51it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.85it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.85it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:22,  2.20it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:22,  2.20it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:18,  2.61it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:18,  2.61it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.02it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.02it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.44it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.44it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.84it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.30it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.75it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.75it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.29it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.29it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.57it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.57it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.19it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.19it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.19it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.81it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.81it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.81it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.48it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.48it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:10<00:03, 10.48it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:02, 12.64it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:02, 12.64it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:02, 12.64it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:10<00:02, 12.64it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:10<00:01, 15.58it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:10<00:01, 15.58it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:10<00:01, 15.58it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:10<00:01, 15.58it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:10<00:01, 19.05it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:10<00:01, 19.05it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:10<00:01, 19.05it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:10<00:01, 19.05it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 20.64it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 20.64it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 20.64it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 20.64it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 22.99it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 29.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 29.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 29.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 29.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:10<00:00, 29.02it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:10<00:00, 31.91it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:10<00:00, 35.58it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:10<00:00, 41.71it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:10<00:00, 41.71it/s] 

    Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:10<00:00, 41.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   2%|▏         | 1/58 [00:00<00:30,  1.86it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.41 GB):   2%|▏         | 1/58 [00:00<00:30,  1.86it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.41 GB):   3%|▎         | 2/58 [00:01<00:29,  1.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.13 GB):   3%|▎         | 2/58 [00:01<00:29,  1.90it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.13 GB):   5%|▌         | 3/58 [00:01<00:28,  1.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.19 GB):   5%|▌         | 3/58 [00:01<00:28,  1.94it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.19 GB):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.26 GB):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.26 GB):   9%|▊         | 5/58 [00:02<00:24,  2.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.33 GB):   9%|▊         | 5/58 [00:02<00:24,  2.12it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.33 GB):  10%|█         | 6/58 [00:02<00:22,  2.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.39 GB):  10%|█         | 6/58 [00:02<00:22,  2.28it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.39 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.46it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.45 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.46it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.45 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.52 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.67it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.52 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.59 GB):  16%|█▌        | 9/58 [00:03<00:16,  2.91it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.59 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.15it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.61 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.61 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.64 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=43.64 GB):  21%|██        | 12/58 [00:04<00:12,  3.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.67 GB):  21%|██        | 12/58 [00:04<00:12,  3.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.67 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.70 GB):  22%|██▏       | 13/58 [00:04<00:11,  4.01it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.70 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=43.73 GB):  24%|██▍       | 14/58 [00:04<00:10,  4.37it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=43.73 GB):  26%|██▌       | 15/58 [00:04<00:08,  4.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.76 GB):  26%|██▌       | 15/58 [00:04<00:08,  4.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=43.76 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=43.79 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.39it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=43.79 GB):  29%|██▉       | 17/58 [00:05<00:06,  5.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.82 GB):  29%|██▉       | 17/58 [00:05<00:06,  5.98it/s]Capturing num tokens (num_tokens=1792 avail_mem=43.82 GB):  31%|███       | 18/58 [00:05<00:06,  6.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=43.84 GB):  31%|███       | 18/58 [00:05<00:06,  6.57it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.87 GB):  31%|███       | 18/58 [00:05<00:06,  6.57it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.87 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.88 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s]Capturing num tokens (num_tokens=960 avail_mem=43.91 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.15it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=43.91 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.85it/s]Capturing num tokens (num_tokens=896 avail_mem=43.91 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.85it/s]Capturing num tokens (num_tokens=832 avail_mem=43.91 GB):  38%|███▊      | 22/58 [00:05<00:03,  9.85it/s]Capturing num tokens (num_tokens=832 avail_mem=43.91 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.25it/s]Capturing num tokens (num_tokens=768 avail_mem=43.91 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.25it/s]Capturing num tokens (num_tokens=704 avail_mem=43.91 GB):  41%|████▏     | 24/58 [00:05<00:03, 11.25it/s]

    Capturing num tokens (num_tokens=704 avail_mem=43.91 GB):  45%|████▍     | 26/58 [00:05<00:02, 12.56it/s]Capturing num tokens (num_tokens=640 avail_mem=43.91 GB):  45%|████▍     | 26/58 [00:05<00:02, 12.56it/s]Capturing num tokens (num_tokens=576 avail_mem=43.93 GB):  45%|████▍     | 26/58 [00:06<00:02, 12.56it/s]Capturing num tokens (num_tokens=576 avail_mem=43.93 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.15it/s]Capturing num tokens (num_tokens=512 avail_mem=43.93 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.15it/s]Capturing num tokens (num_tokens=480 avail_mem=43.92 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.15it/s]Capturing num tokens (num_tokens=448 avail_mem=43.92 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.15it/s]

    Capturing num tokens (num_tokens=448 avail_mem=43.92 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.90it/s]Capturing num tokens (num_tokens=416 avail_mem=43.92 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.90it/s]Capturing num tokens (num_tokens=384 avail_mem=43.92 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.90it/s]Capturing num tokens (num_tokens=352 avail_mem=43.91 GB):  53%|█████▎    | 31/58 [00:06<00:01, 15.90it/s]Capturing num tokens (num_tokens=352 avail_mem=43.91 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.03it/s]Capturing num tokens (num_tokens=320 avail_mem=43.92 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.03it/s]Capturing num tokens (num_tokens=288 avail_mem=43.95 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.03it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.91 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.03it/s]Capturing num tokens (num_tokens=256 avail_mem=43.91 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.87it/s]Capturing num tokens (num_tokens=240 avail_mem=43.92 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.87it/s]Capturing num tokens (num_tokens=224 avail_mem=43.93 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.87it/s]Capturing num tokens (num_tokens=208 avail_mem=43.89 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.87it/s]Capturing num tokens (num_tokens=208 avail_mem=43.89 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.63it/s]Capturing num tokens (num_tokens=192 avail_mem=43.88 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.63it/s]Capturing num tokens (num_tokens=176 avail_mem=43.88 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.63it/s]

    Capturing num tokens (num_tokens=160 avail_mem=43.89 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.63it/s]Capturing num tokens (num_tokens=160 avail_mem=43.89 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.81it/s]Capturing num tokens (num_tokens=144 avail_mem=43.88 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.81it/s]Capturing num tokens (num_tokens=128 avail_mem=43.89 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.81it/s]Capturing num tokens (num_tokens=112 avail_mem=43.89 GB):  74%|███████▍  | 43/58 [00:06<00:00, 22.81it/s]Capturing num tokens (num_tokens=112 avail_mem=43.89 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.44it/s]Capturing num tokens (num_tokens=96 avail_mem=43.88 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.44it/s] Capturing num tokens (num_tokens=80 avail_mem=43.87 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.44it/s]

    Capturing num tokens (num_tokens=64 avail_mem=43.86 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.44it/s]Capturing num tokens (num_tokens=64 avail_mem=43.86 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.48it/s]Capturing num tokens (num_tokens=48 avail_mem=43.86 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.48it/s]Capturing num tokens (num_tokens=32 avail_mem=43.85 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.48it/s]Capturing num tokens (num_tokens=28 avail_mem=43.83 GB):  84%|████████▍ | 49/58 [00:07<00:00, 25.48it/s]Capturing num tokens (num_tokens=28 avail_mem=43.83 GB):  90%|████████▉ | 52/58 [00:07<00:00, 23.49it/s]Capturing num tokens (num_tokens=24 avail_mem=43.84 GB):  90%|████████▉ | 52/58 [00:07<00:00, 23.49it/s]

    Capturing num tokens (num_tokens=20 avail_mem=43.83 GB):  90%|████████▉ | 52/58 [00:07<00:00, 23.49it/s]Capturing num tokens (num_tokens=16 avail_mem=43.83 GB):  90%|████████▉ | 52/58 [00:07<00:00, 23.49it/s]Capturing num tokens (num_tokens=16 avail_mem=43.83 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.95it/s]Capturing num tokens (num_tokens=12 avail_mem=43.82 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.95it/s]Capturing num tokens (num_tokens=8 avail_mem=43.81 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.95it/s] Capturing num tokens (num_tokens=4 avail_mem=43.81 GB):  95%|█████████▍| 55/58 [00:07<00:00, 24.95it/s]Capturing num tokens (num_tokens=4 avail_mem=43.81 GB): 100%|██████████| 58/58 [00:07<00:00, 26.28it/s]Capturing num tokens (num_tokens=4 avail_mem=43.81 GB): 100%|██████████| 58/58 [00:07<00:00,  7.99it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--reasoning-parser` defines the parser used to interpret responses.

### OpenAI Compatible API

Using the OpenAI compatible API, the contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id

messages = [
    {
        "role": "user",
        "content": "What is 1+3?",
    }
]
```

#### Non-Streaming Request


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)
print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \(\boxed{4}\)</strong>


#### Streaming Request


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


Optionally, you can buffer the reasoning content to the last reasoning chunk (or the first chunk after the reasoning content).


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True, "stream_reasoning": False},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Finally, I conclude that the result of 1 + 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:** \(\boxed{4}\)</strong>


The reasoning separation is enable by default when specify . 
**To disable it, set the `separate_reasoning` option to `False` in request.**


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": False},
)

print_highlight("==== Original Output ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. Start with the number **1**.<br>2. Add the number **3** to it.<br>3. The sum is **4**.<br><br>Therefore, the final answer is:<br><br>\[<br>\boxed{4}<br>\]</strong>


### SGLang Native API 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]

print_highlight("==== Original Output ====")
print_highlight(gen_response)

parse_url = f"http://localhost:{port}/separate_reasoning"
separate_reasoning_data = {
    "text": gen_response,
    "reasoning_parser": "deepseek-r1",
}
separate_reasoning_response_json = requests.post(
    parse_url, json=separate_reasoning_data
).json()
print_highlight("==== Reasoning ====")
print_highlight(separate_reasoning_response_json["reasoning_text"])
print_highlight("==== Text ====")
print_highlight(separate_reasoning_response_json["text"])
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I'll add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of the numbers 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>Therefore, the final answer is \(\boxed{4}\).</strong>



```python
terminate_process(server_process)
```

### Offline Engine API


```python
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import print_highlight

llm = sgl.Engine(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
sampling_params = {
    "max_new_tokens": 1024,
    "skip_special_tokens": False,
    "temperature": 0.6,
    "top_p": 0.95,
}
result = llm.generate(prompt=input, sampling_params=sampling_params)

generated_text = result["text"]  # Assume there is only one prompt

print_highlight("==== Original Output ====")
print_highlight(generated_text)

parser = ReasoningParser("deepseek-r1")
reasoning_text, text = parser.parse_non_stream(generated_text)
print_highlight("==== Reasoning ====")
print_highlight(reasoning_text)
print_highlight("==== Text ====")
print_highlight(text)
```

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.18s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.10s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.12s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.91s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:08,  2.30s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:08,  2.30s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:19,  1.45s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:19,  1.45s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:55,  1.03s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:32,  1.58it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:32,  1.58it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:26,  1.92it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.27it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.27it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:18,  2.66it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:18,  2.66it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:15,  3.08it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:15,  3.08it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.47it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.89it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.74it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.74it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  5.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  5.26it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:07,  5.81it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:07,  5.81it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.42it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.42it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.10it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.10it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.10it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.40it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.40it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.40it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 13.37it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:09<00:02, 14.92it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:09<00:02, 14.92it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:09<00:02, 14.92it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:09<00:02, 14.92it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:09<00:01, 17.67it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:09<00:01, 17.67it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:09<00:01, 17.67it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:10<00:01, 17.67it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:10<00:01, 19.11it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:10<00:01, 19.11it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:10<00:01, 19.11it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:10<00:01, 19.11it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:10<00:01, 20.87it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:10<00:01, 20.87it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:10<00:01, 20.87it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:10<00:01, 20.87it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:10<00:01, 20.87it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 23.97it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 23.97it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 23.97it/s]

    Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 23.97it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 25.06it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 25.06it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 25.06it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:10<00:00, 25.06it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:10<00:00, 25.06it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:10<00:00, 26.78it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:10<00:00, 26.78it/s]

    Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:10<00:00, 26.78it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:10<00:00, 26.78it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 27.46it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 27.46it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 27.46it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 27.46it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 27.46it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:10<00:00, 30.18it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:10<00:00, 30.18it/s]

    Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:10<00:00, 30.18it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:10<00:00, 30.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.65 GB):   2%|▏         | 1/58 [00:00<00:37,  1.50it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.62 GB):   2%|▏         | 1/58 [00:00<00:37,  1.50it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.62 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.62 GB):   3%|▎         | 2/58 [00:01<00:34,  1.64it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.62 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.62 GB):   5%|▌         | 3/58 [00:01<00:31,  1.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.62 GB):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.62 GB):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.62 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.62 GB):   9%|▊         | 5/58 [00:02<00:25,  2.05it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=42.62 GB):  10%|█         | 6/58 [00:03<00:23,  2.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.62 GB):  10%|█         | 6/58 [00:03<00:23,  2.20it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.62 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=27.60 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.32it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=27.60 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=27.60 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.52it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=27.60 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=27.60 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.70it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=27.60 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=27.60 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.87it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=27.60 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=27.59 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.08it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=27.59 GB):  21%|██        | 12/58 [00:04<00:13,  3.30it/s]Capturing num tokens (num_tokens=3072 avail_mem=27.59 GB):  21%|██        | 12/58 [00:04<00:13,  3.30it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=27.59 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=27.59 GB):  22%|██▏       | 13/58 [00:05<00:12,  3.50it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=27.59 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=27.59 GB):  24%|██▍       | 14/58 [00:05<00:11,  3.74it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=27.59 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.73it/s]Capturing num tokens (num_tokens=2304 avail_mem=27.59 GB):  26%|██▌       | 15/58 [00:05<00:11,  3.73it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=27.59 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=27.58 GB):  28%|██▊       | 16/58 [00:05<00:10,  4.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=27.58 GB):  29%|██▉       | 17/58 [00:06<00:09,  4.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=27.58 GB):  29%|██▉       | 17/58 [00:06<00:09,  4.33it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=27.58 GB):  31%|███       | 18/58 [00:06<00:08,  4.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=27.51 GB):  31%|███       | 18/58 [00:06<00:08,  4.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=27.51 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=27.51 GB):  33%|███▎      | 19/58 [00:06<00:07,  5.01it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=27.51 GB):  34%|███▍      | 20/58 [00:06<00:07,  5.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.52 GB):  34%|███▍      | 20/58 [00:06<00:07,  5.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=24.52 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.81it/s]Capturing num tokens (num_tokens=960 avail_mem=24.52 GB):  36%|███▌      | 21/58 [00:06<00:06,  5.81it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=24.52 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.15it/s]Capturing num tokens (num_tokens=896 avail_mem=21.82 GB):  38%|███▊      | 22/58 [00:06<00:05,  6.15it/s]Capturing num tokens (num_tokens=896 avail_mem=21.82 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.55it/s]Capturing num tokens (num_tokens=832 avail_mem=21.73 GB):  40%|███▉      | 23/58 [00:06<00:05,  6.55it/s]

    Capturing num tokens (num_tokens=832 avail_mem=21.73 GB):  41%|████▏     | 24/58 [00:07<00:04,  6.99it/s]Capturing num tokens (num_tokens=768 avail_mem=21.72 GB):  41%|████▏     | 24/58 [00:07<00:04,  6.99it/s]Capturing num tokens (num_tokens=768 avail_mem=21.72 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.46it/s]Capturing num tokens (num_tokens=704 avail_mem=21.72 GB):  43%|████▎     | 25/58 [00:07<00:04,  7.46it/s]

    Capturing num tokens (num_tokens=704 avail_mem=21.72 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.88it/s]Capturing num tokens (num_tokens=640 avail_mem=21.71 GB):  45%|████▍     | 26/58 [00:07<00:04,  7.88it/s]Capturing num tokens (num_tokens=640 avail_mem=21.71 GB):  47%|████▋     | 27/58 [00:07<00:03,  8.38it/s]Capturing num tokens (num_tokens=576 avail_mem=21.71 GB):  47%|████▋     | 27/58 [00:07<00:03,  8.38it/s]

    Capturing num tokens (num_tokens=576 avail_mem=21.71 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.60it/s]Capturing num tokens (num_tokens=512 avail_mem=21.70 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.60it/s]Capturing num tokens (num_tokens=480 avail_mem=21.70 GB):  48%|████▊     | 28/58 [00:07<00:03,  8.60it/s]Capturing num tokens (num_tokens=480 avail_mem=21.70 GB):  52%|█████▏    | 30/58 [00:07<00:03,  9.32it/s]Capturing num tokens (num_tokens=448 avail_mem=21.70 GB):  52%|█████▏    | 30/58 [00:07<00:03,  9.32it/s]

    Capturing num tokens (num_tokens=448 avail_mem=21.70 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.27it/s]Capturing num tokens (num_tokens=416 avail_mem=21.70 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.27it/s]Capturing num tokens (num_tokens=384 avail_mem=21.69 GB):  53%|█████▎    | 31/58 [00:07<00:02,  9.27it/s]Capturing num tokens (num_tokens=384 avail_mem=21.69 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.85it/s]Capturing num tokens (num_tokens=352 avail_mem=21.69 GB):  57%|█████▋    | 33/58 [00:07<00:02,  9.85it/s]

    Capturing num tokens (num_tokens=320 avail_mem=21.68 GB):  57%|█████▋    | 33/58 [00:08<00:02,  9.85it/s]Capturing num tokens (num_tokens=320 avail_mem=21.68 GB):  60%|██████    | 35/58 [00:08<00:02, 10.33it/s]Capturing num tokens (num_tokens=288 avail_mem=21.69 GB):  60%|██████    | 35/58 [00:08<00:02, 10.33it/s]Capturing num tokens (num_tokens=256 avail_mem=21.68 GB):  60%|██████    | 35/58 [00:08<00:02, 10.33it/s]

    Capturing num tokens (num_tokens=256 avail_mem=21.68 GB):  64%|██████▍   | 37/58 [00:08<00:01, 10.81it/s]Capturing num tokens (num_tokens=240 avail_mem=21.68 GB):  64%|██████▍   | 37/58 [00:08<00:01, 10.81it/s]Capturing num tokens (num_tokens=224 avail_mem=21.68 GB):  64%|██████▍   | 37/58 [00:08<00:01, 10.81it/s]Capturing num tokens (num_tokens=224 avail_mem=21.68 GB):  67%|██████▋   | 39/58 [00:08<00:01, 11.27it/s]Capturing num tokens (num_tokens=208 avail_mem=21.67 GB):  67%|██████▋   | 39/58 [00:08<00:01, 11.27it/s]

    Capturing num tokens (num_tokens=192 avail_mem=21.67 GB):  67%|██████▋   | 39/58 [00:08<00:01, 11.27it/s]Capturing num tokens (num_tokens=192 avail_mem=21.67 GB):  71%|███████   | 41/58 [00:08<00:01, 11.69it/s]Capturing num tokens (num_tokens=176 avail_mem=21.66 GB):  71%|███████   | 41/58 [00:08<00:01, 11.69it/s]Capturing num tokens (num_tokens=160 avail_mem=21.66 GB):  71%|███████   | 41/58 [00:08<00:01, 11.69it/s]

    Capturing num tokens (num_tokens=160 avail_mem=21.66 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=144 avail_mem=21.66 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=128 avail_mem=21.66 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=112 avail_mem=21.66 GB):  74%|███████▍  | 43/58 [00:08<00:01, 12.04it/s]Capturing num tokens (num_tokens=112 avail_mem=21.66 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s]Capturing num tokens (num_tokens=96 avail_mem=21.65 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s] Capturing num tokens (num_tokens=80 avail_mem=21.65 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s]Capturing num tokens (num_tokens=64 avail_mem=21.64 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s]Capturing num tokens (num_tokens=48 avail_mem=21.64 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s]

    Capturing num tokens (num_tokens=32 avail_mem=21.64 GB):  79%|███████▉  | 46/58 [00:08<00:00, 15.10it/s]Capturing num tokens (num_tokens=32 avail_mem=21.64 GB):  88%|████████▊ | 51/58 [00:09<00:00, 21.78it/s]Capturing num tokens (num_tokens=28 avail_mem=21.64 GB):  88%|████████▊ | 51/58 [00:09<00:00, 21.78it/s]Capturing num tokens (num_tokens=24 avail_mem=21.63 GB):  88%|████████▊ | 51/58 [00:09<00:00, 21.78it/s]Capturing num tokens (num_tokens=20 avail_mem=21.63 GB):  88%|████████▊ | 51/58 [00:09<00:00, 21.78it/s]Capturing num tokens (num_tokens=16 avail_mem=21.62 GB):  88%|████████▊ | 51/58 [00:09<00:00, 21.78it/s]Capturing num tokens (num_tokens=16 avail_mem=21.62 GB):  95%|█████████▍| 55/58 [00:09<00:00, 24.89it/s]Capturing num tokens (num_tokens=12 avail_mem=21.62 GB):  95%|█████████▍| 55/58 [00:09<00:00, 24.89it/s]

    Capturing num tokens (num_tokens=8 avail_mem=21.62 GB):  95%|█████████▍| 55/58 [00:09<00:00, 24.89it/s] Capturing num tokens (num_tokens=4 avail_mem=21.61 GB):  95%|█████████▍| 55/58 [00:09<00:00, 24.89it/s]Capturing num tokens (num_tokens=4 avail_mem=21.61 GB): 100%|██████████| 58/58 [00:09<00:00, 22.58it/s]Capturing num tokens (num_tokens=4 avail_mem=21.61 GB): 100%|██████████| 58/58 [00:09<00:00,  6.23it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Adding 1 and 3 gives me 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Adding 1 and 3 gives me 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
