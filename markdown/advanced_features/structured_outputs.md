# Structured Outputs

You can specify a JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports three grammar backends:

- [XGrammar](https://github.com/mlc-ai/xgrammar)(default): Supports JSON schema, regular expression, and EBNF constraints.
- [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.
- [Llguidance](https://github.com/guidance-ai/llguidance): Supports JSON schema, regular expression, and EBNF constraints.

We suggest using XGrammar for its better performance and utility. XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). For more details, see [XGrammar technical overview](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

To use Outlines, simply add `--grammar-backend outlines` when launching the server.
To use llguidance, add `--grammar-backend llguidance`  when launching the server.
If no backend is specified, XGrammar will be used as the default.

For better output quality, **It's advisable to explicitly include instructions in the prompt to guide the model to generate the desired format.** For example, you can specify, 'Please generate the output in the following JSON format: ...'.


## OpenAI Compatible API


```python
import openai
import os

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [2026-05-17 22:03:31] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-17 22:03:34] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-17 22:03:37] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-17 22:03:38] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-17 22:03:40] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-17 22:03:40] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.28it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.12it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.05it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.39it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.27it/s]


    2026-05-17 22:03:46,891 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 22:03:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:30,  5.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:30,  5.79s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:21,  2.53s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:21,  2.53s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:21,  1.47s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:21,  1.47s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.03it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:27,  1.89it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:27,  1.89it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:20,  2.47it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:20,  2.47it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.14it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.14it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:12,  3.88it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:12,  3.88it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:10,  4.75it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:10,  4.75it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.65it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.32it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.32it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.32it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.95it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.95it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.95it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.82it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.82it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.82it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:08<00:03, 10.82it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:02, 14.06it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:02, 14.06it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:02, 14.06it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:02, 14.06it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:02, 14.06it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.53it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 26.84it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:08<00:00, 34.67it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:08<00:00, 44.83it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]

    Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:08<00:00, 50.87it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=21.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=21.00 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=20.97 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=20.97 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=20.97 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=20.97 GB):   5%|▌         | 3/58 [00:00<00:17,  3.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=20.96 GB):   5%|▌         | 3/58 [00:00<00:17,  3.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=20.96 GB):   7%|▋         | 4/58 [00:01<00:15,  3.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=20.93 GB):   7%|▋         | 4/58 [00:01<00:15,  3.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=20.93 GB):   9%|▊         | 5/58 [00:01<00:14,  3.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=20.93 GB):   9%|▊         | 5/58 [00:01<00:14,  3.67it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=20.93 GB):  10%|█         | 6/58 [00:01<00:12,  4.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.93 GB):  10%|█         | 6/58 [00:01<00:12,  4.01it/s]Capturing num tokens (num_tokens=5120 avail_mem=20.93 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=20.93 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.34it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=20.93 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=20.93 GB):  14%|█▍        | 8/58 [00:02<00:13,  3.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=20.93 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.41it/s]Capturing num tokens (num_tokens=3840 avail_mem=20.93 GB):  16%|█▌        | 9/58 [00:02<00:14,  3.41it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=20.93 GB):  17%|█▋        | 10/58 [00:02<00:14,  3.28it/s]Capturing num tokens (num_tokens=3584 avail_mem=20.93 GB):  17%|█▋        | 10/58 [00:02<00:14,  3.28it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=20.93 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=20.92 GB):  19%|█▉        | 11/58 [00:03<00:14,  3.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=20.92 GB):  21%|██        | 12/58 [00:03<00:13,  3.32it/s]Capturing num tokens (num_tokens=3072 avail_mem=20.92 GB):  21%|██        | 12/58 [00:03<00:13,  3.32it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=20.92 GB):  22%|██▏       | 13/58 [00:03<00:13,  3.42it/s]Capturing num tokens (num_tokens=2816 avail_mem=20.92 GB):  22%|██▏       | 13/58 [00:03<00:13,  3.42it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=20.92 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.92 GB):  24%|██▍       | 14/58 [00:03<00:11,  3.69it/s]Capturing num tokens (num_tokens=2560 avail_mem=20.92 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=20.91 GB):  26%|██▌       | 15/58 [00:04<00:10,  4.15it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=20.91 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=20.91 GB):  28%|██▊       | 16/58 [00:04<00:09,  4.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=20.91 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=20.91 GB):  29%|██▉       | 17/58 [00:04<00:09,  4.54it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=20.91 GB):  31%|███       | 18/58 [00:04<00:08,  4.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=20.90 GB):  31%|███       | 18/58 [00:04<00:08,  4.86it/s]Capturing num tokens (num_tokens=1536 avail_mem=20.90 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.13it/s]Capturing num tokens (num_tokens=1280 avail_mem=20.88 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.13it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=20.88 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=20.88 GB):  34%|███▍      | 20/58 [00:05<00:07,  5.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=20.88 GB):  36%|███▌      | 21/58 [00:05<00:06,  5.74it/s]Capturing num tokens (num_tokens=960 avail_mem=20.87 GB):  36%|███▌      | 21/58 [00:05<00:06,  5.74it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=20.87 GB):  38%|███▊      | 22/58 [00:05<00:06,  6.00it/s]Capturing num tokens (num_tokens=896 avail_mem=20.86 GB):  38%|███▊      | 22/58 [00:05<00:06,  6.00it/s]Capturing num tokens (num_tokens=896 avail_mem=20.86 GB):  40%|███▉      | 23/58 [00:05<00:05,  6.35it/s]Capturing num tokens (num_tokens=832 avail_mem=20.86 GB):  40%|███▉      | 23/58 [00:05<00:05,  6.35it/s]

    Capturing num tokens (num_tokens=832 avail_mem=20.86 GB):  41%|████▏     | 24/58 [00:05<00:05,  6.60it/s]Capturing num tokens (num_tokens=768 avail_mem=20.85 GB):  41%|████▏     | 24/58 [00:05<00:05,  6.60it/s]

    Capturing num tokens (num_tokens=768 avail_mem=20.85 GB):  43%|████▎     | 25/58 [00:05<00:05,  5.67it/s]Capturing num tokens (num_tokens=704 avail_mem=20.85 GB):  43%|████▎     | 25/58 [00:05<00:05,  5.67it/s]

    Capturing num tokens (num_tokens=704 avail_mem=20.85 GB):  45%|████▍     | 26/58 [00:06<00:06,  5.15it/s]Capturing num tokens (num_tokens=640 avail_mem=20.84 GB):  45%|████▍     | 26/58 [00:06<00:06,  5.15it/s]

    Capturing num tokens (num_tokens=640 avail_mem=20.84 GB):  47%|████▋     | 27/58 [00:06<00:06,  4.91it/s]Capturing num tokens (num_tokens=576 avail_mem=20.84 GB):  47%|████▋     | 27/58 [00:06<00:06,  4.91it/s]

    Capturing num tokens (num_tokens=576 avail_mem=20.84 GB):  48%|████▊     | 28/58 [00:06<00:06,  4.81it/s]Capturing num tokens (num_tokens=512 avail_mem=20.84 GB):  48%|████▊     | 28/58 [00:06<00:06,  4.81it/s]

    Capturing num tokens (num_tokens=512 avail_mem=20.84 GB):  50%|█████     | 29/58 [00:06<00:06,  4.73it/s]Capturing num tokens (num_tokens=480 avail_mem=20.83 GB):  50%|█████     | 29/58 [00:06<00:06,  4.73it/s]

    Capturing num tokens (num_tokens=480 avail_mem=20.83 GB):  52%|█████▏    | 30/58 [00:06<00:05,  4.71it/s]Capturing num tokens (num_tokens=448 avail_mem=20.83 GB):  52%|█████▏    | 30/58 [00:06<00:05,  4.71it/s]Capturing num tokens (num_tokens=448 avail_mem=20.83 GB):  53%|█████▎    | 31/58 [00:07<00:05,  4.90it/s]Capturing num tokens (num_tokens=416 avail_mem=20.80 GB):  53%|█████▎    | 31/58 [00:07<00:05,  4.90it/s]

    Capturing num tokens (num_tokens=416 avail_mem=20.80 GB):  55%|█████▌    | 32/58 [00:07<00:04,  5.55it/s]Capturing num tokens (num_tokens=384 avail_mem=20.80 GB):  55%|█████▌    | 32/58 [00:07<00:04,  5.55it/s]Capturing num tokens (num_tokens=352 avail_mem=20.79 GB):  55%|█████▌    | 32/58 [00:07<00:04,  5.55it/s]Capturing num tokens (num_tokens=320 avail_mem=20.79 GB):  55%|█████▌    | 32/58 [00:07<00:04,  5.55it/s]Capturing num tokens (num_tokens=320 avail_mem=20.79 GB):  60%|██████    | 35/58 [00:07<00:02,  9.72it/s]Capturing num tokens (num_tokens=288 avail_mem=20.78 GB):  60%|██████    | 35/58 [00:07<00:02,  9.72it/s]Capturing num tokens (num_tokens=256 avail_mem=20.78 GB):  60%|██████    | 35/58 [00:07<00:02,  9.72it/s]Capturing num tokens (num_tokens=240 avail_mem=20.78 GB):  60%|██████    | 35/58 [00:07<00:02,  9.72it/s]

    Capturing num tokens (num_tokens=224 avail_mem=20.77 GB):  60%|██████    | 35/58 [00:07<00:02,  9.72it/s]Capturing num tokens (num_tokens=224 avail_mem=20.77 GB):  67%|██████▋   | 39/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=208 avail_mem=20.77 GB):  67%|██████▋   | 39/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=192 avail_mem=20.76 GB):  67%|██████▋   | 39/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=176 avail_mem=20.76 GB):  67%|██████▋   | 39/58 [00:07<00:01, 15.29it/s]Capturing num tokens (num_tokens=176 avail_mem=20.76 GB):  72%|███████▏  | 42/58 [00:07<00:00, 17.09it/s]Capturing num tokens (num_tokens=160 avail_mem=20.75 GB):  72%|███████▏  | 42/58 [00:07<00:00, 17.09it/s]

    Capturing num tokens (num_tokens=144 avail_mem=20.75 GB):  72%|███████▏  | 42/58 [00:07<00:00, 17.09it/s]Capturing num tokens (num_tokens=144 avail_mem=20.75 GB):  76%|███████▌  | 44/58 [00:07<00:00, 16.94it/s]Capturing num tokens (num_tokens=128 avail_mem=20.74 GB):  76%|███████▌  | 44/58 [00:07<00:00, 16.94it/s]Capturing num tokens (num_tokens=112 avail_mem=20.75 GB):  76%|███████▌  | 44/58 [00:07<00:00, 16.94it/s]Capturing num tokens (num_tokens=112 avail_mem=20.75 GB):  79%|███████▉  | 46/58 [00:07<00:00, 17.01it/s]Capturing num tokens (num_tokens=96 avail_mem=20.74 GB):  79%|███████▉  | 46/58 [00:07<00:00, 17.01it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=20.74 GB):  79%|███████▉  | 46/58 [00:07<00:00, 17.01it/s]Capturing num tokens (num_tokens=80 avail_mem=20.74 GB):  83%|████████▎ | 48/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=64 avail_mem=20.74 GB):  83%|████████▎ | 48/58 [00:08<00:00, 14.40it/s]

    Capturing num tokens (num_tokens=48 avail_mem=20.73 GB):  83%|████████▎ | 48/58 [00:08<00:00, 14.40it/s]Capturing num tokens (num_tokens=48 avail_mem=20.73 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.17it/s]Capturing num tokens (num_tokens=32 avail_mem=20.73 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.17it/s]

    Capturing num tokens (num_tokens=28 avail_mem=20.73 GB):  86%|████████▌ | 50/58 [00:08<00:00,  9.17it/s]

    Capturing num tokens (num_tokens=28 avail_mem=20.73 GB):  90%|████████▉ | 52/58 [00:08<00:00,  6.94it/s]Capturing num tokens (num_tokens=24 avail_mem=20.72 GB):  90%|████████▉ | 52/58 [00:08<00:00,  6.94it/s]Capturing num tokens (num_tokens=24 avail_mem=20.72 GB):  91%|█████████▏| 53/58 [00:09<00:00,  6.54it/s]Capturing num tokens (num_tokens=20 avail_mem=20.72 GB):  91%|█████████▏| 53/58 [00:09<00:00,  6.54it/s]

    Capturing num tokens (num_tokens=20 avail_mem=20.72 GB):  93%|█████████▎| 54/58 [00:09<00:00,  6.23it/s]Capturing num tokens (num_tokens=16 avail_mem=20.71 GB):  93%|█████████▎| 54/58 [00:09<00:00,  6.23it/s]Capturing num tokens (num_tokens=16 avail_mem=20.71 GB):  95%|█████████▍| 55/58 [00:09<00:00,  5.95it/s]Capturing num tokens (num_tokens=12 avail_mem=20.71 GB):  95%|█████████▍| 55/58 [00:09<00:00,  5.95it/s]

    Capturing num tokens (num_tokens=12 avail_mem=20.71 GB):  97%|█████████▋| 56/58 [00:09<00:00,  5.72it/s]Capturing num tokens (num_tokens=8 avail_mem=20.70 GB):  97%|█████████▋| 56/58 [00:09<00:00,  5.72it/s] Capturing num tokens (num_tokens=8 avail_mem=20.70 GB):  98%|█████████▊| 57/58 [00:09<00:00,  5.74it/s]Capturing num tokens (num_tokens=4 avail_mem=20.70 GB):  98%|█████████▊| 57/58 [00:09<00:00,  5.74it/s]

    Capturing num tokens (num_tokens=4 avail_mem=20.70 GB): 100%|██████████| 58/58 [00:09<00:00,  5.80it/s]


    [2026-05-17 22:04:07] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-17 22:04:09] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


### JSON

you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.

**Using Pydantic**


```python
from pydantic import BaseModel, Field


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**



```python
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>{"name": "Paris", "population": 2147000}</strong>


### EBNF


```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ],
    temperature=0,
    max_tokens=32,
    extra_body={"ebnf": ebnf_grammar},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris is the capital of France</strong>


### Regular expression


```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
    max_tokens=128,
    extra_body={"regex": "(Paris|London)"},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris</strong>


### Structural Tag


```python
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
schema_get_current_date = tool_get_current_date["function"]["parameters"]


def get_messages():
    return [
        {
            "role": "system",
            "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
        },
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_current_weather>",
                "schema": schema_get_current_weather,
                "end": "</function>",
            },
            {
                "begin": "<function=get_current_date>",
                "schema": schema_get_current_date,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_weather,
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_date,
                    },
                    "end": "</function>",
                },
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources:<br>- get_current_date function<br>- get_current_weather function</strong>


## Native API and SGLang Runtime (SRT)

### JSON

**Using Pydantic**


```python
import requests
import json
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


# Make API request
messages = [
    {
        "role": "user",
        "content": "Here is the information of the capital of France in the JSON format.\n",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json.dumps(CapitalInfo.model_json_schema()),
        },
    },
)
print_highlight(response.json())


response_data = json.loads(response.json()["text"])
# validate the response by the pydantic model
capital_info = CapitalInfo.model_validate(response_data)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '5d51f7403c1140d38182d7e715b14ea7', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.23880509659647942, 'response_sent_to_client_ts': 1779055460.5482175}}</strong>



<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**


```python
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': 'd16b2b834470484d8145479e82ec7b7f', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.24486838467419147, 'response_sent_to_client_ts': 1779055460.8020256}}</strong>


### EBNF


```python
messages = [
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "max_new_tokens": 128,
            "temperature": 0,
            "n": 3,
            "ebnf": (
                "root ::= city | description\n"
                'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
                'description ::= city " is " status\n'
                'status ::= "the capital of " country\n'
                'country ::= "England" | "France" | "Germany" | "Italy"'
            ),
        },
        "stream": False,
        "return_logprob": False,
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '150abae57edb4955a2504c29c109d8df', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12332810275256634, 'response_sent_to_client_ts': 1779055460.970469}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '8b3b799222114565abb386ef97731d9e', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12326386198401451, 'response_sent_to_client_ts': 1779055460.9704797}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '695d118c20e841e38aaddcf8358a6518', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.12321885488927364, 'response_sent_to_client_ts': 1779055460.9704838}}]</strong>


### Regular expression


```python
messages = [
    {
        "role": "user",
        "content": "Paris is the capital of",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': 'f31052685b9f459cb5ca7ee140d15846', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.05620921030640602, 'response_sent_to_client_ts': 1779055461.0336113}}</strong>


### Structural Tag


```python
from transformers import AutoTokenizer

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'Paris is the capital of France.', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 13, 128009], 'meta_info': {'id': '0a3203d1881244fc95bfaa7b4752c23c', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 8, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1497934404760599, 'response_sent_to_client_ts': 1779055462.2703614}}</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=get_current_weather>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_weather,
                            },
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=get_current_date>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_date,
                            },
                            "end": "</function>",
                        },
                    ],
                    "at_least_one": False,
                    "stop_after_first": False,
                },
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'Paris is the capital of France.', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 13, 128009], 'meta_info': {'id': '0ea54bf58d4f4c68886dd259255b76b6', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 8, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.08412185125052929, 'response_sent_to_client_ts': 1779055462.3627787}}</strong>



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl

llm = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", grammar_backend="xgrammar"
)
```

    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.16it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.10it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.07it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.40it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.27it/s]


    2026-05-17 22:04:40,019 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 22:04:40] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:22,  5.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:22,  5.66s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:19,  2.49s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:19,  2.49s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:20,  1.46s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:20,  1.46s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:52,  1.04it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:26,  1.97it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:26,  1.97it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.56it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.56it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.23it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.23it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:12,  3.99it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:12,  3.99it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:10,  4.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:10,  4.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:07<00:10,  4.78it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:07<00:07,  6.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:07<00:07,  6.38it/s]

    Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:07<00:07,  6.38it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:07<00:05,  7.96it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:07<00:05,  7.96it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:07<00:05,  7.96it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:07<00:04,  9.68it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:07<00:03, 11.62it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:07<00:03, 11.62it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:07<00:03, 11.62it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:07<00:03, 11.62it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:07<00:02, 15.31it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:07<00:02, 15.31it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:07<00:02, 15.31it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:07<00:02, 15.31it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:08<00:02, 15.31it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:08<00:01, 21.13it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:08<00:00, 30.24it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]

    Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:08<00:00, 39.17it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:08<00:00, 47.04it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:08<00:00, 55.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.80it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=57.67 GB):   2%|▏         | 1/58 [00:00<00:19,  2.85it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.64 GB):   2%|▏         | 1/58 [00:00<00:19,  2.85it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=57.64 GB):   3%|▎         | 2/58 [00:00<00:19,  2.92it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.64 GB):   3%|▎         | 2/58 [00:00<00:19,  2.92it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=57.64 GB):   5%|▌         | 3/58 [00:01<00:18,  3.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.64 GB):   5%|▌         | 3/58 [00:01<00:18,  3.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=57.64 GB):   7%|▋         | 4/58 [00:01<00:16,  3.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.64 GB):   7%|▋         | 4/58 [00:01<00:16,  3.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.56it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=57.64 GB):  10%|█         | 6/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.64 GB):  10%|█         | 6/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.31it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=57.64 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.64 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.75it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.64 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.23it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.64 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.23it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=57.64 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.64 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=57.64 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=57.63 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.13it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=57.63 GB):  21%|██        | 12/58 [00:02<00:06,  6.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.63 GB):  21%|██        | 12/58 [00:02<00:06,  6.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=57.63 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=57.63 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.19it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=57.62 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=57.62 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=57.62 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=57.62 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.55it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=57.62 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.93it/s]Capturing num tokens (num_tokens=1792 avail_mem=57.62 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.61 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=57.61 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=57.61 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=57.61 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.59it/s]

    Capturing num tokens (num_tokens=960 avail_mem=57.59 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.59it/s] Capturing num tokens (num_tokens=960 avail_mem=57.59 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.82it/s]Capturing num tokens (num_tokens=896 avail_mem=57.59 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.82it/s]Capturing num tokens (num_tokens=832 avail_mem=57.59 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.82it/s]Capturing num tokens (num_tokens=768 avail_mem=57.58 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.82it/s]Capturing num tokens (num_tokens=768 avail_mem=57.58 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=704 avail_mem=57.58 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=640 avail_mem=57.58 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.60it/s]

    Capturing num tokens (num_tokens=576 avail_mem=57.57 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=576 avail_mem=57.57 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.46it/s]Capturing num tokens (num_tokens=512 avail_mem=57.57 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.46it/s]Capturing num tokens (num_tokens=480 avail_mem=57.56 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.46it/s]Capturing num tokens (num_tokens=448 avail_mem=57.56 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.46it/s]Capturing num tokens (num_tokens=416 avail_mem=57.56 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.46it/s]Capturing num tokens (num_tokens=416 avail_mem=57.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.58it/s]Capturing num tokens (num_tokens=384 avail_mem=57.56 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.58it/s]Capturing num tokens (num_tokens=352 avail_mem=57.55 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.58it/s]

    Capturing num tokens (num_tokens=320 avail_mem=57.55 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.58it/s]Capturing num tokens (num_tokens=288 avail_mem=57.54 GB):  55%|█████▌    | 32/58 [00:03<00:01, 23.58it/s]Capturing num tokens (num_tokens=288 avail_mem=57.54 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.47it/s]Capturing num tokens (num_tokens=256 avail_mem=57.54 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.47it/s]Capturing num tokens (num_tokens=240 avail_mem=57.54 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.47it/s]Capturing num tokens (num_tokens=224 avail_mem=57.53 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.47it/s]Capturing num tokens (num_tokens=208 avail_mem=57.53 GB):  62%|██████▏   | 36/58 [00:03<00:00, 26.47it/s]Capturing num tokens (num_tokens=208 avail_mem=57.53 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=192 avail_mem=57.52 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=176 avail_mem=57.52 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=57.51 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=144 avail_mem=57.51 GB):  69%|██████▉   | 40/58 [00:03<00:00, 29.23it/s]Capturing num tokens (num_tokens=144 avail_mem=57.51 GB):  76%|███████▌  | 44/58 [00:03<00:00, 31.45it/s]Capturing num tokens (num_tokens=128 avail_mem=57.50 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.45it/s]Capturing num tokens (num_tokens=112 avail_mem=57.51 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.45it/s]Capturing num tokens (num_tokens=96 avail_mem=57.50 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.45it/s] Capturing num tokens (num_tokens=80 avail_mem=57.50 GB):  76%|███████▌  | 44/58 [00:04<00:00, 31.45it/s]Capturing num tokens (num_tokens=80 avail_mem=57.50 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.21it/s]Capturing num tokens (num_tokens=64 avail_mem=57.50 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.21it/s]Capturing num tokens (num_tokens=48 avail_mem=57.49 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.21it/s]

    Capturing num tokens (num_tokens=32 avail_mem=57.49 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.21it/s]Capturing num tokens (num_tokens=28 avail_mem=57.48 GB):  83%|████████▎ | 48/58 [00:04<00:00, 33.21it/s]Capturing num tokens (num_tokens=28 avail_mem=57.48 GB):  90%|████████▉ | 52/58 [00:04<00:00, 34.43it/s]Capturing num tokens (num_tokens=24 avail_mem=57.48 GB):  90%|████████▉ | 52/58 [00:04<00:00, 34.43it/s]Capturing num tokens (num_tokens=20 avail_mem=57.47 GB):  90%|████████▉ | 52/58 [00:04<00:00, 34.43it/s]Capturing num tokens (num_tokens=16 avail_mem=57.47 GB):  90%|████████▉ | 52/58 [00:04<00:00, 34.43it/s]Capturing num tokens (num_tokens=12 avail_mem=57.47 GB):  90%|████████▉ | 52/58 [00:04<00:00, 34.43it/s]Capturing num tokens (num_tokens=12 avail_mem=57.47 GB):  97%|█████████▋| 56/58 [00:04<00:00, 35.46it/s]Capturing num tokens (num_tokens=8 avail_mem=57.46 GB):  97%|█████████▋| 56/58 [00:04<00:00, 35.46it/s] Capturing num tokens (num_tokens=4 avail_mem=57.46 GB):  97%|█████████▋| 56/58 [00:04<00:00, 35.46it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.46 GB): 100%|██████████| 58/58 [00:04<00:00, 13.27it/s]


### JSON

**Using Pydantic**


```python
import json
from pydantic import BaseModel, Field

prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


sampling_params = {
    "temperature": 0.1,
    "top_p": 0.95,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}")  # validate the output by the pydantic model
    capital_info = CapitalInfo.model_validate_json(output["text"])
    print_highlight(f"Validated output: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Beijing","population":21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Paris","population":2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Dublin","population":527617}</strong>


**JSON Schema Directly**


```python
prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

sampling_params = {"temperature": 0.1, "top_p": 0.95, "json_schema": json_schema}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.<br>Generated text: {"name": "Beijing", "population": 21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.<br>Generated text: {"name": "Paris", "population": 2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.<br>Generated text: {"name": "Dublin", "population": 527617}</strong>


### EBNF



```python
prompts = [
    "Give me the information of the capital of France.",
    "Give me the information of the capital of Germany.",
    "Give me the information of the capital of Italy.",
]

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "ebnf": (
        "root ::= city | description\n"
        'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
        'description ::= city " is " status\n'
        'status ::= "the capital of " country\n'
        'country ::= "England" | "France" | "Germany" | "Italy"'
    ),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France.<br>Generated text: Paris is the capital of France</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Germany.<br>Generated text: Berlin is the capital of Germany</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: London is the capital of Italy</strong>


### Regular expression


```python
prompts = [
    "Please provide information about London as a major global city:",
    "Please provide information about Paris as a major global city:",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95, "regex": "(France|England)"}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about London as a major global city:<br>Generated text: England</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about Paris as a major global city:<br>Generated text: France</strong>


### Structural Tag


```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
prompts = [text]


sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema": schema_get_current_weather,
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": schema_get_current_date,
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_weather,
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_date,
                        },
                        "end": "</function>",
                    },
                ],
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
llm.shutdown()
```
