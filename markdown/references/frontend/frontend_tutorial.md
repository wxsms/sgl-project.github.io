# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    [2026-04-09 10:04:40] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:40] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:40] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:04:40] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.90it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.61it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.49it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.42it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.48it/s]


    2026-04-09 10:04:43,938 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 10:04:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<02:56,  3.10s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:31,  1.63s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:56,  1.02s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:56,  1.02s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:39,  1.38it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:39,  1.38it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:29,  1.82it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:29,  1.82it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:22,  2.27it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:18,  2.77it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:15,  3.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:15,  3.29it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:05<00:12,  3.92it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:05<00:12,  3.92it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:10,  4.54it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:09,  5.15it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:09,  5.15it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:07,  5.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:07,  5.78it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:05<00:06,  6.43it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:05<00:06,  6.43it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:05<00:06,  7.12it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:05<00:06,  7.12it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:05<00:06,  7.12it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:06<00:04,  8.59it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:06<00:04,  8.59it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:06<00:04,  8.59it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:03, 10.23it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:03, 10.23it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:03, 10.23it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:03, 12.43it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:03, 12.43it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:06<00:03, 12.43it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:06<00:03, 12.43it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:06<00:02, 15.95it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:06<00:02, 15.95it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:06<00:02, 15.95it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:06<00:02, 15.95it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:06<00:02, 15.95it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:06<00:01, 20.69it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:06<00:01, 20.69it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:06<00:01, 20.69it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:06<00:01, 20.69it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:06<00:01, 20.69it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:06<00:01, 25.33it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:00, 30.30it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]

    Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:06<00:00, 34.44it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 37.41it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 37.41it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:07<00:00, 37.41it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:07<00:00, 37.41it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:07<00:00, 37.41it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:07<00:00, 37.41it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]

    Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:07<00:00, 40.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 47.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  8.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=117.15 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=117.15 GB):   2%|▏         | 1/58 [00:00<00:22,  2.53it/s]Capturing num tokens (num_tokens=7680 avail_mem=117.12 GB):   2%|▏         | 1/58 [00:00<00:22,  2.53it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=117.12 GB):   3%|▎         | 2/58 [00:00<00:20,  2.71it/s]Capturing num tokens (num_tokens=7168 avail_mem=117.12 GB):   3%|▎         | 2/58 [00:00<00:20,  2.71it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=117.12 GB):   5%|▌         | 3/58 [00:01<00:19,  2.89it/s]Capturing num tokens (num_tokens=6656 avail_mem=117.12 GB):   5%|▌         | 3/58 [00:01<00:19,  2.89it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=117.12 GB):   7%|▋         | 4/58 [00:01<00:17,  3.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=117.12 GB):   7%|▋         | 4/58 [00:01<00:17,  3.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=117.12 GB):   9%|▊         | 5/58 [00:01<00:20,  2.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.09 GB):   9%|▊         | 5/58 [00:01<00:20,  2.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=117.09 GB):  10%|█         | 6/58 [00:02<00:16,  3.20it/s]Capturing num tokens (num_tokens=5120 avail_mem=119.72 GB):  10%|█         | 6/58 [00:02<00:16,  3.20it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=119.72 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.72 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=119.72 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.45it/s]Capturing num tokens (num_tokens=4096 avail_mem=119.73 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.45it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=119.73 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.73 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=119.73 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.75it/s]Capturing num tokens (num_tokens=3584 avail_mem=119.73 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.75it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=119.73 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.73 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=119.73 GB):  21%|██        | 12/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=119.73 GB):  21%|██        | 12/58 [00:02<00:06,  7.15it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=119.73 GB):  21%|██        | 12/58 [00:02<00:06,  7.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=119.73 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.46it/s]Capturing num tokens (num_tokens=2560 avail_mem=119.73 GB):  24%|██▍       | 14/58 [00:02<00:05,  8.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=119.73 GB):  24%|██▍       | 14/58 [00:03<00:05,  8.46it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=119.73 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.88it/s]Capturing num tokens (num_tokens=2048 avail_mem=119.73 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.73 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.88it/s]Capturing num tokens (num_tokens=1792 avail_mem=119.73 GB):  31%|███       | 18/58 [00:03<00:03, 11.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=119.72 GB):  31%|███       | 18/58 [00:03<00:03, 11.53it/s]Capturing num tokens (num_tokens=1280 avail_mem=119.73 GB):  31%|███       | 18/58 [00:03<00:03, 11.53it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=119.73 GB):  31%|███       | 18/58 [00:03<00:03, 11.53it/s]Capturing num tokens (num_tokens=1024 avail_mem=119.73 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.55it/s]Capturing num tokens (num_tokens=960 avail_mem=119.72 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.55it/s] Capturing num tokens (num_tokens=896 avail_mem=119.72 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.55it/s]Capturing num tokens (num_tokens=832 avail_mem=119.72 GB):  36%|███▌      | 21/58 [00:03<00:02, 14.55it/s]Capturing num tokens (num_tokens=832 avail_mem=119.72 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=768 avail_mem=119.71 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=704 avail_mem=119.71 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.60it/s]

    Capturing num tokens (num_tokens=640 avail_mem=119.70 GB):  41%|████▏     | 24/58 [00:03<00:01, 17.60it/s]Capturing num tokens (num_tokens=640 avail_mem=119.70 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.59it/s]Capturing num tokens (num_tokens=576 avail_mem=119.70 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.59it/s]Capturing num tokens (num_tokens=512 avail_mem=119.70 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.59it/s]Capturing num tokens (num_tokens=480 avail_mem=119.69 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.59it/s]Capturing num tokens (num_tokens=448 avail_mem=119.69 GB):  47%|████▋     | 27/58 [00:03<00:01, 20.59it/s]Capturing num tokens (num_tokens=448 avail_mem=119.69 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.40it/s]Capturing num tokens (num_tokens=416 avail_mem=119.69 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.40it/s]Capturing num tokens (num_tokens=384 avail_mem=119.68 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.40it/s]

    Capturing num tokens (num_tokens=352 avail_mem=119.68 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.40it/s]Capturing num tokens (num_tokens=320 avail_mem=119.67 GB):  53%|█████▎    | 31/58 [00:03<00:01, 24.40it/s]Capturing num tokens (num_tokens=320 avail_mem=119.67 GB):  60%|██████    | 35/58 [00:03<00:00, 27.61it/s]Capturing num tokens (num_tokens=288 avail_mem=119.67 GB):  60%|██████    | 35/58 [00:03<00:00, 27.61it/s]Capturing num tokens (num_tokens=256 avail_mem=119.67 GB):  60%|██████    | 35/58 [00:03<00:00, 27.61it/s]Capturing num tokens (num_tokens=240 avail_mem=119.66 GB):  60%|██████    | 35/58 [00:03<00:00, 27.61it/s]

    Capturing num tokens (num_tokens=240 avail_mem=119.66 GB):  66%|██████▌   | 38/58 [00:04<00:00, 23.71it/s]Capturing num tokens (num_tokens=224 avail_mem=119.66 GB):  66%|██████▌   | 38/58 [00:04<00:00, 23.71it/s]Capturing num tokens (num_tokens=208 avail_mem=119.65 GB):  66%|██████▌   | 38/58 [00:04<00:00, 23.71it/s]Capturing num tokens (num_tokens=192 avail_mem=119.65 GB):  66%|██████▌   | 38/58 [00:04<00:00, 23.71it/s]Capturing num tokens (num_tokens=192 avail_mem=119.65 GB):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Capturing num tokens (num_tokens=176 avail_mem=119.65 GB):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Capturing num tokens (num_tokens=160 avail_mem=119.65 GB):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Capturing num tokens (num_tokens=144 avail_mem=119.64 GB):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]Capturing num tokens (num_tokens=128 avail_mem=119.65 GB):  71%|███████   | 41/58 [00:04<00:00, 24.23it/s]

    Capturing num tokens (num_tokens=128 avail_mem=119.65 GB):  78%|███████▊  | 45/58 [00:04<00:00, 27.63it/s]Capturing num tokens (num_tokens=112 avail_mem=119.65 GB):  78%|███████▊  | 45/58 [00:04<00:00, 27.63it/s]Capturing num tokens (num_tokens=96 avail_mem=119.64 GB):  78%|███████▊  | 45/58 [00:04<00:00, 27.63it/s] Capturing num tokens (num_tokens=80 avail_mem=119.64 GB):  78%|███████▊  | 45/58 [00:04<00:00, 27.63it/s]Capturing num tokens (num_tokens=64 avail_mem=119.63 GB):  78%|███████▊  | 45/58 [00:04<00:00, 27.63it/s]Capturing num tokens (num_tokens=64 avail_mem=119.63 GB):  84%|████████▍ | 49/58 [00:04<00:00, 30.37it/s]Capturing num tokens (num_tokens=48 avail_mem=119.63 GB):  84%|████████▍ | 49/58 [00:04<00:00, 30.37it/s]Capturing num tokens (num_tokens=32 avail_mem=119.63 GB):  84%|████████▍ | 49/58 [00:04<00:00, 30.37it/s]Capturing num tokens (num_tokens=28 avail_mem=119.63 GB):  84%|████████▍ | 49/58 [00:04<00:00, 30.37it/s]Capturing num tokens (num_tokens=24 avail_mem=119.62 GB):  84%|████████▍ | 49/58 [00:04<00:00, 30.37it/s]

    Capturing num tokens (num_tokens=24 avail_mem=119.62 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.78it/s]Capturing num tokens (num_tokens=20 avail_mem=119.62 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.78it/s]Capturing num tokens (num_tokens=16 avail_mem=119.62 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.78it/s]Capturing num tokens (num_tokens=12 avail_mem=119.61 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.78it/s]Capturing num tokens (num_tokens=8 avail_mem=119.61 GB):  91%|█████████▏| 53/58 [00:04<00:00, 31.78it/s] Capturing num tokens (num_tokens=8 avail_mem=119.61 GB):  98%|█████████▊| 57/58 [00:04<00:00, 33.62it/s]Capturing num tokens (num_tokens=4 avail_mem=119.60 GB):  98%|█████████▊| 57/58 [00:04<00:00, 33.62it/s]Capturing num tokens (num_tokens=4 avail_mem=119.60 GB): 100%|██████████| 58/58 [00:04<00:00, 12.66it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34624


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-09 10:05:03] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Argentina - Buenos Aires</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Certainly! Here is a list of three countries along with their respective capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Mexico** - Mexico City</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries along with their respective capitals:<br><br>1. **Italy** - Rome<br>2. **Canada** - Ottawa<br>3. **Australia** - Canberra</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2<br><br>Let's calculate the result:<br><br>2 * 2 = 4<br><br>No need to use a calculator for this simple calculation, but if you wish, you can easily verify this result using any calculator.</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>1. **Balanced Diet**: Consuming a variety of foods in the right proportions to ensure you get all necessary nutrients. This includes:<br>   - Fruits and vegetables: For vitamins, minerals, and fiber.<br>   - Whole grains: For complex carbohydrates and other important nutrients.<br>   - Lean proteins: For amino acids.<br>   - Healthy fats: For brain function and cell maintenance.<br><br>2. **Regular Exercise**: Engaging in physical activity regularly to improve various aspects of health. This involves:<br>   - Moderate to vigorous aerobic activity: Aim for 150 to 300 minutes of moderate activity or 75 to 150 minutes of vigorous activity per week.<br>   - Strength training: Incorporate at least two days a week to enhance muscle and bone strength.<br><br>By combining these two habits, you can significantly enhance your overall health and well-being.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "willow",<br>        "core": " phoenix feather",<br>        "length": 10.5<br>    },<br>    "alive": "Deceased",<br>    "patronus": "stag",<br>    "bogart": "the WhompingWill"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

    100%|██████████| 3/3 [00:00<00:00, 27.04it/s]

    100%|██████████| 3/3 [00:00<00:00, 26.73it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:51: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779dd-3e096adb4beed1af20400667;6a453722-0a7b-41f5-bcd9-1388d4266bb6)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    retry() failed once (0th try, maximum 2 retries). Will delay 0.76s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779dd-3e096adb4beed1af20400667;6a453722-0a7b-41f5-bcd9-1388d4266bb6)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779dd-6f00a60153a9f4da5d1ccf33;8734f61a-c83f-4729-98ea-bc833ad15d40)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    retry() failed once (1th try, maximum 2 retries). Will delay 1.74s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779dd-6f00a60153a9f4da5d1ccf33;8734f61a-c83f-4729-98ea-bc833ad15d40)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779df-5648bf35514867c85d1e741d;614e678f-dcee-4b76-b3a8-6b884765eed0)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e0-2740b42913a66f41197ac613;712e6faf-2f21-4690-8d9f-f0c02b05c3cc)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:20] retry() failed once (0th try, maximum 2 retries). Will delay 0.82s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e0-2740b42913a66f41197ac613;712e6faf-2f21-4690-8d9f-f0c02b05c3cc)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e1-1260467315b51c6c6131f6e4;eacb3b8c-f42c-42ef-aad5-4cead4f38635)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:21] retry() failed once (1th try, maximum 2 retries). Will delay 1.73s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e1-1260467315b51c6c6131f6e4;eacb3b8c-f42c-42ef-aad5-4cead4f38635)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e3-11abccdb6087f1556d31b129;5645acbe-06f0-4aa4-86af-f299bd751cd2)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:23] Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.


    [2026-04-09 10:05:25] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:25] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:05:25] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:05:26] Retrying in 2s [Retry 2/5].


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e8-2cb3202941615fe66d0976db;2825ef7c-87d0-4df6-be6e-3609e99dd136)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:28] retry() failed once (0th try, maximum 2 retries). Will delay 0.96s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e8-2cb3202941615fe66d0976db;2825ef7c-87d0-4df6-be6e-3609e99dd136)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:28] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:05:28] Retrying in 4s [Retry 3/5].


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e9-5fab7a837a4b26de2eefc72a;9aa0f320-74ae-486f-8ca7-ad7c0ac20956)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:29] retry() failed once (1th try, maximum 2 retries). Will delay 1.69s and retry. Error: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779e9-5fab7a837a4b26de2eefc72a;9aa0f320-74ae-486f-8ca7-ad7c0ac20956)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!


    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 761, in hf_raise_for_status
        response.raise_for_status()
      File "/usr/local/lib/python3.10/dist-packages/httpx/_models.py", line 749, in raise_for_status
        raise HTTPStatusError(message, request=request, response=self)
    httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json'
    For more information check: https://httpstatuses.com/503
    
    The above exception was the direct cause of the following exception:
    
    Traceback (most recent call last):
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py", line 2571, in retry
        return fn()
      File "/actions-runner/_work/sglang/sglang/python/sglang/srt/configs/model_config.py", line 749, in <lambda>
        lambda: hf_api.file_exists(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/hf_api.py", line 3436, in file_exists
        get_hf_file_metadata(url, token=token)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
        return fn(*args, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py", line 1576, in get_hf_file_metadata
        response = _httpx_follow_relative_redirects_with_backoff(
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 692, in _httpx_follow_relative_redirects_with_backoff
        hf_raise_for_status(response)
      File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_http.py", line 889, in hf_raise_for_status
        raise _format(HfHubHTTPError, str(e), response) from e
    huggingface_hub.errors.HfHubHTTPError: Server error '503 Service Unavailable' for url 'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/hf_quant_config.json' (Request ID: Root=1-69d779ea-2cf9a5d648492ddf232c5184;5a0343e5-970f-41d6-b7a0-839748ef93e2)
    For more information check: https://httpstatuses.com/503
    
    Internal Error - We're working hard to fix this as soon as possible!
    [2026-04-09 10:05:30] Failed to load hf_quant_config.json for model Qwen/Qwen2.5-VL-7B-Instruct: retry() exceed maximum number of retries.
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:31] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:05:31] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:32] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:05:32] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:32] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:05:32] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:34] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:05:34] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:38] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:05:38] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:40] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:05:40] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:05:46] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:48] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:48] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:05:48] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:49] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:05:49] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:52] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:05:52] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:54] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:54] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:05:54] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:55] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:05:55] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:56] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:05:56] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:05:57] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:05:57] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:01] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:01] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:04] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:09] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:09] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:12] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:12] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:12] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:13] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:15] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:15] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:17] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:18] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:18] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:19] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:19] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:21] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:21] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:25] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:25] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:27] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:27] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:33] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:33] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:35] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:37] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:37] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:39] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:39] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:06:41] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:41] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:41] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:42] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:06:42] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:43] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:43] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:44] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:06:44] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:48] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:06:48] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:51] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:51] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:56] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:06:56] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:06:59] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:06:59] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:06:59] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:00] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:00] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:02] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:02] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:07:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:05] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:05] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:06] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:06] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:08] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:08] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:12] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:12] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:14] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:14] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:20] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:20] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:22] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:22] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:22] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:23] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:23] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:26] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    [2026-04-09 10:07:28] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/audio_tokenizer_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:28] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:28] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:29] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:29] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:30] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:30] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:31] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:31] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:35] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:38] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:38] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:43] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:07:43] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:46] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:47] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:47] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:49] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:49] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:51] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:52] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:07:52] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:53] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:07:53] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:53] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:53] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:55] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:07:55] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:07:59] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:07:59] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:01] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:01] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:07] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:07] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:09] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:10] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:10] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:11] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:11] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:13] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:13] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:15] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:16] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:16] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:17] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:17] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:17] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:17] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:19] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:19] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:23] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:23] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:25] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:25] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:31] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:31] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:33] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:34] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:34] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:35] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:35] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:37] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:37] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:39] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:39] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:39] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:40] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:40] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:41] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:41] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:42] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:08:42] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:46] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:08:46] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:49] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:49] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:55] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:08:55] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:08:57] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:57] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:08:57] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:08:58] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:08:58] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:00] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:09:00] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:03] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:03] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:03] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 2s [Retry 2/5].
    [2026-04-09 10:09:04] Retrying in 2s [Retry 2/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:04] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:09:04] Retrying in 8s [Retry 4/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:06] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 4s [Retry 3/5].
    [2026-04-09 10:09:06] Retrying in 4s [Retry 3/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:10] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 4/5].
    [2026-04-09 10:09:10] Retrying in 8s [Retry 4/5].


    [2026-04-09 10:09:13] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:13] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:13] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:13] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:18] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 8s [Retry 5/5].
    [2026-04-09 10:09:18] Retrying in 8s [Retry 5/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:26] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json


    [2026-04-09 10:09:28] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-09 10:09:29] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.86it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:01,  1.66it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:01<00:01,  1.62it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.51it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:02<00:00,  1.86it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:02<00:00,  1.74it/s]


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    [2026-04-09 10:09:33] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.jinja
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:33] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:34] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:34] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:36] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:36] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    [2026-04-09 10:09:38] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/video_preprocessor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:38] Retrying in 1s [Retry 1/5].


    HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    [2026-04-09 10:09:39] HTTP Error 503 thrown while requesting HEAD https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/processor_config.json
    Retrying in 1s [Retry 1/5].
    [2026-04-09 10:09:39] Retrying in 1s [Retry 1/5].


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-09 10:09:43,447 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 10:09:43] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32354



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-09 10:09:47] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:799: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person standing on the back of a yellow taxi parked on a city street. The individual is using an iron and appears to be ironing a pair of pants that are laid out on a metal stand attached to the back of the taxi. The scene suggests a humorous or unusual situation, as ironing pants on the back of a taxi is not a practical activity. The background includes other vehicles, such as another taxi, and city buildings with visible storefronts and signage.</strong>



```python
terminate_process(server_process)
```
