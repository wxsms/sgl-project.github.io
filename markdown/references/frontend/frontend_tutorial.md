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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:02<00:06,  2.08s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:03,  1.68s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:05<00:01,  1.63s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:06<00:00,  1.59s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:06<00:00,  1.65s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<05:49,  6.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<05:49,  6.12s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:36,  2.80s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:36,  2.80s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:34,  1.72s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:34,  1.72s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:04,  1.19s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:04,  1.19s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:36,  1.43it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:36,  1.43it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:28,  1.78it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:23,  2.13it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:23,  2.13it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:19,  2.54it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:19,  2.54it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:16,  2.97it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:16,  2.97it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:13,  3.39it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:13,  3.39it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:11,  3.86it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:11,  3.86it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:10,  4.31it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:10,  4.31it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.82it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:07,  5.46it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:07,  5.46it/s]

    Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:06,  6.12it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:06,  6.12it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:05,  6.84it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:05,  6.84it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:10<00:05,  6.84it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:04,  8.54it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:04,  8.54it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:10<00:04,  8.54it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:03, 10.25it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:03, 10.25it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:03, 10.25it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:10<00:02, 12.34it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:10<00:02, 12.34it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:10<00:02, 12.34it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:10<00:02, 12.34it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:10<00:01, 16.04it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:10<00:01, 16.04it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:10<00:01, 16.04it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:10<00:01, 16.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:01, 19.06it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:01, 19.06it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:01, 19.06it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 19.06it/s]

    Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:10<00:01, 19.06it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:10<00:01, 23.02it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:11<00:00, 28.49it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:11<00:00, 33.37it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:11<00:00, 36.65it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:11<00:00, 39.67it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.25 GB):   2%|▏         | 1/58 [00:00<00:35,  1.60it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.33 GB):   2%|▏         | 1/58 [00:00<00:35,  1.60it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.33 GB):   3%|▎         | 2/58 [00:01<00:33,  1.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.39 GB):   3%|▎         | 2/58 [00:01<00:33,  1.69it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.39 GB):   5%|▌         | 3/58 [00:01<00:30,  1.82it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.46 GB):   5%|▌         | 3/58 [00:01<00:30,  1.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.46 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.53 GB):   7%|▋         | 4/58 [00:02<00:27,  1.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.53 GB):   9%|▊         | 5/58 [00:02<00:25,  2.11it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.59 GB):   9%|▊         | 5/58 [00:02<00:25,  2.11it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.59 GB):  10%|█         | 6/58 [00:02<00:22,  2.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.65 GB):  10%|█         | 6/58 [00:02<00:22,  2.30it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.65 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.72 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.49it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.72 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.74it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.79 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.79 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.82 GB):  16%|█▌        | 9/58 [00:03<00:16,  3.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.82 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.84 GB):  17%|█▋        | 10/58 [00:04<00:14,  3.24it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.84 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.87 GB):  19%|█▉        | 11/58 [00:04<00:13,  3.52it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.87 GB):  21%|██        | 12/58 [00:04<00:11,  3.87it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.90 GB):  21%|██        | 12/58 [00:04<00:11,  3.87it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.90 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.97 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.97 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.29 GB):  24%|██▍       | 14/58 [00:04<00:09,  4.67it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.29 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.29 GB):  26%|██▌       | 15/58 [00:04<00:08,  5.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.29 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.28 GB):  28%|██▊       | 16/58 [00:05<00:07,  5.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=44.28 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.28 GB):  29%|██▉       | 17/58 [00:05<00:06,  6.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.28 GB):  31%|███       | 18/58 [00:05<00:05,  7.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.27 GB):  31%|███       | 18/58 [00:05<00:05,  7.01it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.26 GB):  31%|███       | 18/58 [00:05<00:05,  7.01it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.26 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.24 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s]Capturing num tokens (num_tokens=960 avail_mem=44.24 GB):  34%|███▍      | 20/58 [00:05<00:04,  8.61it/s] Capturing num tokens (num_tokens=960 avail_mem=44.24 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.23it/s]Capturing num tokens (num_tokens=896 avail_mem=44.23 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.23it/s]

    Capturing num tokens (num_tokens=832 avail_mem=44.22 GB):  38%|███▊      | 22/58 [00:05<00:03, 10.23it/s]Capturing num tokens (num_tokens=832 avail_mem=44.22 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.69it/s]Capturing num tokens (num_tokens=768 avail_mem=44.21 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.69it/s]Capturing num tokens (num_tokens=704 avail_mem=44.21 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.69it/s]Capturing num tokens (num_tokens=704 avail_mem=44.21 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.14it/s]Capturing num tokens (num_tokens=640 avail_mem=44.20 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.14it/s]

    Capturing num tokens (num_tokens=576 avail_mem=44.19 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.14it/s]Capturing num tokens (num_tokens=576 avail_mem=44.19 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.47it/s]Capturing num tokens (num_tokens=512 avail_mem=44.18 GB):  48%|████▊     | 28/58 [00:05<00:02, 14.47it/s]Capturing num tokens (num_tokens=480 avail_mem=44.18 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.47it/s]Capturing num tokens (num_tokens=448 avail_mem=44.17 GB):  48%|████▊     | 28/58 [00:06<00:02, 14.47it/s]

    Capturing num tokens (num_tokens=448 avail_mem=44.17 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=416 avail_mem=44.16 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=384 avail_mem=44.15 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=352 avail_mem=44.14 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.27it/s]Capturing num tokens (num_tokens=352 avail_mem=44.14 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.20it/s]Capturing num tokens (num_tokens=320 avail_mem=44.15 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.20it/s]Capturing num tokens (num_tokens=288 avail_mem=44.11 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.20it/s]

    Capturing num tokens (num_tokens=256 avail_mem=44.10 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.20it/s]Capturing num tokens (num_tokens=256 avail_mem=44.10 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.78it/s]Capturing num tokens (num_tokens=240 avail_mem=44.09 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.78it/s]Capturing num tokens (num_tokens=224 avail_mem=44.09 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.78it/s]Capturing num tokens (num_tokens=208 avail_mem=44.08 GB):  64%|██████▍   | 37/58 [00:06<00:01, 19.78it/s]Capturing num tokens (num_tokens=208 avail_mem=44.08 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.73it/s]Capturing num tokens (num_tokens=192 avail_mem=44.09 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.73it/s]Capturing num tokens (num_tokens=176 avail_mem=44.09 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.73it/s]

    Capturing num tokens (num_tokens=160 avail_mem=44.09 GB):  69%|██████▉   | 40/58 [00:06<00:00, 21.73it/s]Capturing num tokens (num_tokens=160 avail_mem=44.09 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.54it/s]Capturing num tokens (num_tokens=144 avail_mem=44.09 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.54it/s]Capturing num tokens (num_tokens=128 avail_mem=44.09 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.54it/s]Capturing num tokens (num_tokens=112 avail_mem=44.08 GB):  74%|███████▍  | 43/58 [00:06<00:00, 23.54it/s]Capturing num tokens (num_tokens=112 avail_mem=44.08 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.65it/s]Capturing num tokens (num_tokens=96 avail_mem=44.05 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.65it/s] Capturing num tokens (num_tokens=80 avail_mem=44.04 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.65it/s]

    Capturing num tokens (num_tokens=64 avail_mem=44.04 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.65it/s]Capturing num tokens (num_tokens=64 avail_mem=44.04 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.66it/s]Capturing num tokens (num_tokens=48 avail_mem=44.03 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.66it/s]Capturing num tokens (num_tokens=32 avail_mem=44.05 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.66it/s]Capturing num tokens (num_tokens=28 avail_mem=44.04 GB):  84%|████████▍ | 49/58 [00:06<00:00, 25.66it/s]Capturing num tokens (num_tokens=28 avail_mem=44.04 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.65it/s]Capturing num tokens (num_tokens=24 avail_mem=44.03 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.65it/s]Capturing num tokens (num_tokens=20 avail_mem=44.03 GB):  90%|████████▉ | 52/58 [00:06<00:00, 26.65it/s]

    Capturing num tokens (num_tokens=16 avail_mem=44.02 GB):  90%|████████▉ | 52/58 [00:07<00:00, 26.65it/s]Capturing num tokens (num_tokens=16 avail_mem=44.02 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.53it/s]Capturing num tokens (num_tokens=12 avail_mem=44.01 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.53it/s]Capturing num tokens (num_tokens=8 avail_mem=44.00 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.53it/s] Capturing num tokens (num_tokens=4 avail_mem=43.99 GB):  95%|█████████▍| 55/58 [00:07<00:00, 27.53it/s]Capturing num tokens (num_tokens=4 avail_mem=43.99 GB): 100%|██████████| 58/58 [00:07<00:00,  8.14it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32537


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-30 00:10:06] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure! Here are three countries along with their capitals:<br><br>1. **Germany** - Berlin<br>2. **France** - Paris<br>3. **Japan** - Tokyo</strong>


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


<strong style='color: #00008B;'>Sure! Here's a list of three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Japan** - Tokyo</strong>



<strong style='color: #00008B;'>Certainly! Here’s another list of three countries along with their capitals:<br><br>1. **Italy** - Rome<br>2. **Spain** - Madrid<br>3. **Canada** - Ottawa</strong>


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



<strong style='color: #00008B;'>2 * 2. <br><br>When you multiply 2 by 2, you get:<br><br>2 * 2 = 4<br><br>So, the answer is 4. You don't actually need a calculator for this simple multiplication, but if you prefer to use one, you can certainly do so.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: A balanced diet is essential for overall health. It should include a variety of nutrients from fruits, vegetables, lean proteins, whole grains, and healthy fats. Limiting processed foods, sugars, and saturated fats can help prevent chronic diseases like heart disease, diabetes, and certain types of cancer. Staying hydrated is also key.<br><br>2. **Regular Exercise**: Regular physical activity is crucial for maintaining good health. It improves cardiovascular health, enhances muscle strength and flexibility, boosts metabolism, and reduces stress and anxiety. Regular exercise can help prevent diseases like type 2 diabetes, hypertension, and obesity. Incorporating a variety of activities like aerobic exercises, strength training, and flexibility exercises into your routine is recommended. Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, as advised by health authorities.<br><br>By combining these two practices, you can significantly enhance your physical and mental well-being.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "holly",<br>        "core": "phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "his father"<br>}</strong>


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

     33%|███▎      | 1/3 [00:00<00:00,  9.89it/s]

     67%|██████▋   | 2/3 [00:00<00:00, 15.62it/s]

    100%|██████████| 3/3 [00:00<00:00, 25.13it/s]

    



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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-30 00:10:26] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-30 00:10:32] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:03,  1.03it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.11it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.20it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:03<00:00,  1.26it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.61it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.38it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32820



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-30 00:10:50] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:894: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a yellow taxi in a busy urban area. A man is leaning over from the rear of the taxi, using an umbrella stand as an ironing board to iron a pair of Blue Cross Blue Shield clothing items. The man is wearing a yellow shirt with the same logo as the clothing he is ironing. The scene likely takes place in New York City, as suggested by the "Personal Taxi" sign and the cleanliness and style of the urban environment.</strong>



```python
terminate_process(server_process)
```
