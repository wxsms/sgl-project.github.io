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


    [2026-04-08 15:52:12] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:52:12] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:52:12] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:52:12] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.48it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.35it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.26it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.19it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.24it/s]


    2026-04-08 15:52:16,929 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 15:52:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:45,  2.91s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:03<01:20,  1.44s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:03<01:20,  1.44s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:03<00:48,  1.14it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:03<00:48,  1.14it/s]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:03<00:32,  1.67it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:23,  2.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:23,  2.27it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:17,  2.91it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:14,  3.64it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:14,  3.64it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:11,  4.42it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:09,  5.30it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:09,  5.30it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:09,  5.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:06,  6.96it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:05,  8.48it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04, 10.03it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04, 10.03it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04, 10.03it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:03, 11.95it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:03, 11.95it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:03, 11.95it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:03, 12.46it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:03, 12.46it/s]

    Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:03, 12.46it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:03, 12.15it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:03, 12.15it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:03, 12.15it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 13.08it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 13.08it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 13.08it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:05<00:02, 13.89it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:05<00:02, 13.89it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:05<00:02, 13.89it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 15.04it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 15.04it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 15.04it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 15.04it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 17.20it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 17.20it/s]

    Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:05<00:01, 17.65it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:06<00:01, 17.74it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:06<00:01, 17.74it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:06<00:01, 17.74it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 16.85it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 16.85it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 16.85it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 16.85it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:06<00:01, 17.50it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:06<00:01, 17.50it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:06<00:01, 17.50it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:06<00:00, 16.96it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:06<00:00, 16.96it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:06<00:00, 16.96it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 16.57it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 16.57it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 16.57it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:06<00:00, 17.32it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:06<00:00, 17.32it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:06<00:00, 17.32it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:06<00:00, 16.82it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:06<00:00, 16.82it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:06<00:00, 16.82it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:06<00:00, 16.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:06<00:00, 16.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:07<00:00, 16.35it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:07<00:00, 16.35it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:07<00:00, 17.84it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:07<00:00, 17.84it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:07<00:00, 17.84it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:07<00:00, 17.84it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:07<00:00, 18.72it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:07<00:00, 18.72it/s] 

    Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:07<00:00, 18.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.90it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.63 GB):   2%|▏         | 1/58 [00:01<00:58,  1.02s/it]Capturing num tokens (num_tokens=7680 avail_mem=26.60 GB):   2%|▏         | 1/58 [00:01<00:58,  1.02s/it]

    Capturing num tokens (num_tokens=7680 avail_mem=26.60 GB):   3%|▎         | 2/58 [00:01<00:52,  1.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=26.60 GB):   3%|▎         | 2/58 [00:01<00:52,  1.07it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=26.60 GB):   5%|▌         | 3/58 [00:02<00:47,  1.17it/s]Capturing num tokens (num_tokens=6656 avail_mem=26.60 GB):   5%|▌         | 3/58 [00:02<00:47,  1.17it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=26.60 GB):   7%|▋         | 4/58 [00:03<00:43,  1.25it/s]Capturing num tokens (num_tokens=6144 avail_mem=26.60 GB):   7%|▋         | 4/58 [00:03<00:43,  1.25it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=26.60 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=26.61 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=26.61 GB):  10%|█         | 6/58 [00:04<00:33,  1.56it/s]Capturing num tokens (num_tokens=5120 avail_mem=26.61 GB):  10%|█         | 6/58 [00:04<00:33,  1.56it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=26.61 GB):  12%|█▏        | 7/58 [00:04<00:28,  1.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.61 GB):  12%|█▏        | 7/58 [00:04<00:28,  1.81it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.61 GB):  14%|█▍        | 8/58 [00:05<00:23,  2.12it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.61 GB):  14%|█▍        | 8/58 [00:05<00:23,  2.12it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.61 GB):  16%|█▌        | 9/58 [00:05<00:19,  2.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.61 GB):  16%|█▌        | 9/58 [00:05<00:19,  2.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.61 GB):  17%|█▋        | 10/58 [00:05<00:16,  2.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.61 GB):  17%|█▋        | 10/58 [00:05<00:16,  2.96it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.61 GB):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.61 GB):  19%|█▉        | 11/58 [00:05<00:13,  3.50it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.61 GB):  21%|██        | 12/58 [00:06<00:13,  3.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.42 GB):  21%|██        | 12/58 [00:06<00:13,  3.45it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.42 GB):  22%|██▏       | 13/58 [00:06<00:12,  3.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.57 GB):  22%|██▏       | 13/58 [00:06<00:12,  3.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.57 GB):  24%|██▍       | 14/58 [00:06<00:11,  3.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.59 GB):  24%|██▍       | 14/58 [00:06<00:11,  3.83it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=25.59 GB):  26%|██▌       | 15/58 [00:06<00:10,  4.05it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.14 GB):  26%|██▌       | 15/58 [00:06<00:10,  4.05it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.14 GB):  28%|██▊       | 16/58 [00:06<00:10,  4.05it/s]Capturing num tokens (num_tokens=2048 avail_mem=24.47 GB):  28%|██▊       | 16/58 [00:06<00:10,  4.05it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=24.47 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.76it/s]Capturing num tokens (num_tokens=1792 avail_mem=24.47 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.76it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=24.47 GB):  31%|███       | 18/58 [00:07<00:10,  3.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.69 GB):  31%|███       | 18/58 [00:07<00:10,  3.89it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=25.69 GB):  33%|███▎      | 19/58 [00:07<00:10,  3.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=24.70 GB):  33%|███▎      | 19/58 [00:07<00:10,  3.89it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=24.70 GB):  34%|███▍      | 20/58 [00:08<00:09,  3.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=25.56 GB):  34%|███▍      | 20/58 [00:08<00:09,  3.98it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=25.56 GB):  36%|███▌      | 21/58 [00:08<00:08,  4.15it/s]Capturing num tokens (num_tokens=960 avail_mem=25.76 GB):  36%|███▌      | 21/58 [00:08<00:08,  4.15it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=25.76 GB):  38%|███▊      | 22/58 [00:08<00:08,  4.28it/s]Capturing num tokens (num_tokens=896 avail_mem=25.63 GB):  38%|███▊      | 22/58 [00:08<00:08,  4.28it/s]

    Capturing num tokens (num_tokens=896 avail_mem=25.63 GB):  40%|███▉      | 23/58 [00:08<00:07,  4.44it/s]Capturing num tokens (num_tokens=832 avail_mem=24.90 GB):  40%|███▉      | 23/58 [00:08<00:07,  4.44it/s]

    Capturing num tokens (num_tokens=832 avail_mem=24.90 GB):  41%|████▏     | 24/58 [00:08<00:07,  4.57it/s]Capturing num tokens (num_tokens=768 avail_mem=25.69 GB):  41%|████▏     | 24/58 [00:08<00:07,  4.57it/s]Capturing num tokens (num_tokens=768 avail_mem=25.69 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.81it/s]Capturing num tokens (num_tokens=704 avail_mem=25.02 GB):  43%|████▎     | 25/58 [00:09<00:06,  4.81it/s]

    Capturing num tokens (num_tokens=704 avail_mem=25.02 GB):  45%|████▍     | 26/58 [00:09<00:06,  4.79it/s]Capturing num tokens (num_tokens=640 avail_mem=25.88 GB):  45%|████▍     | 26/58 [00:09<00:06,  4.79it/s]Capturing num tokens (num_tokens=640 avail_mem=25.88 GB):  47%|████▋     | 27/58 [00:09<00:05,  5.26it/s]Capturing num tokens (num_tokens=576 avail_mem=25.16 GB):  47%|████▋     | 27/58 [00:09<00:05,  5.26it/s]

    Capturing num tokens (num_tokens=576 avail_mem=25.16 GB):  48%|████▊     | 28/58 [00:09<00:05,  5.02it/s]Capturing num tokens (num_tokens=512 avail_mem=25.16 GB):  48%|████▊     | 28/58 [00:09<00:05,  5.02it/s]Capturing num tokens (num_tokens=512 avail_mem=25.16 GB):  50%|█████     | 29/58 [00:09<00:05,  5.61it/s]Capturing num tokens (num_tokens=480 avail_mem=26.54 GB):  50%|█████     | 29/58 [00:09<00:05,  5.61it/s]

    Capturing num tokens (num_tokens=480 avail_mem=26.54 GB):  52%|█████▏    | 30/58 [00:09<00:04,  5.62it/s]Capturing num tokens (num_tokens=448 avail_mem=25.29 GB):  52%|█████▏    | 30/58 [00:09<00:04,  5.62it/s]Capturing num tokens (num_tokens=448 avail_mem=25.29 GB):  53%|█████▎    | 31/58 [00:10<00:04,  5.81it/s]Capturing num tokens (num_tokens=416 avail_mem=26.04 GB):  53%|█████▎    | 31/58 [00:10<00:04,  5.81it/s]

    Capturing num tokens (num_tokens=416 avail_mem=26.04 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.91it/s]Capturing num tokens (num_tokens=384 avail_mem=25.38 GB):  55%|█████▌    | 32/58 [00:10<00:04,  5.91it/s]Capturing num tokens (num_tokens=384 avail_mem=25.38 GB):  57%|█████▋    | 33/58 [00:10<00:04,  6.11it/s]Capturing num tokens (num_tokens=352 avail_mem=25.40 GB):  57%|█████▋    | 33/58 [00:10<00:04,  6.11it/s]

    Capturing num tokens (num_tokens=352 avail_mem=25.40 GB):  59%|█████▊    | 34/58 [00:10<00:03,  6.17it/s]Capturing num tokens (num_tokens=320 avail_mem=26.53 GB):  59%|█████▊    | 34/58 [00:10<00:03,  6.17it/s]Capturing num tokens (num_tokens=320 avail_mem=26.53 GB):  60%|██████    | 35/58 [00:10<00:03,  6.56it/s]Capturing num tokens (num_tokens=288 avail_mem=25.50 GB):  60%|██████    | 35/58 [00:10<00:03,  6.56it/s]

    Capturing num tokens (num_tokens=288 avail_mem=25.50 GB):  62%|██████▏   | 36/58 [00:10<00:03,  6.60it/s]Capturing num tokens (num_tokens=256 avail_mem=26.53 GB):  62%|██████▏   | 36/58 [00:10<00:03,  6.60it/s]Capturing num tokens (num_tokens=256 avail_mem=26.53 GB):  64%|██████▍   | 37/58 [00:10<00:03,  6.96it/s]Capturing num tokens (num_tokens=240 avail_mem=25.60 GB):  64%|██████▍   | 37/58 [00:10<00:03,  6.96it/s]

    Capturing num tokens (num_tokens=240 avail_mem=25.60 GB):  66%|██████▌   | 38/58 [00:11<00:02,  7.14it/s]Capturing num tokens (num_tokens=224 avail_mem=26.52 GB):  66%|██████▌   | 38/58 [00:11<00:02,  7.14it/s]Capturing num tokens (num_tokens=224 avail_mem=26.52 GB):  67%|██████▋   | 39/58 [00:11<00:02,  7.59it/s]Capturing num tokens (num_tokens=208 avail_mem=25.65 GB):  67%|██████▋   | 39/58 [00:11<00:02,  7.59it/s]

    Capturing num tokens (num_tokens=208 avail_mem=25.65 GB):  69%|██████▉   | 40/58 [00:11<00:02,  7.59it/s]Capturing num tokens (num_tokens=192 avail_mem=26.51 GB):  69%|██████▉   | 40/58 [00:11<00:02,  7.59it/s]Capturing num tokens (num_tokens=192 avail_mem=26.51 GB):  71%|███████   | 41/58 [00:11<00:02,  7.75it/s]Capturing num tokens (num_tokens=176 avail_mem=26.05 GB):  71%|███████   | 41/58 [00:11<00:02,  7.75it/s]

    Capturing num tokens (num_tokens=176 avail_mem=26.05 GB):  72%|███████▏  | 42/58 [00:11<00:01,  8.17it/s]Capturing num tokens (num_tokens=160 avail_mem=26.21 GB):  72%|███████▏  | 42/58 [00:11<00:01,  8.17it/s]Capturing num tokens (num_tokens=160 avail_mem=26.21 GB):  74%|███████▍  | 43/58 [00:11<00:01,  8.59it/s]Capturing num tokens (num_tokens=144 avail_mem=26.07 GB):  74%|███████▍  | 43/58 [00:11<00:01,  8.59it/s]

    Capturing num tokens (num_tokens=144 avail_mem=26.07 GB):  76%|███████▌  | 44/58 [00:11<00:01,  8.09it/s]Capturing num tokens (num_tokens=128 avail_mem=26.52 GB):  76%|███████▌  | 44/58 [00:11<00:01,  8.09it/s]Capturing num tokens (num_tokens=112 avail_mem=25.88 GB):  76%|███████▌  | 44/58 [00:11<00:01,  8.09it/s]Capturing num tokens (num_tokens=112 avail_mem=25.88 GB):  79%|███████▉  | 46/58 [00:11<00:01,  9.02it/s]Capturing num tokens (num_tokens=96 avail_mem=26.50 GB):  79%|███████▉  | 46/58 [00:11<00:01,  9.02it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=26.13 GB):  79%|███████▉  | 46/58 [00:12<00:01,  9.02it/s]Capturing num tokens (num_tokens=80 avail_mem=26.13 GB):  83%|████████▎ | 48/58 [00:12<00:01,  9.46it/s]Capturing num tokens (num_tokens=64 avail_mem=26.49 GB):  83%|████████▎ | 48/58 [00:12<00:01,  9.46it/s]Capturing num tokens (num_tokens=48 avail_mem=26.01 GB):  83%|████████▎ | 48/58 [00:12<00:01,  9.46it/s]

    Capturing num tokens (num_tokens=48 avail_mem=26.01 GB):  86%|████████▌ | 50/58 [00:12<00:00, 10.54it/s]Capturing num tokens (num_tokens=32 avail_mem=26.36 GB):  86%|████████▌ | 50/58 [00:12<00:00, 10.54it/s]Capturing num tokens (num_tokens=28 avail_mem=26.48 GB):  86%|████████▌ | 50/58 [00:12<00:00, 10.54it/s]Capturing num tokens (num_tokens=28 avail_mem=26.48 GB):  90%|████████▉ | 52/58 [00:12<00:00, 10.93it/s]Capturing num tokens (num_tokens=24 avail_mem=26.21 GB):  90%|████████▉ | 52/58 [00:12<00:00, 10.93it/s]

    Capturing num tokens (num_tokens=20 avail_mem=26.46 GB):  90%|████████▉ | 52/58 [00:12<00:00, 10.93it/s]Capturing num tokens (num_tokens=20 avail_mem=26.46 GB):  93%|█████████▎| 54/58 [00:12<00:00, 11.21it/s]Capturing num tokens (num_tokens=16 avail_mem=26.46 GB):  93%|█████████▎| 54/58 [00:12<00:00, 11.21it/s]Capturing num tokens (num_tokens=12 avail_mem=26.15 GB):  93%|█████████▎| 54/58 [00:12<00:00, 11.21it/s]

    Capturing num tokens (num_tokens=12 avail_mem=26.15 GB):  97%|█████████▋| 56/58 [00:12<00:00, 12.47it/s]Capturing num tokens (num_tokens=8 avail_mem=26.35 GB):  97%|█████████▋| 56/58 [00:12<00:00, 12.47it/s] Capturing num tokens (num_tokens=4 avail_mem=26.42 GB):  97%|█████████▋| 56/58 [00:12<00:00, 12.47it/s]Capturing num tokens (num_tokens=4 avail_mem=26.42 GB): 100%|██████████| 58/58 [00:12<00:00, 12.98it/s]Capturing num tokens (num_tokens=4 avail_mem=26.42 GB): 100%|██████████| 58/58 [00:12<00:00,  4.48it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:31263


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-08 15:52:45] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Certainly! Here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Germany - Berlin<br>3. Spain - Madrid</strong>


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


<strong style='color: #00008B;'>Certainly! Here's a list of three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Canada** - Ottawa</strong>



<strong style='color: #00008B;'>Of course! Here's another list of three countries with their capitals:<br><br>1. **Germany** - Berlin<br>2. **India** - New Delhi<br>3. **Mexico** - Mexico City</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>Let's solve it:<br><br>2 * 2 = 4<br><br>You don't actually need a calculator for this simple multiplication, as it's a straightforward calculation. If you want to verify it or need assistance with more complex calculations, a calculator can be helpful.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: Ensure you consume a variety of foods from different food groups in appropriate proportions. This includes:<br>   - Fruits and vegetables: Provide essential vitamins, minerals, and fiber.<br>   - Whole grains: Offer important dietary fiber and nutrients.<br>   - Lean proteins: Include sources like chicken, fish, beans, and lentils.<br>   - Healthy fats: Include sources like avocados, nuts, seeds, and olive oil.<br><br>2. **Regular Exercise**: Engage in physical activity regularly to improve your health. This should include:<br>   - Aerobic activities: Moderate (e.g., brisk walking, cycling) or vigorous (e.g., running, interval training).<br>   - Strength training: At least two days a week to build muscles and boost metabolism.<br>   - Aim for a total of at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous aerobic activity per week, as recommended by the World Health Organization.<br><br>By combining a balanced diet and regular exercise, you can enhance your overall health and well-being.</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "willow",<br>        "core": "peacock feather",<br>        "length": 10.5<br>    },<br>    "alive": "Alive",<br>    "patronus": " stag",<br>    "bogart": "dementor"<br>}</strong>


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

     33%|███▎      | 1/3 [00:00<00:00,  4.76it/s]

     67%|██████▋   | 2/3 [00:00<00:00,  7.56it/s]

    100%|██████████| 3/3 [00:00<00:00, 12.03it/s]

    



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


    [2026-04-08 15:53:03] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-08 15:53:08] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:08] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:08] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:08] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [2026-04-08 15:53:15] Ignore import error when loading sglang.srt.multimodal.processors.gemma4: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [2026-04-08 15:53:16] Ignore import error when loading sglang.srt.models.gemma4_audio: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:16] Ignore import error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:16] Ignore import error when loading sglang.srt.models.gemma4_mm: cannot import name 'Gemma4AudioConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
    [2026-04-08 15:53:16] Ignore import error when loading sglang.srt.models.gemma4_vision: cannot import name 'Gemma4VisionConfig' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:00,  4.87it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:00<00:01,  2.38it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:01<00:01,  1.80it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.48it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.06it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.32it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-04-08 15:53:27,864 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 15:53:27] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:32710



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-04-08 15:53:31] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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



<strong style='color: #00008B;'>The image shows a person standing on the rear of a yellow car, using an ironing board to iron a shirt. The person is wearing a bright yellow-orange shirt. The car appears to be in a busy urban area, with a yellow taxi to the left and a multi-story commercial or office building behind. The car also displays some text, possibly related to a service, but the text is not entirely clear. The person seems to be playing a lighthearted or humorous scene, given the unusual manner in which they are performing the task of ironing the shirt.</strong>



```python
terminate_process(server_process)
```
