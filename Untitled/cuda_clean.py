import torch
for gpu in range(torch.cuda.device_count()):
    with torch.cuda.device(gpu):
        torch.cuda.empty_cache()