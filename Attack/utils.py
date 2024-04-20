import torch

OF_mean = [0.48145466, 0.4578275, 0.40821073]
OF_std = [0.26862954, 0.26130258, 0.27577711]

def denormalize(tensor, mean = OF_mean, std = OF_std):
    mean = torch.tensor(mean, dtype=tensor.dtype).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype).reshape(1, 3, 1, 1).to(tensor.device)
    denormalized_tensor = tensor * std + mean
    return denormalized_tensor.clamp(0,1)

def normalize(tensor, mean = OF_mean, std = OF_std):
    mean = torch.tensor(mean, dtype=tensor.dtype).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype).reshape(1, 3, 1, 1).to(tensor.device)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def normalize_noise(tensor, std = OF_std):
    std = torch.tensor(std, dtype=tensor.dtype).reshape(1, 3, 1, 1).to(tensor.device)
    normalized_tensor = tensor / std
    return normalized_tensor