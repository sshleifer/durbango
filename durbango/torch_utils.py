import torch
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def num_parameters(module, only_trainable: bool = False) -> int:
    """
    Get number of (optionally, trainable) parameters in the module.
    """
    params = filter(lambda x: x.requires_grad, module.parameters()) if only_trainable else module.parameters()
    return sum(p.numel() for p in params)
