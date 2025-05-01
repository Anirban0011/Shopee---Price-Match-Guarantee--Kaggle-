import torch

class CFG:
    r'''
    default config class
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 5
    fold_id = 0
    init_lr = 3e-4