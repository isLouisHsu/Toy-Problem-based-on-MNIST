import numpy as np

import torch
import torch.nn.functional as F

def analyse_margin_angular(ckptpath):

    state = torch.load(ckptpath)['net_state']
    weights = state[cosine_layer.weights]

    weights = F.normalize(weights)

    cosine = weights.mm(weights.t())
    angular = torch.acos(cosine) / np.pi * 180

    return cosine, angular