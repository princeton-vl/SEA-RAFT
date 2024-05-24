import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

    return flow_loss