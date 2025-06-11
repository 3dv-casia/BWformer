
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np



class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 5):
        super().__init__()
        self.cost_3dcenter = cost_3dcenter
        self.cost_class = cost_class
        assert cost_class != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, corner_coord, corner_logits, targets):
        bs, num_queries = corner_coord.shape[:2]

        out_3dcenter = corner_coord.flatten(0, 1)  
        tgt_3dcenter = torch.cat([v['coords'] for v in targets])
        out_prob = corner_logits.flatten(0,1).sigmoid()
        tgt_prob = torch.cat([v['labels'] for v in targets])


        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)
        cost_class = torch.cdist(out_prob, tgt_prob, p=1)
        C = self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['coords']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
       
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return HungarianMatcher(
        cost_3dcenter=5)