import torch 
from torch import nn 

sim = nn.CosineSimilarity()
def contrastive_loss(x1, x2, tau=0.25, assume_normed=False):
    '''
    Samples from x1 are the same class, x2 are different

    impliments: 

                    exp{sim(x_i, x_j) / tau)}
    l_{i,j} = -log ---------------------------
                    Sum_{k=1} exp(sim(x_i, x_k) / tau)

            = -(sim(x_i, x_j)/ tau) + log(sum(exp(sim(x_i,x_k)))
    '''

    pos = sim(x1, x1[torch.randperm(x1.size(0))])/tau 
    
    # Dot product
    neg_non_norm = (x1.unsqueeze(1) @ x2.T).squeeze(1)

    # If the inputs are already normed, we can skip this part
    if not assume_normed:
        norm = x1.norm(dim=1,keepdim=True) @ x2.norm(dim=1,keepdim=True).T
        neg = neg_non_norm / (norm+1e9)
    else:
        neg = neg_non_norm

    return (-pos + torch.logsumexp(neg, dim=1)).mean()