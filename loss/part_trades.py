import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from craft_ae import element_wise_clamp

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def part_trades_loss(model,
                     x_natural,
                     y,
                     optimizer,
                     weighted_eps,
                     step_size=2/255,
                     perturb_steps=10,
                     beta=1.0,
                     distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            eta = step_size * torch.sign(grad.detach())
            x_adv = Variable(x_adv.data + eta, requires_grad=True)
            eta = element_wise_clamp(x_adv.data - x_natural.data, weighted_eps)
            x_adv = Variable(x_natural.data + eta, requires_grad=True)
            x_adv = Variable(torch.clamp(x_adv, 0, 1.0), requires_grad=True)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
