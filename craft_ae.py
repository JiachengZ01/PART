from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torchattacks import PGD, AutoAttack

def element_wise_clamp(eta, epsilon):
    # Element-wise clamp using the epsilon tensor
    eta_clamped = torch.where(eta > epsilon, epsilon, eta)
    eta_clamped = torch.where(eta < -epsilon, -epsilon, eta_clamped)
    return eta_clamped

def craft_adversarial_example(model, 
                              x_natural, 
                              y,
                              step_size=2/255, 
                              epsilon=8/255, 
                              perturb_steps=10,
                              num_classes=10,
                              mode='pgd'):
    if mode == 'pgd':
        attack = PGD(model, 
                     eps=epsilon, 
                     alpha=step_size, 
                     steps=perturb_steps, 
                     random_start=True)
    elif mode == 'aa':
        attack = AutoAttack(model, 
                            norm='Linf', 
                            eps=epsilon, 
                            version='standard')
    if mode == 'mma':
        x_adv = mma(model,
                    data=x_natural,
                    target=y,
                    epsilon=epsilon,
                    step_size=step_size,
                    num_steps=perturb_steps,
                    category='Madry',
                    rand_init=True,
                    k=3,
                    num_classes=num_classes)
    else:
        x_adv = attack(x_natural, y)
    return x_adv

def part_pgd(model,
             X,
             y,
             weighted_eps,
             epsilon=8/255,
             num_steps=10,
             step_size=2/255):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = element_wise_clamp(X_pgd.data - X.data, weighted_eps)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def part_mma(model, 
             data, 
             target, 
             weighted_eps, 
             epsilon, 
             step_size, 
             num_steps, 
             rand_init, 
             k, 
             num_classes):
    model.eval()
    x_adv = data.detach() + torch.from_numpy(
        np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    logits = model(data)
    target_onehot = torch.zeros(target.size() + (len(logits[0]),))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    index = torch.argsort(logits - 10000 * target_var)[:, num_classes - k:]
    x_adv_set = []
    loss_set = []

    for i in range(k):
        x_adv_0 = x_adv.clone().detach()
        for j in range(num_steps):
            x_adv_0.requires_grad_()
            output1 = model(x_adv_0)
            model.zero_grad()
            with torch.enable_grad():
                loss_adv0 = mm_loss(output1, target, index[:, i], num_classes=num_classes)
            loss_adv0.backward()
            eta = step_size * x_adv_0.grad.sign()
            x_adv_0 = x_adv_0.detach() + eta
            eta = element_wise_clamp(x_adv_0 - data, weighted_eps)
            x_adv_0 = data + eta
            x_adv_0 = torch.clamp(x_adv_0, 0.0, 1.0)

        pipy = mm_loss_train(model(x_adv_0), target, index[:, i], num_classes=num_classes)
        loss_set.append(pipy.view(len(pipy), -1))
        x_adv_set.append(x_adv_0)

    loss_pipy = loss_set[0]
    for i in range(k - 1):
        loss_pipy = torch.cat((loss_pipy, loss_set[i + 1]), 1)

    index_choose = torch.argsort(loss_pipy)[:, -1]

    adv_final = torch.zeros(x_adv.size()).cuda()
    for i in range(len(index_choose)):
        adv_final[i, :, :, :] = x_adv_set[index_choose[i]][i]

    return adv_final

def mma(model, 
        data, 
        target, 
        epsilon, 
        step_size, 
        num_steps, 
        category, 
        rand_init, 
        k, 
        num_classes):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    logits = model(data)
    target_onehot = torch.zeros(target.size() + (len(logits[0]),))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    index = torch.argsort(logits - 10000 * target_var)[:, num_classes - k:]

    x_adv_set = []
    loss_set = []
    for i in range(k):
        x_adv_0 = x_adv.clone().detach()
        for j in range(num_steps):
            x_adv_0.requires_grad_()
            output1 = model(x_adv_0)
            model.zero_grad()
            with torch.enable_grad():
                loss_adv0 = mm_loss(output1, target, index[:, i], num_classes=num_classes)
            loss_adv0.backward()
            eta = step_size * x_adv_0.grad.sign()
            x_adv_0 = x_adv_0.detach() + eta
            x_adv_0 = torch.min(torch.max(x_adv_0, data - epsilon), data + epsilon)
            x_adv_0 = torch.clamp(x_adv_0, 0.0, 1.0)

        pipy = mm_loss_train(model(x_adv_0), target, index[:, i], num_classes=num_classes)
        loss_set.append(pipy.view(len(pipy), -1))
        x_adv_set.append(x_adv_0)

    loss_pipy = loss_set[0]
    for i in range(k - 1):
        loss_pipy = torch.cat((loss_pipy, loss_set[i + 1]), 1)

    index_choose = torch.argsort(loss_pipy)[:, -1]

    adv_final = torch.zeros(x_adv.size()).cuda()
    for i in range(len(index_choose)):
        adv_final[i, :, :, :] = x_adv_set[index_choose[i]][i]

    return adv_final

# loss for MM AT
def mm_loss_train(output, target, target_choose, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    return other-real

# loss for MM Attack
def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss