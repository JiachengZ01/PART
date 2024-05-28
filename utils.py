import torch
from torchcam.methods import GradCAM, XGradCAM, LayerCAM
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
import numpy as np
import random
from craft_ae import *
from loss.mart import *
from loss.trades import *
from loss.part_mart import *
from loss.part_trades import *

def parse_fraction(fraction_string):
    if '/' in fraction_string:
        numerator, denominator = fraction_string.split('/')
        return float(numerator) / float(denominator)
    return float(fraction_string)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(args, optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= args.adjust_second:
        lr = args.lr * 0.01
    elif epoch >= args.adjust_first:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def craft_weight_matrix(model, data, device, args, parallel=True):
    batch, img_size1, img_size2 = data.shape[0], data.shape[-2], data.shape[-1]
    weight_matrix_tensor = torch.empty(batch, 3, img_size1, img_size2).to(device)

    if args.model == 'resnet':
        if args.cam == 'gradcam':
            cam_extractor = GradCAM(model.module if parallel else model, 'layer4')
        if args.cam == 'xgradcam':
            cam_extractor = XGradCAM(model.module if parallel else model, 'layer4')
        if args.cam == 'layercam':
            cam_extractor = LayerCAM(model.module if parallel else model, 'layer4')
    elif args.model == 'wideresnet':
        if args.cam == 'gradcam':
            cam_extractor = GradCAM(model.module if parallel else model, 'block3')
        if args.cam == 'xgradcam':
            cam_extractor = XGradCAM(model.module if parallel else model, 'block3')
        if args.cam == 'layercam':
            cam_extractor = LayerCAM(model.module if parallel else model, 'block3')

    for i in range(batch):
        output = model(data[i].unsqueeze(0))
        heatmap = cam_extractor(output.argmax().item(), output)
        mask = to_pil_image(heatmap[0].squeeze(0).cpu().numpy())
        overlay = mask.resize((img_size1, img_size2), resample=Image.BICUBIC)
        cmap_overlay = cm.get_cmap('jet')(np.asarray(overlay) ** 2)
        weight_matrix_tensor[i] = process_overlay(cmap_overlay, device)

    cam_extractor.remove_hooks()

    return weight_matrix_tensor

def process_overlay(overlay, device):
    overlay = (255 * overlay[:, :, :3]).astype(np.double)
    normalized_overlay = overlay / 255
    mean, std = np.mean(normalized_overlay), np.std(normalized_overlay)
    weight_matrix = torch.from_numpy((normalized_overlay - mean) / std)
    return torch.clamp(weight_matrix, 1, weight_matrix.max()).float().permute(2, 0, 1).to(device)

def generate_weighted_eps(weight_matrix, args):
    epsilon = torch.where(weight_matrix > 1, args.epsilon, args.low_epsilon)
    return epsilon

def standard_train(args, model, device, train_loader, optimizer, epoch):

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        # calculate robust loss
        model.eval()
        if args.attack == 'pgd':
            data = craft_adversarial_example(model=model, 
                                             x_natural=data, 
                                             y=label, 
                                             step_size=args.step_size,
                                             epsilon=args.epsilon, 
                                             perturb_steps=args.num_steps,
                                             num_classes=args.num_class,
                                             mode='pgd')
        elif args.attack == 'mma':
            data = craft_adversarial_example(model=model, 
                                             x_natural=data, 
                                             y=label, 
                                             step_size=args.step_size,
                                             epsilon=args.epsilon, 
                                             perturb_steps=args.num_steps,
                                             num_classes=args.num_class,
                                             mode='mma')

        model.train()
        optimizer.zero_grad()

        logits_out = model(data)
        loss = F.cross_entropy(logits_out, label)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))

def mart_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = mart_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            
def trades_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            
def eval_test(args, model, device, test_loader, mode='pgd'):
    model.eval()
    correct = 0
    correct_adv = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data = craft_adversarial_example(model=model, 
                                         x_natural=data, 
                                         y=label,
                                         step_size=args.step_size, 
                                         epsilon=8/255, 
                                         perturb_steps=20,
                                         num_classes=args.num_class,
                                         mode=mode)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), correct_adv,
        len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))
    
def save_cam(model, train_loader, device, args):
    weighted_eps_list = []
    for _, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        # calculate robust loss
        model.eval()
        weight_matrix = craft_weight_matrix(model, data, device, args, parallel=True)
        weighted_eps = generate_weighted_eps(weight_matrix, args)
        weighted_eps_list.append(weighted_eps)
    return weighted_eps_list
