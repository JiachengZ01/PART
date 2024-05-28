from __future__ import print_function
import os
import argparse

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models.resnet import ResNet18
from models.wideresnet import WideResNet

from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Pixel-reweighted Adversarial Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='maximum allowed perturbation', type=parse_fraction)
parser.add_argument('--low-epsilon', default=7/255,
                    help='maximum allowed perturbation for unimportant pixels', 
                    type=parse_fraction)
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--num-class', default=10,
                    help='number of classes')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size', type=parse_fraction)
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--adjust-first', type=int, default=60,
                    help='adjust learning rate on which epoch in the first round')
parser.add_argument('--adjust-second', type=int, default=90,
                    help='adjust learning rate on which epoch in the second round')
parser.add_argument('--rand_init', type=bool, default=True,
                    help="whether to initialize adversarial sample with random noise")

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PART_M',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--save-weights', default=1, type=int, metavar='N',
                    help='save frequency for weighted matrix')

parser.add_argument('--data', type=str, default='CIFAR10', help='data source', choices=['CIFAR10', 'SVHN', 'TinyImagenet'])
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wideresnet'])
parser.add_argument('--warm-up', type=int, default=20, help='warm up epochs')
parser.add_argument('--cam', type=str, default='gradcam', choices=['gradcam', 'xgradcam', 'layercam'])
parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'mma'])

args = parser.parse_args()

if args.data == 'CIFAR100':
    args.num_class = 100
if args.data == 'TinyImagenet':
    args.num_class = 200

def train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        X, y = Variable(data, requires_grad=True), Variable(target)

        model.eval()
        weighted_eps = weighted_eps_list[batch_idx]

        optimizer.zero_grad()

        # calculate robust loss
        loss = part_mart_loss(model=model,
                              x_natural=X,
                              y=y,
                              optimizer=optimizer,
                              weighted_eps= weighted_eps,
                              step_size=args.step_size,
                              perturb_steps=args.num_steps,
                              beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))

def main():
    # settings
    setup_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # setup data loader
    if args.data == 'CIFAR10':
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
        test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()
        if args.model == 'resnet':
            model_dir = './checkpoint/CIFAR10/ResNet_18/PART_M'
            model = ResNet18(num_classes=10).to(device)
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/CIFAR10/WideResnet-34/PART_M'
            model = WideResNet(34, 10, 10).to(device)
        else:
            raise ValueError("Unknown model")
    elif args.data == 'SVHN':
        args.step_size = 1/255
        args.weight_decay = 0.0035
        args.lr = 0.01
        args.batch_size = 128
        train_loader = SVHN(train_batch_size=args.batch_size).train_data()
        test_loader = SVHN(test_batch_size=args.batch_size).test_data()
        if args.model == 'resnet':
            model_dir = './checkpoint/SVHN/ResNet_18/PART_M'
            model = ResNet18(num_classes=10).to(device)
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/SVHN/WideResnet-34/PART_M'
            model = WideResNet(34, 10, 10).to(device)
        else:
            raise ValueError("Unknown model")
    else:
        raise ValueError("Unknown data")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)

    # warm up
    print('warm up starts')
    for epoch in range(1, args.warm_up + 1):
        mart_train(args, model, device, train_loader, optimizer, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'pre_part_m_epoch{}.pth'.format(epoch)))
            print('save the model')
            print('================================================================')
    print('warm up ends')

    weighted_eps_list = save_cam(model, train_loader, device, args)

    for epoch in range(1, args.epochs - args.warm_up + 1):
        if epoch % args.save_weights == 0 and epoch != 1:
            weighted_eps_list = save_cam(model, train_loader, device, args)

        # adjust learning rate for SGD
        adjust_learning_rate(args, optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list)

        # evaluation on natural examples
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pth'.format(epoch)))

    # evaluation on adversarial examples
    print('PGD=============================================================')
    eval_test(args, model, device, test_loader, mode='pgd')
    print('MMA==============================================================')
    eval_test(args, model, device, test_loader, mode='mma')
    print('AA==============================================================')
    eval_test(args, model, device, test_loader, mode='aa')

if __name__ == '__main__':
    main()
