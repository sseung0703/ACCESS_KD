from torchvision import datasets
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from models.wresnet import WideResNet
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def main(args):
    transform = transforms.Compose([transforms.ColorJitter(),
                                    transforms.RandomResizedCrop(32, (0.5, 1.5)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_dataset = datasets.CIFAR10('./Datasets', train=True, transform=transform, download=True)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_dataset = datasets.CIFAR10('./Datasets', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net_depth, net_width = int(args.select_model[4:6]), int(args.select_model[-1])

    Teacher_model = WideResNet(depth=net_depth, num_classes=10, widen_factor=net_width, dropRate=0.3)
    Teacher_model.cuda()
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(Teacher_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(args.total_epochs):

        for iter_, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to('cuda:0'), labels.to('cuda:0')
            outputs, *acts = Teacher_model(images)
            classification_loss = criterion(outputs, labels)
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            if iter_ % 10 == 0:
                print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:02.5f}".format(epoch, args.total_epochs, iter_,
                                                                              train_loader.__len__(),
                                                                              classification_loss.item()))

        with torch.no_grad():
            Teacher_model.eval()
            cumulated_acc = 0
            for x, y in test_loader:
                x, y = x.to('cuda:0'), y.to('cuda:0')
                logits, *activations = Teacher_model(x)
                acc = accuracy(logits.data, y, topk=(1,))[0]
                cumulated_acc += acc
            print("Test Accuracy is {:02.2f} %".format(cumulated_acc / test_loader.__len__() * 100))
            Teacher_model.train()
            if best_acc <= cumulated_acc / test_loader.__len__() * 100:
                best_acc = cumulated_acc / test_loader.__len__() * 100
                torch.save(Teacher_model.state_dict(),
                           './Pretrained/CIFAR10/WRN-{}-{}/Teacher_best.ckpt'.format(net_depth, net_width))

        optim_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=128, help='total training batch size')
    parser.add_argument('--select_model', type=str, default='WRN-40-2', help='What do you want to train?')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)