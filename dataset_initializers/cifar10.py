import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(42)



# 定义数据预处理的转换
def cifar10_loaders():
    transform1 = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.RandomCrop(32, padding=4),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #  transforms.RandomHorizontalFlip(p=0.5)
         ])
    
    transform2 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
         )

    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root='./../dataset', train=True,
                                            download=True, transform=transform1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=False, num_workers=0)

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root='./../dataset', train=False,
                                           download=True, transform=transform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)

    # 定义类别标签


    # 打印训练集和测试集的大小
    print('Cifar-10 训练集大小:', len(trainset))
    print('Cifar-10 测试集大小:', len(testset))

    return [trainloader, testloader]

# 遍历训练集示例
if __name__ == "__main__":
    for i, data in enumerate(trainloader):
        # 在这里可以对数据进行进一步处理或训练模型
        print(i)
        break

