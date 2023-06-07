import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(42)



# 定义数据预处理的转换
def cifar10_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root='./../dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root='./../dataset', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=0)

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

