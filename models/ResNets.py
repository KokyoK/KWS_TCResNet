'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer0 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        self.early_fc_0 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1, padding=0,
                            bias=False)
        self.early_fc_1 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, padding=0,
                            bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer0(out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        
        ####### train_layer_wise 1
        out0 = self.forward_0(x)
        out1 = self.forward_1(x)
        out2 = self.forward_full(x)
        return [out0,out1, out2]
        
        # train_end2end 
        # out = F.relu(self.bn1(self.conv1(x)))
        # feat0 = self.layer0(out)
        # out0 = F.avg_pool2d(feat0, feat0.size()[3])
        # out0 = self.early_fc_0(out0)
        # out0 = F.softmax(out0, dim=1)
        # out0 = out0.view(out0.shape[0], -1)
        # # out0 = self.early_fc_0(feat0)
        # feat1 = self.layer1(feat1)
        # out1 = F.avg_pool2d(feat1, feat1.size()[3])
        # out1 = self.early_fc_0(out1)
        # out1 = F.softmax(out1, dim=1)
        # out1 = out0.view(out1.shape[0], -1)
        #
        # out = self.layer2(feat1)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # return [out0,out1, out]
        ####### train_classifier_wise #################
        # out = F.relu(self.bn1(self.conv1(x)))
        # feat0 = self.layer0(out)
        # feat1 = self.layer1(feat0)
        # out = self.layer2(feat1)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # out = self.forward_full(x)
        # return feat0, feat1, out
        # return out

    def forward_0(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer0(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.early_fc_0(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_1(self, x):
        with torch.no_grad():
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer0(out)
        out = self.layer1(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.early_fc_1(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_full(self, x):
        with torch.no_grad():
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer0(out)
            out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def save(self, is_onnx=0, name="ResNet34"):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "ResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), "saved_model/resnet34/"+name)

    def load(self,name="ResNet34"):
        self.load_state_dict(torch.load("saved_model/resnet34/"+name, map_location=lambda storage, loc: storage))


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))



class res32_ec(nn.Module):
    def __init__(self, num_classes):
        super(res32_ec, self).__init__()
        self.early_fc_0 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1, padding=0,
                            bias=False)
        self.early_fc_1 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, padding=0,
                            bias=False)
        self.index = 0
    def forward(self, x):
        if (self.index == 0):
            out = F.avg_pool2d(x, x.size()[3])
            out = self.early_fc_0(out)
            out = F.softmax(out, dim=1)
            out = out.view(out.shape[0], -1)
        if(self.index == 1):
            out = F.avg_pool2d(x, x.size()[3])
            out = self.early_fc_1(out)
            out = F.softmax(out, dim=1)
            out = out.view(out.shape[0], -1)
        return out

    def save(self, is_onnx=0, name="ResNet34_EE"):
        torch.save(self.state_dict(), "saved_model/resnet34/" + name)

    def load(self, name="ResNet34"):
        self.load_state_dict(torch.load("saved_model/resnet34/" + name, map_location=lambda storage, loc: storage))
            




if __name__ == "__main__":
    model = resnet32()
