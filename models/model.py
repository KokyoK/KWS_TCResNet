import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNets import *

# Pytorch implementation of Temporal Convolutions (TC-ResNet).
# Original code (Tensorflow) by Choi et al. at https://github.com/hyperconnect/TC-ResNet/blob/master/audio_nets/tc_resnet.py
#
# Input data represents frequencies (MFCCs) in different channels.
#                      _________________
#                     /                /|
#               freq /                / /
#                   /_______________ / /
#                1 |_________________|/
#                          time




class S2_Block(nn.Module):
    """ S2 ConvBlock used in Temporal Convolutions
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______-> CONV -> BN -> RELU ->________|
    """

    def __init__(self, in_ch, out_ch):
        super(S2_Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                               padding=(0, 4), bias=False)
        self.bn0 = nn.BatchNorm2d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                               padding=(0, 4), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, affine=True)
        # Residual convolution layer
        self.conv_res = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=False)
        self.bn_res = nn.BatchNorm2d(out_ch, affine=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        identity = self.conv_res(x)
        identity = self.bn_res(identity)
        identity = F.relu(identity)
        out += identity
        out = F.relu(out)

        return out


class TCResNet8(nn.Module):
    """ TC-ResNet8 implementation.

        Input dim:  [batch_size x n_mels x 1 x 101]
        Output dim: [batch_size x n_classes]

        Input -> Conv Layer -> (S2 Block) x 3 -> Average pooling -> FC Layer -> Output
    """

    def __init__(self, k, n_mels, n_classes):
        super(TCResNet8, self).__init__()

        # First Convolution layer
        self.conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
                                    padding=(0, 1), bias=False)

        # S2 Blocks
        self.s2_block0 = S2_Block(int(16 * k), int(24 * k))
        self.s2_block0_speaker = S2_Block(int(16 * k), int(24 * k))

        self.s2_block1 = S2_Block(int(24 * k), int(32 * k))
        self.s2_block1_speaker = S2_Block(int(24 * k), int(32 * k))

        self.s2_block2 = S2_Block(int(32 * k), int(48 * k))
        self.s2_block2_speaker = S2_Block(int(32 * k), int(48 * k))

        # Features are [batch x 48*k channels x 1 x 13] at this point
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=10, kernel_size=1, padding=0,
                            bias=False)
        # self.fc_s = nn.Conv2d(in_channels=int(48 * k), out_channels=1481, kernel_size=1, padding=0,
        #                     bias=False)
        self.early_fc_0 = nn.Conv2d(in_channels=int(24 * k), out_channels=10, kernel_size=1, padding=0,
                            bias=False)
        self.early_fc_1 = nn.Conv2d(in_channels=int(32 * k), out_channels=10, kernel_size=1, padding=0,
                            bias=False)


    def forward(self,x):
        out0 = self.forward_0(x)
        out1 = self.forward_1(x)
        out2 = self.forward_full(x)
        return [out0,out1, out2]

    def forward_0(self, x):
        out = self.conv_block(x)
        out = self.s2_block0(out)
        out = F.avg_pool2d(out, kernel_size = (1,51))
        out = self.early_fc_0(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_1(self, x):
        with torch.no_grad():
            out = self.conv_block(x)
            out = self.s2_block0(out)
        out = self.s2_block1(out)
        out = F.avg_pool2d(out, kernel_size=(1, 26))
        out = self.early_fc_1(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_full(self, x):
        # print("nn input shape: ",x.shape)
        with torch.no_grad():
            out = self.conv_block(x)
            out = self.s2_block0(out)
            out = self.s2_block1(out)
        out = self.s2_block2(out)
        out = F.avg_pool2d(out, kernel_size=(1, 13))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out




    def save(self, is_onnx=0, name="TCResNet8"):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), "saved_model/"+name)

    def load(self,name="TCResNet8"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))




'''
ResNet18
'''
class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(BasicBlock, self).__init__()
        # normal block
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        # identity mapping
        # if the in channel does not match out channel, an additional convolution and batch normalization is needed
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet18(torch.nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(My_CNN, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.repeat_shortcut_block(ResBlock, channels=64, num_blocks=1, stride=1)
        self.layer2 = self.repeat_shortcut_block(ResBlock, channels=128, num_blocks=2, stride=2)
        self.layer3 = self.repeat_shortcut_block(ResBlock, channels=256, num_blocks=2, stride=2)
        self.layer4 = self.repeat_shortcut_block(ResBlock, channels=512, num_blocks=2, stride=2)

        self.early_fc_1 = nn.Linear(64, num_classes)
        self.early_fc_2 = nn.Linear(128, num_classes)
        self.early_fc_3 = nn.Linear(256, num_classes)
        self.fc = nn.Linear(512, num_classes)

    # repeat same residual block
    def repeat_shortcut_block(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):

        map1, out1 = self.forward_1(x)
        map2, out2 = self.forward_2(x)
        map3, out3 = self.forward_3(x)
        map4, out4 = self.forward_full(x)

        return [map1, map2, map3, map4],[out1, out2, out3, out4]


    def forward_1(self, x):
        out = self.conv1(x)
        map = self.layer1(out)
        ###
        out = F.avg_pool2d(map, 32)
        out = out.view(out.size(0), -1)
        out = self.early_fc_1(out)
        return map, out

    def forward_2(self, x):
        with torch.no_grad():
            out = self.conv1(x)
            out = self.layer1(out)
        map = self.layer2(out)
        out = F.avg_pool2d(map, 16)
        out = out.view(out.size(0), -1)
        out = self.early_fc_2(out)
        return map, out

    def forward_3(self, x):
        with torch.no_grad():
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
        map = self.layer3(out)
        out = F.avg_pool2d(map, 8)
        out = out.view(out.size(0), -1)
        out = self.early_fc_3(out)
        return map, out

    def forward_full(self, x):
        with torch.no_grad():
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        map = self.layer4(out)
        out = F.avg_pool2d(map, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return map,out
        #
        # out = self.conv1(x)     # 64,32,32
        # out = self.layer1(out)  # 64,32,32
        # out = self.layer2(out)  # 128,16,16
        # out = self.layer3(out)  # 256,8,8
        # out = self.layer4(out)  # 512,4,4
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # return out
    def save(self,name):
        torch.save(self.state_dict(), "saved_model/"+name+".pt")

    def load(self, name="saved_model/Res18.pt"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))

'''
ResNet32
'''
import torch
import torch.nn as nn






# # 创建ResNet-32模型实例
# model = ResNet32()
# print(model)


# 创建ResNet-32模型实例







if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    model_tcresnet8 = TCResNet8(1, 40, 12)
    result_tcresnet8 = model_tcresnet8(x)
    print(result_tcresnet8)
