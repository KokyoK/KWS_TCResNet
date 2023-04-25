import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0,
                            bias=False)
        self.fc_s = nn.Conv2d(in_channels=int(48 * k), out_channels=37, kernel_size=1, padding=0,
                            bias=False)

    def forward(self, x):
        # print("nn input shape: ",x.shape)
        out = self.conv_block(x)

        out_inter = self.s2_block0(out)


        #### keyword recog
        out = self.s2_block1(out_inter)
        k_map = self.s2_block2(out)
        out = self.avg_pool(k_map)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        out_k = out.view(out.shape[0], -1)

        #### speaker recog
        with torch.no_grad():
            out = self.conv_block(x)
            out_inter = self.s2_block0(out)

        out_s = self.s2_block1_speaker(out_inter)
        s_map = self.s2_block2_speaker(out_s)
        out_s = self.avg_pool(s_map)
        out_s = self.fc_s(out_s)
        out_s = F.softmax(out_s, dim=1)
        out_s = out_s.view(out_s.shape[0], -1)

        return out_k,out_s ,k_map,s_map


    def save(self, is_onnx=0, name="TCResNet8"):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), "saved_model/"+name)

    def load(self,name="TCResNet8"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))


if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    model_tcresnet8 = TCResNet8(1, 40, 12)
    result_tcresnet8 = model_tcresnet8(x)
    print(result_tcresnet8)
