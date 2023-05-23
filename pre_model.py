import torch
import torch.nn as nn
import torch.nn.functional as F


class DeciNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeciNet, self).__init__()

        # First convolution layer
        # self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
        #                        padding=(0, 4), bias=False)
        # self.bn0 = nn.BatchNorm2d(out_ch, affine=True)
        # Second convolution layer
        # self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
        #                        padding=(0, 4), bias=False)
        # self.bn1 = nn.BatchNorm2d(out_ch, affine=True)
        # Residual convolution layer
        # self.conv_res = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
        #                           padding=(0, 4), bias=False)
        # self.bn_res = nn.BatchNorm2d(out_ch, affine=True)
        self.linear1 = nn.Linear(400, 20)
        self.linear2 = nn.Linear(20, 2)

    def forward(self, x):
        out = F.avg_pool2d(x, kernel_size = (1,10))
        out = out.view(out.shape[0],-1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = F.relu(out)
        # out = torch.min(out, torch.ones_like(out))
        out = F.softmax(out)
        # out = F.sigmoid(out)
        # out = self.linear2(out)
        # out = F.sigmoid(out)
        # out = F.sigmoid(out)
        # out = F.relu(out)
        # out = torch.min(out, torch.ones_like(out))
        # out =
        # out = out.round()

        return out

    def save(self, is_onnx=0, name="DeciNet"):
        # if (is_onnx):
        #     dummy_input = torch.randn(16, 40, 1, 101)
        #     torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
        #                       output_names=["output0"])
        # else:
        torch.save(self.state_dict(), "saved_model/"+name)

    def load(self,name="DeciNet"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))


if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    model = DeciNet(40,3)
    model.load()
    model.eval()
    result = model(x)
    print(result)
