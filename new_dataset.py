import os


from numpy.lib.function_base import i0
from pre_model import DeciNet
import torch.utils.data as data
import torch.nn.functional as F
import torchaudio.functional as F_audio
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchaudio
import random
import speech_dataset as sd
import torch
torch.manual_seed(42)
from model import TCResNet8
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = "gpu" if torch.cuda.is_available() else "cpu"
train_on_gpu = True if device=="gpu" else False
e_count = 3
TRAIN = True
# debug = True
debug = False

class DifficultySpeechDataset(data.Dataset):

    def __init__(self,data_list):
        self.data_list = data_list

    def load_data(self, data_element):

        return (data_element)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.data_list[idx])
        cur_element = self.load_data(self.data_list[idx])
        # cur_element = (cur_element[0], self.word_list.index(cur_element[1]), self.speaker_list.index(cur_element[2]))
        return cur_element



class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=[1,0], reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        input = torch.argmax(input,dim=1)
        sub = (F.relu(input-target)).sum()

        if self.alpha is not None:
            alpha_t = self.alpha[0]
            focal_loss = alpha_t * focal_loss

        beta = 0.9
        return (1-beta) * focal_loss + sub * beta
        # if self.reduction == 'mean':
        #     return focal_loss
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss


def prepare_new_set(root_dir,word_list,speaker_list):
    ap = sd.AudioPreprocessor()
    train, dev, test = sd.split_dataset(root_dir, word_list, speaker_list)
    #########################
    if debug:
        train = train[:5]
        dev = dev[:1]
        test = test[:1]
    ###################

    # Dataset
    train_data = sd.SpeechDataset(train, "train", ap, word_list, speaker_list)
    dev_data = sd.SpeechDataset(dev, "train", ap, word_list, speaker_list)
    test_data = sd.SpeechDataset(test, "train", ap, word_list, speaker_list)
    # Dataloaders
    train_dataloader = data.DataLoader(train_data, batch_size=1, shuffle=False)
    dev_dataloader = data.DataLoader(dev_data, batch_size=1, shuffle=False)
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=False)
    loaders = [train_dataloader, dev_dataloader, test_dataloader]
    new_loaders=[]
    o_model = TCResNet8(k=1, n_mels=40, n_classes=len(word_list))
    o_model.load("e_297_valacc_94.401_94.718_94.676.pt")
    o_model.eval()


    for loader in loaders:
        Bar = enumerate(loader)
        new_data = []
        # print(len()
        for i, (feat, label) in Bar:
            accs = torch.zeros([3])
            train_data = feat.to(device)
            label = label.to(device)
            outs = o_model(train_data)
            for layer in range(3):
                acc_layer = torch.argmax(outs[layer], dim=1)
                if (acc_layer == label):
                    accs[layer] = 1
            feat = feat.squeeze(0)
            label = label[0]
            diff = 0
            if sum(accs) == 3:
                diff = 1
            new_data.append((i, feat, label, diff))
            # if(i==2):
            #     break
            # print(i,"/",len(train_dataloader))
        Diff_dataset = DifficultySpeechDataset(new_data)
        new_loader = data.DataLoader(Diff_dataset, batch_size=16, shuffle=False)
        new_loaders.append(new_loader)
    torch.save(new_loaders,"loader")
    return new_loaders

def evaluate_testset(model, loaders):
    criterion = nn.CrossEntropyLoss()
    test_dataloader = loaders[2]
    model.eval()
    model.mode = "eval"
    model.load()
    total_infer_time = 0
    valid_loss = 0
    valid_kw_correct = 0
    loss_all = 0
    for batch_idx, (idx, feat, label, diff) in enumerate(test_dataloader):
        if train_on_gpu:
            feat = feat.cuda()
            label = label.cuda()
            diff = diff.cuda()

        out = model(x=feat)
        loss_0 = criterion(out, diff)
        loss_full = loss_0
        loss = loss_full

        loss_all+=loss

        # out = torch.min(out, torch.ones_like(out))
        out = out.round()
        valid_loss += loss.item() * feat.size(0)
        # ba = torch.zeros(1)
        # b_hit = torch.zeros(1)
        b_hit = float(torch.sum((out == diff)).item()) / out.shape[1]
        # b_hit = float(torch.sum(torch.argmax(out, 1) == diff).item())
        ba = b_hit / float(feat.shape[0])
        valid_kw_correct += b_hit

    print("#############################################################")
    print("TEST | Loss_ALL: {:.4f}  | Decision ACC: {:.4f} \t| |  ".format(
        loss_all, valid_kw_correct/len(test_dataloader)))
    print("#############################################################")
def train(model,loaders,num_epoch):
    criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss()
    [train_dataloader, dev_dataloader, test_dataloader] = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    # Training
    step_idx = 0
    previous_valid_accuracy = 0

    for epoch in range(num_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        train_kw_accuracy = 0
        valid_kw_correct = 0
        correct = 0
        model.train()

        for batch_idx, (idx, feat, label, diff) in enumerate(train_dataloader):
            if train_on_gpu:
                feat = feat.cuda()
                label = label.cuda()
                diff = diff.cuda()
            # if diff.sum() == 48:
            #     a = torch.rand(1)
            #     if a < 0.5:
            #         continue

            out = model(x=feat)

            # cal_loss
            loss_0 = criterion(out, diff)
            loss_full = loss_0
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                # loss_full.backward(retain_graph=True)
                loss_full.backward(retain_graph=True)
                # loss_id.backward(retain_graph=True)
                # loss_o.backward(retain_graph=True)
            loss = loss_full
            optimizer.step()
            train_loss += loss.item() * feat.size(0)

            out = torch.argmax(out, dim=1)
            b_hit = torch.sum((out == diff)).item()
            # b_hit = float(torch.sum((out == diff)).item()) / out.shape[0]
            # b_hit = float(torch.sum(torch.argmax(out, 1) == diff).item())
            ba = b_hit / float(feat.shape[0])
            correct += b_hit

            # if (batch_idx % 100 == 0):
            #     print("Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | Decision ACC: {:.4f} |  ".format(
            #         epoch, step_idx, loss, ba))

            # train_kw_correct += b_hit
            step_idx += 1

        # Validation (1 epoch)
        model.eval()
        model.mode = "eval"
        total_infer_time = 0

        for batch_idx, (idx, feat, label, diff) in enumerate(dev_dataloader):
            if train_on_gpu:
                feat = feat.cuda()
                label = label.cuda()
                diff = diff.cuda()

            out = model(x=feat)
            loss_0 = criterion(out, diff)

            loss_full = loss_0
            loss = loss_full


            valid_loss += loss.item() * feat.size(0)
            out = torch.argmax(out, dim=1)
            # out = torch.min(out, torch.ones_like(out))
            # ba = torch.zeros(1)
            b_hit = torch.sum((out == diff)).item()
            # b_hit = float(torch.sum((out == diff)).item()) / out.shape[1]
            # b_hit = float(torch.sum(torch.argmax(out, 1) == diff).item())
            ba = b_hit / float(feat.shape[0])
            valid_kw_correct += b_hit

            # if (batch_idx % 100 == 0):
            #     print(
            #         "Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | Decision ACC: {:.4f} \t| |  ".format(
            #             epoch, step_idx, loss, ba))

            # valid_kw_correct += b_hit
            step_idx += 1

        # Loss statistics
        train_loss = train_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(dev_dataloader.dataset)


        train_kw_accuracy = 100.0 * (correct / len(train_dataloader.dataset))
        valid_kw_accuracy = 100.0 * (valid_kw_correct / len(dev_dataloader.dataset))
        # print(output.shape)
        # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
        # print(f1_scores)
        print("===========================================================================")
        print("EPOCH #{}     | TRAIN KWS ACC: {:.2f}%\t|  TRAIN LOSS : {:.2f}".format(epoch,
                                                                               train_kw_accuracy,
                                                                               train_loss))
        print("             | VAL KWS ACC :  {:.2f}%\t|  VAL LOSS   : {:.2f} | ".format(
            valid_kw_accuracy, valid_loss))
        # print("Validation path count:   ", path_count)
        # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")
        ##################### 看要不要存

        save = 0

        if (valid_kw_accuracy > previous_valid_accuracy):
            save = 1

        if (save == 1):
            previous_valid_accuracy = valid_kw_accuracy
            print("Saving current model...")
            model.save()
            model.save(is_onnx=0, name='e_{}_valacc_{:.3f}.pt'.format(epoch, valid_kw_accuracy))
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/

if __name__ == "__main__":
    # root_dir = "dataset/huawei_modify/WAV_new/"
    # word_list = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']
    speaker_list = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]
    root_dir = "dataset/google_origin/"
    word_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    # loaders = prepare_new_set(root_dir,word_list,speaker_list)
    loaders = torch.load("loader")

    print("New dataset prepared")
    # train_loader = loaders[0]
    Bar = enumerate(loaders[0])
    for i, (idx, feat, label, diff) in Bar:
        break
    NUM_EPOCH = 200

    pre_model = DeciNet(in_ch=40,out_ch=3)
    if TRAIN:
        train(pre_model,loaders, NUM_EPOCH)
    else:
        # train, dev, test = sd.split_dataset(ROOT_DIR, WORD_LIST, SPEAKER_LIST)
        # ap = sd.AudioPreprocessor()
        # test_data = sd.SpeechDataset(test, "eval", ap, WORD_LIST, SPEAKER_LIST)
        # test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)
        evaluate_testset(pre_model, loaders)


