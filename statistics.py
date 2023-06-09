import torch
import torch.nn.functional as F
import tensorflow as tf
from matplotlib import transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
# from sklearn.metrics import f1_score
from torch.utils.data import random_split, DataLoader, ConcatDataset
from models.model import TCResNet8
torch.manual_seed(42)
import numpy as np
import sys
from dataset_initializers import speech_dataset as sd,cifar10

from models.model import *
# from torch_cluster import knn_graph, graclus_cluster

# sys.stdout = print_log
# device = torch.device('cpu')
root_dir = "dataset/google_origin/"
word_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
speaker_list=[]

dataset_name="cifar10"
model_name="res32"
model = resnet32()
loaders = cifar10.cifar10_loaders()
e_count = 3
# model = TCResNet8(k=1, n_mels=40, n_classes=len(word_list))
model.load("e_30_valacc_62.070_66.290_69.750.pt")
model.eval()
def run_statistic(model, loaders):

    loader = loaders[1]
    device = "gpu" if torch.cuda.is_available() else "cpu"


    modes = ["train", "val", "test"]


    err_label = torch.zeros([3,10])
    grads = torch.zeros([len(loader),3])
    accs = torch.zeros([len(loader),3])
    lossfunc = torch.nn.CrossEntropyLoss()
    for i, (train_data, label) in enumerate(loader):
        train_data = train_data.to(device)
        label = label.to(device)
        outs = model(train_data)
        # optimizer.zero_grad()
        losses = [0, 0, 0]
        for layer in range(3):
            outs[layer].retain_grad()
            losses[layer] = lossfunc(outs[layer], label)
            acc_layer = torch.argmax(outs[layer],dim=1)
            if (acc_layer == label):
                accs[i,layer] = 1
            else:
                err_label[layer,label] += 1
            losses[layer].backward()
            # grads[i,layer] = torch.norm(outs[layer].grad)
            losses[layer] = losses[layer].detach()
            grads[i, layer] = losses[layer]
        # optimizer.step()
        # print(i," / ", len(loader))
        if(i % 500 == 0):
            print(i, " / ", len(loader))
    grads = grads.numpy()
    np.save("loss_test_{}_{}.npy".format(model_name,dataset_name), grads)
    np.save("acc_test_{}_{}.npy".format(model_name,dataset_name), accs)
    print("")



# optimizer=0
def prepare_run():
    # ap = sd.AudioPreprocessor()  # Computes Log-Mel spectrogram
    # train_files, dev_files, test_files = sd.split_dataset(root_dir, word_list, speaker_list)
    #
    # train_data = sd.SpeechDataset(train_files, "train", ap, word_list, speaker_list)
    # dev_data = sd.SpeechDataset(dev_files, "dev", ap, word_list, speaker_list)
    # test_data = sd.SpeechDataset(test_files, "test", ap, word_list, speaker_list)
    #
    # train_dataloader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    # dev_dataloader = data.DataLoader(dev_data, batch_size=1, shuffle=True)
    # test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)

    e_num = 3
    model = resnet32()
    loaders = cifar10.cifar10_loaders()

    # model = TCResNet8(k=1, n_mels=40, n_classes=len(word_list))
    model.load("e_30_valacc_62.070_66.290_69.750.pt")
    model.eval()
    ###########################################start train######################
    # loaders = [train_dataloader,dev_dataloader,test_dataloader]
    # for layer in range(e_num):
    run_statistic(model=model, loaders=loaders)


def display():
    ces = np.load("loss_test_{}_{}.npy".format(model_name,dataset_name))[:,:3]
    acs = np.load("acc_test_{}_{}.npy".format(model_name,dataset_name))[:,:3]
    # errs = np.load("err.npy")
    # '''
    counts = []
    # 定义数据
    d = 1000
    gmin = ces.min()
    gmax = ces.max()
    for i in range(3):
        ce = ces[:,i]
        ac = acs[:,i]
        # gmin = ce.min()
        # gmax = ce.max()
        interval = (gmax - gmin) / d
        x = np.linspace(gmin, gmax, d)
        count = np.linspace(0, 0, d)
        for k in range(ce.shape[0]):
            for i in range(d):
                if (ce[k] > gmin + interval * i and ce[k] <= gmin + (i + 1) * interval):
                    count[i] += 1
        ####counts 2 log ratio ###
        # count =np.log( count / ces.shape[0])
        # count[count == -np.inf] = 9999
        # count[count == 9999] = np.min(count)
        counts.append(count)
    plt.figure(figsize=(8, 6))
    plt.plot(x, counts[0], color='#FF0000', label='exit0', linewidth=1.0)
    plt.plot(x, counts[1], color='#00FF00', label='exit1', linewidth=1.0)
    plt.plot(x, counts[2], color='#0000FF', label='exit2', linewidth=1.0)
    # 设置图形标题和坐标轴标签
    plt.title("")
    plt.xlabel(" loss ")
    plt.ylabel("num_samples  ")

    plt.legend(fontsize=18)
    plt.savefig('figure.png', dpi=600)
    # 显示图形
    plt.show()

    # '''
    # fig, ax = plt.subplots()
    # x = np.linspace(0, 9, 10)
    # rects=[0,0,0]
    # for i in range(3):
    #     errs = torch.Tensor(errs)
    #     rects[i]=ax.bar(x,errs[i])
    # plt.show()

    print("")


def change_rate_evaluate(model, test_dataloader):
    # Final test
    test_loss = 0.0
    test_correct = [0, 0, 0]
    model.eval()

    valid_loss = 0
    criterion = nn.CrossEntropyLoss()
    step_idx = 0
    l = len(test_dataloader)
    c01_list = torch.zeros([l])
    c12_list = torch.zeros([l])
    outs_list = torch.zeros([l,3])
    for batch_idx, (audio_data, label_kw) in enumerate(test_dataloader):
        # if train_on_gpu:
        # audio_data = audio_data.cuda()
        # label_kw = label_kw.cuda()

        outs = model(x=audio_data)
        [out0, out1, out2] = outs
        change_01 = criterion(out0, out1)
        change_12 = criterion(out1, out2)
        c01_list[batch_idx] = change_01
        c12_list[batch_idx] = change_12
        for i in range(3):
            outs_list[batch_idx,i] = torch.Tensor(torch.argmax(outs[i])==label_kw)
        # loss_2 = criterion(out2, label_kw)

        # cal_loss
        # loss_0 = criterion(out0, label_kw)
        # loss_1 = criterion(out1, label_kw)
        # loss_2 = criterion(out2, label_kw)
        #
        # loss_full = loss_0 + loss_1 + loss_2
        # loss = loss_full
        #
        # valid_loss += loss.item() * audio_data.size(0)
        ba = torch.zeros([e_count])
        b_hit = torch.zeros([e_count])
        for i in range(e_count):
            b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
            ba[i] = b_hit[i] / float(audio_data.shape[0])
            test_correct[i] += b_hit[i]

        # if (batch_idx % 100 == 0):
        #     print("Loss_ALL: {:.4f}  | KWS ACC: {:.4f} \t| {:.4f}\t| {:.4f}\t|  ".format(
        #         epoch, step_idx, loss, ba[0], ba[1], ba[2]))

        # valid_kw_correct += b_hit
        step_idx += 1
    print("finished inference")
    draw_pair_change(c01_list,c12_list,outs_list)
    valid_loss = valid_loss / len(test_dataloader.dataset)
    valid_kw_accuracy = torch.zeros([e_count])
    for i in range(e_count):
        valid_kw_accuracy[i] = 100.0 * (test_correct[i] / len(test_dataloader.dataset))
    print("===========================================================================")

    print("             | VAL KWS ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(
        valid_kw_accuracy[0], valid_kw_accuracy[1], valid_kw_accuracy[2], valid_loss))
    print("===========================================================================")
def draw_pair_change(c01s,c12s,outs):
    avg_c01_true = 0
    avg_c12_true = 0
    avg_c01_false = 0
    avg_c12_false = 0
    outs_sum = torch.sum(outs, dim=1)
    outs_t = (outs_sum == 3)
    count_true = 0
    len = outs_t.shape[0]
    for i in range(len):
        # if (outs_t[i] == True):
        if (outs[i,0] == 1):
            avg_c01_true += c01s[i]
            avg_c12_true += c12s[i]
            count_true += 1
        else:
            avg_c01_false += c01s[i]
            avg_c12_false += c12s[i]
    avg_c01_true /= count_true
    avg_c12_true /= count_true
    avg_c01_false /= (len-count_true)
    avg_c12_false /= (len-count_true)
    #######################    统计完成  ####################
    import matplotlib.pyplot as plt

    # 数据
    x = ['change rate between exit 0&1n', 'change rate between exit 1&2']
    y1 = [ avg_c01_true.detach().numpy(),avg_c12_true.detach().numpy()]  # 第一个柱子的高度
    y2 = [avg_c01_false.detach().numpy(),avg_c12_false.detach().numpy()]  # 第二个柱子的高度

    # 设置柱子宽度
    bar_width = 0.15

    # 创建柱状图
    plt.bar(x, y2, width=bar_width, label='False classification', alpha=1)

    plt.bar(x, y1, width=bar_width, label='True classification')

    # 添加标题和标签
    plt.title('Change Rate')
    plt.xlabel('Exit index')
    plt.ylabel('change rate')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

    print("")
    # '''
    # counts = []
    # # 定义数据
    # d = 1000
    # gmin = ces.min()
    # gmax = ces.max()
    # for i in range(3):
    #     ce = ces[:, i]
    #     ac = acs[:, i]
    #     # gmin = ce.min()
    #     # gmax = ce.max()
    #     interval = (gmax - gmin) / d
    #     x = np.linspace(gmin, gmax, d)
    #     count = np.linspace(0, 0, d)
    #     for k in range(ce.shape[0]):
    #         for i in range(d):
    #             if (ce[k] > gmin + interval * i and ce[k] <= gmin + (i + 1) * interval):
    #                 count[i] += 1
    #     ####counts 2 log ratio ###
    #     count = np.log(count / ces.shape[0])
    #     # count[count == -np.inf] = 9999
    #     # count[count == 9999] = np.min(count)
    #     counts.append(count)
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, counts[0], color='#FF0000', label='exit0', linewidth=1.0)
    # plt.plot(x, counts[1], color='#00FF00', label='exit1', linewidth=1.0)
    # plt.plot(x, counts[2], color='#0000FF', label='exit2', linewidth=1.0)
    # # 设置图形标题和坐标轴标签
    # plt.title("")
    # plt.xlabel("grad norm val")
    # plt.ylabel("num_samples ratio log ")
    #
    # plt.legend(fontsize=18)
    # plt.savefig('figure.png', dpi=600)
    # # 显示图形
    # plt.show()
    #
    # # '''
    # # fig, ax = plt.subplots()
    # # x = np.linspace(0, 9, 10)
    # # rects=[0,0,0]
    # # for i in range(3):
    # #     errs = torch.Tensor(errs)
    # #     rects[i]=ax.bar(x,errs[i])
    # # plt.show()
    #
    # print("")



if __name__ == '__main__':
    # prepare_run()
    # display()
    change_rate_evaluate(model,loaders[1])
