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
from dataset_initializers import speech_dataset as sd

# from torch_cluster import knn_graph, graclus_cluster

# sys.stdout = print_log
# device = torch.device('cpu')
root_dir = "dataset/google_origin/"
word_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
speaker_list=[]



def run_statistic(model, loaders):

    loader = loaders[2]
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
            grads[i,layer] = torch.norm(outs[layer].grad)
            losses[layer] = losses[layer].detach()
            # grads[i, layer] = losses[layer]
        # optimizer.step()
        # print(i," / ", len(loader))
        if(i % 500 == 0):
            print(i, " / ", len(loader))
    grads = grads.numpy()
    np.save("grad_test.npy", grads)
    np.save("acc_test.npy", accs)
    print("")



# optimizer=0
def prepare_run():
    ap = sd.AudioPreprocessor()  # Computes Log-Mel spectrogram
    train_files, dev_files, test_files = sd.split_dataset(root_dir, word_list, speaker_list)

    train_data = sd.SpeechDataset(train_files, "train", ap, word_list, speaker_list)
    dev_data = sd.SpeechDataset(dev_files, "dev", ap, word_list, speaker_list)
    test_data = sd.SpeechDataset(test_files, "test", ap, word_list, speaker_list)

    train_dataloader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    dev_dataloader = data.DataLoader(dev_data, batch_size=1, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)

    e_num = 3  # exit number = 4


    model = TCResNet8(k=1, n_mels=40, n_classes=len(word_list))
    model.load("e_297_valacc_94.401_94.718_94.676.pt")
    model.eval()
    ###########################################start train######################
    loaders = [train_dataloader,dev_dataloader,test_dataloader]
    # for layer in range(e_num):
    run_statistic(model=model, loaders=loaders)


def display():
    ces = np.load("grad_test.npy")[:,:3]
    acs = np.load("acc_test.npy")[:,:3]
    errs = np.load("err.npy")
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
        count =np.log( count / ces.shape[0])
        # count[count == -np.inf] = 9999
        # count[count == 9999] = np.min(count)
        counts.append(count)
    plt.figure(figsize=(8, 6))
    plt.plot(x, counts[0], color='#FF0000', label='exit0', linewidth=1.0)
    plt.plot(x, counts[1], color='#00FF00', label='exit1', linewidth=1.0)
    plt.plot(x, counts[2], color='#0000FF', label='exit2', linewidth=1.0)
    # 设置图形标题和坐标轴标签
    plt.title("")
    plt.xlabel("grad norm val")
    plt.ylabel("num_samples ratio log ")

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



if __name__ == '__main__':
    # prepare_run()
    display()

