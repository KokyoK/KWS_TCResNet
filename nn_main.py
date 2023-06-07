import torch
torch.manual_seed(42)
import torch.utils.data as data
from dataset_initializers import speech_dataset as sd
from dataset_initializers import cifar10
import utility_ee as util
from models.model import *

import os
dataset_name = "cifar10"
# dataset_name = "google_kws"
TRAIN = True
######################################################## KWS 相关 ###########################
ROOT_DIR = "dataset/google_origin/"
WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
# ROOT_DIR = "dataset/huawei_modify/WAV_new/"
# WORD_LIST = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']
# SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
SPEAKER_LIST = []
# SPEAKER_LIST = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]
# ROOT_DIR = "dataset/lege/"
# WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
######################################################## Cifar-10 相关 ###########################
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######################################################## ########################################
NUM_EPOCH = 2000


if __name__ == "__main__":

    # model = TCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST))
    model = resnet32()
    # print(model)



    if dataset_name=="google_kws":
        loaders = sd.kws_loaders(ROOT_DIR, WORD_LIST,SPEAKER_LIST)
        if TRAIN :
            util.train(model, loaders, NUM_EPOCH)
        else:
            util.evaluate_testset(model, loaders[1])
            # util.change_rate_evaluate(model_fp32, loaders[2])
    elif dataset_name=="cifar10":
        loaders = cifar10.cifar10_loaders()
        if TRAIN:
            util.train(model,  loaders, NUM_EPOCH)
        else:
            train, dev, test = sd.split_dataset(ROOT_DIR, WORD_LIST, SPEAKER_LIST)

            # util.evaluate_testset(model_fp32, test_dataloader)
            util.change_rate_evaluate(model_fp32, test_dataloader)


        



