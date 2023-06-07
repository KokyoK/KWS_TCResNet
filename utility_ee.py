import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.utils.data as data
from dataset_initializers import speech_dataset as sd
from models import model as md

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
time_check = True

# train_on_gpu = False
time_check = False


torch.set_printoptions(precision=4)
e_count = 3

if not train_on_gpu:
    print('CUDA is not available.  Using CPU ...')
else:
    print('CUDA is available!  Using GPU ...') 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_check = True







def evaluate_testset(model, test_dataloader):
    # Final test
    test_loss = 0.0
    test_correct = [0,0,0]
    model.load("e_297_valacc_94.401_94.718_94.676.pt")
    model.eval()

    valid_loss = 0
    criterion = nn.CrossEntropyLoss()
    step_idx = 0
    for batch_idx, (audio_data, label_kw) in enumerate(test_dataloader):
        if train_on_gpu:
            audio_data = audio_data.cuda()
            label_kw = label_kw.cuda()

        outs = model(x=audio_data)
        [out0, out1, out2] = outs
        # cal_loss

        loss_0 = criterion(out0, label_kw)
        loss_1 = criterion(out1, label_kw)
        loss_2 = criterion(out2, label_kw)

        loss_full = loss_0 + loss_1 + loss_2
        loss = loss_full

        valid_loss += loss.item() * audio_data.size(0)
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


    valid_loss = valid_loss / len(test_dataloader.dataset)
    valid_kw_accuracy = torch.zeros([e_count])
    for i in range(e_count):
        valid_kw_accuracy[i] = 100.0 * (test_correct[i] / len(test_dataloader.dataset))
    # print(output.shape)
    # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
    # print(f1_scores)
    print("===========================================================================")

    print("             | VAL KWS ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(
        valid_kw_accuracy[0], valid_kw_accuracy[1], valid_kw_accuracy[2], valid_loss))
    # print("Validation path count:   ", path_count)
    # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
    print("===========================================================================")

    
    # print("================================================")
    # print(" FINAL ACCURACY : {:.4f}% - TEST LOSS : {:.4f}".format(test_accuracy, test_loss))
    # print(" Time for avg test set inference:    ",total_infer_time/len(test_dataloader.dataset))
    # print(" Flops for avg test set inference:    ",total_flops / len(test_dataloader.dataset))
    # # print(" Test Set path count:   ", path_count)
    # print("================================================")

class OrgLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, map_k, map_s):
        mul = torch.matmul(map_k.squeeze(dim=2), map_s.squeeze(dim=2).permute(0,2,1))
        o_loss = torch.norm(mul, p='fro') ** 2 / (48*48)
        return o_loss

def train(model,loaders,num_epoch):
    """
    Trains TCResNet
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()

    [train_dataloader,dev_dataloader]=loaders


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-6)
    # Training
    step_idx = 0

    previous_valid_accuracy = [0,0,0]

    for epoch in range(num_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        valid_kw_correct = [0,0,0]
        valid_id_correct = 0
        train_kw_correct = [0,0,0]
        train_id_correct = 0


        model.train()

        for batch_idx, (audio_data,label_kw) in enumerate(train_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()

            optimizer.zero_grad()

            outs = model(x=audio_data)
            [out0, out1, out2]=outs
            # cal_loss

            loss_0 = criterion(out0, label_kw)
            loss_1 = criterion(out1, label_kw)
            loss_2 = criterion(out2, label_kw)

            loss_full = loss_0 +loss_1+loss_2


            with torch.autograd.set_detect_anomaly(True):
                # loss_full.backward(retain_graph=True)
                loss_full.backward(retain_graph=True)
                # loss_id.backward(retain_graph=True)
                # loss_o.backward(retain_graph=True)
            loss = loss_full

            optimizer.step()

            train_loss += loss.item()*audio_data.size(0)
            ba = torch.zeros([e_count ])
            b_hit = torch.zeros([e_count ])
            for i in range(e_count ):
                b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                ba[i] = b_hit[i]/float(audio_data.shape[0])
                train_kw_correct[i]+= b_hit[i]

            
            if (batch_idx%100 == 0):
                print("Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | ACC: {:.4f} \t| {:.4f}\t| {:.4f}|  ".format(epoch, step_idx, loss, ba[0],ba[1],ba[2]))

            # train_kw_correct += b_hit
            step_idx += 1


        # Validation (1 epoch)
        model.eval()
        model.mode = "eval"
        total_infer_time = 0

        for batch_idx, (audio_data, label_kw) in enumerate(dev_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()

            optimizer.zero_grad()

            outs = model(x=audio_data)
            [out0, out1, out2] = outs
            # cal_loss

            loss_0 = criterion(out0, label_kw)
            loss_1 = criterion(out1, label_kw)
            loss_2 = criterion(out2, label_kw)

            loss_full = loss_0 + loss_1 + loss_2
            loss = loss_full

            optimizer.step()

            valid_loss += loss.item() * audio_data.size(0)
            ba = torch.zeros([e_count ])
            b_hit = torch.zeros([e_count ])
            for i in range(e_count ):
                b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                ba[i] = b_hit[i]/float(audio_data.shape[0])
                valid_kw_correct[i]+= b_hit[i]

            if (batch_idx % 100 == 0):
                print("Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.4f} \t| {:.4f}\t| {:.4f}\t|  ".format(epoch, step_idx, loss, ba[0],ba[1],ba[2]))

            # valid_kw_correct += b_hit
            step_idx += 1

        # Loss statistics
        train_loss = train_loss/len(train_dataloader.dataset)
        valid_loss = valid_loss/len(dev_dataloader.dataset)
        train_kw_accuracy = torch.zeros([e_count])
        valid_kw_accuracy = torch.zeros([e_count])
        for i in range(e_count):
            train_kw_accuracy[i] = 100.0 * (train_kw_correct[i] / len(train_dataloader.dataset))
            valid_kw_accuracy[i] = 100.0 * (valid_kw_correct[i] / len(dev_dataloader.dataset))
        # print(output.shape)
        # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
        # print(f1_scores)
        print("===========================================================================")
        print("EPOCH #{}     | TRAIN KWS ACC: {:.2f}%\t| {:.2f}%\t |{:.2f}% |  TRAIN LOSS : {:.2f}".format(epoch, train_kw_accuracy[0], train_kw_accuracy[1],train_kw_accuracy[2],train_loss))
        print("             | VAL KWS ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(valid_kw_accuracy[0],valid_kw_accuracy[1],valid_kw_accuracy[2],valid_loss))
        # print("Validation path count:   ", path_count)
        # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")
        ##################### 看要不要存
        save = 1
        for i in range(e_count):
            if (valid_kw_accuracy[i] < previous_valid_accuracy[i]):
                save = 0

        if (save == 1):
            previous_valid_accuracy = valid_kw_accuracy
            print("Saving current model...")
            model.save()
            model.save(is_onnx=0, name='e_{}_valacc_{:.3f}_{:.3f}_{:.3f}.pt'.format( epoch,valid_kw_accuracy[0],valid_kw_accuracy[1],valid_kw_accuracy[2]))
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))
       
        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_testset(model, test_dataloader)
    
# on test set
def change_rate_evaluate(model, test_dataloader):
    # Final test
    test_loss = 0.0
    test_correct = [0, 0, 0]
    model.load("e_297_valacc_94.401_94.718_94.676.pt")
    model.eval()

    valid_loss = 0
    criterion = nn.CrossEntropyLoss()
    step_idx = 0
    l = len(test_dataloader)
    c01_list = torch.zeros([l])
    c12_list = torch.zeros([l])
    outs_list = torch.zeros([l,3])
    for batch_idx, (audio_data, label_kw) in enumerate(test_dataloader):
        if train_on_gpu:
            audio_data = audio_data.cuda()
            label_kw = label_kw.cuda()

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
    draw(c01_list,c12_list,outs_list)
    valid_loss = valid_loss / len(test_dataloader.dataset)
    valid_kw_accuracy = torch.zeros([e_count])
    for i in range(e_count):
        valid_kw_accuracy[i] = 100.0 * (test_correct[i] / len(test_dataloader.dataset))
    # print(output.shape)
    # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
    # print(f1_scores)
    print("===========================================================================")

    print("             | VAL KWS ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(
        valid_kw_accuracy[0], valid_kw_accuracy[1], valid_kw_accuracy[2], valid_loss))
    # print("Validation path count:   ", path_count)
    # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
    print("===========================================================================")

    # print("================================================")
    # print(" FINAL ACCURACY : {:.4f}% - TEST LOSS : {:.4f}".format(test_accuracy, test_loss))
    # print(" Time for avg test set inference:    ",total_infer_time/len(test_dataloader.dataset))
    # print(" Flops for avg test set inference:    ",total_flops / len(test_dataloader.dataset))
    # # print(" Test Set path count:   ", path_count)
    # print("================================================")



def draw(c01s,c12s,outs):
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






