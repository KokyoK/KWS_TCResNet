import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.utils.data as data
from dataset_initializers import speech_dataset as sd
from models import model as md
from models.ResNets import res32_ec
from torch.optim import lr_scheduler

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
    # model.load("e_297_valacc_94.401_94.718_94.676.pt")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)
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
        # model.mode = "eval"
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

            if (batch_idx % 1000 == 0):
                print("Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.4f} \t| {:.4f}\t| {:.4f}\t|  ".format(epoch, batch_idx, loss, ba[0],ba[1],ba[2]))

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
        print("EPOCH #{}     | TRAIN ACC: {:.2f}%\t| {:.2f}%\t |{:.2f}% |  TRAIN LOSS : {:.2f}".format(epoch, train_kw_accuracy[0], train_kw_accuracy[1],train_kw_accuracy[2],train_loss))
        print("             | VAL ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(valid_kw_accuracy[0],valid_kw_accuracy[1],valid_kw_accuracy[2],valid_loss))
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
            model.save(is_onnx=0, name='e__{}_valacc_{:.3f}_{:.3f}_{:.3f}.pt'.format( epoch,valid_kw_accuracy[0],valid_kw_accuracy[1],valid_kw_accuracy[2]))
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))
       
        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_testset(model, dev_dataloader)


def train_layer_wise(model, loaders, num_epoch):
    """
    Trains TCResNet
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    [train_dataloader, dev_dataloader] = loaders
    criterion = nn.CrossEntropyLoss()

    # Training
    step_idx = 0

    previous_valid_accuracy = [0, 0, 0]

    for e_idx in range(e_count):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch * 300, eta_min=1e-3)

        for epoch in range(num_epoch):
            train_loss = 0.0
            valid_loss = 0.0
            valid_kw_correct = [0, 0, 0]
            train_kw_correct = [0, 0, 0]
            model.train()
            for batch_idx, (audio_data, label_kw) in enumerate(train_dataloader):
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
                losses = [loss_0,loss_1,loss_2]
                loss_full = loss_0 + loss_1 + loss_2

                # with torch.autograd.set_detect_anomaly(True):
                    # loss_full.backward(retain_graph=True)
                losses[e_idx].backward(retain_graph=True)
                    # loss_id.backward(retain_graph=True)
                    # loss_o.backward(retain_graph=True)
                loss = losses[e_idx]

                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * audio_data.size(0)
                ba = torch.zeros([e_count])
                b_hit = torch.zeros([e_count])
                for i in range(e_count):
                    b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                    ba[i] = b_hit[i] / float(audio_data.shape[0])
                    train_kw_correct[i] += b_hit[i]

                if (batch_idx % 50 == 0):
                    print(
                        "Layer {} | Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | ACC: {:.4f} \t| {:.4f}\t| {:.4f}|  ".format(e_idx,epoch,
                                                                                                                      step_idx,
                                                                                                                      loss,
                                                                                                                      ba[0],
                                                                                                                      ba[1],
                                                                                                                      ba[
                                                                                                                          2]))
                # train_kw_correct += b_hit
                step_idx += 1

            # Validation (1 epoch)
            model.eval()
            # model.mode = "eval"
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
                ba = torch.zeros([e_count])
                b_hit = torch.zeros([e_count])
                for i in range(e_count):
                    b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                    ba[i] = b_hit[i] / float(audio_data.shape[0])
                    valid_kw_correct[i] += b_hit[i]

                if (batch_idx % 50 == 0):
                    print("Layer {} | Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.4f} \t| {:.4f}\t| {:.4f}\t|  ".format(
                        e_idx, epoch, step_idx, loss, ba[0], ba[1], ba[2]))

                # valid_kw_correct += b_hit
                step_idx += 1

            # Loss statistics
            train_loss = train_loss / len(train_dataloader.dataset)
            valid_loss = valid_loss / len(dev_dataloader.dataset)
            train_kw_accuracy = torch.zeros([e_count])
            valid_kw_accuracy = torch.zeros([e_count])
            for i in range(e_count):
                train_kw_accuracy[i] = 100.0 * (train_kw_correct[i] / len(train_dataloader.dataset))
                valid_kw_accuracy[i] = 100.0 * (valid_kw_correct[i] / len(dev_dataloader.dataset))
            # print(output.shape)
            # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
            # print(f1_scores)
            print("===========================================================================")
            print("Layer {} | EPOCH #{}     | TRAIN ACC: {:.2f}%\t| {:.2f}%\t |{:.2f}% |  TRAIN LOSS : {:.2f}".format(e_idx,epoch,
                                                                                                           train_kw_accuracy[
                                                                                                               0],
                                                                                                           train_kw_accuracy[
                                                                                                               1],
                                                                                                           train_kw_accuracy[
                                                                                                               2],
                                                                                                           train_loss))
            print("                         | VAL ACC :  {:.2f}%\t| {:.2f}%\t |{:.2f}% |  VAL LOSS   : {:.2f}".format(
                valid_kw_accuracy[0], valid_kw_accuracy[1], valid_kw_accuracy[2], valid_loss))
            # print("Validation path count:   ", path_count)
            # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
            print("===========================================================================")
            ##################### 看要不要存
            save = 0
            # for i in range(e_count):
            if (valid_kw_accuracy[e_idx] > previous_valid_accuracy[e_idx]):
                save = 1

            if (save == 1):
                previous_valid_accuracy = valid_kw_accuracy
                print("Saving current model...")
                model.save()
                model.save(is_onnx=0, name='l_{}_e_{}_valacc_{:.3f}_{:.3f}_{:.3f}.pt'.format(
                                                                        e_idx, epoch, valid_kw_accuracy[0],
                                                                                        valid_kw_accuracy[1],
                                                                                        valid_kw_accuracy[2]))
                # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
                # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))

            # Update scheduler (for decaying learning rate)
            # scheduler.step()

    # Final test
    evaluate_testset(model, dev_dataloader)

def train_layer_wise_ee(model, loaders, num_epoch,ratios=[0.5,0.8,1]):
    """
    Trains TCResNet
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    [train_dataloader, dev_dataloader] = loaders
    criterion = nn.CrossEntropyLoss()

    # Training
    step_idx = 0
    thresholds = [0.3,0.3,0.3]
    thresh_idxs = [0,0,0]
    previous_valid_accuracy = [0, 0, 0]
    model_names = ["EE0", "EE1", "FULL"]
    for e_idx in range(e_count):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch * 300, eta_min=1e-3)

        for epoch in range(num_epoch):
            train_loss = 0.0
            valid_loss = 0.0
            valid_kw_correct = [0, 0, 0]
            train_kw_correct = [0, 0, 0]

            model.train()
            for batch_idx, (audio_data, label_kw) in enumerate(train_dataloader):
                if train_on_gpu:
                    audio_data = audio_data.cuda()
                    label_kw = label_kw.cuda()
                outs = model(x=audio_data)
                # [out0, out1, out2] = outs
                out = outs[e_idx]
                
                # calculate threshold and partition the batch
                batch_size = label_kw.shape[0]
                temp_thresholds=[0,0,0]
                # forward through all exit
                ratio = [0,ratios[0]]
                [temp_thresholds[0], thresh_idxs[0]], [exit_out, non_exit_out], [exit_label, non_exit_label] \
                    = partition_batch(out=out, label=label_kw, thresh=thresholds[e_idx], ratio=ratio)
                # if not training the first layer
                if (e_idx != 0):
                    for i in range(1,e_count):
                        # ratio = cal_ratio(ratios, e_idx)
                        ratio = [ratios[i-1], ratios[i]]
                        # if non_exit_out.shape[0]!= 0:
                        [temp_thresholds[i], thresh_idxs[i]], [exit_out, non_exit_out], [exit_label, non_exit_label] \
                            = partition_batch(out=out, label=label_kw, thresh=thresholds[i], ratio=ratio)
                thresholds[e_idx] = temp_thresholds[e_idx]


                # cal_loss
                loss_exit = 0 if exit_out.shape[0]==0 else criterion(exit_out, exit_label)
                loss_non_exit = 0 if non_exit_out.shape[0]==0 else criterion(non_exit_out,non_exit_label)
                # loss = criterion(out, label_kw)
                loss = 0.6 * loss_exit + 0.4 * loss_non_exit
                # loss_0 = criterion(out0, label_kw)
                # loss_1 = criterion(out1, label_kw)
                # loss_2 = criterion(out2, label_kw)
                # losses = [loss_0,loss_1,loss_2]
                # loss_full = loss_0 + loss_1 + loss_2
                # loss = losses[e_idx]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * audio_data.size(0)
                ba = torch.zeros([e_count])
                b_hit = torch.zeros([e_count])
                for i in range(e_count):
                    b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                    ba[i] = b_hit[i] / float(audio_data.shape[0]) * 100
                    train_kw_correct[i] += b_hit[i]

                if (batch_idx % 50 == 0):
                    print(
                    "{} | Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | ALL ACC: {:.2f}%  {:.2f}%  {:.2f}%\t| Exit Ratio {:.2f}% | Thresh {:.3f} Learning Rate: {:.7f} ".format(
                        model_names[e_idx], epoch, step_idx, loss, ba[0], ba[1], ba[2], thresh_idxs[e_idx]/batch_size*100, thresholds[e_idx],
                        optimizer.state_dict()['param_groups'][0]['lr']))
                # train_kw_correct += b_hit
                step_idx += 1
                # break

            # Validation (1 epoch)
            model.eval()
            # model.mode = "eval"
            total_infer_time = 0
            hit_exit = torch.zeros([e_count])   # 
            exit_count = torch.zeros([e_count])   # count of current exit samples
            hit_non_exit = torch.zeros([e_count])
            valid_acc_exit = torch.zeros([e_count])
            valid_acc_non_exit = torch.zeros([e_count])
            valid_exit_ratio = torch.zeros([e_count])
            for batch_idx, (audio_data, label_kw) in enumerate(dev_dataloader):
                if train_on_gpu:
                    audio_data = audio_data.cuda()
                    label_kw = label_kw.cuda()
                outs = model(x=audio_data)
                [out0, out1, out2] = outs
                
                batch_size = label_kw.shape[0]

                # forward through all exit

                exit_outs, exit_labels, exit_indices = infer_batch(outs, label_kw, thresholds)

                b_exit_hit = []
                b_non_exit_hit=[]
                for i in range(e_count):
                    if (exit_outs[i].dim()==2):
                        # print(exit_outs[i].shape)
                        b_exit_hit.append(float(torch.sum(torch.argmax(exit_outs[i], 1) == exit_labels[i]).item()))
                        hit_exit[i] += b_exit_hit[i]
                        exit_count[i] += exit_outs[i].shape[0]
                    else:           # batch has output
                        b_exit_hit.append(torch.Tensor(0))
                    # b_non_exit_hit = float(torch.sum(torch.argmax(non_exit_out, 1) == non_exit_label).item())
                    # hit_non_exit[i] += b_non_exit_hit[i]

                # cal_loss
                loss_0 = criterion(out0, label_kw)
                loss_1 = criterion(out1, label_kw)
                loss_2 = criterion(out2, label_kw)

                loss_full = loss_0 + loss_1 + loss_2
                # loss = loss_full
                loss = criterion(out0, label_kw)

                valid_loss += loss.item() * audio_data.size(0)
                ba = torch.zeros([e_count])
                b_hit = torch.zeros([e_count])
                for i in range(e_count):
                    b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                    ba[i] = b_hit[i] / float(audio_data.shape[0])*100
                    valid_kw_correct[i] += b_hit[i]

                if (batch_idx % 50 == 0):
                    print("{} | Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.2f} {:.2f} {:.2f}|  ".format(
                        model_names[e_idx], epoch, step_idx, loss, ba[0], ba[1],ba[2]))

                # valid_kw_correct += b_hit
                step_idx += 1
                # break

            # Loss statistics for current EPOCH
            train_loss = train_loss / len(train_dataloader.dataset)
            valid_loss = valid_loss / len(dev_dataloader.dataset)
            train_kw_accuracy = torch.zeros([e_count])
            valid_kw_accuracy = torch.zeros([e_count])
            
            valid_acc_non_exit[e_idx] = -1 if(len(dev_dataloader.dataset) - exit_count[e_idx])==0 \
                else 100 * hit_non_exit[e_idx] / (len(dev_dataloader.dataset) - exit_count[e_idx])
            
            infer_valid_acc = torch.sum(hit_exit) / len(dev_dataloader.dataset)
            for i in range(e_count):
                train_kw_accuracy[i] = 100.0 * (train_kw_correct[i] / len(train_dataloader.dataset))
                valid_kw_accuracy[i] = 100.0 * (valid_kw_correct[i] / len(dev_dataloader.dataset))
                valid_acc_exit[i] = 100 * hit_exit[i] / exit_count[i]
                valid_exit_ratio[i] = exit_count[i] / len(dev_dataloader.dataset)
            # print(output.shape)
            # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
            # print(f1_scores)
            print("===========================================================================")
            print("{} | EPOCH #{}     | TRAIN ACC: {}%\t|  TRAIN LOSS : {:.2f} |".format(
                model_names[e_idx], epoch, train_kw_accuracy,train_loss))
            print("  | INFER ACC: {:.2f}%\t| ".format(infer_valid_acc*100))
            for i in range(e_count):
                print("            EXIT {} | VAL ACC :  {:.2f}%\t ｜ Exit Ratio: {:.2f}% | Thresholds: {:.4f} |ACC_exit: {:.2f}%\t| ACC_non_exit {:.2f}%\t | VAL LOSS : {:.2f}".format(
                i,  valid_kw_accuracy[i],valid_exit_ratio[i]*100, thresholds[i], valid_acc_exit[i],valid_acc_non_exit[i], valid_loss))
            # print("Validation path count:   ", path_count)
            # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
            print("===========================================================================")
            ##################### 看要不要存
            save = 0
            # for i in range(e_count):
            if (valid_kw_accuracy[e_idx] > previous_valid_accuracy[e_idx]):
                save = 1

            if (save == 1):
                previous_valid_accuracy = valid_kw_accuracy
                print("Saving current model...")
                model.save()
                model.save(is_onnx=0, name='l_{}_e_{}_valacc_{:.3f}_{:.3f}_{:.3f}.pt'.format(
                                                                        e_idx, epoch, valid_kw_accuracy[0],
                                                                                        valid_kw_accuracy[1],
                                                                                        valid_kw_accuracy[2]))
                # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
                # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))

            # Update scheduler (for decaying learning rate)
            # scheduler.step()

    # Final test
    evaluate_testset(model, dev_dataloader)


def train_classifier_wise(model, loaders, num_epoch):
    model_names = ["BACKBONE","EE0","EE1"]
    early_classifier = res32_ec(num_classes=10)
    [train_dataloader, dev_dataloader] = loaders
    criterion = nn.CrossEntropyLoss()
    # train backbone
    for i_model in range(len(model_names)):
        previous_valid_accuracy = 0
        step_idx = 0
        if i_model==0:
            cur_model = model
        else:
            cur_model = early_classifier
        optimizer = torch.optim.SGD(cur_model.parameters(), lr = 1e-1, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epoch * 300, eta_min=1e-3)
        for epoch in range(num_epoch):
            train_loss = 0.0
            valid_loss = 0.0
            valid_kw_correct = 0
            train_kw_correct = 0
            cur_model.train()
            for batch_idx, (audio_data, label_kw) in enumerate(train_dataloader):
                if train_on_gpu:
                    cur_model.cuda()
                    early_classifier.cuda()
                    audio_data = audio_data.cuda()
                    label_kw = label_kw.cuda()
                outs = model(x=audio_data)
                [feat0, feat1 ,out_b] = outs
                if i_model == 0:
                    out = out_b
                else:
                    early_classifier.index = i_model - 1
                    out = early_classifier(outs[i_model - 1])
                loss = criterion(out, label_kw)
                # with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * audio_data.size(0)
                b_hit = float(torch.sum(torch.argmax(out, 1) == label_kw).item())
                ba = b_hit / float(audio_data.shape[0])
                train_kw_correct += b_hit
                if (batch_idx % 50 == 0):
                    print(
                        "{} | Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | ACC: {:.2f}%\t| Learning Rate: {:.7f} ".format(
                            model_names[i_model],epoch,step_idx,loss,ba*100, optimizer.state_dict()['param_groups'][0]['lr']))
                # train_kw_correct += b_hit
                step_idx += 1
                # break

            # Validation (1 epoch)
            cur_model.eval()
            with torch.no_grad():
                for batch_idx, (audio_data, label_kw) in enumerate(dev_dataloader):
                    if train_on_gpu:
                        audio_data = audio_data.cuda()
                        label_kw = label_kw.cuda()
                    outs = model(x=audio_data)
                    [feat0, feat1 ,out_b] = outs
                    if i_model == 0:
                        out = out_b
                    else:
                        early_classifier.index = i_model - 1
                        out = early_classifier(outs[i_model - 1])
                    loss = criterion(out, label_kw)
                    valid_loss += loss.item() * audio_data.size(0)
                    b_hit = float(torch.sum(torch.argmax(out, 1) == label_kw).item())
                    ba = b_hit / float(audio_data.shape[0])
                    if (batch_idx % 1000 == 0):
                        print("{} | Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.2f} |  ".format(
                            model_names[i_model],epoch, step_idx, loss, ba*100))
                    valid_kw_correct += b_hit
                    step_idx += 1
                # break
            # Loss statistics
            train_loss = train_loss / len(train_dataloader.dataset)
            valid_loss = valid_loss / len(dev_dataloader.dataset)
            train_kw_accuracy= 100.0 * (train_kw_correct / len(train_dataloader.dataset))
            valid_kw_accuracy = 100.0 * (valid_kw_correct / len(dev_dataloader.dataset))

            print("===========================================================================")
            print("{} | EPOCH #{}     | TRAIN ACC: {:.2f}%\t|  TRAIN LOSS : {:.2f} |".format(
                model_names[i_model],epoch,train_kw_accuracy,train_loss))
            print("                         | VAL ACC :  {:.2f}%\t|  VAL LOSS   : {:.2f}".format(
                valid_kw_accuracy, valid_loss))
            # print("Validation path count:   ", path_count)
            # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
            print("===========================================================================")
            ##################### 看要不要存
            save = 1

            if (valid_kw_accuracy< previous_valid_accuracy):
                    save = 0

            if (save == 1):
                previous_valid_accuracy = valid_kw_accuracy
                print("Saving current model...")
                cur_model.save()
                cur_model.save(is_onnx=0, name='{}_e_{}_valacc_{:.3f}.pt'.format(
                                                model_names[i_model],epoch, valid_kw_accuracy))
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))

        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_testset(model, loaders[1])


def train_ee(model, loaders, num_epoch,ratios):
    early_classifier = res32_ec(num_classes=10)
    model.eval()
    model.load("BACKBONE_e_1523_valacc_94.260_2000.pt")
    # early_classifier.load("EE1_e_1528_valacc_88.610_2000.pt")
    early_classifier.train()
    [train_dataloader, dev_dataloader] = loaders
    criterion = nn.CrossEntropyLoss()
    model_names = ["EE0", "EE1"]
    previous_valid_accuracy = 0
    step_idx = 0
    optimizer = torch.optim.SGD(early_classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(early_classifier.parameters(), lr=1e-1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch * 300, eta_min=1e-3)
    for epoch in range(num_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        valid_kw_correct = 0
        train_kw_correct = 0
        threshold = 0.2
        early_classifier.train()
        for batch_idx, (audio_data, label_kw) in enumerate(train_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()
                model.cuda()
                early_classifier.cuda()
            [feat0, feat1, out_b] = model(x=audio_data)
            early_classifier.index = 0
            out0 = early_classifier(feat0)
            batch_size = label_kw.shape[0]
            idxes = torch.arange(0,  batch_size)

            # true_prob_0 = out0[idxes, label_kw]
            # sorted_prob, sorted_indices = true_prob_0.sort(dim=0,descending=True)
            max_prob_0, _ = out0.max(dim=1)
            sorted_prob, sorted_indices = max_prob_0.sort(dim=0, descending=True)

            exit_indices = sorted_indices[:round(ratios[0] * batch_size)]
            # cur_thresh = out0[exit_indices[-1]][label_kw[exit_indices[-1]]]
            cur_thresh = out0[exit_indices[-1]].max()
            threshold = 0.9 * threshold + 0.1 * cur_thresh
            thresh_idx = (sorted_prob > threshold).sum()

            exit_indices = sorted_indices[:thresh_idx]
            non_exit_indices = sorted_indices[thresh_idx:]
            exit_out = out0[exit_indices]
            non_exit_out = out0[non_exit_indices]

            loss_exit = 0.6 * criterion(exit_out, label_kw[exit_indices])
            loss_non_exit = 0 if non_exit_out.shape[0]==0 else 0.4*criterion(non_exit_out, label_kw[non_exit_indices])
            # loss = loss_exit + loss_non_exit
            loss = criterion(out0, label_kw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * audio_data.size(0)
            b_hit = float(torch.sum(torch.argmax(out0, 1) == label_kw).item())
            ba = b_hit / float(audio_data.shape[0])
            train_kw_correct += b_hit
            if (batch_idx % 50 == 0):
                print(
                    "{} | Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | ALL ACC: {:.2f}%\t| Exit Ratio {:.2f}% | Thresh {:.3f} Learning Rate: {:.7f} ".format(
                        model_names[0], epoch, step_idx, loss, ba * 100, thresh_idx/batch_size*100, threshold,
                        optimizer.state_dict()['param_groups'][0]['lr']))
            step_idx += 1
            # break

        ############ eval ############
        early_classifier.eval()
        hit_exit = 0
        exit_all = 0
        hit_non_exit = 0
        with torch.no_grad():
            for batch_idx, (audio_data, label_kw) in enumerate(dev_dataloader):
                if train_on_gpu:
                    audio_data = audio_data.cuda()
                    label_kw = label_kw.cuda()
                outs = model(x=audio_data)
                [feat0, feat1, out_b] = outs
                out0 = early_classifier(feat0)
                batch_size = label_kw.shape[0]
                idxes = torch.arange(0, batch_size)

                max_prob_0,_ = out0.max(dim=1)
                sorted_prob, sorted_indices = max_prob_0.sort(dim=0, descending=True)
                thresh_idx = (sorted_prob > threshold).sum()

                exit_indices = sorted_indices[:thresh_idx]
                non_exit_indices = sorted_indices[thresh_idx:]
                exit_out = out0[exit_indices]
                non_exit_out = out0[non_exit_indices]

                b_exit_hit = float(torch.sum(torch.argmax(exit_out, 1) == label_kw[exit_indices]).item())
                b_non_exit_hit = float(torch.sum(torch.argmax(non_exit_out, 1) == label_kw[non_exit_indices]).item())
                exit_all += exit_out.shape[0]
                hit_exit += b_exit_hit
                hit_non_exit += b_non_exit_hit

                loss = criterion(out0, label_kw)
                valid_loss += loss.item() * audio_data.size(0)
                b_hit = float(torch.sum(torch.argmax(out0, 1) == label_kw).item())
                ba = b_hit / float(audio_data.shape[0])
                if (batch_idx % 100 == 0):
                    print("{} | Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.2f} |  ".format(
                        model_names[0], epoch, step_idx, loss, ba * 100))
                valid_kw_correct += b_hit
                step_idx += 1
            # break
            # Loss statistics
        train_loss = train_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(dev_dataloader.dataset)
        train_kw_accuracy = 100.0 * (train_kw_correct / len(train_dataloader.dataset))
        valid_kw_accuracy = 100.0 * (valid_kw_correct / len(dev_dataloader.dataset))
        valid_acc_exit = 100 * hit_exit / exit_all
        valid_acc_non_exit = -1 if(len(dev_dataloader.dataset) - exit_all)==0 \
            else 100 * hit_non_exit / (len(dev_dataloader.dataset) - exit_all)

        print("===========================================================================")
        print("{} | EPOCH #{}     | TRAIN ACC: {:.2f}%\t|  TRAIN LOSS : {:.2f} |".format(
            model_names[0], epoch, train_kw_accuracy,train_loss))
        print("                         | VAL ACC :  {:.2f}%\t | ACC_exit: {:.2f}%\t| ACC_non_exit {:.2f}%\t|VAL LOSS   : {:.2f}".format(
            valid_kw_accuracy, valid_acc_exit,valid_acc_non_exit, valid_loss))
        # print("Validation path count:   ", path_count)
        # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")


def partition_batch(out, label, thresh, ratio):
    par_batch_size = out.shape[0]
    max_prob_0, _ = out.max(dim=1)
    sorted_prob, sorted_indices = max_prob_0.sort(dim=0, descending=True)
    exit_indices = sorted_indices[round(ratio[0] * par_batch_size):round(ratio[1] * par_batch_size)]
    # cur_thresh = out[exit_indices[-1]][label_kw[exit_indices[-1]]]
    cur_thresh = out[exit_indices[-1]].max() if par_batch_size!=0 else thresh
    threshold = 0.9 * thresh + 0.1 * cur_thresh
    thresh_idx = (sorted_prob > threshold).sum()

    exit_indices = sorted_indices[:thresh_idx]
    non_exit_indices = sorted_indices[thresh_idx:]
    exit_out = out[exit_indices]
    non_exit_out = out[non_exit_indices]
    exit_label = label[exit_indices]
    non_exit_label = label[non_exit_indices]
    return [threshold,thresh_idx], [exit_out, non_exit_out], [exit_label, non_exit_label]

def infer_batch(outs, label, thresholds):
    exit_outs=[]
    exit_labels=[]
    exit_indices=[]
    for i in range(len(outs)):
        out = outs[i]
        max_prob,_ = out.max(dim=1)
        for index in exit_indices:
                max_prob[index] = 0
        if i == (len(outs)-1):
            t = max_prob >0
        else:
            t = max_prob>=thresholds[i]
        exit_index = torch.squeeze(torch.nonzero(t))
        exit_indices.append(exit_index)
        exit_outs.append(out[exit_index])
        exit_labels.append(label[exit_index])

    return exit_outs, exit_labels, exit_indices

# def cal_ratio(ratios,e_idx):
#     if e_idx ==0:
#         return ratios[0]
#     else:
#         ratio = 1
#         for i in range(e_idx-1):
#             ratio *= (1-ratios[i])
#         ratio *= ratios[e_idx]
#         return ratio

























