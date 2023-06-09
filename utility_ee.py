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
    evaluate_testset(model, test_dataloader)


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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
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

                with torch.autograd.set_detect_anomaly(True):
                    # loss_full.backward(retain_graph=True)
                    losses[e_idx].backward(retain_graph=True)
                    # loss_id.backward(retain_graph=True)
                    # loss_o.backward(retain_graph=True)
                loss = loss_full

                optimizer.step()

                train_loss += loss.item() * audio_data.size(0)
                ba = torch.zeros([e_count])
                b_hit = torch.zeros([e_count])
                for i in range(e_count):
                    b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                    ba[i] = b_hit[i] / float(audio_data.shape[0])
                    train_kw_correct[i] += b_hit[i]

                if (batch_idx % 100 == 0):
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

                if (batch_idx % 100 == 0):
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
            save = 1
            for i in range(e_count):
                if (valid_kw_accuracy[i] < previous_valid_accuracy[i]):
                    save = 0

            if (save == 1):
                previous_valid_accuracy = valid_kw_accuracy
                print("Saving current model...")
                model.save()
                model.save(is_onnx=0, name='e_{}_valacc_{:.3f}_{:.3f}_{:.3f}.pt'.format(epoch, valid_kw_accuracy[0],
                                                                                        valid_kw_accuracy[1],
                                                                                        valid_kw_accuracy[2]))
                # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
                # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))

            # Update scheduler (for decaying learning rate)
            # scheduler.step()

    # Final test
    evaluate_testset(model, test_dataloader)




