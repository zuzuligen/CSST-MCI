from dataset import MyDataset
import torch
import torch.nn as nn
from model import Seq2Seq
import numpy as np
import time
from torch.utils.data import DataLoader
from RCgenerate import MoveState
# from PFCgenerate import MoveState   # PFC case
import os
import re

outdir = './results/'
class parameters():
    def __init__(self):
        self.device="cuda:0"
        self.scale_level = 2  #3 for PFC case
        #parameters for generate new moves
        self.TotalState = 4  # Total State for optical element
        self.movelength = 5  # Number of move in one sequence
        self.TotalmoveN = 1
        #RC
        self.stepsize = 0.02
        self.threshold = 0.1
        #PFC case
        # self.stepsize1 = 0.02
        # self.stepsize2 = 0.002
        # self.threshold1 = 0.1
        # self.threshold2 = 0.01

        #parameters for training
        self.batch_size = 30
        self.epochs = 1000
        self.learning_rate= 0.00002
        self.input_size= 256
        self.num_layers= 3
        self.output_size= 4


para = parameters()

def extract_mse_from_filename(filename):
    pattern = r'MSE=([\d.e+-]+)\.pkl'
    match = re.search(pattern, filename)
    if match:
        mse_str = match.group(1)
        mse = float(mse_str)
        return mse
    return None

def find_min_mse_pkl(folder_path):
    min_mse = float('inf')
    min_mse_filename = None
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            mse = extract_mse_from_filename(filename)
            # Update the minimum MSE and the corresponding file name
            if mse is not None and mse < min_mse:
                min_mse = mse
                min_mse_filename = filename
    return min_mse_filename


def find_esix_indices(data):
    e_06_range = data > 1e-6
    e_06_indices = np.where(e_06_range)[0]
    return e_06_indices


def train(msemodel_path, pkl_str, labels, epochs, learning_rate):
    device = para.device
    batch_size = para.batch_size
    input_size = para.input_size
    hidden_size = 128
    output_size = para.output_size
    num_layers = para.num_layers
    datasets = MyDataset(labels)

    train_loader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=False)

    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size)
    model.load_state_dict(torch.load(msemodel_path+pkl_str))
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    bestmse = 10000
    process = []
    print('Begin Training')
    # loop over the dataset multiple times
    for epoch in range(epochs):
        timestart = time.time()

        model.train()
        allloss = []
        for i, data in enumerate(train_loader, 0):
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)

            optimizer.zero_grad()
            predict = model(imgs)

            #For PFC case
            # predict[:, :, 2:4] *= 10
            # targets[:, :, 2:4] *= 10

            loss1 = loss_fn(predict, targets)
            loss = torch.log(loss1)

            loss.backward()
            optimizer.step()
            allloss.append(loss1.item())

            print('\rEpoch:{} train:[{}/{}] lr:{} loss:{} logloss:{}'.format(epoch, (i + 1), len(train_loader),
                                                                            optimizer.param_groups[0]['lr'],
                                                                            loss1.item(), loss.item()), end='')
        #scheduler.step()
        timeend = time.time()
        process.append([epoch, np.mean(allloss)])
        if (epoch % 10) == 0 or (np.mean(allloss) < bestmse):
            torch.save(model.state_dict(),
                       r'{}/stepN={}_Epoch={}_MSE={}.pkl'.format(outdir + 'model/' + 'step' + str(stepN),
                                                                              stepN,epoch,np.mean(allloss)))
            if np.mean(allloss) < bestmse:
                bestmse = np.mean(allloss)

        np.save(os.path.join(outdir, 'processdata' + str(stepN) + '.npy'), process)

        print("Running time per epoch（minute）", (timeend - timestart) / 60)
    print('Finished Training')


def test( msemodel_path,pkl_str, labels):
    # pkl_str : str
    device = para.device
    input_size = para.input_size
    hidden_size = 128
    output_size = para.output_size
    num_layers = para.num_layers
    batch_size = 1
    # model initialize
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size)
    # Load Trained Weights
    model.load_state_dict(torch.load(msemodel_path+pkl_str))
    model.to(device)
    datasets = MyDataset(labels)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=True)
    loss_fn = nn.MSELoss()
    l = []
    Preds = []
    Tars = []
    model.eval()
    loader = dataloader
    for data in loader:
        torch.no_grad()
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.to(device)
            targets = targets.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, targets)
        preds = preds.detach().cpu().numpy()
        tars = targets.detach().cpu().numpy()
        Preds.append(preds)
        Tars.append(tars)

        print(" Loss: {}".format(loss.item()))
        l.append(loss.item())

    l = np.array(l)
    indicies = find_esix_indices(l)

    Tars = np.array(Tars)

    loss_maxmse = []
    test_maxmse_info = []
    for index in indicies:
        test_maxmse_info.append(Tars[index])
        loss_maxmse.append(l[index])

    test_maxmse_info = np.array(test_maxmse_info)
    loss_maxmse = np.array(loss_maxmse)

    return test_maxmse_info, loss_maxmse


if __name__ == '__main__':
    time1 = time.time()

    stepN = 0  # Step for Active Leaning
    merged_matrix = np.load('TestMoves.npy')  #random sequences for testing
    labels = np.array(merged_matrix)
    pkl_str = 'model.pkl'
    msemodel_path = os.path.join(outdir,'model/')

    while (stepN < 10):
        print('stepN', stepN)
        # 1.test
        test_maxmse_info, loss_maxmse = test( msemodel_path,pkl_str, labels)  # labels大小需要是（n，5,4）
        np.save(f'{outdir}info/{stepN}_test_maxmse_info', test_maxmse_info)
        np.save(f'{outdir}info/{stepN}_test_maxmse_lossinfo', loss_maxmse)

        #2.Generate new moves based on active learning
        newinfo = test_maxmse_info.reshape(-1, 4)
        newmoves = []
        for info in newinfo:
            MS = MoveState(info, para.TotalState, para.movelength, para.TotalmoveN, para.stepsize,para.threshold)
            # PFC case
            #MS = MoveState(info, para.TotalState, para.movelength, para.TotalmoveN, para.stepsize1,para.stepsize2, para.threshold1,para.threshold2)
            move = MS.move()
            move = np.round(move,para.scale_level)
            newmoves.append(move)
        newmoves = np.array(newmoves)
        np.save(outdir+'info/'+f'{stepN}_newmoves', newmoves)

        #3.Train based on new moves
        train(msemodel_path, pkl_str, newmoves, epochs=para.epochs, learning_rate=para.learning_rate)

        #4.find new model with best mse
        msemodel_path = outdir +'model/'+ 'step' + str(stepN) + '/'
        #5.Find the model with the smallest mse under msemodel_path
        pkl_str = find_min_mse_pkl(msemodel_path)
        np.save(outdir + 'info/' + f'{stepN}_minmsemodel', pkl_str)

        #5.continue
        stepN += 1
        labels = newmoves

    time2 = time.time()
    hour = (time2 - time1) / 3600
    print("Running time for all epochs：", hour)














