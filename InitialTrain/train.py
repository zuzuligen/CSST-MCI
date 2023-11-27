from dataset import MyDataset
import torch
import torch.nn as nn
from model import Seq2Seq
import numpy as np
import time
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt



outdir = './psf/'
class parameters():
    def __init__(self):
        self.device="cuda:0"

        self.TotalState = 4  # Total State for optical element
        self.movelength = 5  # Number of move in one sequence
        self.TotalmoveN = 2000
        # RC
        self.stepsize = 0.02
        self.threshold = 0.1
        # PFC case
        # self.stepsize1 = 0.02
        # self.stepsize2 = 0.002
        # self.threshold1 = 0.1
        # self.threshold2 = 0.01

        self.batch_size= 30
        self.epochs = 10000
        self.learning_rate= 0.00002
        self.input_size= 256
        self.num_layers= 3
        self.output_size= 4


para = parameters()


def plot_loss(loss,  outdirs=outdir):
    plt.clf()
    epoch = range(len(loss))
    plt.plot(epoch, loss, "r", label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Process")
    plt.legend(loc=1)
    plt.savefig(outdirs + "Process.jpg")
    # plt.show()


def train( labels, epochs, learning_rate):
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
    #model.load_state_dict(torch.load('./DATA/model.pkl',map_location='cuda:0'))
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

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

            # 清空梯度
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
        # 更新学习率
        #scheduler.step()

        timeend = time.time()
        process.append([epoch, np.mean(allloss)])

        if (epoch % 10) == 0 or (np.mean(allloss) < bestmse):
            torch.save(model.state_dict(),
                       r'{}/Epoch={}_MSE={}.pkl'.format(outdir + 'model' ,epoch,np.mean(allloss)))
            if np.mean(allloss) < bestmse:
                bestmse = np.mean(allloss)

        plot_loss(loss=[i[1] for i in process])
        np.save(os.path.join(outdir, 'processdata.npy'), process)

        print("Running time per epoch（minute）", (timeend - timestart) / 60)
    print('Finished Training')



if __name__ == '__main__':
    time1 = time.time()
    labels = np.load('./DATA/moves.npy')
    train(labels,para.epochs,para.learning_rate)
    time2 = time.time()
    hour = (time2 - time1) / 3600
    print("所有epoch的运行时间(hour)：", hour)














