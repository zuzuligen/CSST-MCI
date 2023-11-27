import torch
from torch import nn

class Bottleneck(nn.Module):
    #The multiple of the expansion in each stage
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''
        :param inplanes:  The number of channels before the block
        :param planes: The number of channels when processing in the middle of a block
                planes*self.extention: Output
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):

        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        if self.downsample is not None:
            residual=self.downsample(x)

        out = out+residual
        out=self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, embed_dim=256, block=Bottleneck):
        self.inplane = 64
        super(ResNet, self).__init__()

        self.block=block
        self.layers=layers

        self.conv1=nn.Conv2d(1, self.inplane, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)

        #full connected layer
        self.fc=nn.Linear(98304, embed_dim)
       # self.fc = nn.Linear(393216, embed_dim)   # PFC case


    def forward(self,x):
        x1 = x.view(x.size(0) * x.size(1), 1, x.size(2), x.size(3))

        #stem part：conv+bn+maxpool
        out=self.conv1(x1)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block part
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)

        out=torch.flatten(out,1)
        out=self.fc(out)

        out = out.view(x.size(0), x.size(1), -1)
        return out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block
        :param plane: The dimension of the intermediate operation of each module which is generally equal to a quarter of the output dimension
        :param block_num: Number of repetitions
        :param stride:
        :return:
        '''
        block_list=[]
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )

        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)

# if __name__ == '__main__':
#     resnet = ResNet([3, 4, 6, 3], 256)
#     x = torch.randn(1, 5, 192, 256)
#     X = resnet(x)
#     print(X.shape)
