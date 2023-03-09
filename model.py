import torch
from torch import nn
import torch.nn.functional as F
from config import get_parameter

params = get_parameter()
class FC(nn.Module):
    def __init__(self, in_dim = 1024, out_dim = 2 * params['_M'] + 1):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)
    
    def forward(self, input_x):
        input_x = self.fc1(input_x)
        input_x = torch.tanh(input_x)
        input_x = self.fc2(input_x) #[B, F, 2, T, 31]
        return input_x

def init_layer(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform(m.weight, gain = nn.init.calculate_gain('relu'))

class CRN_Net(nn.Module):
    def __init__(self):
        super(CRN_Net, self).__init__()
        self.in_channels = [2, 16, 32, 64]#, 128]
        self.out_channels = [16, 32, 64, 128]#, 256]
        self.kernel_size = (3, 3)
        self.stride = (1, 2)
        self.input_size = 128 #2944 #1408 #
        self.hidden_size = 128 #2944 #1408 #
        self.num_layers = 1

        self.Convs = nn.Sequential()
        convNum = len(self.in_channels)
        for i in range(convNum):
            self.Convs.add_module('Cov-{}'.format(i), 
                nn.Conv2d(in_channels = self.in_channels[i],
                    out_channels = self.out_channels[i],
                    kernel_size = self.kernel_size,
                    stride = self.stride))
        self.rnn1 = getattr(nn, 'GRU')(self.input_size, self.hidden_size, 
                            self.num_layers, batch_first=True, bidirectional=False)
        self.rnn2 = getattr(nn, 'GRU')(self.input_size, self.hidden_size, 
                            self.num_layers, batch_first=True, bidirectional = False)
        self.deConvs = nn.Sequential()
        for i in range(convNum):
            self.deConvs.add_module('deConv-{}'.format(i),
                nn.ConvTranspose2d(in_channels = self.out_channels[-1 - i], 
                                out_channels = self.in_channels[-1 - i], 
                                kernel_size = self.kernel_size, 
                                stride = self.stride))
        self.out_lr1 = FC(31)
        # self.init_weight()
    
    def init_weight(self):
        init_layer(self.Convs)
        init_layer(self.rnn1)
        init_layer(self.rnn2)
        init_layer(self.deConvs)
        init_layer(self.out_lr1)

    def forward(self, x):
        # print(x.shape)
        batchsize = x.shape[0]
        F_cnt = x.shape[1]
        x = x.reshape([batchsize * F_cnt] + [x.shape[i] for i in range(2, len(x.shape))]) #[B*F, 2, T, 2M+1]
        x = self.Convs(x) #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = x.permute(0, 2, 1, 3) #[B*F, T, C, 2M+1]
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  #[B*F, T, C*2M+1]
        # print(x.shape)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x) 
        x = x.reshape(x.shape[0], x.shape[1], self.out_channels[-1], -1) #[B*F, T, C, 2M+1]
        x = x.permute(0, 2, 1, 3)  #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = self.deConvs(x)  #[B*F, 2, T, 2M+1]
        # print(x.shape)
        # print(x.shape)
        x = self.out_lr1(x)
        x = x.reshape([batchsize, F_cnt] + [x.shape[i] for i in range(1, len(x.shape))])  ##[B, F, 2, T, 2M+1]
        return x

class conv_trans(nn.Module):
    def __init__(self):
        super(conv_trans, self).__init__()
        self.in_channels = [2, 16, 32, 64]#, 128]
        self.out_channels = [16, 32, 64, 128]#, 256]
        self.kernel_size = (3, 3)
        self.stride = (1, 2)
        self.input_size = 128 #2944 #1408 #
        self.hidden_size = 128 #2944 #1408 #
        self.num_layers = 1
        self.nhead = 1

        self.Convs = nn.Sequential()
        convNum = len(self.in_channels)
        for i in range(convNum):
            self.Convs.add_module('Conv-{}'.format(i),
                    nn.Conv2d(in_channels = self.in_channels[i],
                    out_channels = self.out_channels[i],
                    kernel_size = self.kernel_size,
                    stride = self.stride))
        self.transencoderlayer = nn.TransformerEncoderLayer(d_model = self.input_size, 
                                        nhead = self.nhead, batch_first = True)
        self.transencoder = nn.TransformerEncoder(encoder_layer = self.transencoderlayer, 
                                        num_layers = self.num_layers)
       
        self.deConvs = nn.Sequential()
        for i in range(convNum):
            self.deConvs.add_module('deConv-{}'.format(i),
                nn.ConvTranspose2d(in_channels = self.out_channels[-1 - i], 
                                out_channels = self.in_channels[-1 - i], 
                                kernel_size = self.kernel_size, 
                                stride = self.stride))
        self.out_lr1 = FC(31)
        # self.init_weight()
    
    def init_weight(self):
        init_layer(self.Convs)
        init_layer(self.deConvs)
        init_layer(self.out_lr1)

    def forward(self, x):
        # print(x.shape)
        batchsize = x.shape[0]
        F_cnt = x.shape[1]
        x = x.reshape([batchsize * F_cnt] + [x.shape[i] for i in range(2, len(x.shape))]) #[B*F, 2, T, 2M+1]
        x = self.Convs(x) #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = x.permute(0, 2, 1, 3) #[B*F, T, C, 2M+1]
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  #[B*F, T, C*2M+1]
        # print(x.shape)
        x = self.transencoder(x)
        x = x.reshape(x.shape[0], x.shape[1], self.out_channels[-1], -1) #[B*F, T, C, 2M+1]
        x = x.permute(0, 2, 1, 3)  #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = self.deConvs(x)  #[B*F, 2, T, 2M+1]
        # print(x.shape)
        # print(x.shape)
        x = self.out_lr1(x)
        x = x.reshape([batchsize, F_cnt] + [x.shape[i] for i in range(1, len(x.shape))])  ##[B, F, 2, T, 2M+1]
        return x

class ConvATN(nn.Module):
    def __init__(self):
        super(ConvATN, self).__init__()
        self.in_channels = [2, 16, 32, 64]#, 128]
        self.out_channels = [16, 32, 64, 128]#, 256]
        self.kernel_size = (3, 3)
        self.stride = (1, 2)
        self.input_size = 128 #2944 #1408 #
        self.n_head = 4

        self.Convs = nn.Sequential()
        num_convs = len(self.in_channels)
        for i in range(num_convs):
            self.Convs.add_module('Conv-{}'.format(i),
                        nn.Conv2d(in_channels = self.in_channels[i], out_channels = self.out_channels[i], 
                                    kernel_size = self.kernel_size, stride = self.stride))
        self.atn = nn.MultiheadAttention(embed_dim = self.input_size, 
                                    num_heads = self.n_head, batch_first = True)
        self.deConvs = nn.Sequential()
        for i in range(num_convs):
            self.deConvs.add_module('deConvs-{}'.format(i),
                        nn.ConvTranspose2d(in_channels = self.out_channels[-1 - i], 
                                        out_channels = self.in_channels[-1 - i], 
                                        kernel_size = self.kernel_size, stride = self.stride))
        self.out_lr1 = FC(31)
    
    def forward(self, x):
        batchsize = x.shape[0]
        F_cnt = x.shape[1]
        x = x.reshape([batchsize * F_cnt] + [x.shape[i] for i in range(2, len(x.shape))]) #[B*F, 2, T, 2M+1]
        x = self.Convs(x) #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = x.permute(0, 2, 1, 3) #[B*F, T, C, 2M+1]
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  #[B*F, T, C*2M+1]
        # print(x.shape)
        x, _ = self.atn(x, x, x)
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], self.out_channels[-1], -1) #[B*F, T, C, 2M+1]
        x = x.permute(0, 2, 1, 3)  #[B*F, C, T, 2M+1]
        # print(x.shape)
        x = self.deConvs(x)  #[B*F, 2, T, 2M+1]
        # print(x.shape)
        # print(x.shape)
        x = self.out_lr1(x)
        x = x.reshape([batchsize, F_cnt] + [x.shape[i] for i in range(1, len(x.shape))])  ##[B, F, 2, T, 2M+1]
        return x

if __name__ == '__main__':
    model = ConvATN()
    input = torch.randn((8, 5, 2, 155, 31))
    out = model(input)
    print(out.shape)
