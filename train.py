from config import get_parameter
from dataset import get_train_dataloader
from encoder import generate_base_matrix, generate_bessel_matrix
from model import CRN_Net, ConvATN, conv_trans
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
params = get_parameter()

M = params['_M']
if_sh = '_nosh'
envs = '_reverb'
logdir = '/mnt/hd16t/wdh/anc/simulate/simulate_fbnn/logs/' + params['model_path']
print(logdir)
print('if_sh =', len(if_sh)==0, envs)
print(params)

os.makedirs(os.path.join('./models', params['model_path']), exist_ok = True)
base_matrix = torch.from_numpy(generate_base_matrix(M = M)).type(torch.complex64).to(device)
base_matrix_inv = torch.linalg.pinv(base_matrix).type(torch.complex64).to(device)
bessel_matrix = torch.from_numpy(generate_bessel_matrix(M = M, R1 = 1.0)).type(torch.complex64).to(device)

xv = np.linspace(0.1, 5.9, 100)
yv = np.linspace(0.1, 5.9, 100)
X, Y = np.meshgrid(xv, yv)
grid = np.concatenate([np.expand_dims(X, [-1]), np.expand_dims(Y, [-1])], axis = -1)
grid = grid.reshape(-1 , 2)
inside_pos = []
for i in range(len(grid)):
    xr = grid[i][0]
    yr = grid[i][1]
    xr = xr #(xg / 100) * 6
    yr = yr #(yg / 100) * 6
    r = np.sqrt((xr - 3) ** 2 + (yr - 3) ** 2)
    if r < 1:
        inside_pos.append(i)
inside_pos = np.array(inside_pos)

def get_sh(F_noise):
    #F_noise:[B, F, T, Q]
    F_noise = F_noise.permute(0, 2, 1, 3) #[B, T, F, Q]
    BetaJ = torch.matmul(base_matrix_inv, F_noise.unsqueeze(-1)).squeeze(-1) #[B, T, F, 2 * M + 1]
    # print('mulbase', BetaJ[1, 1, :, :])
    BetaJ = BetaJ / bessel_matrix #[B, T, F, 2 * M + 1]
    BetaJ = BetaJ.permute(0, 2, 1, 3) #[B, F, T, 2M+1]
    BetaJ = torch.cat([BetaJ.real.unsqueeze(-3), BetaJ.imag.unsqueeze(-3)], 
                        axis = -3)#[B, F, 2, T, 2M+1]
    # BetaJ = BetaJ * 0.1
    return BetaJ

def get_nosh(F_noise):
    BetaJ = torch.cat([F_noise.real.unsqueeze(-3), F_noise.imag.unsqueeze(-3)], 
                        axis = -3)#[B, F, 2, T, Q]
    return BetaJ

def getdata_from_rirs(F_data, rirs):
    ###
    #F_data:[B, F, T, (NO)]
    #rirs:[F, NE, NO]
    ###
    if len(F_data.shape) == 3:
        F_data = torch.cat([F_data.unsqueeze(-1) for i in range(rirs.shape[-1])], axis = -1)
    F_data = F_data.permute(0, 2, 1, 3)
    getdata = torch.sum(F_data.unsqueeze(-2) * rirs, dim = -1) #[B, T, F, NE]
    getdata = getdata.permute(0, 2, 1, 3) #[B, F, T, NE]
    return getdata

def train():
    if params['model_type'] == 'ConvATN':
        model = ConvATN().to(device)
    elif params['model_type'] == 'conv_trans':
        model = conv_trans().to(device)
    else:
        model = CRN_Net().to(device)

    print(model)

    F_pri_rirs = np.load('../F_pri_rirs'+envs+'.npy')
    F_pri_rirs = torch.from_numpy(F_pri_rirs).type(torch.complex64).to(device) #[F, NE, NO]

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'], amsgrad=True)
    scheduler = ExponentialLR(optimizer, gamma = params['gamma'])
    writer = SummaryWriter(logdir)

    dataloader = get_train_dataloader()

    total_step = 0
    for epoch in range(params['epoch']):
        print('epoch', epoch)
        step = 0
        for batch in dataloader:
            data_in = batch['input'].to(device)
            data_trg = batch['target'].to(device)
           
            F_data_in = torch.stft(data_in, n_fft=params['n_fft'], win_length = params['n_fft'],
                                    hop_length=int(0.5 * params['n_fft']), center = False, return_complex = True)
            # print(F_data_in.shape)
            F_data_trg = torch.stft(data_trg, n_fft=params['n_fft'], win_length = params['n_fft'],
                                    hop_length=int(0.5 * params['n_fft']), center = False, return_complex = True)
            in_mics = getdata_from_rirs(F_data_in, F_pri_rirs)
            trg_mics = getdata_from_rirs(F_data_trg, F_pri_rirs) #[B, F, T, NE]

            if len(if_sh) == 0:
                input_data = get_sh(in_mics) #[B, F, 2, T, 2M+1]
                target_data = get_sh(trg_mics)
            else:
                input_data = get_nosh(in_mics)
                target_data = get_nosh(trg_mics) #[B, F, 2, T, NE]

            input_data_b = input_data[:, :20, :, :, :] #
            target_data_b = target_data[:, :20, :, :, :]
            
            out_data_b = model(input_data_b)

            loss = loss_fn(out_data_b, target_data_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            left_data = torch.zeros_like(target_data) + target_data
            left_data[:, :20, :, :, :] -= out_data_b  #[:, 3:8, :, :, :]
            
            snr = 10 * np.log10(np.sum(np.abs(left_data.cpu().detach().numpy()) ** 2)/ 
                                np.sum(np.abs(target_data.cpu().detach().numpy()) ** 2))
            writer.add_scalar(tag = 'train/loss', scalar_value=loss.item(), global_step = total_step)
            writer.add_scalar(tag = 'train/snr', scalar_value=snr, global_step = total_step)
            step += 1
            total_step += 1
            if step % 300 == 0:
                print(step, loss.item())
        if (epoch + 1) % 3 == 0:
            print(os.path.join(
                            os.path.join('./models', params['model_path'],
                            params['model_path']+'_'+str(epoch)+'.pth')))
            torch.save(model.state_dict(), os.path.join(
                            os.path.join('./models', params['model_path'],
                            params['model_path']+'_'+str(epoch)+'.pth')))
            scheduler.step()
    print(os.path.join(
                        os.path.join('./models', params['model_path'],
                        params['model_path']+'_final.pth')))
    torch.save(model.state_dict(), os.path.join(
                        os.path.join('./models', params['model_path'],
                        params['model_path']+'_final.pth')))
    writer.close()


def get_ish(sh):
    ####
    # sh:[B, F, T, 2M+1]
    ####
    sh = sh.permute(0, 2, 1, 3)  #[B, T, F, 2M+1]
    sh = sh * bessel_matrix
    sig = torch.matmul(base_matrix, sh.unsqueeze(-1)).squeeze(-1)  #[B, T, F, NE]
    sig = sig.permute(0, 2, 1, 3) #[B, F, T, NE]
    return sig

import soundfile as sf
def test_data(wavname = None):
    if params['model_type'] == 'ConvATN':
        model = ConvATN().to(device)
    elif params['model_type'] == 'conv_trans':
        model = conv_trans().to(device)
    else:
        model = CRN_Net().to(device)
    model_path = os.path.join(os.path.join('./models', params['model_path'],
                        params['model_path']+'_final.pth'))
    print(model_path)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict = state_dict)

    loss_fn = nn.MSELoss().to(device)

    wf = sf.SoundFile(wavname)
    test_data = np.array(wf.read(dtype = np.float32))
    test_data = test_data / np.max(np.abs(test_data))
    test_in = test_data[:-params['n_fft']]
    test_trg = test_data[params['n_fft']:]
    
    input_length = params['n_fft'] * 128
    data_length = len(test_in)
    print(data_length)

    F_pri_rirs = np.load('../F_pri_rirs'+envs+'.npy')
    F_pri_rirs = torch.from_numpy(F_pri_rirs).type(torch.complex64).to(device) #[F, NE, NO]
    F_sec_rirs = np.load('../F_sec_rirs'+envs+'.npy')
    F_sec_rirs = torch.from_numpy(F_sec_rirs).type(torch.complex64) #[F, NE, NO]
    F_sec_rirs_inv = torch.linalg.pinv(F_sec_rirs).to(device)  #[F, NO, NE]

    print(F_pri_rirs.shape)
    print(F_sec_rirs.shape)

    cal_loss = 0
    cnt_iter = 0
    rec_out_ls = []
    rec_data_trg = []
    for i in range(0, data_length, input_length):
        if i + input_length > data_length:
            break
        data_in = torch.from_numpy(test_in[i:i + input_length]).unsqueeze(0).to(device)
        data_trg = torch.from_numpy(test_trg[i:i + input_length]).unsqueeze(0).to(device)
        F_data_in = torch.stft(data_in, n_fft=params['n_fft'], win_length = params['n_fft'],
                                hop_length=int(0.5 * params['n_fft']), center = False, return_complex = True)
        # print(F_data_in.shape)
        F_data_trg = torch.stft(data_trg, n_fft=params['n_fft'], win_length = params['n_fft'],
                                hop_length=int(0.5 * params['n_fft']), center = False, return_complex = True)
                                #[B, F, T]
        in_mics = getdata_from_rirs(F_data_in, F_pri_rirs)
        trg_mics = getdata_from_rirs(F_data_trg, F_pri_rirs) #[B, F, T, NE]

        if len(if_sh) == 0:
            input_data = get_sh(in_mics) #[B, F, 2, T, 2M+1]
            target_data = get_sh(trg_mics)
        else:
            input_data = get_nosh(in_mics)
            target_data = get_nosh(trg_mics) #[B, F, 2, T, NE]

        input_data_b = input_data[:, :20, :, :, :] #[:, 3:8, :, :, :]
        target_data_b = target_data[:, :20, :, :, :]
        
        out_data_b = model(input_data_b) #[B, F_cnt, 2, T, NE]]
        loss = loss_fn(out_data_b, target_data_b)
        cal_loss += loss.item()
        cnt_iter += 1

        out_data = torch.zeros_like(target_data)
        out_data[:, :20, :, :, :] = out_data_b  #[:, 3:8, :, :, :]
        out_data = torch.complex(out_data[:, :, 0, :, :], out_data[:, :, 1, :, :]) #[B, F, T, 2M+1/NE]
        
        if len(if_sh) == 0:
            out_mics = get_ish(out_data) #[B, F, T, NE]
        else:
            out_mics = out_data
        #[F, NE, NO] [F, NO, 1]=[F, NE, 1] 
        out_ls = torch.matmul(F_sec_rirs_inv, 
                    out_mics.permute(0, 2, 1, 3).unsqueeze(-1)).squeeze(-1) #[B, T, F, NO]
        out_ls = out_ls.permute(0, 2, 1, 3) #[B, F, T, NO]
        # print(out_ls.shape)
        rec_out_ls.append(out_ls.cpu().detach().numpy())
        rec_data_trg.append(F_data_trg.cpu().detach().numpy())

    rec_out_ls = np.concatenate(rec_out_ls, axis = -2)
    rec_data_trg = np.concatenate(rec_data_trg, axis = -1)
    np.save('out_ls'+if_sh+'.npy', rec_out_ls)
    np.save('data_trg'+if_sh+'.npy', rec_data_trg)
    print(rec_out_ls.shape)
    print(rec_data_trg.shape)
    print('loss:', cal_loss / cnt_iter)

def test():
    out_ls = torch.from_numpy(np.load('out_ls'+if_sh+'.npy')).type(torch.complex64)[:, :, -128:, :].to(device)
    data_trg = torch.from_numpy(np.load('data_trg'+if_sh+'.npy')).type(torch.complex64)[:, :, -128:].to(device)

    F_pri_rirs_ins = np.load('../F_pri_rirs_ins'+envs+'.npy')[:, inside_pos, :]
    F_pri_rirs_ins = torch.from_numpy(F_pri_rirs_ins).type(torch.complex64).to(device) #[F, NE, NO]
    F_sec_rirs_ins = np.load('../F_sc_rirs'+envs+'.npy')[:, inside_pos, :]
    F_sec_rirs_ins = torch.from_numpy(F_sec_rirs_ins).type(torch.complex64).to(device) #[F, NE, NO]
    # F_pri_rirs = np.load('../F_pri_rirs'+envs+'.npy')
    # F_pri_rirs = torch.from_numpy(F_pri_rirs).type(torch.complex64).to(device) #[F, NE, NO]
    # F_sec_rirs = np.load('../F_sec_rirs'+envs+'.npy')
    # F_sec_rirs = torch.from_numpy(F_sec_rirs).type(torch.complex64).to(device) #[F, NE, NO]

    trg_ins = getdata_from_rirs(data_trg, F_pri_rirs_ins)  #[B, F, T, NE]
    out_ins = getdata_from_rirs(out_ls, F_sec_rirs_ins)
    trg_ins = trg_ins.permute(0, 3, 1, 2).squeeze(0)
    out_ins = out_ins.permute(0, 3, 1, 2).squeeze(0) #[NE, F, T]
    left_ins = trg_ins - out_ins
    snr = 10 * torch.log10(torch.sum(torch.abs(left_ins) ** 2) / torch.sum(torch.abs(trg_ins) ** 2))
    print(snr)

if __name__ == '__main__':
    # train()
    data_root = '/mnt/hd16t/wdh/anc/simulate/data/train_datas_ori'
    scene_list = os.listdir(data_root)
    for scene in scene_list[7:8]:
        wav_path = os.path.join(data_root, scene, 'ch15.wav')
        print(wav_path)
        test_data(wavname = wav_path)
        test()
