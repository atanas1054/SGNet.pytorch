import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

from lib.utils.eval_utils import eval_ethucy, eval_ethucy_cvae
from lib.losses import cvae, cvae_multi
from sklearn.utils import shuffle

#torch.autograd.set_detect_anomaly(True)

def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    count = 0
    total_goal_loss = 0
    total_dec_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            # if batch_idx > 1:
            #     break
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = target_traj, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj, first_history_index[0])

            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            train_loss = goal_loss + cvae_loss  + KLD_loss.mean()

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= count
    total_cvae_loss/= count
    total_KLD_loss/= count
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss


def self_train(model, val_gen, train_gen, criterion, mmd_criterion, optimizer, device):

    total_goal_loss = 0
    total_dec_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    count = 0
    ps_labels = []
    ps_input = []
    ps_first_history_index = []
    target_features = []

    val_input = []
    val_labels = []
    val_first_history_index = []
    train_labels = []
    train_input = []
    train_first_history_index = []
    source_features = []

    val_loader = tqdm(val_gen, total=len(val_gen))
    train_loader = tqdm(train_gen, total=len(train_gen))


    # get pseudo-labels
    with torch.set_grad_enabled(False):
          for batch_idx, data in enumerate(val_loader):
             model.train()
             first_history_index = data['first_history_index']
             assert torch.unique(first_history_index).shape[0] == 1
             input_traj = data['input_x'].to(device)
             pred_trajs = []
             for s in range(30):
                 cvae_dec_hidden, all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = None, start_index = first_history_index, training =  False)
                 #pred_traj = torch.mean(cvae_dec_traj, 3)
                 pred_traj = cvae_dec_traj[:,:,:,0,:]
                 pred_trajs.append(pred_traj)

             pred_trajs = torch.stack(pred_trajs)
             pred_traj_mean = torch.mean(pred_trajs, 0)
             pred_trajs_ = pred_trajs[:,:,-1,:,:]
             #samples, batch, 12, 2 -> samples, batch, 12*2
             pred_trajs_ = pred_trajs_.view(pred_trajs_.size(0), pred_trajs_.size(1), -1)
             #swap samples with batch
             pred_trajs_ = pred_trajs_.transpose(0, 1)
             # batch, samples, 12*2 -> batch, 12*2*samples
             pred_trajs_ = pred_trajs_.reshape(pred_trajs_.shape[0], pred_trajs_.shape[1]*pred_trajs_.shape[2])
             pred_trajs_var = torch.var(pred_trajs_, 1)

             #filter pseudo trajectories above certain variance
             pseudo_labels_certain = pred_trajs_var < 99
             ps_labels.append(pred_traj_mean[pseudo_labels_certain])

             ps_input.append(input_traj[pseudo_labels_certain])
             ps_first_history_index.append(first_history_index[pseudo_labels_certain])



    # get val data
    #for batch_idx, data in enumerate(val_loader):

        #first_history_index = data['first_history_index']
        #assert torch.unique(first_history_index).shape[0] == 1
        #input_traj = data['input_x'].to(device)
        #target_traj = data['target_y'].to(device)
        #val_labels.append(target_traj)
        #val_input.append(input_traj)
        #val_first_history_index.append(first_history_index)

    # get train data
    for batch_idx, data in enumerate(train_loader):
        # batch_size = data['input_x'].shape[0]
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1
        input_traj = data['input_x'].to(device)
        target_traj = data['target_y'].to(device)
        train_labels.append(target_traj)
        train_input.append(input_traj)
        train_first_history_index.append(first_history_index)


    #input_combined = ps_input + train_input
    #labels_combined = ps_labels + train_labels

    #first_history_index_combined = ps_first_history_index + train_first_history_index

    #input_combined_shuffled, labels_combined_shuffled, first_history_index_shuffled = shuffle(np.array(input_combined),
                                                                                              #np.array(labels_combined),
                                                                                              #first_history_index_combined)

    model.train()
    with torch.set_grad_enabled(True):
        #for i, l, h in zip(ps_input, ps_labels, ps_first_history_index):
        for i_t, l_t, h_t, i_s, l_s, h_s in zip(ps_input, ps_labels, ps_first_history_index, train_input, train_labels, train_first_history_index):

            first_history_index_t = h_t
            input_traj_t = i_t
            target_traj_t = l_t
            first_history_index_s = h_s
            input_traj_s = i_s
            target_traj_s = l_s

            batch_size = input_traj.shape[0]
            count += batch_size

            #target
            cvae_dec_hidden_t, all_goal_traj_t, cvae_dec_traj_t, KLD_loss_t, _ = model(input_traj_t, map_mask=None, targets=target_traj_t,
                                                              start_index=first_history_index_t, training=False)
            cvae_loss_t = cvae_multi(cvae_dec_traj_t, target_traj_t, first_history_index_t[0])

            goal_loss_t = criterion(all_goal_traj_t[:, first_history_index_t[0]:, :, :],
                                  target_traj_t[:, first_history_index_t[0]:, :, :])

            #source
            cvae_dec_hidden_s, all_goal_traj_s, cvae_dec_traj_s, KLD_loss_s, _ = model(input_traj_s, map_mask=None,
                                                                                       targets=target_traj_s,
                                                                                       start_index=first_history_index_s,
                                                                                       training=False)
            cvae_loss_s = cvae_multi(cvae_dec_traj_s, target_traj_s, first_history_index_s[0])

            goal_loss_s = criterion(all_goal_traj_s[:, first_history_index_s[0]:, :, :],
                                    target_traj_s[:, first_history_index_s[0]:, :, :])
            features_min = torch.min(torch.tensor([cvae_dec_hidden_t.shape[0], cvae_dec_hidden_s.shape[0]]))
            #print(features_min)

            #print("target features ", cvae_dec_hidden_t.shape)
            #print("source features ", cvae_dec_hidden_s.shape)
            mmd_loss = mmd_criterion(torch.mean(cvae_dec_hidden_t[:features_min,:,:], 1), torch.mean(cvae_dec_hidden_s[:features_min,:,:], 1))

            #total_loss = (goal_loss_s) + (cvae_loss_s) + (KLD_loss_s.mean()) + mmd_loss
            total_loss = goal_loss_s + cvae_loss_s + KLD_loss_s.mean() + (10*mmd_loss)

            #total_goal_loss += goal_loss.item() * batch_size
            #total_cvae_loss += cvae_loss.item() * batch_size
            #total_KLD_loss += KLD_loss.mean() * batch_size

            # optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        #total_goal_loss /= count
        #total_cvae_loss /= count
        #total_KLD_loss /= count

        return total_goal_loss, total_cvae_loss, total_KLD_loss


def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    count = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            # if batch_idx > 1:
            #     break
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            cvae_dec_hidden, all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = None, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            

            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

    val_loss = total_goal_loss/count \
         + total_cvae_loss/count+ total_KLD_loss/ count

    return val_loss

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            # if batch_idx > 1:
            #     break
            
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            cvae_dec_hidden, all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = None, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()
            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']
            

    
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    

    test_loss = total_goal_loss/count + total_cvae_loss/count + total_KLD_loss/count
    # print("Test Loss %4f\n" % (test_loss))
    # print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
    return test_loss, ADE_08, FDE_08, ADE_12, FDE_12

def evaluate(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    all_file_name = []
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):            
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            scene_name = data['scene_name'] 
            timestep = data['timestep']
            current_img = timestep
            #import pdb; pdb.set_trace()
            # filename = datapath + '/test/biwi_eth.txt'
            # data = pd.read_csv(filename, sep='\t', index_col=False, header=None)
            # data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
            # frame_id_min = data['frame_id'].min()
            # filename path = os.path.join(datapath, dataset ,str((current_img[1][0]+int(frame_id_min)//10)*10).zfill(5) + '.png')

            all_goal_traj, cvae_dec_traj, KLD_loss = model(input_traj, target_traj, first_history_index, False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            #import pdb;pdb.set_trace()
            # Decoder
            # batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU =\
            #     eval_jaad_pie(input_traj_np, target_traj_np, all_dec_traj_np)
            batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

            if batch_idx == 0:
                all_input = input_traj_np
                all_target = target_traj_np
                all_prediction = cvae_dec_traj_np
            else:
                all_input = np.vstack((all_input,input_traj_np))
                all_target = np.vstack((all_target,target_traj_np))
                all_prediction = np.vstack((all_prediction,cvae_dec_traj_np))
            all_file_name.extend(current_img)

            

    
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    
    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

    return all_input,all_target,all_prediction,all_file_name

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
