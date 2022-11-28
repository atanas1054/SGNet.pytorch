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

from lib.utils.eval_utils import eval_ethucy
from sklearn.utils import shuffle


def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    count = 0
    total_goal_loss = 0
    total_dec_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            # target_bbox_st = data['target_y_st'].to(device)

            all_goal_traj, all_dec_traj = model(input_traj, first_history_index[0])

            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            dec_loss = criterion(all_dec_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size


            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= count
    total_dec_loss /= count

    
    return total_goal_loss, total_dec_loss, total_goal_loss + total_dec_loss

def self_train(model, val_gen, train_gen, criterion, optimizer, device):

    model.eval()
    total_goal_loss = 0
    total_dec_loss = 0
    count = 0
    ps_labels = []
    ps_input = []
    ps_first_history_index = []
    val_input = []
    val_labels = []
    val_first_history_index = []
    train_labels = []
    train_input = []
    train_first_history_index = []
    
    val_loader = tqdm(val_gen, total=len(val_gen))
    train_loader = tqdm(train_gen, total=len(train_gen))

    # get pseudo-labels
    # with torch.set_grad_enabled(False):
    #      for batch_idx, data in enumerate(val_loader):
    #         first_history_index = data['first_history_index']
    #         assert torch.unique(first_history_index).shape[0] == 1
    #         input_traj = data['input_x'].to(device)
    #         all_goal_traj_pred, all_dec_traj_pred = model(input_traj, first_history_index[0])
    #         if not (torch.any(all_dec_traj_pred.isnan())) and not (torch.any(all_dec_traj_pred < 0)):
    #             ps_labels.append(all_dec_traj_pred)
    #             ps_input.append(input_traj)
    #             ps_first_history_index.append(first_history_index)

    #get val data
    for batch_idx, data in enumerate(val_loader):
        # batch_size = data['input_x'].shape[0]
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1
        input_traj = data['input_x'].to(device)
        target_traj = data['target_y'].to(device)
        val_labels.append(target_traj)
        val_input.append(input_traj)
        val_first_history_index.append(first_history_index)

    # get train data
    for batch_idx, data in enumerate(train_loader):
        # batch_size = data['input_x'].shape[0]
        first_history_index = data['first_history_index']
        assert torch.unique(first_history_index).shape[0] == 1
        input_traj = data['input_x'].to(device)
        print(input_traj.shape)
        target_traj = data['target_y'].to(device)
        train_labels.append(target_traj)
        train_input.append(input_traj)
        train_first_history_index.append(first_history_index)

    input_combined = val_input + train_input
    labels_combined = val_labels + train_labels
    first_history_index_combined = val_first_history_index + train_first_history_index

    input_combined_shuffled, labels_combined_shuffled, first_history_index_shuffled = shuffle(np.array(input_combined), np.array(labels_combined), 
                                                                                             first_history_index_combined)

    model.train()
    with torch.set_grad_enabled(True):
        for i, l, h in zip(input_combined_shuffled, labels_combined_shuffled, first_history_index_shuffled):
            first_history_index = h
            input_traj = i
            target_traj = l
            batch_size = input_traj.shape[0]
            count += batch_size

            all_goal_traj, all_dec_traj = model(input_traj, first_history_index[0])

            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :],
                                  target_traj[:, first_history_index[0]:, :, :])
            dec_loss = criterion(all_dec_traj[:, first_history_index[0]:, :, :],
                                 target_traj[:, first_history_index[0]:, :, :])

            train_loss = (goal_loss + dec_loss)

            total_goal_loss += goal_loss.item() * batch_size
            total_dec_loss += dec_loss.item() * batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    total_goal_loss /= count
    total_dec_loss /= count

    return total_goal_loss, total_dec_loss, total_goal_loss + total_dec_loss

def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_dec_loss = 0
    count = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            # target_bbox_st = data['target_y_st'].to(device)

            all_goal_traj, all_dec_traj = model(input_traj, first_history_index[0])


            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            dec_loss = criterion(all_dec_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size

    val_loss = total_goal_loss/count + total_dec_loss/count
    return val_loss  

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_dec_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):

            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            # target_bbox_st = data['target_y_st'].to(device)

            all_goal_traj, all_dec_traj = model(input_traj, first_history_index[0])
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            dec_loss = criterion(all_dec_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size

            all_dec_traj_np = all_dec_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # Decoder
            batch_ADE_08, batch_FDE_08, batch_ADE_12, batch_FDE_12 =\
                eval_ethucy(input_traj_np, target_traj_np, all_dec_traj_np)

            ADE_08 += batch_ADE_08
            ADE_12 += batch_ADE_12
            FDE_08 += batch_FDE_08
            FDE_12 += batch_FDE_12
            
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    

    test_loss = total_goal_loss/count + total_dec_loss/count

    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
    return test_loss, ADE_08, FDE_08, ADE_12, FDE_12