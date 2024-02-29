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
import copy

import lib.utils as utl
from configs.ethucy import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss, MMD_loss, coral_loss
from lib.utils.ethucy_train_utils_cvae import train, val, test, self_train
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset,model_name,str(args.dropout), str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    student_model = build_model(args)
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                           min_lr=1e-10, verbose=1)
    student_model = nn.DataParallel(student_model)
    teacher_model = copy.deepcopy(student_model)
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    if osp.isfile(args.checkpoint):

        checkpoint = torch.load(args.checkpoint, map_location=device)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = args.lr
        print("Learning rate: ")
        print(optimizer.param_groups[0]['lr'])
        args.start_epoch += checkpoint['epoch']
        del checkpoint

    criterion = rmse_loss().to(device)
    mmd_criterion = MMD_loss().to(device)
    coral_criterion = coral_loss().to(device)
    mse_criterion = nn.MSELoss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size = 1)
    val_gen = utl.build_data_loader(args, 'val', batch_size = 1)
    target_val = utl.build_data_loader(args, 'val_t', batch_size = 1)
    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    # train
    min_loss = 1e6
    min_ADE_08 = 10e5
    min_FDE_08 = 10e5
    min_ADE_12 = 10e5
    min_FDE_12 = 10e5
    best_model = None
    best_model_metric = None
    #parameter for controlling the weight of pseudo labels
    alpha = 2

    tb = SummaryWriter(args.exp_name)

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        #print("Number of training samples:", len(train_gen))

        # train
        #train_goal_loss, train_cvae_loss, train_KLD_loss = train(teacher_model, train_gen, criterion, optimizer, device)

        # self-train
        #train_goal_loss, train_cvae_loss, train_KLD_loss = self_train(teacher_model, student_model, target_val, train_gen, criterion, mmd_criterion, optimizer, device, alpha)
        alpha += (1 / args.epochs)
        # print('Train Epoch: ', epoch, 'Goal loss: ', train_goal_loss, 'Decoder loss: ', train_dec_loss, 'CVAE loss: ', train_cvae_loss, \
        #     'KLD loss: ', train_KLD_loss, 'Total: ', total_train_loss) 
        #print('Train Epoch: {} \t Goal loss: {:.4f}\t  CVAE loss: {:.4f}\t KLD loss: {:.4f}\t Total: {:.4f}'.format(
                #epoch,train_goal_loss, train_cvae_loss, train_KLD_loss, train_goal_loss + train_cvae_loss + train_KLD_loss ))


        # val
        #val_loss = val(teacher_model, val_gen, criterion, device)

        #tb.add_scalar("val_loss", val_loss, epoch)
        #lr_scheduler.step(val_loss)

        #save checkpoints if performance increases
        # if val_loss < min_loss:
        #     try:
        #         os.remove(best_model)
        #     except:
        #         pass
        #
        #     min_loss = val_loss
        #     saved_model_name = 'epoch_' + str(format(epoch, '03')) + '_bestValLoss_%.4f' % min_loss + '.pth'
        #
        #     print("Saving checkpoints: " + saved_model_name)
        #     if not os.path.isdir(save_dir):
        #         os.mkdir(save_dir)
        #
        #     save_dict = {'epoch': epoch,
        #                  'model_state_dict': model.state_dict(),
        #                  'optimizer_state_dict': optimizer.state_dict()}
        #     torch.save(save_dict, os.path.join(save_dir, saved_model_name))
        #     best_model = os.path.join(save_dir, saved_model_name)


        # test every 10 epochs
        if epoch % 1 == 0:
            ADE_08, FDE_08, ADE_12, FDE_12 = test(teacher_model, test_gen, criterion, device)
            #print("Test Loss: {:.4f}".format(test_loss))
            print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

            #tb.add_scalar("test_loss", test_loss, epoch)
            tb.add_scalar("ADE_08", ADE_08, epoch)
            tb.add_scalar("FDE_08", FDE_08, epoch)
            tb.add_scalar("ADE_12", ADE_12, epoch)
            tb.add_scalar("FDE_12", FDE_12, epoch)

if __name__ == '__main__':
    main(parse_args())
