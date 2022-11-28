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

import lib.utils as utl
from configs.jaad import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils_cvae import train_d, val, test_d, self_train
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))


    model = build_model(args)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):    
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']
  

    

    criterion = rmse_loss().to(device)
    
    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')

    print("Number of train samples:", train_gen.__len__())
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())

    
    # train
    min_loss = 1e6
    best_model = None

    tb = SummaryWriter()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
            #print("Number of training samples:", len(train_gen))

            # self-train on val and train on train
            self_train_goal_loss, self_train_dec_loss, self_train_total_loss = self_train(model, val_gen, train_gen, criterion,
                                                                                          optimizer, device)

            # train
            #train_goal_loss, train_dec_loss, total_train_loss = train_d(model, train_gen, criterion, optimizer, device)

            #print('Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
                    #epoch, train_goal_loss, train_dec_loss, total_train_loss))


            # val
            #val_loss = val(model, val_gen, criterion, device)
            #

            # test
            test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = test_d(model, test_gen, criterion, device)
            print("MSE_15: %4f;  MSE_05: %4f;  MSE_10: %4f;   CMSE: %4f\n" % (MSE_15, MSE_05, MSE_10, CMSE))
            lr_scheduler.step(test_loss)

            tb.add_scalar("MSE_05", MSE_05, epoch)
            tb.add_scalar("MSE_15", MSE_15, epoch)
            tb.add_scalar("MSE_10", MSE_10, epoch)
            tb.add_scalar("CMSE", CMSE, epoch)
            tb.add_scalar("CFMSE", CFMSE, epoch)

            # save checkpoints if loss decreases
            if MSE_05 < min_loss:
                try:
                    os.remove(best_model)
                except:
                    pass

                min_loss = MSE_05
                saved_model_name = 'epoch_' + str(format(epoch,'03')) + '_bestMSE05_%.4f'%min_loss + '.pth'

                print("Saving checkpoints: " + saved_model_name )
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)

                save_dict = {   'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}
                torch.save(save_dict, os.path.join(save_dir, saved_model_name))
                best_model = os.path.join(save_dir, saved_model_name)

    tb.close()

if __name__ == '__main__':
    main(parse_args())