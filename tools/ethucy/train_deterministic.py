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

from configs.ethucy import parse_sgnet_args as parse_args
import lib.utils as utl
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.ethucy_train_utils import train, val, test, self_train
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                           min_lr=1e-10, verbose=1)
    model = model.to(device)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']
        del checkpoint


    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size = 1)
    val_gen = utl.build_data_loader(args, 'val', batch_size = 1)
    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    print("Number of train samples:", train_gen.__len__())
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

    tb = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):

        #train_goal_loss, train_dec_loss, total_train_loss = self_train(model, val_gen, criterion, optimizer, device)
        self_train_goal_loss, self_train_dec_loss, self_train_total_loss = self_train(model, val_gen, train_gen,
                                                                                      criterion,
                                                                                      optimizer, device)

        #print('Self-Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
            #epoch, train_goal_loss, train_dec_loss, total_train_loss))

        #train_goal_loss, train_dec_loss, total_train_loss = train(model, train_gen, criterion, optimizer, device)

        #print('Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
            #epoch, train_goal_loss, train_dec_loss, total_train_loss))

        # val
        #val_loss = val(model, val_gen, criterion, device)
        # lr_scheduler.step(val_loss)


        # test
        test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, test_gen, criterion, device)
        print("ADE_08: %4f;  FDE_08: %4f;  ADE_10: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
        lr_scheduler.step(test_loss)

        tb.add_scalar("test_loss", test_loss, epoch)
        tb.add_scalar("ADE_08", ADE_08, epoch)
        tb.add_scalar("FDE_08", FDE_08, epoch)
        tb.add_scalar("ADE_12", ADE_12, epoch)
        tb.add_scalar("FDE_12", FDE_12, epoch)

        # save checkpoints if performance increases
        if ADE_08 < min_loss:
            try:
                os.remove(best_model)
            except:
                pass

            min_loss = ADE_08
            saved_model_name = 'epoch_' + str(format(epoch, '03')) + '_bestADE08_%.4f' % min_loss + '.pth'

            print("Saving checkpoints: " + saved_model_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(save_dir, saved_model_name))
            best_model = os.path.join(save_dir, saved_model_name)

if __name__ == '__main__':
    main(parse_args())
