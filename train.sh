# !/bin/bash
echo " Running Training EXP"

python tools/ethucy/train_cvae.py --gpu 0 --dataset ETH --target_dataset HOTEL --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/A2B_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ETH/SGNet_CVAE/0.5/1/epoch_091_bestValLoss_12.7729.pth && echo "A2B launched." &
P0=$!

wait $P0
python tools/ethucy/train_cvae.py --gpu 0 --dataset ETH --target_dataset UNIV --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/A2C_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ETH/SGNet_CVAE/0.5/1/epoch_091_bestValLoss_12.7729.pthh && echo "A2C launched." &
P1=$!

wait $P1
python tools/ethucy/train_cvae.py --gpu 0 --dataset ETH --target_dataset ZARA1 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/A2D_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ETH/SGNet_CVAE/0.5/1/epoch_091_bestValLoss_12.7729.pth && echo "A2D launched." &
P2=$!

wait $P2
python tools/ethucy/train_cvae.py --gpu 0 --dataset ETH --target_dataset ZARA2 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/A2E_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ETH/SGNet_CVAE/0.5/1/epoch_091_bestValLoss_12.7729.pth && echo "A2E launched." &
P3=$!

wait $P3
python tools/ethucy/train_cvae.py --gpu 0 --dataset HOTEL --target_dataset ETH --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/B2A_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/HOTEL/SGNet_CVAE/0.5/1/epoch_095_bestValLoss_3.2360.pth && echo "B2A launched." &
P4=$!

wait $P4
python tools/ethucy/train_cvae.py --gpu 0 --dataset HOTEL --target_dataset UNIV --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/B2C_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/HOTEL/SGNet_CVAE/0.5/1/epoch_095_bestValLoss_3.2360.pth && echo "B2C launched." &
P5=$!

wait $P5
python tools/ethucy/train_cvae.py --gpu 0 --dataset HOTEL --target_dataset ZARA1 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/B2D_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/HOTEL/SGNet_CVAE/0.5/1/epoch_095_bestValLoss_3.2360.pth && echo "B2D launched." &
P6=$!

wait $P6
python tools/ethucy/train_cvae.py --gpu 0 --dataset HOTEL --target_dataset ZARA2 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/B2E_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/HOTEL/SGNet_CVAE/0.5/1/epoch_095_bestValLoss_3.2360.pth && echo "B2E launched." &
P7=$!

wait $P7
python tools/ethucy/train_cvae.py --gpu 0 --dataset UNIV --target_dataset ETH --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/C2A_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/UNIV/SGNet_CVAE/0.5/1/epoch_014_bestValLoss_6.6584.pth  && echo "C2A launched." &
P8=$!

wait $P8
python tools/ethucy/train_cvae.py --gpu 0 --dataset UNIV --target_dataset HOTEL --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/C2B_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/UNIV/SGNet_CVAE/0.5/1/epoch_014_bestValLoss_6.6584.pth  && echo "C2B launched." &
P9=$!

wait $P9
python tools/ethucy/train_cvae.py --gpu 0 --dataset UNIV --target_dataset ZARA1 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/C2D_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/UNIV/SGNet_CVAE/0.5/1/epoch_014_bestValLoss_6.6584.pth  && echo "C2D launched." &
P10=$!

wait $P10
python tools/ethucy/train_cvae.py --gpu 0 --dataset UNIV --target_dataset ZARA2 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/C2E_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/UNIV/SGNet_CVAE/0.5/1/epoch_014_bestValLoss_6.6584.pth  && echo "C2E launched." &
P11=$!

wait $P11
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA1 --target_dataset ETH --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/D2A_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA1/SGNet_CVAE/0.5/1/epoch_026_bestValLoss_5.7619.pth  && echo "D2A launched." &
P12=$!

wait $P12
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA1 --target_dataset HOTEL --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/D2B_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA1/SGNet_CVAE/0.5/1/epoch_026_bestValLoss_5.7619.pth  && echo "D2B launched." &
P13=$!

wait $P13
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA1 --target_dataset UNIV --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/D2C_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA1/SGNet_CVAE/0.5/1/epoch_026_bestValLoss_5.7619.pth  && echo "D2C launched." &
P14=$!

wait $P14
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA1 --target_dataset ZARA2 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/D2E_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA1/SGNet_CVAE/0.5/1/epoch_026_bestValLoss_5.7619.pth  && echo "D2E launched." &
P15=$!

wait $P15
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA2 --target_dataset ETH --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/E2A_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA2/SGNet_CVAE/0.5/1/epoch_015_bestValLoss_4.3694.pth  && echo "E2A launched." &
P16=$!

wait $P16
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA2 --target_dataset HOTEL --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/E2B_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA2/SGNet_CVAE/0.5/1/epoch_015_bestValLoss_4.3694.pth  && echo "E2B launched." &
P17=$!

wait $P17
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA2 --target_dataset UNIV --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/E2C_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA2/SGNet_CVAE/0.5/1/epoch_015_bestValLoss_4.3694.pth  && echo "E2C launched." &
P18=$!

wait $P18
python tools/ethucy/train_cvae.py --gpu 0 --dataset ZARA2 --target_dataset ZARA1 --model SGNet_CVAE --lr 0.00001 --exp_name 'runs/E2D_SGNetCVAE_MMD' --epochs 10 --checkpoint tools/ethucy/checkpoints/ZARA2/SGNet_CVAE/0.5/1/epoch_015_bestValLoss_4.3694.pth && echo "E2D launched." &
P19=$!