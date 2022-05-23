export CUDA_VISIBLE_DEVICES='1,2,3,4'

#python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 train_LPRNet_ddp.py --sync-bn

nohup python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 train_LPRNet_ddp.py --sync-bn >log/train_d05_21.log 2>&1 &

