#nohup python3 train.py >log/train_d04_15_baby-v3.log 2>&1 &

nohup python3 -m torch.distributed.launch --master_port 9998 --nproc_per_node 4 train.py --sync-bn >log/train_d06_10_v5x_resume.log 2>&1 &
