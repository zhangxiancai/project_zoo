export CUDA_VISIBLE_DEVICES='0,1,2,3,4'
nohup python3 train.py  >log/train_d04_07_A0_clean_lr0.0000025.log 2>&1 &