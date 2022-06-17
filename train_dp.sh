export CUDA_VISIBLE_DEVICES='0'
nohup python3 train_LPRNet_dp.py  >log/train_d04_20_double_plate_straight.log 2>&1 &