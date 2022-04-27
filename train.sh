export CUDA_VISIBLE_DEVICES='0,1,2,3,4'
nohup python3 train.py  >log/train_d04_11_A0s-stu.log 2>&1 &