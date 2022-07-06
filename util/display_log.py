import re

import matplotlib.pyplot as plt # 画图
import numpy as np

def display_train_log():
    '''
    根据train_1234.log文件画loss，acc_val,lr
    :return:
    '''
    log='/home/xiancai/classification-pytorch/log/train_1234.log'
    with open(log,'r') as f:
        ls=f.readlines()

    # decode train_1234.log
    res_ill=[]
    res_a=[]
    for i in ls:
        if re.search('iters: ',i): # re.match只匹配字符串的开始, re.search匹配整个字符串
            spls=re.split('iters: |lr: |loss: ',i)
            del spls[0]
            rs=[]
            for ind,spl in enumerate(spls):
                rs.append(float(spl.split(',')[0])) # iter,lr,loss
            res_ill.append(rs)

        if re.search('acc_val: ',i):
            res_a.append(float(re.split('acc_val: ',i)[1].strip())) # acc_val   list: + = extend

    # draw
    scal_iters=50
    res_ill = np.array(res_ill)
    res_a = np.array(res_a)
    res_a=res_a[...,None]
    ys=np.concatenate((res_ill,res_a),axis=1) # iter,lr,loss,acc  ~*4
    # ys=ys[200000//100:,...]
    ns=['lr','loss','acc_val']

    for ind in range(3):
        plt.plot(ys[::scal_iters,0],ys[::scal_iters,ind+1])
        plt.xlabel('batchs')
        plt.ylabel(ns[ind])
        plt.savefig(f'/home/xiancai/classification-pytorch/log/png/train_1234_{ns[ind]}.png')
        plt.close()


if __name__=='__main__':
    display_train_log()