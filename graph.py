'''
pytorch计算图
'''

import torch
#
class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        '''
        :param ctx:
        :param x:上层结点值
        :return:
        '''
        ctx.save_for_backward(x) #
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx,grad_out):
        '''
        :param ctx:
        :param grad_out: 当前结点梯度
        :return: 上层结点梯度
        '''
        x,=ctx.saved_tensors
        grad_x=grad_out.clone()
        grad_x[x<0] = 0 #
        return grad_x

if __name__=='__main__':

    img=torch.tensor([-1,2,3])
    w=torch.tensor(2.0,requires_grad=True)
    relu=MyReLU.apply
    y1=img * w
    a1=relu(y1)
    loss=torch.sum(a1)
    loss.backward()
    print(w.grad)