class Config(object):

    def __init__(self):

        # model
        self.model_name='resnet18'
        self.num_classes = 2
        self.finetune = True # 是否载入预训练模型
        self.load_model_path  = 'result/2022_02_17/resnet18_best_0.9749_epoch119.pth'

        # loss
        self.loss = 'focal_loss' # focal_loss

        # data
        self.test_root = '' # txt文件里的img所加的路径前缀
        self.test_list = ['/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_val.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/kag_val.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_val.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/wiki_val.txt',
                          '/data1/xiancai/FACE_DATA/dataset_gender_classify/seep_val.txt',
                          '/data1/xiancai/FACE_DATA/dataset_gender_classify/imdb_val.txt']
        # self.test_list = ['/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_val.txt']
        self.test_batch_size=128
        self.train_root = ''
        self.train_list = ['/data1/xiancai/FACE_DATA/dataset_gender_classify/AFAD_train.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/kag_train.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_train.txt',
                      '/data1/xiancai/FACE_DATA/dataset_gender_classify/wiki_train.txt' ,
                           '/data1/xiancai/FACE_DATA/dataset_gender_classify/seep_train.txt',
                           '/data1/xiancai/FACE_DATA/dataset_gender_classify/imdb_train.txt']
        # self.train_list = ['/data1/xiancai/FACE_DATA/dataset_gender_classify/UTKface_train.txt']

        self.train_batch_size = 128      # batch size
        self.num_workers = 8  # how many workers for loading data
        self.input_shape = (3, 112, 112)

        # opt, lr
        self.optimizer = 'sgd'            # optimizer should be sgd, adam
        self.weight_decay = 5e-4

        self.lr = 0.1         # initial learning rate
        self.lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
        self.milestones = [40, 80]  # adjust lr
        self.warmup = 0 # 不使用warmup
        self.max_epoch = 120   # max epoch

        # count
        self.print_freq = 100             # print info every N batch

        # save
        import time
        import os
        self.save_path=f'/home/xiancai/classification-pytorch/checkpoints/{str(int(time.time()))}/' # the path to save model
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)