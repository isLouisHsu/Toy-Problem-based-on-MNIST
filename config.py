from easydict import EasyDict

configer = EasyDict()

configer.ckptdir = './ckpt'
configer.logdir = './log'

configer.inputsize = (1, 28, 28)    # (C, H, W)
configer.batchsize = 256
configer.n_epoch = 50
configer.valid_freq = 1

configer.lrbase = 0.0001
configer.adjstep = [35]
configer.gamma = 0.1

configer.cuda = True

