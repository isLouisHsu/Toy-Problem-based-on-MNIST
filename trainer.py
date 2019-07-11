import os
import time
import numpy as np
from torchstat import stat
from collections import OrderedDict
from sklearn.metrics import adjusted_mutual_info_score

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from processbar import ProcessBar
from utils import getTime, accuracy

class SupervisedTrainer(object):
    """ Train Templet
    """

    def __init__(self, configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1):

        self.configer = configer
        self.valid_freq = valid_freq

        self.net = net
        
        ## directory for log and checkpoints
        self.logdir = os.path.join(configer.logdir, self.net._get_name())
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = configer.ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        
        ## datasets
        self.trainset = trainset
        self.validset = validset
        self.trainloader = DataLoader(trainset, configer.batchsize, True)
        self.validloader = DataLoader(validset, configer.batchsize, True)

        ## for optimization
        self.criterion = criterion
        self.optimizer = optimizer(params, configer.lrbase)
        self.lr_scheduler = lr_scheduler(self.optimizer, configer.adjstep, configer.gamma)
        self.writer = SummaryWriter(self.logdir)
        
        ## initialize
        self.valid_loss = float('inf')
        self.elapsed_time = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        self.save_times = 0
        self.num_to_keep = num_to_keep

        ## if resume
        if resume:
            self.load_checkpoint()

        ## print information
        if configer.cuda and cuda.is_available(): self.net.cuda()
            
        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("train samples:   {}k".format(len(trainset)/1000))
        print("valid samples:   {}k".format(len(validset)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(trainset)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("val frequency:   {}".format(self.valid_freq))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")

    def train(self):
        
        n_epoch = self.configer.n_epoch - self.cur_epoch
        print("Start training! current epoch: {}, remain epoch: {}".format(self.cur_epoch, n_epoch))

        bar = ProcessBar(n_epoch)
        loss_train = 0.; loss_valid = 0.
        acc_train  = 0.; acc_valid  = 0.

        for i_epoch in range(n_epoch):
            
            if self.configer.cuda and cuda.is_available(): cuda.empty_cache()

            self.cur_epoch += 1
            bar.step()

            self.lr_scheduler.step(self.cur_epoch)
            cur_lr = self.lr_scheduler.get_lr()[-1]
            self.writer.add_scalar('{}/lr'.format(self.net._get_name()), cur_lr, self.cur_epoch)

            loss_train, acc_train = self.train_epoch()
            # print("----------------------------------------------------------------------------------------------")
            
            if self.valid_freq != 0 and self.cur_epoch % self.valid_freq == 0:
                loss_valid, acc_valid = self.valid_epoch()
            # print("----------------------------------------------------------------------------------------------")

            self.writer.add_scalars('{}/loss'.format(self.net._get_name()), {'train': loss_train, 'valid': loss_valid}, self.cur_epoch)
            self.writer.add_scalars('{}/acc'.format(self.net._get_name()),  {'train': acc_train,  'valid': acc_valid }, self.cur_epoch)

            # print_log = "{} || Elapsed: {:.4f}h || Epoch: [{:3d}]/[{:3d}] || lr: {:.6f},| train loss: {:4.4f}, valid loss: {:4.4f}".\
            #         format(getTime(), self.elapsed_time/3600, self.cur_epoch, self.configer.n_epoch, 
            #             cur_lr, loss_train, loss_valid)
            # print(print_log)

            if self.valid_freq == 0:
                self.save_checkpoint()
                
            else:
                if loss_valid < self.valid_loss:
                    self.valid_loss = loss_valid
                    self.save_checkpoint()
                
            # print("==============================================================================================")


    def train_epoch(self):
        
        self.net.train()
        avg_loss = []; avg_acc = []
        start_time = time.time()
        n_batch = len(self.trainset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.trainloader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            loss_i = self.criterion(y_pred, y)
            acc_i  = accuracy(y_pred, y)

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            avg_loss += [loss_i.detach().cpu().numpy()]; avg_acc += [acc_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/train/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/train/acc_i'.format(self.net._get_name()), acc_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
            self.elapsed_time += duration_time
            total_time = duration_time * self.configer.n_epoch * len(self.trainset) // self.configer.batchsize
            left_time = total_time - self.elapsed_time

        avg_loss = np.mean(np.array(avg_loss))
        avg_acc  = np.mean(np.array(avg_acc))
        return avg_loss, avg_acc


    def valid_epoch(self):
        
        self.net.eval()
        avg_loss = []; avg_acc = []
        start_time = time.time()
        n_batch = len(self.validset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.validloader):

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            y_pred = self.net(X)
            loss_i = self.criterion(y_pred, y)
            acc_i  = accuracy(y_pred, y)

            avg_loss += [loss_i.detach().cpu().numpy()]; avg_acc += [acc_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/valid/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/valid/acc_i'.format(self.net._get_name()), acc_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()

        avg_loss = np.mean(np.array(avg_loss))
        avg_acc  = np.mean(np.array(avg_acc))
        return avg_loss, avg_acc
    

    def save_checkpoint(self):
        
        checkpoint_state = {
            'save_time': getTime(),

            'cur_epoch': self.cur_epoch,
            'cur_batch': self.cur_batch,
            'elapsed_time': self.elapsed_time,
            'valid_loss': self.valid_loss,
            'save_times': self.save_times,
            
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times))
        torch.save(checkpoint_state, checkpoint_path)
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times-self.num_to_keep))
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        self.save_times += 1
        

    def load_checkpoint(self, index):
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), index))
        checkpoint_state = torch.load(checkpoint_path, map_location='cuda' if cuda.is_available() else 'cpu')
        
        self.cur_epoch = checkpoint_state['cur_epoch']
        self.cur_batch = checkpoint_state['cur_batch']
        self.elapsed_time = checkpoint_state['elapsed_time']
        self.valid_loss = checkpoint_state['valid_loss']
        self.save_times = checkpoint_state['save_times']

        self.net.load_state_dict(checkpoint_state['net_state'])
        self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state'])


class MarginTrainer(SupervisedTrainer):

    def __init__(self, configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=False):

        super(MarginTrainer, self).__init__(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep, resume, valid_freq)
        
        m1m2m3 = '_'.join(list(map(str, [self.criterion.margin.m1, self.criterion.margin.m2, self.criterion.margin.m3])))
        self.logdir = os.path.join(self.logdir, m1m2m3)
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.writer.close(); self.writer = SummaryWriter(self.logdir)
            
        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("train samples:   {}k".format(len(trainset)/1000))
        print("valid samples:   {}k".format(len(validset)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(trainset)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("val frequency:   {}".format(self.valid_freq))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")

        self.show_embedding = show_embedding

    def train_epoch(self):
        
        self.net.train()
        avg_loss = []; avg_acc = []
        start_time = time.time()
        n_batch = len(self.trainset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.trainloader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            cosine = self.net(X)
            loss_i = self.criterion(cosine, y)

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            avg_loss += [loss_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/train/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
            self.elapsed_time += duration_time
            total_time = duration_time * self.configer.n_epoch * len(self.trainset) // self.configer.batchsize
            left_time = total_time - self.elapsed_time

        avg_loss = np.mean(np.array(avg_loss))
        avg_acc  = np.mean(np.array(avg_acc))
        return avg_loss, avg_acc

    def valid_epoch(self):
        
        self.net.eval()
        avg_loss = []; avg_acc = []
        start_time = time.time()
        n_batch = len(self.validset) // self.configer.batchsize

        if self.show_embedding:
            mat = None
            metadata = None
            label_img = None

        for i_batch, (X, y) in enumerate(self.validloader):

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            cosine = self.net(X)
            loss_i = self.criterion(cosine, y)

            avg_loss += [loss_i.detach().cpu().numpy()]
            self.writer.add_scalar('{}/valid/loss_i'.format(self.net._get_name()), loss_i, self.cur_epoch*n_batch + i_batch)

            if self.show_embedding:
                mat = torch.cat([mat, cosine], dim=0) if mat is not None else cosine
                metadata = torch.cat([metadata, y], dim=0) if metadata is not None else y
                label_img = torch.cat([label_img, X], dim=0) if label_img is not None else X

            duration_time = time.time() - start_time
            start_time = time.time()

        if self.show_embedding:
            self.writer.add_embedding(mat, metadata, label_img, global_step=self.cur_epoch)

        avg_loss = np.mean(np.array(avg_loss))
        avg_acc  = np.mean(np.array(avg_acc))
        return avg_loss, avg_acc
    

class UnsupervisedTrainer():

    def __init__(self, configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1):

        self.configer = configer
        self.valid_freq = valid_freq

        self.net = net
        
        ## directory for log and checkpoints
        self.logdir = os.path.join(configer.logdir, self.net._get_name())
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = configer.ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        
        ## datasets
        self.trainset = trainset
        self.validset = validset
        self.trainloader = DataLoader(trainset, configer.batchsize, True)
        self.validloader = DataLoader(validset, configer.batchsize, True)

        ## for optimization
        self.criterion = criterion
        self.optimizer = optimizer(params, configer.lrbase)
        self.lr_scheduler = lr_scheduler(self.optimizer, configer.adjstep, configer.gamma)
        self.writer = SummaryWriter(self.logdir)
        
        ## initialize
        self.valid_loss = float('inf')
        self.valid_ami  = 0   # adjusted mutual info score
        self.elapsed_time = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        self.save_times = 0
        self.num_to_keep = num_to_keep

        ## if resume
        if resume:
            self.load_checkpoint()

        ## print information
        if configer.cuda and cuda.is_available(): 
            self.net.cuda()
            self.criterion.cuda()
            
        print("==============================================================================================")
        print("model:           {}".format(self.net._get_name()))
        print("logdir:          {}".format(self.logdir))
        print("ckptdir:         {}".format(self.ckptdir))
        print("train samples:   {}k".format(len(trainset)/1000))
        print("valid samples:   {}k".format(len(validset)/1000))
        print("batch size:      {}".format(configer.batchsize))
        print("batch per epoch: {}".format(len(trainset)/configer.batchsize))
        print("epoch:           [{:4d}]/[{:4d}]".format(self.cur_epoch, configer.n_epoch))
        print("val frequency:   {}".format(self.valid_freq))
        print("learing rate:    {}".format(configer.lrbase))
        print("==============================================================================================")

    def predict(self, x):
        """
        Params:
            x: {tensor(N, n_features)}
        Returns:
            y: {tensor(N)}
        Notes:
            self.criterion.m: {tensor(num_clusters, n_features)}
        """
        x = torch.cat(list(map(lambda x: self.criterion._f(x, 
                    self.criterion.m, self.criterion.s1).unsqueeze(0), x)), dim=0)
        y = torch.argmin(x, dim=1)
        return y
    
    def train(self):
        
        n_epoch = self.configer.n_epoch - self.cur_epoch
        print("Start training! current epoch: {}, remain epoch: {}".format(self.cur_epoch, n_epoch))

        bar = ProcessBar(n_epoch)
        loss_train = 0.; loss_valid = 0.
        ami_train  = 0.; ami_valid  = 0.

        for i_epoch in range(n_epoch):

            if self.configer.cuda and cuda.is_available(): cuda.empty_cache()

            self.cur_epoch += 1
            bar.step()

            self.lr_scheduler.step(self.cur_epoch)
            cur_lr = self.lr_scheduler.get_lr()[-1]
            self.writer.add_scalar('{}/lr'.format(self.net._get_name()), cur_lr, self.cur_epoch)

            loss_train, ami_train = self.train_epoch()

            if self.valid_freq != 0 and self.cur_epoch % self.valid_freq == 0:
                loss_valid, ami_valid = self.valid_epoch()

            self.writer.add_scalars('{}/loss'.format(self.net._get_name()), {'train': loss_train, 'valid': loss_valid}, self.cur_epoch)
            self.writer.add_scalars('{}/ami'.format(self.net._get_name()),  {'train': ami_train,  'valid': ami_valid }, self.cur_epoch)

            if self.valid_freq == 0:
                self.save_checkpoint()
            
            else:
                if ami_valid > self.valid_ami:
                    self.valid_ami = ami_valid
                    self.valid_loss = loss_valid
                    self.save_checkpoint()

    def train_epoch(self):

        self.net.train()
        avg_loss = []; avg_ami = []
        start_time = time.time()
        n_batch = len(self.trainset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.trainloader):

            self.cur_batch += 1

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            feature = self.net(X)
            total_i, intra_i, inter_i = self.criterion(feature)
            ami_i  = adjusted_mutual_info_score(y.detach().cpu().numpy(), 
                            self.predict(feature).detach().cpu().numpy())

            self.optimizer.zero_grad()
            total_i.backward()
            self.optimizer.step()

            avg_loss += [total_i.detach().cpu().numpy()]; avg_ami += [ami_i]
            self.writer.add_scalars('{}/train/loss_i'.format(self.net._get_name()), {'total_i': total_i, 'intra_i': intra_i, 'inter_i': inter_i, }, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/train/ami_i'.format(self.net._get_name()),  ami_i,  self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
            self.elapsed_time += duration_time
            total_time = duration_time * self.configer.n_epoch * len(self.trainset) // self.configer.batchsize
            left_time = total_time - self.elapsed_time

        avg_loss = np.mean(np.array(avg_loss))
        avg_ami  = np.mean(np.array(avg_ami))
        return avg_loss, avg_ami

    def valid_epoch(self):

        self.net.eval()
        avg_loss = []; avg_ami = []
        start_time = time.time()
        n_batch = len(self.validset) // self.configer.batchsize

        for i_batch, (X, y) in enumerate(self.validloader):

            X = Variable(X.float()); y = Variable(y.long())
            if self.configer.cuda and cuda.is_available(): X = X.cuda(); y = y.cuda()
            
            feature = self.net(X)
            total_i, intra_i, inter_i  = self.criterion(feature)
            ami_i  = adjusted_mutual_info_score(y.detach().cpu().numpy(), 
                            self.predict(feature).detach().cpu().numpy())
            
            avg_loss += [total_i.detach().cpu().numpy()]; avg_ami += [ami_i]
            self.writer.add_scalars('{}/valid/loss_i'.format(self.net._get_name()), {'total_i': total_i, 'intra_i': intra_i, 'inter_i': inter_i, }, self.cur_epoch*n_batch + i_batch)
            self.writer.add_scalar('{}/valid/ami_i'.format(self.net._get_name()),  ami_i,  self.cur_epoch*n_batch + i_batch)

            duration_time = time.time() - start_time
            start_time = time.time()
        
        avg_loss = np.mean(np.array(avg_loss))
        avg_ami  = np.mean(np.array(avg_ami))
        return avg_loss, avg_ami

    def save_checkpoint(self):
        
        checkpoint_state = {
            'save_time': getTime(),

            'cur_epoch': self.cur_epoch,
            'cur_batch': self.cur_batch,
            'elapsed_time': self.elapsed_time,
            'valid_loss': self.valid_loss,
            'valid_ami':self.valid_ami,
            'save_times': self.save_times,
            
            'net_state': self.net.state_dict(),
            'criterion_state': self.criterion.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times))
        torch.save(checkpoint_state, checkpoint_path)
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), self.save_times-self.num_to_keep))
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        self.save_times += 1
        
    def load_checkpoint(self, index):
        
        checkpoint_path = os.path.join(self.ckptdir, "{}_{:04d}.pkl".\
                            format(self.net._get_name(), index))
        checkpoint_state = torch.load(checkpoint_path, map_location='cuda' if cuda.is_available() else 'cpu')
        
        self.cur_epoch = checkpoint_state['cur_epoch']
        self.cur_batch = checkpoint_state['cur_batch']
        self.elapsed_time = checkpoint_state['elapsed_time']
        self.valid_loss = checkpoint_state['valid_loss']
        self.valid_ami = checkpoint_state['valid_ami']
        self.save_times = checkpoint_state['save_times']

        self.net.load_state_dict(checkpoint_state['net_state'])
        self.criterion.load_state_dict(checkpoint_state['criterion_state'])
        self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler_state'])

    def show_embedding_features(self, dataset):
        
        embedding_images   = torch.zeros(len(dataset), 1, 28, 28)
        embedding_features = torch.zeros(len(dataset), self.criterion.m.shape[1])
        embedding_labels   = torch.zeros(len(dataset))

        self.net.eval()

        dataloader = DataLoader(dataset)
        for i, (X, y) in enumerate(dataloader):

            if self.configer.cuda and cuda.is_available():
                X = X.cuda()
            embedding_features[i] = self.net(X)

            embedding_images[i] = X[0]
            embedding_labels[i] = y

        self.writer.add_embedding(embedding_features, 
                    metadata=embedding_labels, label_img=embedding_images)