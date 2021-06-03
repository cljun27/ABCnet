import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d as networks
from util.loss_functions import TVLoss, SegLoss, CorLoss


class Pix2Pix3dModel(BaseModel):
    def name(self):
        return 'Pix2Pix3dModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_W = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_G = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depthSize, opt.fineSize, opt.fineSize)
        self.input_minA_norm = self.Tensor(1)
        self.input_maxA_norm = self.Tensor(1)
        self.input_minB_norm = self.Tensor(1)
        self.input_maxB_norm = self.Tensor(1)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            opt.lrdd = opt.lr / 2
            self.old_lrdd = opt.lrdd
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionReg = smooothing_loss()
            self.criterionCor = CorLoss()
            self.criterionSeg = SegLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lrdd, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['Ori' if AtoB else 'BF']
        input_B = input['BF' if AtoB else 'Ori']
        input_W = input['WM' ]
        input_G = input['GM' ]
        input_minA_norm = input['min_img_norm']
        input_maxA_norm = input['max_img_norm']
        input_minB_norm = input['min_bf_norm']
        input_maxB_norm = input['max_bf_norm']
        self.input_A.resize_(input_Ori.size()).copy_(input_Ori)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_W.resize_(input_W.size()).copy_(input_W)
        self.input_G.resize_(input_G.size()).copy_(input_G)
        self.input_minA_norm.resize_(input_minA_norm.size()).copy_(input_minA_norm)
        self.input_maxA_norm.resize_(input_maxA_norm.size()).copy_(input_maxA_norm)
        self.input_minB_norm.resize_(input_minB_norm.size()).copy_(input_minB_norm)
        self.input_maxB_norm.resize_(input_maxB_norm.size()).copy_(input_maxB_norm)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_Ori)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)
        self.tissue_W = self.input_W.type(torch.ByteTensor)
        self.tissue_G = self.input_G.type(torch.ByteTensor)
        self.tissue_W = Variable(self.tissue_W)
        self.tissue_G = Variable(self.tissue_G)
        self.tissue_W = self.tissue_W.cuda()
        self.tissue_G = self.tissue_G.cuda()

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_Ori, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return "blksdf"
        #return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionCor(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G_S1 = self.criterionSeg(self.real_A, self.fake_B, self.tissue_W, self.input_minA_norm, self.input_maxA_norm, self.input_minB_norm, self.input_maxB_norm) * self.opt.lambda_C
        # print(self.loss_G_S1)
        self.loss_G_S2 = self.criterionSeg(self.real_A, self.fake_B, self.tissue_G, self.input_minA_norm, self.input_maxA_norm, self.input_minB_norm, self.input_maxB_norm) * self.opt.lambda_C

        self.loss_G_smooth = self.criterionReg(self.fake_B) * self.opt.lambda_B

        self.loss_G = self.loss_G_GAN + self.loss_G_S1 + self.loss_G_S2 + self.loss_G_smooth + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('G_S1', self.loss_G_S1.data),
                            ('G_S2', self.loss_G_S2.data),
                            ('G_smooth', self.loss_G_smooth.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data)
                            ])

    def get_current_visuals(self):
        self.tissue_W = self.input_W.type(torch.ByteTensor)
        self.tissue_G = self.input_G.type(torch.ByteTensor)
        self.tissue_W = Variable(self.tissue_W)
        self.tissue_G = Variable(self.tissue_G)
        self.tissue_W = self.tissue_W.cuda()
        self.tissue_G = self.tissue_G.cuda()
        self.real_Ar = self.real_A*(self.input_maxA_norm - self.input_minA_norm) + self.input_minA_norm
        self.fake_Br = self.fake_B*(self.input_maxB_norm - self.input_minB_norm) + self.input_minB_norm
        self.fake_A = self.real_Ar/self.fake_Br
        c_w = torch.masked_select(self.fake_A, self.tissue_W)
        c_g = torch.masked_select(self.fake_A, self.tissue_G)
        c_var_w = torch.var(c_w)
        c_mean_w = torch.mean(c_w)
        c_var_g = torch.var(c_g)
        c_mean_g = torch.mean(c_g)
        real_A = util.tensor2im3d(self.real_A.data)
        fake_B = util.tensor2im3d(self.fake_B.data)
        real_B = util.tensor2im3d(self.real_B.data)
        fake_A = util.tensor2im3d(self.fake_A.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),  ('fake_A', fake_A)])

    def get_corrcoef(self):
        real = self.real_B.data.cpu().float().numpy()
        real = np.squeeze(real)
        fake = self.fake_B.data.cpu().float().numpy()
        fake = np.squeeze(fake)
        real = np.ndarray.flatten(real)
        print (real.shape)
        fake = np.ndarray.flatten(fake)
        cc = np.corrcoef(real, fake)
        return cc

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

        lrd = self.opt.lr / self.opt.niter_decay
        lrd = lrd / 2
        lrddd = self.old_lrdd - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lrddd

        print('update learning rate: %f -> %f' % (self.old_lrdd, lrddd))
        self.old_lrdd = lrddd
