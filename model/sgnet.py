import model.common as common
import torch.nn as nn
#from model.common import SFTLayer
import pdb
import torch
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pdb
import cv2
from model.pac import PacConvTranspose2d
"""
given LR bayer  -> output HR noise-free RGB
"""
class m_res(nn.Module):
    def __init__(self, opt):
        super(m_res, self).__init__()
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type 
        
        self.r1 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r2 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r3 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r4 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r5 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r6 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.final = common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias)
    def forward(self, x):
        output = self.r1(x)
        output = self.r2(output)
        output = self.r3(output)
        output = self.r4(output)
        output = self.r5(output)
        output = self.r6(output)
        return self.final(output)



class green_res(nn.Module):
    def __init__(self, opt):
        super(green_res, self).__init__()
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type 
        self.head = common.ConvBlock( 2 , dm_n_feats, 5, act_type=act_type, bias=True)
        self.r1 = common.RRDB(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.r2 = common.RRDB(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        #self.r3 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        #self.r4 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        #self.r5 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        #self.r6 = common.RRDB2(dm_n_feats, dm_n_feats, 3, 1, bias, norm_type, act_type, 0.2)
        self.final = common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias)
        
        self.up = nn.Sequential(
           common.Upsampler(2, dm_n_feats, norm_type, act_type, bias=bias),
           common.ConvBlock(dm_n_feats, 1 , 3, bias=True),
           nn.LeakyReLU(0.2, inplace = True)
        )

    def forward(self, x):
        output = self.r1(self.head(x))
        output = self.r2(output)
        output = self.final(output) +self.head(x)
        output = self.up(output)
        #output = self.r3(output)
        #output = self.r4(output)
        #output = self.r5(output)
        #output = self.r6(output)
        return output #self.final(output)
   
class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()

        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type

        # define sr module
        if denoise:
            m_sr_head = [common.ConvBlock(6, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(4, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(sr_n_feats, sr_n_feats, 3,
                                         1, bias, norm_type, act_type, 0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(sr_n_feats, 4, 3, bias=True)]

        # branch for sr_raw output
        m_sr_tail = [nn.PixelShuffle(2)]

        # define demosaick module
        m_dm_head = [common.ConvBlock(4, dm_n_feats, 5,
                                      act_type=act_type, bias=True)]

        if block_type.lower() == 'rrdb':
            m_dm_resblock = m_res(opt) #[common.RRDB(dm_n_feats, dm_n_feats, 3,
                                         #1, bias, norm_type, act_type, 0.2)
                             #for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(dm_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        #m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias)]
        m_dm_up = [common.Upsampler(2, dm_n_feats, norm_type, act_type, bias=bias)]
                   #common.ConvBlock(dm_n_feats, 3, 3, bias=True)]

        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.Sequential(*m_sr_resblock)),
                                      *m_sr_up)
        self.sr_output = nn.Sequential(*m_sr_tail)
        self.model_dm1 = nn.Sequential(*m_dm_head)
        self.model_dm2 = m_dm_resblock
        self.model_dm3 = nn.Sequential(*m_dm_up)


        greenresblock = green_res(opt)
        self.green = greenresblock
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
        #self.sft = SFTLayer()
        self.combine = nn.Sequential(
                common.ConvBlock(dm_n_feats+1, dm_n_feats, 1 , bias=True),
                nn.LeakyReLU(0.2 , inplace = True)
        )
        self.greenup = nn.Sequential(
                common.ConvBlock(1, 4, 1 , bias=True),
                nn.LeakyReLU(0.2 , inplace = True),
                common.ConvBlock(4, 8, 1 , bias=True),
                nn.LeakyReLU(0.2 , inplace = True)

        )

        self.pac = PacConvTranspose2d(64,64,kernel_size=5, stride=2, padding=2, output_padding=1)
        self.final = common.ConvBlock( dm_n_feats , 3, 3 , bias=True)
        self.norm = nn.InstanceNorm2d(1)
    def density(self , x):
        x = torch.clamp(x**(1/2.2)*255,0.,255.).detach()
        b,w,h = x.shape
        
        im= np.array(x[0].cpu()).astype(np.uint8)
        im = Image.fromarray(im)
        im_blur = im.filter(ImageFilter.GaussianBlur(radius=3))
        im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
        im_minus = np.uint8(im_minus)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        im_sum = torch.from_numpy(cv2.dilate(im_minus, kernel).astype(np.float))
        im_sum = im_sum.unsqueeze(0)
        #print(im_sum.shape)    
        for i in range(1,b):
            im= np.array(x[i].cpu()).astype(np.uint8)
            im = Image.fromarray(im)
            im_blur = im.filter(ImageFilter.GaussianBlur(radius=5))
            im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
            im_minus = np.uint8(im_minus)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
            im_minus = cv2.dilate(im_minus, kernel).astype(np.float)
            im_minus = torch.from_numpy(im_minus).unsqueeze(0)
            #print(im_minus.shape)
            im_sum = torch.cat([im_sum,im_minus] , 0 )
        return im_sum.unsqueeze(1).float().cuda()
        
    def forward(self, x ):
        # estimate density map
        dmap = x.clone()
        dmap = (dmap[:,0,:,:] + (dmap[:,1,:,:]+dmap[:,2,:,:])/2 + dmap[:,3,:,:])/3
        dmap = self.density(dmap)
        dmap = self.norm(dmap).detach()
        
        # super resoliton in the task, JDDS
        x = self.model_sr(torch.cat([x,dmap],1))     
        
        x1 = x
        sr_raw = self.sr_output(x)

        # demosaic and denoise
        x = self.model_dm1(x)
        green_output = self.green(x1[:,1:3,:,:].detach())
        x =  x + self.model_dm2(x)
        
        #pdb.set_trace()
        
        g_combine = self.greenup(green_output)
        x = self.pac(x,g_combine)
        x = self.final(x)
        return sr_raw, x , green_output


