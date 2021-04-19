import model.common as common
import torch.nn as nn
import model.ops as ops
import torch
"""
given 5 channel LR img   -> output HR noise-free RGB
"""
class ConvBlock2(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm='batch'):
        super(ConvBlock2, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class srcnn(nn.Module):

    def __init__(self):
        super(srcnn, self).__init__()
        self.num_channels = 64
        self.dropout_rate = 0.2
        
        self.layers = torch.nn.Sequential(
            ConvBlock2(3, self.num_channels, 9, 1, 4, norm=None), # 144*144*64 # conv->batchnorm->activation
            ConvBlock2(self.num_channels, self.num_channels // 2, 1, 1, 0, norm=None), # 144*144*32
            ConvBlock2(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None) # 144*144*1
        )

    def forward(self, s):
        out = self.layers(s)
        return out

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
    
class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 32
        n_feats = 64
        kernel_size = 3 
        scale = 2
        act = 'relu'
        #self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                 n_feats, kernel_size, act_type=act,bias = True, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(scale, n_feats, norm_type=False, act_type=False),
            conv(n_feats, 3, kernel_size)
        ]
        
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        return x 
    
    
class carn(nn.Module):   
    def __init__(self):
        super(carn, self).__init__()
        
        
        group = 2

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = ops.UpsampleBlock(64, scale=2, 
                                          multi_scale= False,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x):
        scale = 2
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
    
class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        
        self.conv1 = nn.Sequential(
            common.ConvBlock(5, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 12 , 3,  act_type=act_type, bias=True),
            
        )
        
        self.conv2 = nn.Sequential(
            common.ConvBlock(6, 64 , 3,  act_type=act_type, bias=True),
            common.ConvBlock(64, 3 , 3,  act_type=act_type, bias=True)
        )
        
        self.ps = nn.PixelShuffle(2)
        self.size = 64
        s = self.size
        mask1 = torch.zeros(s,s)
        mask2 = torch.zeros(s,s)
        mask3 = torch.zeros(s,s)
        for i in range(0,s):
            for j in range(0,s):
                if i %2==1 and j%2==0:
                    mask1[i,j] = 1
        for i in range(0,s):
            for j in range(0,s):
                if (i %2==0 and j%2==0) or (i %2==1 and j%2==1):
                    mask2[i,j] = 1
                    
        for i in range(0,s):
            for j in range(0,s):
                if i %2==0 and j%2==1:
                    mask3[i,j] = 1
        self.mask1 = mask1.cuda()
        self.mask2 = mask2.cuda()
        self.mask3 = mask3.cuda()
        self.srcnn = srcnn()
        self.carn = carn()
        self.edsr = EDSR()
    def forward(self, x):
        x1 = self.conv1(x)
        b,c,w,h = x.shape
        x_ps = self.ps(x[:,0:4,:,:])
        c456 =  self.ps(x1)
        
        if x_ps.shape[2]==256:
            s=256 
            mask1 = torch.zeros(s,s)
            mask2 = torch.zeros(s,s)
            mask3 = torch.zeros(s,s)

            for i in range(0,s):
                for j in range(0,s):
                    if i %2==1 and j%2==0:
                        mask1[i,j] = 1
            for i in range(0,s):
                for j in range(0,s):
                    if (i %2==0 and j%2==0) or (i %2==1 and j%2==1):
                        mask2[i,j] = 1
                    
            for i in range(0,s):
                for j in range(0,s):
                    if i %2==0 and j%2==1:
                        mask3[i,j] = 1
            self.mask1 = mask1.cuda()
            self.mask2 = mask2.cuda()
            self.mask3 = mask3.cuda()

        c1 = self.mask1*x_ps[:,0,:,:]
        c2 = self.mask2*x_ps[:,0,:,:]
        c3 = self.mask3*x_ps[:,0,:,:]
        c1 =  c1.unsqueeze(1)
        c2 =  c2.unsqueeze(1)
        c3 =  c3.unsqueeze(1)


        x2 = torch.cat([c1,c2,c3,c456],1)
        output = self.conv2(x2)
        output = self.edsr(output)
        return output
        #x2 = torch.zeros(b,4,w,h)
        
