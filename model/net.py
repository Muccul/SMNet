import torch
import torch.nn as nn
import torch.nn.functional as F
from  model.non_local_block import NONLocalBlock2D
from model.region_non_local_block import RegionNONLocalBlock


class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, inter_channels=32, dilation=[1, 1, 1]):
        super(DenseBlock, self).__init__()

        num_convs = len(dilation)
        concat_channels = in_channels + num_convs * inter_channels
        self.conv_list, channels_now = nn.ModuleList(), in_channels

        for i in range(num_convs):
            conv = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channels, kernel_size=3,
                          stride=1, padding=dilation[i], dilation=dilation[i]),
                nn.ReLU(inplace=True),
            )
            self.conv_list.append(conv)

            channels_now += inter_channels

        assert channels_now == concat_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, local_skip=True):
        feature_list = [x,]

        for conv in self.conv_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = conv(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs)

        if local_skip:
            block_outputs = fusion_outputs + x
        else:
            block_outputs = fusion_outputs

        return block_outputs



class NEDB(nn.Module):
    def __init__(self, block_num=4, inter_channel=32, channel=64, dilation=[1, 1, 1, 1]):
        super(NEDB, self).__init__()
        concat_channels = channel + block_num * inter_channel
        channels_now = channel
        self.non_local = NONLocalBlock2D(channels_now, bn_layer=False)
        self.dense = DenseBlock(in_channels=channel, inter_channels=inter_channel, dilation=dilation)
        
    def forward(self, x):
        out = self.non_local(x)
        out = self.dense(out, local_skip=False)
        out += x
        
        return out
    
    
class RNEDB(nn.Module):
    def __init__(self, block_num=4, inter_channel=32, channel=64, grid=[8, 8], dilation=[1, 1, 1, 1]):
        super(RNEDB, self).__init__()
        concat_channels = channel + block_num * inter_channel
        channels_now = channel
        self.region_non_local = RegionNONLocalBlock(channels_now, grid=grid)
        self.dense = DenseBlock(in_channels=channel, inter_channels=inter_channel, dilation=dilation)
        
    def forward(self, x):
        out = self.region_non_local(x)
        out = self.dense(out, local_skip=False)
        out += x
        
        return out
    
    
    
class Derain(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, inter_ch=32):
        super(Derain, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1)

        self.up_1 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.up_2 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.up_3 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])

        self.down_3 = NEDB(block_num=4, inter_channel=32, channel=64, dilation=[1, 1, 1, 1])
        self.down_2 = RNEDB(
            block_num=4, inter_channel=32, channel=64, grid=[2, 2], dilation=[1, 1, 1, 1])
        self.down_1 = RNEDB(
            block_num=4, inter_channel=32, channel=64, grid=[4, 4], dilation=[1, 1, 1, 1])

        self.down_2_fusion = nn.Conv2d(base_ch + base_ch, base_ch, 1, 1, 0)
        self.down_1_fusion = nn.Conv2d(base_ch + base_ch, base_ch, 1, 1, 0)

        self.fusion = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 1, 1, 0),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

        
    def forward(self, x):
        feature_neg_1 = self.conv1(x)

        feature_0 = self.conv2(feature_neg_1)
        
        up_1_banch = self.up_1(feature_0)
        up_1, indices_1 = nn.MaxPool2d(2, 2, return_indices=True)(up_1_banch)
        
        up_2 = self.up_2(up_1)
        up_2, indices_2 = nn.MaxPool2d(2, 2, return_indices=True)(up_2)

        up_3 = self.up_3(up_2)
        up_3, indices_3 = nn.MaxPool2d(2, 2, return_indices=True)(up_3)
        
        down_3 = self.down_3(up_3)
        down_3 = nn.MaxUnpool2d(2, 2)(down_3, indices_3, output_size=up_2.size())
        down_3 = torch.cat([up_2, down_3], dim=1)
        down_3 = self.down_2_fusion(down_3)

        down_2 = self.down_2(down_3)
        down_2 = nn.MaxUnpool2d(2, 2)(down_2, indices_2, output_size=up_1.size())
        down_2 = torch.cat([up_1, down_2], dim=1)
        down_2 = self.down_1_fusion(down_2)

        down_1 = self.down_1(down_2)
        down_1 = nn.MaxUnpool2d(2, 2)(down_1, indices_1, output_size=feature_0.size())
        down_1 = torch.cat([feature_0, down_1], dim=1)
        cat_block_feature = torch.cat([down_1, up_1_banch], 1)
        feature = self.fusion(cat_block_feature)
#         feature = feature + feature_neg_1
        output = self.final_conv(feature)
        
        return output, feature
    
    

class SuperResolution(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, inter_ch=64):
        super(SuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1)

        self.dense_1 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_2 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_3 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_4 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_5 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_6 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_7 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_8 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])
        self.dense_9 = DenseBlock(in_channels=base_ch, inter_channels=inter_ch, dilation=[1, 1, 1, 1])

        self.dilate_1 = nn.Sequential(nn.Conv2d(in_channels=base_ch,out_channels=base_ch,
                                               kernel_size=3,padding=2,groups=1,bias=False,dilation=2),
                                     nn.ReLU(inplace=True))
        self.dilate_2 = nn.Sequential(nn.Conv2d(in_channels=base_ch, out_channels=base_ch,
                                               kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
                                     nn.ReLU(inplace=True))
        self.dilate_3 = nn.Sequential(nn.Conv2d(in_channels=base_ch, out_channels=base_ch,
                                               kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
                                     nn.ReLU(inplace=True))


        self.fusion = nn.Sequential(
            nn.Conv2d(base_ch * 9, base_ch, 1, 1, 0),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
        )
        
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(base_ch, base_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(base_ch, in_channels, 3, 1, 1),
        ])

        
        
    def forward(self, x):
        x = self.conv1(x)

        feature_0 = self.conv2(x)
        
        dout_1 = self.dense_1(feature_0)
        dout_2 = self.dense_2(dout_1)
        dout_3 = self.dense_3(dout_2)
        dout_3 = self.dilate_1(dout_3)
        dout_4 = self.dense_4(dout_3)
        dout_5 = self.dense_5(dout_4)
        dout_6 = self.dense_6(dout_5)
        dout_6 = self.dilate_2(dout_6)
        dout_7 = self.dense_7(dout_6)
        dout_8 = self.dense_8(dout_7)
        dout_9 = self.dense_9(dout_8)
        dout_9 = self.dilate_3(dout_9)
        
        out = torch.cat(
            (dout_1, dout_2, dout_3, dout_4, dout_5, dout_6, dout_7, dout_8, dout_9), dim=1)

        out = self.fusion(out)
        out += x
        
        output = self.UPNet(out)
        
        return output
    

class SMNet(nn.Module):
    def __init__(self, in_channels=3):
        super(SMNet, self).__init__()
        self.derain = Derain(in_channels=in_channels)
        self.sr = SuperResolution(in_channels=in_channels)
    

    def forward(self, x, visual=False):
        mask, feature = self.derain(x)
        x = x - mask
        # x = torch.cat([feature, x], dim=-1)
        srred = self.sr(x)
        srred = F.interpolate(srred, size=mask.shape[-2:], mode='bilinear', align_corners=False)
    
        if visual:
            return mask, x, srred
        
        return srred



if __name__ == '__main__':
    x = torch.rand(1, 1, 256, 256).cuda()     # x -> b*c*h*w
    model = SMNet(in_channels=1).cuda()
    print(model(x).shape)