import torch.nn as nn
import torch as t


class EncoderBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(EncoderBlock, self).__init__()
        self.conv_bn_relu_x2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.down_sample = nn.MaxPool2d((2,2))

    def forward(self,x):
        """
        :param x: input tensor to the current EncoderBlock
        :return: output tensor to the deeper EncoderBlock
        """
        output = self.conv_bn_relu_x2(x)
        output = self.down_sample(output)
        return output


class DecoderBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DecoderBlock, self).__init__()
        self.conv_bn_relu_x2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.up_sample = nn.Upsample(scale_factor=2)

    def forward(self,x):
        '''
        :param x: input tensor to the current DecoderBlock (without residual connection)
        :return: output to the upper DecoderBlock
        '''

        output = self.up_sample(x)
        output = self.conv_bn_relu_x2(output)
        return output


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BottleNeck, self).__init__()
        self.conv_bn_relu_x2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        :param x: input tensor to the Bottleneck block
        :return: output to the upper DecoderBlock
        '''
        output = self.conv_bn_relu_x2(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_img_channel, filter_num, class_num):
        super(UNet, self).__init__()
        # 4 EncoderBlocks
        self.e1=EncoderBlock(in_ch=in_img_channel, out_ch=filter_num)  # 3-64
        self.e2=EncoderBlock(in_ch=filter_num, out_ch=filter_num*2)    # 64-128
        self.e3=EncoderBlock(in_ch=filter_num*2, out_ch=filter_num*4)  # 128-256
        self.e4=EncoderBlock(in_ch=filter_num*4, out_ch=filter_num*8)  # 256-512

        # Bottleneck
        self.bottleneck = BottleNeck(in_ch=filter_num*8, out_ch=filter_num*16)  # 512-1024

        # 4 DecoderBlocks
        self.d1=DecoderBlock(in_ch=filter_num*16, out_ch=filter_num*8)  # 1024-512
        self.d2=DecoderBlock(in_ch=filter_num*8, out_ch=filter_num*4)   # 1024-512-256
        self.d3=DecoderBlock(in_ch=filter_num*4, out_ch=filter_num*2)   # 512-256-128
        self.d4=DecoderBlock(in_ch=filter_num*2, out_ch=filter_num)     # 256-128-64

        # output block
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=filter_num, out_channels=class_num, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(class_num),
            nn.ReLU()
        )

    def forward(self,x):
        output_e1 = self.e1(x)
        output_e2 = self.e2(output_e1)
        output_e3 = self.e3(output_e2)
        output_e4 = self.e4(output_e3)
        output_bottleneck = self.bottleneck(output_e4)
        output_d1 = self.d1(output_bottleneck)
        output_d2 = self.d2(output_d1)
        output_d3 = self.d3(output_d2)
        output_d4 = self.d4(output_d3)
        final_output = self.output_block(output_d4)

        return final_output

if __name__ == '__main__':
    demo_unet = UNet(in_img_channel=3, filter_num=64, class_num=5)
    demo_input = t.rand((1,3,224,224))
    demo_output = demo_unet.forward(demo_input)
    print(demo_output.size())