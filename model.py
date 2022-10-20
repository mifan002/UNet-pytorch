import torch.nn as nn
import torch as t


class EncoderBlock(nn.Module):
    """An EncoderBlock in the compression pass of UNet.

    An EncoderBlock has 2 * (conv2d, BatchNorm2d, ReLU) and 1 MaxPool2d(for downsampling).

    A tensor passes through a EncoderBlock: (b, in_ch, w, h) -> (b, out_ch, w/2, h/2).

    Attributes:
        conv_bn_relu_x2: a sequential container consisting of 2 * (conv2d, BatchNorm2d, ReLU)
        down_sample: a MaxPool2d layer for downsampling
    """
    def __init__(self,in_ch,out_ch):
        """create a EncoderBlock.

        In the compression pass of UNet, how many EncoderBlocks do you need depends on the depth.

        :param in_ch: input channel number to the current EncoderBlock
        :param out_ch: output channel number to the current EncoderBlock
        """
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
        """put the tensor x into the current EncoderBlock and return the output.

        The input will be firstly pass through the 2 * (conv2d, BatchNorm2d, ReLU), and then be passed into
        down_sample module to shrink its width and height dimension by 2.

        :param x: input tensor to the current EncoderBlock
        :return: output tensor to the deeper EncoderBlock
        """
        output = self.conv_bn_relu_x2(x)
        output = self.down_sample(output)
        return output


class DecoderBlock(nn.Module):
    """An DecoderBlock in decompression pass of UNet.

    An DecoderBlock has 2 * (conv2d, BatchNorm2d, ReLU) and 1 Upsample(for upsampling).

    A tensor passes through a DecoderBlock: (b, in_ch, w, h) -> (b, out_ch, w*2, h*2).

    Attributes:
        conv_bn_relu_x2: a sequential container consisting of 2 * (conv2d, BatchNorm2d, ReLU)
        up_sample: a Upsample layer for upsampling
    """
    def __init__(self,in_ch,out_ch):
        """create a DecoderBlock.

        In the decompression pass of UNet, how many DecoderBlocks do you need depends on the depth.

        :param in_ch: input channel number to the current DecoderBlock
        :param out_ch: output channel number to the current DecoderBlock
        """
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
        """put the tensor x into the current DecoderBlock and return the output.

        The input will be firstly upsampled, i.e. recovered its width and height dimension by 2, then be passed
        through the 2 * (conv2d, BatchNorm2d, ReLU).

        :param x: input tensor to the current DecoderBlock (without residual connection)
        :return: output to the upper DecoderBlock
        """
        output = self.up_sample(x)
        output = self.conv_bn_relu_x2(output)
        return output


class BottleNeck(nn.Module):
    """The unique Bottleneck structure in the bottom of UNet.

    The Bottleneck only has 2 * (conv2d, BatchNorm2d, ReLU), without any downsampling or upsampling module.

    A tensor passes through this BottleNeck structure: (b, in_ch, w, h) -> (b, out_ch, w, h).

    Attributes:
        conv_bn_relu_x2: a sequential container consisting of 2 * (conv2d, BatchNorm2d, ReLU)
    """
    def __init__(self, in_ch, out_ch):
        """create the unique BottleNeck structure.

        Only need 1 in UNet.

        :param in_ch: input channel number to the current BottleNeck
        :param out_ch: output channel number to the current BottleNeck
        """
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
        """put the tensor x into the BottleNeck and return the output.

        The input will be passed through 2 * (conv2d, BatchNorm2d, ReLU), no upsampling or downsampling needed additionally.
        .
        :param x: input tensor to the Bottleneck block
        :return: output to the upper DecoderBlock
        """
        output = self.conv_bn_relu_x2(x)
        return output


class UNet(nn.Module):
    """Unet with depth=4

    Net structure: 4*EncoderBlocks+1*BottleNeck+4*DecoderBlocks+1*OutputBlock.

    Attributes:
        e1-e4: 4 EncoderBlocks that compress the input image (b, c, w, h) -> (b, filter_num*8, w/16, h/16).
        bottleneck: compress the encoded tensor further from (b, filter_num*8, w/16, h/16) -> (b, filter_num*16, w/16, h/16).
        d1-d4: 4 DecoderBlocks that recover the encoded tensor (b, filter_num*16, w/16, h/16) -> (b, filter_num, w, h).
        output_block: generate probability map: (b, filter_num, w, h) -> (b, class_num, w, h)
    """
    def __init__(self, in_img_channel, filter_num, class_num):
        super(UNet, self).__init__()
        # 4 EncoderBlocks
        self.e1=EncoderBlock(in_ch=in_img_channel, out_ch=filter_num)  # 3 -> filter_num
        self.e2=EncoderBlock(in_ch=filter_num, out_ch=filter_num*2)       # filter_num -> 2*filter_num
        self.e3=EncoderBlock(in_ch=filter_num*2, out_ch=filter_num*4)    # 2*filter_num -> 4*filter_num
        self.e4=EncoderBlock(in_ch=filter_num*4, out_ch=filter_num*8)    # 4*filter_num -> 8*filter_num

        # Bottleneck
        self.bottleneck = BottleNeck(in_ch=filter_num*8, out_ch=filter_num*16)  # 8*filter_num -> 16*filter_num

        # 4 DecoderBlocks
        self.d1=DecoderBlock(in_ch=filter_num*16, out_ch=filter_num*8)  # 16*filter_num -> 8*filter_num
        self.d2=DecoderBlock(in_ch=filter_num*8, out_ch=filter_num*4)    # 8*filter_num -> 4*filter_num
        self.d3=DecoderBlock(in_ch=filter_num*4, out_ch=filter_num*2)    # 4*filter_num -> 2*filter_num
        self.d4=DecoderBlock(in_ch=filter_num*2, out_ch=filter_num)       # 2*filter_num -> filter_num

        # output block
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=filter_num, out_channels=class_num, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(class_num),
            nn.Sigmoid()
        )                                                                 # filter_num -> class_num

        # loss block
        self.loss = nn.BCELoss()

    def forward(self,x, y):
        output_e1 = self.e1(x)
        output_e2 = self.e2(output_e1)
        output_e3 = self.e3(output_e2)
        output_e4 = self.e4(output_e3)
        output_bottleneck = self.bottleneck(output_e4)
        output_d1 = self.d1(output_bottleneck)
        output_d2 = self.d2(output_d1)
        output_d3 = self.d3(output_d2)
        output_d4 = self.d4(output_d3)
        prob_map = self.output_block(output_d4)
        loss = self.loss(prob_map, y)

        return loss, prob_map

if __name__ == '__main__':
    demo_unet = UNet(in_img_channel=3, filter_num=64, class_num=5)
    demo_input = t.rand((1,3,224,224))
    demo_output = demo_unet.forward(demo_input)
    print(demo_output.size())