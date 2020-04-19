import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_chan, out_chan, pool, act="relu", dropout=True, conv_bias=True):

        super(Block, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan

        if act == "relu":
            self.activation1 = nn.ReLU
            self.activation2 = nn.ReLU
        elif act == "prelu":
            self.activation1 = nn.PReLU
            self.activation2 = nn.PReLU
        elif act == "leak":
            self.activation1 = nn.LeakyReLU
            self.activation2 = nn.LeakyReLU

        self.conv_1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=conv_bias),
                                    nn.BatchNorm2d(out_chan),
                                    self.activation1(),
                                    nn.Conv2d(out_chan, out_chan, 3, padding=1, bias=conv_bias),
                                    nn.BatchNorm2d(out_chan))
        if dropout:
            self.conv_1.add_module("Drop", nn.Dropout2d(0.2))

        if pool == 1:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        elif pool == 2:
            self.pooling = nn.Conv2d(in_chan, out_chan, 1, 2, bias=False)

        if in_chan != out_chan:
            self.scaler = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1, stride=1, bias=False),
                                        nn.BatchNorm2d(out_chan))

    def forward(self, x):

        if hasattr(self, "scaler"):
            scaled = self.scaler(x)
        else:
            scaled = x

        # scaled = conv1x1(self.in_chan,self.out_chan,1)

        block_out = self.conv_1(x)

        if hasattr(self, "pooling"):
            return self.pooling(self.activation2()(scaled + block_out))
        else:
            return self.activation2()(scaled + block_out)


class Net(nn.Module):

    def __init__(self, n_classes=10, model_shape=[(64, 3), (128, 3), (256, 3)], act="relu",
                 adapt_pool_size=2, dropout=True, pool_type=0, conv_bias=True, init_scale=False, multiple_answers=False,
                 save_outputs=False):
        """
        An implementation of resnet using some extra techniques to try and improve training accuracy and lower
        overfitting. Default param values appear to be optimal from current testing.

        :param n_classes: The number of classes we are predicting
        :param in_chans:  Initial input channel's
        :param model_shape: The shape of the network - expected list of tuples in format of [(channels,repeats),(chan...
        i.e [(64,3),(128,3)] would produce a network of 3 blocks of 64 layer convolutions followed by 3 blocks of 128
        layer colvolutions.
        :param act: Activation type (options: "relu" for ReLU "leak" for LeakyRelu "prelu" for PReLU)
        :param adapt_pool_size: final pool size before flatten, default 2
        :param dropout: True/False -? Use dropout?
        :param pool_type: 0 for MaxPooling or 1 for Conv2d with side of 2.
        :param conv_bias: use bias on convolutional layers
        :param init_scale: use a conv2d as initial downsampler.
        :param multiple_answers: Default: FALSE - expected values 0-5 - used to stack multiple predictions from the
        last n blocks.
        :param save_outputs: save outputs into a dictionary from each step for later analysis
        """

        super(Net, self).__init__()

        self.save_outputs = save_outputs
        if save_outputs:
            self.outputs = {}
        if multiple_answers:
            self.multiple_answers = multiple_answers

        chans = 3

        if init_scale:
            out_chans = model_shape[-1][0]
            self.init = nn.Sequential(nn.Conv2d(chans, out_chans, kernel_size=7, stride=2, padding=3, bias=False),
                                      nn.BatchNorm2d(out_chans),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            chans = out_chans

        seq_final = nn.Sequential()

        # this is to use maxpooling to downsize
        if pool_type == 0:
            pool_true = 1
        # this is to use a convolution
        elif pool_type == 1:
            pool_true = 2

        for i, blocks in enumerate(model_shape):

            seq = nn.Sequential()

            if blocks[1] > 1:
                seq.add_module(f"Init Block {i}",
                               Block(chans, blocks[0], False, act=act, dropout=dropout, conv_bias=conv_bias))

                if blocks[1] > 2:

                    for x in range(blocks[1] - 2):
                        seq.add_module(f"Inner Block {i}:{x}",
                                       Block(blocks[0], blocks[0], False, act=act, dropout=dropout,
                                             conv_bias=conv_bias))

                seq.add_module(f"Exit Block {i}",
                               Block(blocks[0], blocks[0], pool_true, act=act, dropout=dropout, conv_bias=conv_bias))
            else:
                seq.add_module(f"Lonely Block {i}",
                               Block(chans, blocks[0], pool_true, act=act, dropout=dropout, conv_bias=conv_bias))

            chans = blocks[0]

            seq_final.add_module(f"Main Block {i}", seq)

        self.conv_block = seq_final

        self.avgpool = nn.AdaptiveAvgPool2d((adapt_pool_size, adapt_pool_size))

        inshape = model_shape[-1][0] * adapt_pool_size * adapt_pool_size

        if dropout:

            self.linear = nn.Sequential(nn.Flatten(),
                                        nn.Linear(inshape, int(inshape / 4)),
                                        nn.Linear(int(inshape / 4), n_classes))

        else:

            self.linear = nn.Sequential(nn.Flatten(),
                                        nn.Dropout2d(0.4),
                                        nn.Linear(inshape, int(inshape / 4)),
                                        nn.Linear(int(inshape / 4), n_classes))

    def forward(self, x):

        if hasattr(self, "init"):
            x = self.init(x)

        for i, seq in enumerate(self.conv_block.children()):
            x = seq(x)
            if self.save_outputs:
                self.outputs[i] = x

        x = self.conv_block(x)
        x = self.avgpool(x)
        x = self.linear(x)

        return x
