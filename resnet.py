import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_chan, out_chan, pool=True, act="relu", dropout=False):
        """
        Main block for the resnet type architecture.

        :param in_chan: Number of chanel in
        :param out_chan: Number of chanel out
        :param pool: Use pooling True/False
        :param act: Activation type (options: "relu" for ReLU "leak" for LeakyRelu "prelu" for PReLU
        :param dropout: Use dropout True/False
        """
        super(Block, self).__init__()

        # set channels
        self.in_chan = in_chan
        self.out_chan = out_chan

        # set activation
        if act == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif act == "prelu":
            self.activation1 = nn.PReLU()
            self.activation2 = nn.PReLU()
        elif act == "leak":
            self.activation1 = nn.LeakyReLU()
            self.activation2 = nn.LeakyReLU()

        # create sequential block
        self.conv_1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 3, padding=1),
                                    nn.BatchNorm2d(out_chan),
                                    self.activation1,
                                    nn.Conv2d(out_chan, out_chan, 3, padding=1),
                                    nn.BatchNorm2d(out_chan))

        # if need dropout (overfit helper)
        if dropout:
            self.conv_1.add_module("Drop", nn.Dropout2d(0.2))

        # if need pool (downsampling)
        if pool:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # channel reduction
        self.scaler = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1, stride=1, bias=False),
                                    nn.BatchNorm2d(out_chan))

    def forward(self, x):

        # scale feature maps down from X input to match
        scaled = self.scaler(x)

        # convolve the x input tensor
        block_out = self.conv_1(x)

        # pool if needed and add X input to blocks output
        if hasattr(self, "pooling"):
            return self.pooling(self.activation2(scaled + block_out))
        else:
            return self.activation2(scaled + block_out)


class Net(nn.Module):

    def __init__(self, n_classes=10, model_shape=[(64, 3), (128, 3), (256, 3)], act="relu",
                 adapt_pool_size=2, dropout=True):
        """
        An implementation of resnet using some extra techniques to try and improve training accuracy and lower
        overfitting.

        :param n_classes: The number of classes we are predicting
        :param in_chans:  Initial input channel's
        :param model_shape: The shape of the network - expected list of tuples in format of [(channels,repeats),(chan...
        i.e [(64,3),(128,3)] would preduce a network of 3 blocks of 64 layer convolutions followed by 3 blocks of 128
        layer colvolutions.
        :param act: Activation type (options: "relu" for ReLU "leak" for LeakyRelu "prelu" for PReLU)
        :param adapt_pool_size: final pool size before flatten, default 2
        :param dropout: True/False -? Use dropout?
        """
        super(Net, self).__init__()

        seq_final = nn.Sequential()
        chans = 3

        #create blocks and add to seqential.
        for i, blocks in enumerate(model_shape):

            seq = nn.Sequential()

            if blocks[1] > 1:
                seq.add_module(f"Init Block {i}", Block(chans, blocks[0], False, act=act, dropout=dropout))

                if blocks[1] > 2:

                    for x in range(blocks[1] - 2):
                        seq.add_module(f"Inner Block {i}:{x}",
                                       Block(blocks[0], blocks[0], False, act=act, dropout=dropout))

                seq.add_module(f"Exit Block {i}", Block(blocks[0], blocks[0], True, act=act, dropout=dropout))
            else:
                seq.add_module(f"Lonely Block {i}", Block(chans, blocks[0], True, act=act, dropout=dropout))

            chans = blocks[0]

            seq_final.add_module(f"Main Block {i}", seq)

        self.conv_block = seq_final

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

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

        x = self.conv_block(x)
        x = self.avgpool(x)
        x = self.linear(x)
        return x
