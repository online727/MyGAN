import torch
import torch.nn as nn


class Generator_Linear(nn.Module):
    # based on linear layers
    def __init__(self):
        super(Generator_Linear, self).__init__()
        self.gen = nn.Sequential(
            # 256 is the input size of the generator
            nn.Linear(100, 256),
            # nn.BatchNorm1d(256),
            # dropout layer, 0.3 is the probability of an element to be zeroed
            # nn.Dropout(0.3),
            nn.LeakyReLU(True),

            nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.4),
            nn.LeakyReLU(True),

            nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.Dropout(0.5),
            nn.LeakyReLU(True),

            # 784 is the output size of the generator, because the image's size is 28 * 28
            nn.Linear(1024, 784),

            # tanh layer, the output of the generator is in the range of [-1, 1]
            nn.Tanh()
        )
    
    def forward(self, image_noise):
        # image_noise's shape is (batch_size, 256)
        output = self.gen(image_noise)

        # output's shape is (batch_size, 784)
        # reshape the output to (batch_size, 1, 28, 28)
        output = output.view(-1, 1, 28, 28)
        return output
    

class Generator_Conv(nn.Module):
    # based on convolutional layers, up-sampling method is used
    def __init__(self):
        super(Generator_Conv, self).__init__()

        self.expand = nn.Sequential(
            # 256 is the input size of the generator
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(True),

            nn.Linear(256, 484),
            nn.BatchNorm1d(484),
            nn.Dropout(0.5),
            nn.LeakyReLU(True),
        )
        # the output size of self.expand is 256, because the image's size is 22 * 22
        # this size is set to match the up-sampling process that follows, which will increase the size to 28 * 28

        '''
        the calculation of the output size of the ConvTranspose2d layer is:
        output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        in this case, the stride is 1, padding is 0, output_padding is 0
        so the output_size = input_size - 1 + kernel_size
        '''
        self.gen = nn.Sequential(
            # 2-d transpose convolutional layer, input channels is 1, output channels is 4, kernel size is 3
            nn.ConvTranspose2d(1, 4, kernel_size=3),
            # the output size of this layer is 22 - 1 + 3 = 24 * 24
            nn.BatchNorm2d(4),
            nn.LeakyReLU(True),

            # 2-d transpose convolutional layer, input channels is 4, output channels is 8, kernel size is 3
            nn.ConvTranspose2d(4, 8, kernel_size=3),
            # the output size of this layer is 24 - 1 + 3 = 26 * 26
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),

            # 2-d transpose convolutional layer, input channels is 8, output channels is 4, kernel size is 3
            nn.ConvTranspose2d(8, 4, kernel_size=3),
            # the output size of this layer is 26 - 1 + 3 = 28 * 28
            nn.BatchNorm2d(4),
            nn.LeakyReLU(True),

            # 2-d transpose convolutional layer, input channels is 4, output channels is 1, kernel size is 1
            # this layer is used to reduce the number of channels to 1 to match greyscale images
            nn.ConvTranspose2d(4, 1, kernel_size=1),
            # the output size of this layer is 28 -1 + 1 = 28 * 28
            nn.BatchNorm2d(1),
            nn.LeakyReLU(True),

            # tanh layer, the output of the generator is in the range of [-1, 1]
            nn.Tanh()
        )
    
    def forward(self, image_noise):
        # image_noise's shape is (batch_size, 484)
        output = self.expand(image_noise)

        # output's shape is (batch_size, 484)
        # reshape the output to (batch_size, 1, 22, 22)
        output = output.view(-1, 1, 22, 22)

        # up-sampling the output to (batch_size, 1, 28, 28)
        output = self.gen(output)

        return output


def get_Generator(
        from_old_model=None,
        model_path=None,
        device='cuda',
        G_type='L'
        ):
    if G_type == 'L':
        model = Generator_Linear()
    elif G_type == 'C':
        model = Generator_Conv()
    else:
        raise ValueError('G_type should be either Linear or Conv')
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    
    model.to(device)
    return model