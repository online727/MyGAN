import torch
import torch.nn as nn


class Discriminator_Linear(nn.Module):
    def __init__(self):
        super(Discriminator_Linear, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dis(x).squeeze()
        return x


class Discriminator_Conv(nn.Module):
    def __init__(self):
        super(Discriminator_Conv, self).__init__()
        
        self.features = nn.Sequential(
            # 2-d convolutional layer, input channels is 1 (grayscale images), output channels is 32, kernel size is 3
            nn.Conv2d(1, 32, kernel_size=3),

            # batch normalization, 32 is the number of output channels of the last Conv2d layer
            # nn.BatchNorm2d(32),

            # Leaky ReLU, 0.2 is the negative values' slope.
            nn.LeakyReLU(0.2),

            # 2-d convolutional layer, input channels is 32, output channels is 64, kernel size is 3
            nn.Conv2d(32, 64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        '''
        self.features's output's shape is (batch_size, 64, 24, 24)
        64 is the number of output channels of the last Conv2d layer
        24 is the height and width of the output of the last Conv2d layer

        compute the size of the output of the Conv2d layer: (padding (填充) = 0, stride (步长) = 1)
        new size = (old size - kernel size + 2 * padding) / stride + 1

        so after the first Conv2d layer, the size of the output is (28 - 3 + 0) / 1 + 1 = 26, output channels is 32
        after the second Conv2d layer, the size of the output is (26 - 3 + 0) / 1 + 1 = 24, output channels is 64

        so at the end the size of output is 64 * 24 * 24
        '''

        self.classifier = nn.Sequential(
            nn.Linear(64*24*24, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        # image's shape is (batch_size, 1, 28, 28)
        # get images' features
        features = self.features(image)

        # features' shape is (batch_size, 64, 24, 24)
        # reshape features to (batch_size, 64 * 24 * 24)
        features = features.view(features.shape[0], -1)

        # get output
        output = self.classifier(features).squeeze()
        return output


def get_Discriminator(
        from_old_model=None,
        model_path=None, 
        device='cuda',
        D_type='Linear'
        ):

    if D_type == 'Linear':
        model = Discriminator_Linear()
    elif D_type == 'Conv':
        model = Discriminator_Conv()
    else:
        raise ValueError('D_type should be either Linear or Conv')

    if from_old_model:
        model.load_state_dict(torch.load(model_path))

    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    
    model.to(device)
    return model