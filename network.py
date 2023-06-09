import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def get_network(opt, class_embedding, stage):
    if opt.network == 'c3d':
        return C3D(class_embedding, fixconvs=opt.fixconvs, stage=stage)
    if opt.network == 'r3d':
        network = models.video.r3d_18
    else:
        network = models.video.r2plus1d_18
    return ResNet18(network, class_embedding, fixconvs=opt.fixconvs, stage=stage)


class ResNet18(nn.Module):
    def __init__(self, network, class_embedding, fixconvs=False, stage=0):
        super(ResNet18, self).__init__()
        self.stage = stage
        self.model = network(weights=True)
        wv_dim = 300
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, wv_dim)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.scnet = nn.Parameter(torch.Tensor(class_embedding), requires_grad=False)


    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.dropout(x)
        x = self.regressor(x)
        x = F.normalize(x)
        if self.stage == 1:
            scnet = torch.transpose(self.scnet,0,1)
            scnet = F.normalize(scnet)
            x = torch.matmul(x, scnet)
        return x


class C3D(nn.Module):
    """
    References
    ----------
    [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
    Proceedings of the IEEE international conference on computer vision. 2015.

    C3D code taken from: https://github.com/DavideA/c3d-pytorch
    """
    def __init__(self, class_embedding, fixconvs=False, stage=0, dataset='ucf'):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, 400)

        self.dropout = nn.Dropout(p=0.10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # self.load_state_dict(torch.load('workplace/c3d.pickle'))
        # print('torch.load(workplace/c3d.pickle)')

        self.regressor = nn.Linear(4096, 300)

        if fixconvs:
            for model in [self.conv1, self.conv2,
                          self.conv3a, self.conv3b,
                          self.conv4a, self.conv4b,
                          self.conv5a, self.conv5b,
                          self.fc6]:
                for param in model.parameters():
                    param.requires_grad = False
        
        self.stage = stage
        self.scnet = nn.Parameter(torch.Tensor(class_embedding), requires_grad=False)


    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)

        h = h.reshape(bs, nc, -1)
        h = torch.mean(h, 1)
        h = h.reshape(bs, -1)

        h = self.regressor(h)
        h = torch.nn.functional.normalize(h, dim=-1)

        if self.stage == 1:
            scnet = torch.transpose(self.scnet,0,1)
            scnet = F.normalize(scnet)
            h = torch.matmul(h, scnet)
        return h