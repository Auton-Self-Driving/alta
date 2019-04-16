import torch
import torch.nn as nn
import torchvision.models as models

from .block import FcBlock


class Pretrained(nn.Module):

    def __init__(self, model_name='alexnet', pre_trained=False):
        super(Pretrained, self).__init__()
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=pre_trained)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pre_trained)
        elif model_name == 'squeezenet':
            model = models.squeezenet1_0(pretrained=pre_trained)
        elif model_name == 'densenet':
            model = models.densenet161(pretrained=pre_trained)
        elif model_name == 'inception':
            model = models.inception_v3(pretrained=pre_trained)
        else:
            model = models.alexnet(pretrained=pre_trained)
        # self.model = torch.nn.Sequential(
        #     *list(self.model.children()))

        self.core = model.features
        self.fc = model.classifier
        # if hasattr(self.model[-1], "in_features"):
        #     self.no_features = self.model[-1].in_features
        # else:
        #     self.no_features = self.model[-1][1].in_features

        self.no_features = self.get_in_features(self.fc)
        self.fc = torch.nn.Sequential(
            FcBlock(self.no_features, 2048),
            FcBlock(2048, 512)
        )

    def forward(self, image):
        x = self.core(image)
        x = self.fc(x.contiguous().view(-1, self.no_features))
        return x

    def get_in_features(self, classifier):
        for _, mod in enumerate(classifier.modules()):
            if type(mod) == nn.Linear:
                no_features = mod.in_features
                break
        return no_features
