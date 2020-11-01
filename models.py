from config import *

def classifier(out_size, num_classes):
    classifier = nn.Sequential(nn.Linear(out_size, int(out_size/2)), nn.BatchNorm1d(int(out_size/2)) , nn.ReLU(),
                               nn.Dropout(p=0.1), nn.Linear(int(out_size/2), num_classes))
    return(classifier)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def googlenet(num_classes):
    net = torchvision.models.googlenet(pretrained=False)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    net.aux1.fc2 = classifier(net.aux1.fc2.in_features, num_classes)
    net.aux2.fc2 = classifier(net.aux2.fc2.in_features, num_classes)
    return(net)

def Resnet34(num_classes):
    net = torchvision.models.resnet34(pretrained=False)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Resnet50(num_classes):
    net = torchvision.models.resnet50(pretrained=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Resnet152(num_classes):
    net = torchvision.models.resnet152(pretrained=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Wide_resnet50(num_classes):
    net = torchvision.models.wide_resnet50_2(pretrained=False)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)

def Resnext101(num_classes):
    net = torchvision.models.resnext101_32x8d(pretrained=False)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)
