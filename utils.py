from config import *

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class FocalLossBalanced(nn.Module):
    def __init__(self, gamma, Class_dist, beta = 0.9):
        super().__init__()
        self.gamma = gamma
        self.Class_dist = Class_dist
        self.beta = beta

    def forward(self, input, target):
        empty = torch.zeros(input.size()).cuda()
        #print(empty.shape)
        for i in range(target.shape[0]):
            empty[i][target[i]] = 1
        target = empty
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        beta = self.beta
        Class_dist = self.Class_dist
        effective_num = 1.0 - np.power(beta, Class_dist)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights)*len(Class_dist)
        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(loss.shape[0], 1)*loss
        return weights.mean()

def accuracy(net, test, cuda=True):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in tqdm(test):
            images, labels = data
            if cuda:
                net = net.cuda()
                images = images.type(torch.cuda.FloatTensor)
                #images = images.type(torch.cuda.HalfTensor)
                labels = labels.type(torch.cuda.LongTensor)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    net.train()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total

def accuracy_perclass(net, test, num_classes = 26, cuda=True):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    TP = [0]*num_classes
    Total= [0]*num_classes
    with torch.no_grad():
        for data in tqdm(test):
            images, labels = data
            if cuda:
                net = net.cuda()
                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for ind in range(labels.size(0)):
                for i in range(num_classes):
                    if i == labels[ind]:
                        TP[i] += int(labels[ind] == predicted[ind])
                        Total[i] += 1
            #print(predicted, labels, TP, Total)
                        
    for i in range(num_classes):
        TP[i] /= Total[i]
    print("Accuracy is: ", sum(TP)/sum(Total))
    
    net.train()
    return TP

def average_model_ensembling(Models, test_data, class_num = 23,  cuda = True):
    softner = nn.Softmax(dim = 0)
    correct = 0
    total = 0
    loss = 0
    CM = np.zeros((class_num,class_num))

    with torch.no_grad():
        for data in tqdm(test_data):
            images, labels = data
            outputs = torch.zeros(labels.size(0), class_num)
            for net in Models:
                net.eval()
                if cuda:
                    net = net.cuda()
                    images = images.type(torch.cuda.FloatTensor)
                outputs += softner(net(images).cpu())/len(Models)
                net.train()
            _, predicted = torch.max(outputs.data, 1)
            for ind in range(len(predicted)):
                if labels[ind] < class_num and predicted[ind] < class_num:
                    CM[labels[ind]][predicted[ind]] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: ' , (
        100 * correct / total))
    return CM

def confusion_matrix(net, test_loader, Dataset):
    import seaborn as sn

    CM = average_model_ensembling([net], test_loader)
    Names = Dataset['Class name'].unique()
    perm = list(range(0,12))
    perm += [14, 15, 16, 17, 19,20,21,22, 23, 24, 25, 12,13, 18]
    Names = list(Names[perm[i]] for i in range(0,26))[:23]
    CM2 = np.zeros((23,23))
    for i in range(len(CM)):
        for j in range(len(CM[0])):
            CM2[i][j] = int(CM[i][j] / sum(CM[i])*100)/100
    df_cm = pd.DataFrame(CM2, index = [i for i in Names],
                    columns = [i for i in Names])
    plt.figure(figsize = (15,15))
    sn.heatmap(df_cm, annot=True)
    plt.show()
