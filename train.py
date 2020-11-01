from config import *
from Dataset import *
from models import *
from utils import *

def mixup_data(x, y, alpha=1, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def train(net, optimizer, train_loader, criterion,  n_epoch = 2,
          train_acc_period = 5,
          test_acc_period = 5, aux_logits = False, mixed = False,
          cuda=True, visualize = True, lr_decay = 0, learning_rate = 0.1):
    loss_train = []
    loss_test = []
    total = 0
    
    if cuda:
        net = net.cuda()
        #net, optimizer = amp.initialize(net, optimizer, opt_level = 'O1')
    for epoch in tnrange(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        if lr_decay > 0:
            adjust_learning_rate_cossine(optimizer, epoch, lr_decay, learning_rate)
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # get the inputs
            if cuda:
                #inputs = inputs.type(torch.cuda.HalfTensor)
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()
            if mixed:
            	inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, cuda)
            	inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
            # forward + backward + optimize
            #try:
            outputs = net(inputs)
            #except:
            #    print("problem with input: ", i)
            #    continue
            if not cuda:
                labels = labels.type(torch.LongTensor)
            if aux_logits:
                loss = criterion(outputs[0].float(), labels)
                loss += 0.2*criterion(outputs[1].float(), labels)
                loss += 0.2*criterion(outputs[2].float(), labels)
                outputs = outputs[0]
            else:
                if mixed:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs.float(), labels)
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #scaled_loss.backward()
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            # print statistics
            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss
            _, predicted = torch.max(outputs.data, 1)
            if mixed:
                correct = ((lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()))/10
            else:
                correct = (predicted == labels).sum().item()/labels.size(0)
            running_acc = 0.3*correct + 0.66*running_acc
            loss_train.append(running_loss)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                    print('[%d, %5d] acc: %.3f' %(epoch + 1, i + 1, running_acc))
                    running_loss = 0.0
                    total = 0
            inputs = None
            outputs = None
            labels = None
      
    if visualize:
        print('Finished Training')
    return(loss_train[-1])

if __name__ == '__main__':
    model_dict = {'googlenet': googlenet,
                    'resnet34': Resnet34,
                    'resnet50': Resnet50,
                    'resnet152': Resnet152,
                    'wide_resnet50': Wide_resnet50,
                    'resnext101': Resnext101,
                    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--class_num', type=int, default=23, help='number of classes')
    parser.add_argument('--model', type=str, default='resnet152', help='learning rate')

    parser.add_argument('--data_path', type=str, help='path to downloaded marvel dataset')
    parser.add_argument('--weight', type=str, default='', help='initial weight path in .pth format')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--mixed', type=bool, default=False, help='mixed training technique')
    parser.add_argument('--save', type=str, default='', help='save model weights under path')
    opt = parser.parse_args()

    train_loader , test_loader = get_loaders(get_pd_dataset(opt.data_path), opt.batch_size)
    criterion = LabelSmoothingCrossEntropy(0.1)

    learning_rate = opt.learning_rate
    model = model_dict[opt.model]
    net = model(opt.class_num)
    if opt.weight != '':
        net.load_state_dict(torch.load(opt.weight))

    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate, momentum = opt.momentum)
    train(net, optimizer, train_loader, criterion,  n_epoch = opt.epoch, visualize = True, mixed = opt.mixed,
                        cuda = torch.cuda.is_available(), train_acc_period = 20)
    
    accuracy(net, test_loader)

    if opt.save != '':
        torch.save(net.state_dict(), opt.save)
