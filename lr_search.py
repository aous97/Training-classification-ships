from Dataset import *
from models import *
from utils import *
from config import *


def train(net, optimizer, train_loader, criterion,  n_epoch = 2,
          train_acc_period = 5,
          test_acc_period = 5, aux_logits = True,
          cuda=False, visualize = True, lr_decay = 0, learning_rate = 0.1):
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
                loss = criterion(outputs.float(), labels)
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #scaled_loss.backward()
            loss.backward()
            optimizer.step()
            
            total += labels.size(0)
            # print statistics
            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss
            _, predicted = torch.max(outputs.data, 1)
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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--class_num', type=int, default=23, help='number of classes')
    parser.add_argument('--model', type=str, default='resnet152', help='learning rate')

    parser.add_argument('--data_path', type=str, help='path to downloaded marvel dataset')
    parser.add_argument('--weight', type=str, default='', help='initial weight path in .pth format')
    opt = parser.parse_args()

    Dataset = get_pd_dataset(opt.data_path)
    batch_size = opt.batch_size
    reorder = list(range(0,12))
    reorder += [23,24,12,13,14,15,25,16,17,18,19,20,21,22]

    Classes = [500]*26
    Classes[12] = 0
    Classes[13] = 0
    Classes[18] = 0
    test_Classes = [100]*26
    test_Classes[12] = 2
    test_Classes[13] = 2
    test_Classes[18] = 2
    train_loader = imbalanced_trainloader(Dataset, opt.batch_size, Classes, augmented = True, reorder = reorder)
    test_loader = imbalanced_trainloader(Dataset, opt.batch_size, test_Classes, train = 2, reorder = reorder)
    criterion = LabelSmoothingCrossEntropy(0.1)

    model = model_dict[opt.model]

    Lr = [1e-4, 1e-3, 1e-2,  1e-1]
    accuracys = []
    loss = []
    for lr in Lr:
        learning_rate = lr
        net = model(opt.class_num)
        if opt.weight != '':
            net.load_state_dict(torch.load(opt.weight))
        #net = nn.DataParallel(net)
        optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate, momentum = opt.momentum) 
        loss.append(train(net, optimizer, train_loader, criterion,  n_epoch = 1, visualize =True,
                        cuda = True, lr_decay = False, train_acc_period = 50, aux_logits = False))
        accuracys.append(accuracy(net, test_loader))

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.plot(Lr, loss)
    plt.ylabel('model loss')
    plt.xlabel('learning rates')
    plt.xscale('log')
    plt.subplot(132)
    plt.plot(Lr, accuracys)
    plt.ylabel('model accuracy')
    plt.xlabel('learning rates')
    plt.xscale('log')
    plt.show()
