#evaluate model on single image or show confusion matrix
from Dataset import *
from models import *
from utils import *
from config import *

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])

names23 = ['Container Ship',
            'Bulk Carrier',
            'Passengers Ship',
            'Ro-ro/passenger Ship',
            'Ro-ro Cargo',
            'Tug',
            'Vehicles Carrier',
            'Reefer',
            'Yacht',
            'Sailing Vessel',
            'Heavy Load Carrier',
            'Wood Chips Carrier',
            'Patrol Vessel',
            'Platform',
            'Standby Safety Vessel',
            'Combat Vessel',
            'Icebreaker',
            'Replenishment Vessel',
            'Tankers',
            'Fishing Vessels',
            'Supply Vessels',
            'Carrier/Floating',
            'Dredgers']

names26 = ['Container Ship', 'Bulk Carrier', 'Passengers Ship',
       'Ro-ro/passenger Ship', 'Ro-ro Cargo', 'Tug', 'Vehicles Carrier',
       'Reefer', 'Yacht', 'Sailing Vessel', 'Heavy Load Carrier',
       'Wood Chips Carrier', 'Livestock Carrier', 'Fire Fighting Vessel',
       'Patrol Vessel', 'Platform', 'Standby Safety Vessel',
       'Combat Vessel', 'Training Ship', 'Icebreaker',
       'Replenishment Vessel', 'Tankers', 'Fishing Vessels',
       'Supply Vessels', 'Carrier/Floating', 'Dredgers']

def get_label(vect):
    if len(vect) == 23:
        names = names23
    elif len(vect) == 26:
        names = names26
    else:
        print("Invalid class number for classification")
        assert(False)
    
    m = 0
    for i in range(len(vect)):
        if vect[i] > vect[m]:
            m = i
    return(m, names[m])

def get_pred_vect(model, img):
    image_obj = transform(img).unsqueeze(0)
    image_obj = Variable(image_obj.type(torch.cuda.FloatTensor))
    pred = model(image_obj)
    Softner = nn.Softmax(dim = 0)
    return(Softner(pred[0]).tolist())

if __name__ == '__main__':
    model_dict = {'googlenet': googlenet,
                    'resnet34': Resnet34,
                    'resnet50': Resnet50,
                    'resnet152': Resnet152,
                    'wide_resnet50': Wide_resnet50,
                    'resnext101': Resnext101,
                    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet152', help='learning rate')
    parser.add_argument('--class_num', type=int, default=23, help='number of classes')

    parser.add_argument('--data_path', type=str, help='path to downloaded marvel dataset')
    parser.add_argument('--weight', type=str, help='initial weight path in .pth format')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    parser.add_argument('--test_image', type=str, default='', help='path to a test image')
    opt = parser.parse_args()

    model = model_dict[opt.model]
    net = model(opt.class_num).eval()
    net.load_state_dict(torch.load(opt.weight))

    if opt.test_image != '':
        net = net.cuda()
        image_path = opt.test_image
        img = Image.open(image_path).convert('RGB').resize((256,256))
        label = get_label(get_pred_vect(net, img))
        print(label)
        img.show()
    else:
        _, test_loader = get_loaders(get_pd_dataset(opt.data_path), opt.batch_size)
        Dataset = get_pd_dataset(opt.data_path)

        confusion_matrix(net, test_loader, Dataset)

