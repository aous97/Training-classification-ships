from config import *

def get_pd_dataset(data_path):
    if (os.path.isfile(data_path+"/FINAL.dat")):
        os.rename(data_path+"/FINAL.dat", data_path+"/FINAL.csv")
    labeldf = pd.read_csv(data_path+"/FINAL.csv")
    labeldf['img'] = labeldf['993293']
    labeldf['train_test'] = labeldf['1']
    labeldf['Label'] = labeldf['1.1']
    labeldf['Class name'] = labeldf['Container Ship']
    labeldf['path'] = labeldf[data_path+'/W0_1/993293.jpg']
    labeldf = labeldf[['img', 'train_test', 'Label', 'Class name', 'path']]
    labeldf = labeldf[labeldf.path != '-']
    val_df = labeldf[['path']].drop_duplicates()
    val_df2 = labeldf.loc[val_df.index.values.tolist()]
    Dataset = val_df2.reset_index()[['img', 'train_test', 'Label', 'Class name', 'path']]
    return(Dataset)

def crop_resize(image_obj, coords, x_size = 256, y_size = 256):
    width = image_obj.size[0]
    height = image_obj.size[1]
    coords = (coords[0]*width, coords[1]*height, coords[2]*width, coords[3]*height)
    cropped_image = image_obj.crop(coords)
    resized_image = cropped_image.resize((x_size, y_size), Image.ANTIALIAS)
    return(resized_image.convert('RGB'))

class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, Table, augmented = False, randomize = True, reorder = list(range(0,26))):
        self.image_paths = Table['path']
        self.labels = Table['Label']
        self.randomizer = list(Table.index)
        self.reorder = reorder
        self.augmented = augmented
        if randomize:
            np.random.shuffle(self.randomizer)
        self.transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
    def __getitem__(self, index):
        image = Image.open(self.image_paths[self.randomizer[index]]).convert('RGB')
        
        if self.augmented:
            ratio = np.random.random()*(4/3-3/4)+3/4
            area = np.random.random()*0.5+0.5
            rot = (np.random.random()*2-1)*10
            h = np.sqrt(area/ratio)
            w = h*ratio
            x = (1 - w)*np.random.random()
            y = (1 - h)*np.random.random()
            crop_rect = (x,y, x+w, y+h)
            image = crop_resize(image, crop_rect)
            if np.random.random() >= 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            converter = PIL.ImageEnhance.Color(image)
            image = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            converter = PIL.ImageEnhance.Brightness(image)
            image = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            image = image.rotate(rot)
            
        x = self.transform(image)
        #x = x.type(torch.HalfTensor)
        label = self.reorder[self.labels[self.randomizer[index]]-1]
        return([x, label])
    def __len__(self):
        return(len(self.image_paths))

def imbalanced_trainloader(Dataset, batch_size, caps, augmented = False, train = 1, reorder = list(range(0,26))):
    train_dataset = Dataset[Dataset.train_test == train]
    train_dataset = train_dataset.reset_index()[['img', 'train_test', 'Label', 'Class name', 'path']]
    train_dataset_balanced = train_dataset[train_dataset.Label == 1].reset_index()
    start = int(np.random.rand()*(len(train_dataset_balanced)-caps[0]))
    train_dataset_balanced = train_dataset_balanced[train_dataset_balanced.index > start]
    train_dataset_balanced = train_dataset_balanced[train_dataset_balanced.index < start + caps[0]]
    for i in range(2,27):
        temp = train_dataset[train_dataset.Label == i].reset_index()
        start = int(np.random.rand()*(len(temp)-caps[i-1]))
        temp = temp[temp.index > start]
        temp = temp[temp.index < start + caps[i-1]]
        train_dataset_balanced = pd.concat([train_dataset_balanced, temp])
    train_dataset_balanced = train_dataset_balanced.reset_index()[['img', 'Label', 'Class name', 'path']]
    custom_train_dataset_balanced = Custom_dataset(train_dataset_balanced, augmented, reorder = reorder)
    return(torch.utils.data.DataLoader(custom_train_dataset_balanced, batch_size, num_workers = 4, pin_memory = True))

def get_loaders(Dataset, batch_size):
    reorder = list(range(0,12))
    reorder += [23,24,12,13,14,15,25,16,17,18,19,20,21,22]

    train_dataset = Dataset[Dataset.train_test == 1]
    test_dataset = Dataset[Dataset.train_test == 2]

    batch_size = 10
    Balance = list(len(test_dataset[test_dataset.Label == i+1]) for i in reorder)

    Classes = [2000]*26
    Classes[12] = 0
    Classes[13] = 0
    Classes[18] = 0
    test_Classes = [1000]*26
    test_Classes[12] = 0
    test_Classes[13] = 0
    test_Classes[18] = 0

    train_loader = imbalanced_trainloader(Dataset, batch_size, Classes, augmented = True,
                     reorder = reorder)
    test_loader = imbalanced_trainloader(Dataset, batch_size, test_Classes,
                     train = 2, reorder = reorder)
    return(train_loader, test_loader)

def augment(image):
    ratio = np.random.random()*(4/3-3/4)+3/4
    area = np.random.random()*0.5+0.5
    rot = (np.random.random()*2-1)*10
    h = np.sqrt(area/ratio)
    w = h*ratio
    x = (1 - w)*np.random.random()
    y = (1 - h)*np.random.random()
    crop_rect = (x,y, x+w, y+h)
    image = crop_resize(image, crop_rect)
    if np.random.random() >= 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    converter = PIL.ImageEnhance.Color(image)
    image = converter.enhance((1.4-0.6)*np.random.random()+0.6)
    converter = PIL.ImageEnhance.Brightness(image)
    image = converter.enhance((1.4-0.6)*np.random.random()+0.6)
    image = image.rotate(rot)
    return(image)


def imbalanced_trainloader_cache(Dataset, caps, batch_size,  train = 1, augmented = True,
                                 x_size = 256,  y_size = 256,  reorder = list(range(0,26))):
    train_dataset = Dataset[Dataset.train_test == train]
    train_dataset = train_dataset.reset_index()[['img', 'train_test', 'Label', 'Class name', 'path']]
    train_dataset_balanced = train_dataset[train_dataset.Label == 1].reset_index()
    start = int(np.random.rand()*(len(train_dataset_balanced)-caps[0]))
    train_dataset_balanced = train_dataset_balanced[train_dataset_balanced.index > start]
    train_dataset_balanced = train_dataset_balanced[train_dataset_balanced.index < start + caps[0]]
    for i in range(2,27):
        temp = train_dataset[train_dataset.Label == i].reset_index()
        start = int(np.random.rand()*(len(temp)-caps[i-1]))
        temp = temp[temp.index > start]
        temp = temp[temp.index < start + caps[i-1]]
        train_dataset_balanced = pd.concat([train_dataset_balanced, temp])
    train_dataset_balanced = train_dataset_balanced.reset_index()[['img', 'Label', 'Class name', 'path']]
    size = len(train_dataset_balanced)
    tensor_x = torch.empty (size, 3, x_size, y_size)
    images = []
    labels = []
    L = list(train_dataset_balanced.index)
    np.random.shuffle(L)
    for i in tqdm(L):
        path = train_dataset_balanced['path'][i]
        image = copy.deepcopy(Image.open(path).convert('RGB'))
        if augmented:
            image = augment(image)
        images.append(transform(image))
        labels.append(reorder[train_dataset_balanced['Label'][i]-1])
    for i in range(size):
        tensor = images[0]
        tensor_x[i] = tensor
        images2 = images[1:]
        images = images2[:]
    tensor_y = torch.Tensor(labels)
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    dataloader = (torch.utils.data.DataLoader(dataset, batch_size, num_workers = 4, pin_memory = True))
    return(dataloader)
