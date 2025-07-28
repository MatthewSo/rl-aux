import torchvision.transforms as transforms

# CIFAR Transforms
cifar_trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
cifar_trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])

def f(x):
    if x.shape[0] == 1:
        return x.repeat(3,1,1)
    return x
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
cifar_256_train_transform = transforms.Compose([
    transforms.Resize(256),                # Resize to a bit larger than final crop
    transforms.RandomCrop(224),            # Random crop to 224x224
    transforms.RandomHorizontalFlip(),     # Data augmentation
    transforms.ToTensor(),
    transforms.Lambda(lambda x: f(x)),
    transforms.Normalize(mean, std),
])

cifar_256_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: f(x)),
    transforms.Normalize(mean, std),
])

# ImageNet Transforms
imagenet_trans_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

imagenet_trans_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# Places365 Transforms
places365_trans_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

places365_trans_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# SVHN Transforms
svhn_trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4377, 0.4438, 0.4728),
                         std=(0.1980, 0.2010, 0.1970)),
])

svhn_trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4377, 0.4438, 0.4728),
                         std=(0.1980, 0.2010, 0.1970)),
])

imgnet_mean = (0.485, 0.456, 0.406)
imgnet_std  = (0.229, 0.224, 0.225)

common_train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imgnet_mean, imgnet_std),
])

common_test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imgnet_mean, imgnet_std),
])

to_tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])