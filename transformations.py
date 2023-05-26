import torchvision.transforms as T
from PIL import Image, ImageOps


IMAGE_SIZE = 32
NORMALIZE = [[0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]] # CIFAR-10


class Solarization():
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)
    

class TwoCropTransform:
    """ Create two crops of the same image """
    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2

    def __call__(self, x):
        return (self.transform1(x), self.transform2(x))


TRANSFORM_SIMCLR = T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=IMAGE_SIZE//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*NORMALIZE)
            ])


TRANSFORM1_BYOL = T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=IMAGE_SIZE//20*2+1, sigma=(0.1, 2.0)), # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            T.ToTensor(),
            T.Normalize(*NORMALIZE)
        ])


TRANSFORM2_BYOL = T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=IMAGE_SIZE//20*2+1, sigma=(0.1, 2.0))], p=0.1),
            T.RandomApply([Solarization()], p=0.2),
            T.ToTensor(),
            T.Normalize(*NORMALIZE)
        ])


TRANSFORM_SIMSIAM = T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=IMAGE_SIZE//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*NORMALIZE)
        ])


TRANSFORM_SWAV = None


TRANSFORM_LINEAR = T.Compose([
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*NORMALIZE)
            ])


TRANSFORM_EVAL = T.Compose([
                T.Resize(int(IMAGE_SIZE*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                T.CenterCrop(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(*NORMALIZE)
            ])
            

transform_dict = {
    'byol': [TRANSFORM1_BYOL, TRANSFORM2_BYOL],
    'simclr': [TRANSFORM_SIMCLR, TRANSFORM_SIMCLR],
    'simsiam': [TRANSFORM_SIMSIAM, TRANSFORM_SIMSIAM],
    'supcon': [TRANSFORM_SIMCLR, TRANSFORM_SIMCLR],
    'swav': [TRANSFORM_SWAV, TRANSFORM_SWAV],
}