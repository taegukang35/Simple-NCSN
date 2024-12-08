from utils import *
from trainer import NCSNTrainer
from model import CondRefineNetDilated
from mymodel import MyCondRefineNetDilated

#Adam optim params
lr = 1e-3
beta1 = 0.9
beta2 = 0.999

batch_size = 32 #128 too much memory for me
num_iters = 100001

#data params
dataset = "MNIST"
image_size = 32
channels = 1
logit_transform = False
random_flip = False
num_classes = 10
ngf = 64

#save params
checkpoints_folder = './mycheckpoints'
save_every = 1000

mnist_config = config(
            dataset, 
            image_size, 
            channels, 
            logit_transform, 
            random_flip,
            num_classes,
            ngf
)

# ncsn = CondRefineNetDilated(mnist_config)
myncsn = MyCondRefineNetDilated(mnist_config)

dataloader = get_train_set(batch_size)

trainer = NCSNTrainer(myncsn, lr, dataloader, num_iters, beta1, beta2, checkpoints_folder, save_every)
trainer.train_ncsn()