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
checkpoints_folder = './checkpoints'
save_every = 100

mnist_config = config(
            dataset, 
            image_size, 
            channels, 
            logit_transform, 
            random_flip,
            num_classes,
            ngf
)

ncsn = CondRefineNetDilated(mnist_config)
#ncsn = MyCondRefineNetDilated(mnist_config)

dataloader = get_train_set(batch_size)

trainer = NCSNTrainer(ncsn, lr, dataloader, num_iters, beta1, beta2, checkpoints_folder, save_every)
# trainer.train_ncsn()

model_path = './mycheckpoints/my_ncsn_44'

state_dict = torch.load(model_path, map_location=torch.device('cuda'))
trained_ncsn = MyCondRefineNetDilated(mnist_config).to('cuda')
trained_ncsn.load_state_dict(state_dict)
trained_ncsn.eval()
image_shape = (32, 32, 1)
eps = 2e-5
T = 100
nrow = 8 
#trainer.save_sample_grid(nrow, trained_ncsn, eps, T)
trainer.annealed_langevin_dynamics_perstep(trained_ncsn, eps, T)