import torch 
from torch import nn 
from torch.optim import LBFGS
from torch.nn.functional import mse_loss 
from torch.nn import MSELoss
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]
)

class InverseNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

un = InverseNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#--GLOBAL VARIABLES 

# device handle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model construction / images

style_layers = [0, 5, 10, 19] #indices of the style layers in model.features 
content_layers = [28] #indices of the content layers in model.features

content_image = preprocess(Image.open("trump.jpg"))
style_image = preprocess(Image.open("guernica.jpg"))

content_image_batch = content_image.unsqueeze(0).to(device)
style_image_batch = style_image.unsqueeze(0).to(device)

model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
model.eval()
model.to(device)

# optimization

iters = 1000
lr = 0.02
style_weights = [0.2,0.2,0.2,0.2]
alpha = 1 # content weight
beta = 1 #style weight

#--END GLOBAL VARIABLES

def gram_matrix(input_):
    a,b,c,d = input_.size()
    a = input_.view(a*b, c*d)
    gram = torch.mm(a, a.t())
    return gram

def content_loss(input_, reference):
    """
    input : torch.Tensor with all the activations in the current layer 
    """ 
    return mse_loss(input_, reference)

def style_loss(input_, reference_g): 
    """
    input : torch.Tensor with all the activations in the current layer
    eference_g : torch.Tensor
    """ 
    a,b,c,d = input_.size()
    g = gram_matrix(input_)
    return 0.25*(1/((b*c*d)**2))*torch.sum(torch.square(g - reference_g))

class loss_layer(nn.Module):

    def __init__(self,target,style=False):
        super(loss_layer, self).__init__()
        self.style = style
        if style:
            self.target = gram_matrix(target.detach()) 
        else: 
            self.target = target.detach()
    
    def forward(self, x):
        """ 
        Layer does nothing except records the style / content loss of the previous layer
        """ 
        if not self.style:
            self.loss = content_loss(x, self.target)
        else:
            self.loss = style_loss(x, self.target)
        return x

def create_network(mod):
    """
    mod : vgg network. Must contain .features attribute
    """
    
    style_losses = [] #list that contains references to the intermediate loss layers
    content_losses = []

    new_model = torch.nn.Sequential()
    stop = max(style_layers + content_layers)

    c = content_image_batch
    s = style_image_batch

    for i, layer in enumerate(mod.features):

        if i > stop:
            break
        
        if isinstance(layer, nn.MaxPool2d):
            #change all max-pool layers to avg_pool layers
            b = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            new_model.add_module("%d_AvgPool" % (i), b) 
            c = b(c)
            s = b(s)

        if isinstance(layer, nn.Conv2d):
            #keep the conv2d layers as they are
            new_model.add_module("%d_Conv2D" % (i), layer)
            c = layer(c)
            s = layer(s)
            if i in style_layers:
                b = loss_layer(s,style=True)
                new_model.add_module("%d_style_loss" % (i), b)
                style_losses.append(b)
            if i in content_layers:
                b = loss_layer(c)
                new_model.add_module("%d_content_loss" % (i), b)
                content_losses.append(b)

        if isinstance(layer, nn.ReLU):
            #change relu layers to inplace=False
            b = nn.ReLU(inplace=False)
            c = layer(c)
            s = layer(s)
            new_model.add_module("%d_ReLU" % (i), b)

    return new_model, style_losses, content_losses

new_model, style_losses, content_losses = create_network(model)

#run style transfer
gen = torch.rand_like(style_image_batch,requires_grad=True).to(device)
optim = torch.optim.Adam([gen], lr=lr)
losses = []
for iter in range(iters):
    optim.zero_grad()
    new_model(gen)
    total_loss = 0
    for i, s in enumerate(style_losses):
        total_loss += style_weights[i]*s.loss*beta
    for i, c in enumerate(content_losses):
        total_loss += c.loss*alpha
    losses.append(total_loss.detach().cpu().numpy().item())
    total_loss.backward()
    optim.step()


#show result 
un(gen[0].detach())
plt.imshow(gen[0].permute(1, 2, 0).detach().cpu().numpy())

#uncomment to save result 
#plt.savefig("result.png")
