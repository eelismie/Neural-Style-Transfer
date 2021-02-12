import torch 
from torch import nn 
from torch.optim import LBFGS
from torch.nn.functional import mse_loss 
from torch.nn import MSELoss
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]
)

#GLOBAL VARIABLES 

# model construction / images
style_layers = [0, 5, 10, 19] #indices of the style layers in model.features 
content_layers = [28] #indices of the content layers in model.features

content_image = preprocess(Image.open("download.jpeg"))
style_image = preprocess(Image.open("ref2.jpg"))

content_image_batch = content_image.unsqueeze(0)
style_image_batch = style_image.unsqueeze(0) 

model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
model.eval()


# optimi


#________________  

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
    g = gram_matrix(input_)
    return mse_loss(g, reference_g)


class loss_layer(nn.Module):

    def __init__(self,target,style=False):
        super(loss_layer, self).__init__()
        self.style = style
        if style:
            self.target = gram_matrix(target) 
        else: 
            self.target = target
    
    def forward(self, x):
        """ 
        Layer does nothing except records the style / content loss of the previous layer
        """ 
        if not self.style:
            self.loss = content_loss(x, self.target)
        else:
            self.loss = style_loss(x, self.target)
        return x

def get_activations(model, content_im, style_im, style_layers, content_layers):

    """
    return two dicts containing the layer indices and the corresponding activations at that layer 
    for the style_layers and content_layers
    """

    style_activations = []
    content_activations = []
    c = content_im
    s = style_im
    for i, layer in enumerate(model.children()):
        c = layer(c)
        s = layer(s)
        if (i in style_layers):
            style_activations.append((i, s.detach()))
        if (i in content_layers):
            content_activations.append((i, c.detach()))

    return dict(style_activations), dict(content_activations)


def create_network(mod):
    """
    mod : vgg network. Must contain .features attribute
    """
    loss_layers = [] #list that contains references to the intermediate loss layers

    style_activations, content_activations = get_activations(mod.features, content_image_batch, style_image_batch, style_layers, content_layers)

    new_model = torch.nn.Sequential()
    stop = max(style_layers + content_layers)

    for i, layer in enumerate(mod.features):
        if i > stop:
            break
        if isinstance(layer, nn.MaxPool2d):
            new_model.add_module("%d_AvgPool" % (i), nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)) #in the paper it's mentioned that better results are obtained with average pooling
        if isinstance(layer, nn.Conv2d):
            new_model.add_module("%d_Conv2D" % (i), layer)
            if i in style_activations.keys():
                b = loss_layer(style_activations[i],style=True)
                new_model.add_module("%d_style_loss" % (i), b)
            if i in content_activations.keys():
                b = loss_layer(content_activations[i])
                new_model.add_module("%d_content_loss" % (i), b)
        if isinstance(layer, nn.ReLU):
            b = nn.ReLU(inplace=False)
            new_model.add_module("%d_ReLU" % (i), b)

    return new_model, loss_layers 

model, loss_layers = create_network(model)
