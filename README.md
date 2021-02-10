# Neural-Style-Transfer
Exploration and Implementation of style transfer techniques from the paper "A Neural Algorithm of Artistic Style" (arXiv:1508.06576)


## Aims

The goal of this project is simply to learn the techniques behind neural style transfer, and to create an easy-to-use interface that allows the user to experiment with different configurations. 

## Implementation Details

The high level description of the flow of the program is as follows. We have a style reference image and we have a content reference image, which we would like to mix. We define a number of parameters, such as the weight of the style and content losses, the weight of the style losses, and the split between the content ans style layers of the network.

The first step is simply to compute the activations of the style and content images at the conv2d layers of the VGG-19 network. We accomplish this by passing both images in the network and saving the outputs at each layer with a forward hook. We store these values in a way that allows them to be efficiently accessed during training. 

The computation of the correlations (the gram matrix) can be done in a vectorized way by collapsing the input tensors with tensor.view(a*b, c*d) and doing a matrix multiplication with the transpose.   

We initialize the optimization algorithm by generating a random noise image or with a copy of the content image. At each forward pass, we compute the style losses at the style layers and the content losses on the content layers. After the forward pass, the total loss is computed as weighted sum of the content and style losses. 

We can then call .backward on the total loss, which stores gradient information in the initial input image. We then update the image with an optimization algortihm of our choice.
