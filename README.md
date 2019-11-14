# Vincent-2.0
It is a Neural Network that can transfer a picture to amazing Vincent van Gogh's oil paintings. 

Here I try to implement the algorithm and neural network in the paper

A Neural Algorithm of Artistic Style https://arxiv.org/pdf/1508.06576v2.pdf by Tensorflow and Python. In this paper, the authors claim a method to
obtain a representation of the style of a picture and apply and combine this style into another picture.

For Vincent, the main idea is: 

1. Use the pre-trained classification network VGG 19 to extract the features in the different layers of the picture (The imagenet-vgg-verydeep-19.mat is needed to run this program, because it is too large so please download it by http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat and put it in the document Vincent)

2. The style is in the low layer of the picture, the content is the in the high layer of the picture

3. Use the gradient descent to fine the image by the loss function

4. After iteration, we can keep a part of the content of the picture and apply the style of the other picture in this picture.

So, in practice, we firstly input a image into the VGG network, we can get the each convolution layer response, let us say $X^l$. We can also have a target image $\hat{X}^l$. Thus, we can have the error: 

<img src="https://latex.codecogs.com/gif.latex?$$E_c^l&space;=&space;\frac{1}{2}||X^l&space;-&space;\hat{X}^l||^2$$" title="$$E_c^l = \frac{1}{2}||X^l - \hat{X}^l||^2$$" />
It is easy to see we can get the derivative between the $E_c^l$ and $dx^l$ which is the pixel by the back-propagation. Thus, we can make the content approach the target by update the pixel.

The paper provides the loss function:

Loss function for the content:

<img src="https://latex.codecogs.com/gif.latex?$$L_{content}(p,x,l)&space;=&space;\frac{1}{2}&space;\sum_{i,j}(F^l_{ij}-P^l_{ij})^2$$" title="$$L_{content}(p,x,l) = \frac{1}{2} \sum_{i,j}(F^l_{ij}-P^l_{ij})^2$$" />
Loss function for the style:

<img src="https://latex.codecogs.com/gif.latex?$$L_{sytle}(a,x)&space;=&space;w_l&space;\frac{1}{4N_l^2&space;M_l^2}&space;\sum_{i,j}(G^l_{ij}-A^l_{ij})^2$$" title="$$L_{sytle}(a,x) = w_l \frac{1}{4N_l^2 M_l^2} \sum_{i,j}(G^l_{ij}-A^l_{ij})^2$$" />
Total loss function:

<img src="https://latex.codecogs.com/gif.latex?$$L_{total}(p,a,x)&space;=&space;\alpha&space;L_{content}(p,x)&space;&plus;&space;\beta&space;L_{style}(a,x)$$" title="$$L_{total}(p,a,x) = \alpha L_{content}(p,x) + \beta L_{style}(a,x)$$" />
And the Vincent was implemented by above idea.
