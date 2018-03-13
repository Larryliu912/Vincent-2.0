
import numpy as np
import scipy.misc

from stylize import stylize

import histsimilar

from PIL import Image

#This default value refer to https://github.com/anishathalye/neural-style
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 2101
initial_noiseblend = 1.0
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

img_cor = []
content = './images/ex5.jpg'
styles = []

#Get all the style as a array
for i in range(1, 18):
    img_cor.append(['./styles/%d.jpg' % i, histsimilar.calc_similar_by_path(content, './styles/%d.jpg' % i)])
img_cor.sort(key=lambda x:x[1],reverse=True)

#Take the possible related style
styles.append(img_cor[0])
print(img_cor[0][0])
styles.append(img_cor[1])
print(img_cor[1][0])


def main():
    content_image = imread(content)
    style_images = [imread(style[0]) for style in styles] #Accepted all style
    style_blend_weights = [style[1] for style in styles ]
    target_shape = content_image.shape
    #target size set end

    for i in range(len(style_images)): #Do this loop for all style_images
        style_scale = STYLE_SCALE #What this value?
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *   #each style_image should be resized by the style scale of the target_shape by the colums of the pic (Why is not full size but only width)
                target_shape[1] / style_images[i].shape[1])

    total_blend_weight = sum(style_blend_weights)  #sum up all the given style_blend_weights
    style_blend_weights = [weight/total_blend_weight #Let each the weight be divided by the total_blend_weight
                           for weight in style_blend_weights]

    for iteration, image in stylize(   #Get in the stylize
        network=VGG_PATH,
        initial_noiseblend=initial_noiseblend,
        content=content_image,
        styles=style_images,
        iterations=ITERATIONS,
        content_weight=CONTENT_WEIGHT,
        content_weight_blend=CONTENT_WEIGHT_BLEND,
        style_weight=STYLE_WEIGHT,
        style_blend_weights=style_blend_weights,
        tv_weight=1e2,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        pooling=POOLING,
        checkpoint_iterations=100
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            output_file = './outputs/ex7output%s.jpg' % iteration
        if output_file:
            imsave(output_file, combined_rgb) #save the output_file

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
