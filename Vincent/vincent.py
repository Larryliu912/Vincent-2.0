
import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import histsimilar


COR = []
OUTPUT_DIR = './outputs/'
STYLE_IMAGE = []
CONTENT_IMAGE = './images/ex1.jpg'
for i in range(1, 17):
    COR.append(['./styles/%d.jpg' % i, histsimilar.calc_similar_by_path(CONTENT_IMAGE, './styles/%d.jpg' % i)])
COR.sort(key=lambda x:x[1],reverse=True)

STYLE_IMAGE.append(COR[0][0])
print(COR[0][0])
STYLE_IMAGE.append(COR[1][0])
print(COR[1][0])
if COR[1][1] > 0.4:
    STYLE_IMAGE.append(COR[2][0])
    print(COR[2][0])


img = scipy.misc.imread(CONTENT_IMAGE).astype(np.float)

IMAGE_WIDTH = img.shape[1]
IMAGE_HEIGHT = img.shape[0]
COLOR_CHANNELS = 3

# Noise ratio. Percentage of weight of the noise for intermixing with the content image.
NOISE_RATIO = 1.0
# Number of iterations to run.
ITERATIONS = 2100
EPSILON = 2.0
BETA = 5

ALPHA = 5000

# Pick the VGG 19-layer model by from the paper "Very Deep Convolutional
# Networks for Large-Scale Image Recognition".
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# We get the mean from the VGG model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape(3,)

def generate_noise_image():
    initial = tf.random_normal([1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS]) * 0.256  # inital random Gassian Noise image
    input_image = tf.Variable(initial)
    return input_image

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def load_vgg_model(path):
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

def content_loss_func(sess, model):
    def _content_loss(p, x):
        return 0.5 * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

def style_loss_func(sess, model):
    #Style loss function as defined in the paper.

    def _gram_matrix(F, N, M):
        #The gram matrix G.
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        #The style loss calculation.
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    #Layer weights was set as the paper recommendation
    layers = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2),
    ]

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = sum([W[l] * E[l] for l in range(len(layers))])
    return loss

if __name__ == '__main__':
    with tf.Session() as sess:
        # Load the images.
        content_image = imread(CONTENT_IMAGE)
        target_shape = content_image.shape
        content_image = np.reshape(content_image, ((1,) + content_image.shape))

        # Load the model.
        model = load_vgg_model(VGG_MODEL)

        # Generate the Gassian White noise
        input_image = generate_noise_image()

        sess.run(tf.global_variables_initializer())
        # Construct content_loss using content_image.
        sess.run(model['input'].assign(content_image))
        content_loss = content_loss_func(sess, model)
        style_loss = 0
        # Construct style_loss using style_image.
        for img in STYLE_IMAGE:
            style_image = imread(img)
            if style_image.shape[0] < style_image.shape[1]:
                style_image = scipy.misc.imresize(style_image,   #each style_image should be resized by the style scale of the target_shape by the colums of the pic
                        target_shape[0] / style_image.shape[0])
            else:
                style_image = scipy.misc.imresize( style_image,  #each style_image should be resized by the style scale of the target_shape by the colums of the pic
                        target_shape[1] / style_image.shape[1])
            style_image = style_image[0:target_shape[0], 0:target_shape[1]]
            style_image = np.reshape(style_image, ((1,) + style_image.shape))
            sess.run(model['input'].assign(style_image))
            style_loss += 0.5*style_loss_func(sess, model)


        # Instantiate equation 7 of the paper.
        total_loss = BETA * content_loss + ALPHA * style_loss

        optimizer = tf.train.AdamOptimizer(EPSILON)
        train_step = optimizer.minimize(total_loss)

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))
        for it in range(ITERATIONS):
            #start train
            train_step.run()
            if it%100 == 0:
                # Print every 100 iteration.
                mixed_image = sess.run(model['input'])
                print('Iteration %d' % (it))
                print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
                print('cost: ', sess.run(total_loss))

                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)

                filename = './outputs/%d.png' % (it)
                save_image(filename, mixed_image)