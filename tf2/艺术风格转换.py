import numpy as np
import tensorflow
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

target_image_path = "cat.jpg"
style_reference_image_path = "style.jpg"

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

def preprocess_image(image_path):  # 对图片预处理
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


from keras import backend as backend

target_image = backend.constant(preprocess_image(target_image_path))
style_reference_image = backend.constant(preprocess_image(style_reference_image_path))
combination_image = backend.placeholder((1, img_height, img_width, 3))

input_tensor = backend.concatenate([target_image,
                                    style_reference_image,
                                    combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)


def content_loss(base, combination):
    return backend.sum(backend.square(combination - base))

def gram_matrix(x):  # 用于计算 gram matrix
    #将表示深度的最后一维置换到前面，再flatten后就是上文中n*m的矩阵A
    features = backend.batch_flatten(tensorflow.keras.backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])  # 各隐藏层名称和输出的映射

content_layer = 'block5_conv2'  # 第5个block
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']  # 所有5个block

style_weight = 1.
content_weight = 0.025

loss = backend.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss =loss+content_weight * content_loss(target_image_features,
                                      combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style = style_loss(style_reference_features, combination_features)
    loss += (style_weight * (style / len(style_layers)))


grads = backend.gradients(loss, combination_image)[0]
fetch_loss_and_grads = backend.function([combination_image], [loss, grads])

class Evaluator(object):  # 建立一个类，同时计算loss和gradient
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()



from scipy.optimize import fmin_l_bfgs_b
# from scipy.misc import imsave
import imageio

import time

def deprocess_image(x): # 对生成的图片进行后处理
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # 'BGR'->'RGB'
    x = np.clip(x, 0, 255).astype('uint8')
    return x

result_prefix = 'style_transfer_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(1, iterations + 1):
    print('start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('Current loss value: ', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imageio.imsave(fname, img)
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, time.time() - start_time))

