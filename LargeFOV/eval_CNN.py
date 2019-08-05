import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import keras.backend as K
import h5py 
from keras.models import Sequential, load_model, Model

FOVSize = 150

def f1_acc(y_true, y_pred):

    # import numpy as np

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    # if K.isnan(f1_score):
    #     print('c1:', c1)
    #     print('c2:', c2)
    #     print('c3:', c3)

    return f1_score

model = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV{}.h5'.format(FOVSize), custom_objects={'f1_acc': f1_acc})

def make_and_save_filter_img(layer_number, pool = None):
  layer_name = 'conv2d_{}'.format(layer_number)
  if pool:
    layer_name = pool
  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[1], (1, 150, 150, 1)))
  s = intermediate_output.shape
  first = int(s[3] / 2)
  sec = int(s[3] - 1)
  plt.subplot(4,3,1)
  plt.imshow(TestNo[1])
  plt.title('original')
  plt.axis('off')
  plt.subplot(4,3,2)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,3)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestNo[10], (1, 150, 150, 1)))
  plt.subplot(4,3,4)
  plt.imshow(TestNo[10])
  plt.axis('off')
  plt.subplot(4,3,5)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,6)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[25], (1, 150, 150, 1)))
  plt.subplot(4,3,7)
  plt.imshow(TestYes[25])
  plt.axis('off')
  plt.subplot(4,3,8)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,9)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')

  intermediate_output = intermediate_layer_model.predict(np.reshape(TestYes[36], (1, 150, 150, 1)))
  plt.subplot(4,3,10)
  plt.imshow(TestYes[36])
  plt.axis('off')
  plt.subplot(4,3,11)
  plt.imshow(np.reshape(intermediate_output[:, :, :, first], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.subplot(4,3,12)
  plt.imshow(np.reshape(intermediate_output[:, :, :, sec], (intermediate_output.shape[1], intermediate_output.shape[2])))
  plt.axis('off')
  plt.savefig('intermediate_output_{}_layer:{}.png'.format(FOVSize, layer_number))
make_and_save_filter_img(1)
make_and_save_filter_img(2)
make_and_save_filter_img(3)
make_and_save_filter_img(4, pool = "max_pooling2d_2")
make_and_save_filter_img(4, pool = "max_pooling2d_3")
# make_and_save_filter_img(4)
# make_and_save_filter_img(5)
# make_and_save_filter_img(6)


no_preds = high_acc.predict(np.reshape(TestNo, (len(TestNo), FOVSize, FOVSize, 1)))
yes_preds = high_acc.predict(np.reshape(TestYes, (len(TestYes), FOVSize, FOVSize, 1)))
vals_y = high_acc.predict(np.reshape(ValYes, (len(ValYes), FOVSize, FOVSize, 1)))
vals_n = high_acc.predict(np.reshape(ValNo, (len(ValNo), FOVSize, FOVSize, 1)))
plt.subplot(121)
plt.hist(no_preds, bins = 15)
plt.title('no craters')
plt.subplot(122)
plt.hist(yes_preds, bins = 15)
plt.title('with craters')
plt.savefig('CNN_{}_hist.png'.format(FOVSize))

x = len([i for i in no_preds if i < .5]) / len(no_preds)
y = len([i for i in yes_preds if i > .5]) / len(yes_preds)

print('no:')
print(x)
print('yes:')
print(y)
print('total:')
print((x + y) / 2)
print('val:')
print((len([i for i in vals_y if i > .5]) + len([i for i in vals_n if i < .5])) / (len(vals_y) + len(vals_n)))
