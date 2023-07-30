from __future__ import absolute_import, division, print_function
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.metrics as sm
import yaml
from keras.models import load_model
import matplotlib.pyplot as plt
# 加载数据
from utils import Trainer, Parser
from visual import plot_confusion_matrix


test_data = np.load("D:\\pythonProject\\AM2F\\datatest.npz")
test_x = test_data["x"]
test_y = test_data["y"]
parser = Parser()
parser.create_parser()
pargs = parser.parser.parse_args()
if pargs.config is not None:
    with open(pargs.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(pargs).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.parser.set_defaults(**default_arg)
args = parser.parser.parse_args()
print("参数:", pargs)
mode = args.mode
trainer = Trainer(args.train_args)
batch_size, data_type, device_id, epoch, model_name, train_path, validate_path, window_size = trainer._read_args(trainer.train_args)
trainer._initial_gpu_env(device_id)
trainer._choose_dataset(data_type, test_x, test_x)
print(test_x.shape)



import numpy as np
import tensorflow as tf

def _fft(bottom, sequential, compute_size):
    return tf.fft(bottom)
def _ifft(bottom, sequential, compute_size):
    return tf.ifft(bottom)

def _generate_sketch_matrix(rand_h, rand_s, output_dim):


    #  Generate a sparse matrix for tensor count sketch

    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix    # [input_dim, output_dim]

def MCB(bottom1, bottom2, output_dim=128,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7, sequential=True,
    compute_size=128):


    # count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim1)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim1) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim2)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim2) - 1  # 1/-1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    #  Flatten the input tensors and count sketch
    bottom1_flat = tf.reshape(bottom1, [-1, input_dim1])
    bottom2_flat = tf.reshape(bottom2, [-1, input_dim2])

    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
        bottom1_flat, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
        bottom2_flat, adjoint_a=True, adjoint_b=True))

    #  FFT
    fft1 = _fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)),
                sequential, compute_size)
    fft2 = _fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)),
                sequential, compute_size)

    fft_product = tf.multiply(fft1, fft2)

    #  iFFT
    mcb_flat = tf.real(_ifft(fft_product, sequential, compute_size))
    output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1,  1, 0]),
                          [0,  0, output_dim])
    mcb = tf.reshape(mcb_flat, output_shape)
    # set static shape for the output
    mcb.set_shape(bottom1.get_shape().as_list()[:-1] + [output_dim])

    return mcb

model = load_model("D:/pythonProject/AM2F/attention_gg.h5",custom_objects={'MCB':MCB })
print(model.summary())
# model.fit(train_x, train_y)

# predict model
result_prob = model.predict({'gyrx_input': trainer.gyr_x_v,
                             'gyry_input': trainer.gyr_y_v,
                             'gyrz_input': trainer.gyr_z_v,
                             'laccx_input': trainer.lacc_x_v,
                             'laccy_input': trainer.lacc_y_v,
                             'laccz_input': trainer.lacc_z_v,

                         })
y_predict = np.argmax(result_prob, axis=-1)
print("预测", y_predict)
test_y = np.argmax(test_y, axis=-1)
print("真实", test_y)

cm = sm.confusion_matrix(test_y, y_predict)
sub_matrix = cm[:3, :3]
plot_confusion_matrix(sub_matrix, ['0'])
print("---------------混淆矩阵\n", sub_matrix)

cp = sm.classification_report(test_y, y_predict,digits=4)
print("---------------分类报告\n", cp)
acc = np.sum(y_predict == test_y) / test_y.size
print(acc)





