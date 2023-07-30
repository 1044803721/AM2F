from __future__ import absolute_import, division, print_function

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
