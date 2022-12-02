import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided


def toeplitz_from_vector(column, row):
    column = np.asarray(column).ravel()
    if row is None:
        row = column.conjugate()
    else:
        row = np.asarray(row).ravel()
    vals = np.concatenate((column[::-1], row[1:]))
    out_shp = len(column), len(row)
    n = vals.strides[0]
    return as_strided(vals[len(column) - 1:], shape=out_shp, strides=(-n, n)).copy()


def single_channel(kernel, input_size):
    kernel_rows, kernel_cols = kernel.shape
    input_rows, input_cols = input_size
    output_rows, output_cols = input_rows - kernel_rows + 1, input_cols - kernel_cols + 1

    toeplitz = [toeplitz_from_vector((kernel[row, 0], *np.zeros(input_cols - kernel_cols)), (*kernel[row], *np.zeros(input_cols - kernel_cols))) for row in range(kernel_rows)]

    col, row = toeplitz[0].shape

    weights = np.zeros((output_rows, col, input_rows, row))

    for i, B in enumerate(toeplitz):
        for j in range(output_rows):
            weights[j, :, i + j, :] = B

    weights.shape = (output_rows * col, input_rows * row)
    return weights


def multiple_channel(kernel, input_size, padding=1):
    r, m, n = input_size
    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[2] - 1) + 2 * padding,
                   input_size[2] - (kernel_size[3] - 1) + 2 * padding)
    weights = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))
    x_off = padding * (n + 2 * padding) + padding
    y_off = 0
    pad = np.zeros(((m + 2 * padding) * (n + 2 * padding), m * n))
    for i in range(m):
        pad[x_off:x_off + n, y_off:y_off + n] = np.eye(n, n)
        x_off += n + 2 * padding
        y_off += n

    for i, ks in enumerate(kernel):
        for j, k in enumerate(ks):
            toeplitz = single_channel(k, (m + 2 * padding, n + 2 * padding))
            weights[i, :, j, :] = toeplitz @ pad

    weights = weights.reshape((np.prod(output_size), np.prod(input_size)))

    return weights

@staticmethod
def multiple_channel_with_stride(kernel, input_size, stride, padding=1):
    r, m, n = input_size
    t = kernel.shape[0]
    weights = multiple_channel(kernel, (r, m, n), padding=padding)

    m_out = m - (kernel.shape[-2] - 1) + 2 * padding
    n_out = n - (kernel.shape[-1] - 1) + 2 * padding

    choose_elems = np.zeros(n_out)
    choose_elems[::stride] = 1
    mask = np.zeros((m_out, n_out), dtype='float32')
    mask[::stride] = choose_elems
    mask = np.reshape(mask, (-1))
    mask = np.tile(mask, t)
    weights = weights[mask > 0]
    return torch.from_numpy(weights.astype('float32'))
