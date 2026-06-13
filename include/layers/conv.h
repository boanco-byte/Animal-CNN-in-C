#ifndef CONV_H
#define CONV_H
#include "tensor.h"

Tensor *conv(Tensor *t, float *kernel, int kernel_size, int out_channels);

Tensor *conv_backward(Tensor *input, Tensor *output_grad, float *kernel, int kernel_size, int out_channels, float learning_rate);

#endif