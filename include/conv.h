#ifndef CONV_H
#define CONV_H
#include "tensor.h"

Tensor *conv(Tensor *t, float *kernel, float *bias, int kernel_size, int out_channels);

#endif