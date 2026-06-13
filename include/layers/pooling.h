#ifndef POOLING_H
#define POOLING_H

#include "tensor.h"

Tensor *max_pooling(Tensor *t, int pool_size, int stride);

Tensor *max_pooling_backward(Tensor *output_grad, Tensor *input, int pool_size, int stride);

#endif // POOLING_H