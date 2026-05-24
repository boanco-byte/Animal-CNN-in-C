#ifndef POOLING_H
#define POOLING_H

#include "tensor.h"

Tensor *max_pooling(Tensor *t, int pool_size, int stride);
Tensor *average_pooling(Tensor *t, int pool_size, int stride);

#endif // POOLING_H