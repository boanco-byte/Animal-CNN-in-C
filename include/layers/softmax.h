#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor.h"

void softmax_forward(Tensor *input);

int tensor_argmax(Tensor *tensor);

#endif // SOFTMAX_H 