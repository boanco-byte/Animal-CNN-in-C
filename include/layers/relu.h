#ifndef RELU_H
#define RELU_H
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

void relu(Tensor *t);

Tensor *relu_backward(Tensor *output_grad, Tensor *input);

#endif // RELU_H