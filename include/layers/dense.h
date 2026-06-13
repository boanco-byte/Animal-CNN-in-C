#ifndef DENSE_H
#define DENSE_H
#include "tensor.h"

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
} DenseLayer;

DenseLayer *dense_layer_create(int input_size, int output_size);

void dense_layer_free(DenseLayer *layer);

Tensor *dense_forward(DenseLayer *layer, Tensor *input);

Tensor *dense_backward(DenseLayer *layer, Tensor *input, Tensor *output_grad, float learning_rate);
#endif // DENSE_H