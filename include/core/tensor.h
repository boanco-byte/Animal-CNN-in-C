#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include "image.h"

typedef struct {
    int width;
    int height;
    int channels;
    float *data;
} Tensor;

Tensor *tensor_create(int width, int height, int channels);

void tensor_free(Tensor *t);

Tensor *image_to_tensor(Image *img);

Tensor *flatten(Tensor *t);

Tensor *flatten_backward(Tensor *output_grad, Tensor *input);


#endif // TENSOR_H