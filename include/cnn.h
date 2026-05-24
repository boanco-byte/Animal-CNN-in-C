#ifndef CNN_H
#define CNN_H
#include "image.h"
#include "tensor.h"
#include "conv.h"
#include "relu.h"
#include "pooling.h"
#include "dense.h"
#include "softmax.h"

typedef struct {
    float *conv1_kernel;

    int conv1_kernel_size;
    int conv1_out_channels;

    int pool_size;
    int pool_stride;

    DenseLayer *fc1;

} CNN;

CNN *cnn_create();

void cnn_free(CNN *cnn);

Tensor *cnn_forward(CNN *cnn, Image *img);

#endif // CNN_H