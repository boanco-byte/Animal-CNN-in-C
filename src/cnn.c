#include "cnn.h"
#include "image.h"
#include "tensor.h"
#include "conv.h"
#include "relu.h"
#include "pooling.h"
#include "dense.h"
#include "softmax.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

CNN *cnn_create() {
    CNN *cnn = (CNN *)malloc(sizeof(CNN));
    if (!cnn) {
        fprintf(stderr, "Failed to allocate memory for CNN\n");
        return NULL;
    }

    // ===== Conv layer config =====
    cnn->conv1_kernel_size = 3;
    cnn->conv1_out_channels = 8;

    int in_channels = 3;

    int kernel_count =
        cnn->conv1_out_channels *
        in_channels *
        cnn->conv1_kernel_size *
        cnn->conv1_kernel_size;

    cnn->conv1_kernel =
        (float *)malloc(kernel_count * sizeof(float));

    if (!cnn->conv1_kernel) {
        fprintf(stderr, "Failed to allocate memory for conv1 kernel\n");
        free(cnn);
        return NULL;
    }

    for (int i = 0; i < kernel_count; i++) {
        cnn->conv1_kernel[i] =
            ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    // ===== Pooling config =====
    cnn->pool_size = 2;
    cnn->pool_stride = 2;

    // ===== FC layer =====
    // SAME padding conv:
    // 32x32 -> 32x32
    // pool 2 stride 2:
    // 32x32 -> 16x16

    int fc_input_size =
        16 *
        16 *
        cnn->conv1_out_channels;

    int fc_output_size = 10;

    cnn->fc1 =
        dense_layer_create(fc_input_size, fc_output_size);

    if (!cnn->fc1) {
        fprintf(stderr, "Failed to create fully connected layer\n");

        free(cnn->conv1_kernel);
        free(cnn);

        return NULL;
    }

    return cnn;
}

void cnn_free(CNN *cnn) {
    if (!cnn) return;

    if (cnn->conv1_kernel) {
        free(cnn->conv1_kernel);
    }

    if (cnn->fc1) {
        dense_layer_free(cnn->fc1);
    }

    free(cnn);
}

Tensor *cnn_forward(CNN *cnn, Image *img) {
    if (!cnn || !img) {
        fprintf(stderr, "Invalid arguments to cnn_forward\n");
        return NULL;
    }

    // ===== Image -> Tensor =====
    Tensor *t = image_to_tensor(img);
    if (!t) {
        return NULL;
    }

    // ===== Conv =====
    Tensor *conv_out = conv(
        t,
        cnn->conv1_kernel,
        NULL,
        cnn->conv1_kernel_size,
        cnn->conv1_out_channels
    );

    tensor_free(t);

    if (!conv_out) {
        return NULL;
    }

    // ===== ReLU (in-place) =====
    relu(conv_out);

    // ===== Pooling =====
    Tensor *pool_out = max_pooling(
        conv_out,
        cnn->pool_size,
        cnn->pool_stride
    );

    tensor_free(conv_out);

    if (!pool_out) {
        return NULL;
    }

    // ===== Flatten =====
    Tensor *flattened = flatten(pool_out);

    tensor_free(pool_out);

    if (!flattened) {
        return NULL;
    }

    // ===== Dense =====
    Tensor *fc_out = dense_forward(
        cnn->fc1,
        flattened
    );

    tensor_free(flattened);

    if (!fc_out) {
        return NULL;
    }

    // ===== Softmax =====
    softmax_forward(fc_out);

    return fc_out;
}