#include <stdio.h>
#include <stdlib.h>

#include "dense.h"
#include "tensor.h"

DenseLayer *dense_layer_create(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) {
        fprintf(stderr, "Invalid dense layer size\n");
        return NULL;
    }

    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    if (!layer) {
        fprintf(stderr, "Failed to allocate memory for dense layer\n");
        return NULL;
    }

    layer->weights = NULL;
    layer->biases = NULL;

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float *)malloc(output_size * sizeof(float));

    if (!layer->weights || !layer->biases) {
        fprintf(stderr, "Failed to allocate memory for dense layer weights or biases\n");
        dense_layer_free(layer);
        return NULL;
    }
    
    for (int i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    for (int i = 0; i < output_size; i++) {
        layer->biases[i] = 0.0f;
    }

    return layer;
}

void dense_layer_free(DenseLayer *layer) {
    if (!layer) return;

    if (layer->weights) {
        free(layer->weights);
    }

    if (layer->biases) {
        free(layer->biases);
    }

    free(layer);
}

Tensor *dense_forward(DenseLayer *layer, Tensor *input) {
    if (!layer || !input || !input->data) {
        fprintf(stderr, "Invalid arguments to dense_forward\n");
        return NULL;
    }

    if (input->width * input->height * input->channels != layer->input_size) {
        fprintf(stderr, "Input size mismatch in dense_forward\n");
        return NULL;
    }

    Tensor *output = tensor_create(1, 1, layer->output_size);
    if (!output) {
        return NULL;
    }

    int input_size = layer->input_size;
    int output_size = layer->output_size;
    float *in_data = input->data;

    for (int o = 0; o < output_size; o++) {
        float sum = layer->biases[o];
        for (int i = 0; i < input_size; i++) {
            sum += in_data[i] * layer->weights[o * input_size + i];
        }
        output->data[o] = sum;
    }

    return output;
}

Tensor *dense_backward(DenseLayer *layer, Tensor *input, Tensor *output_grad, float learning_rate) {
    if (!layer || !input || !input->data || !output_grad || !output_grad->data) {
        fprintf(stderr, "Invalid arguments to dense_backward\n");
        return NULL;
    }

    int input_size = input->width * input->height * input->channels;
    int output_size = output_grad->width * output_grad->height * output_grad->channels;

    if (input_size != layer->input_size) {
        fprintf(stderr, "Input size mismatch in dense_backward\n");
        return NULL;
    }

    if (output_size != layer->output_size) {
        fprintf(stderr, "Output size mismatch in dense_backward\n");
        return NULL;
    }

    Tensor *input_grad = tensor_create(input->width, input->height, input->channels);
    if (!input_grad) return NULL;

    for (int i = 0; i < input_size; i++) {
        input_grad->data[i] = 0.0f;
    }

    for (int i = 0; i < output_size; i++) {

        for (int j = 0; j < input_size; j++) {
            input_grad->data[j] += output_grad->data[i] * layer->weights[i * input_size + j];
            layer->weights[i * input_size + j] -= input->data[j] * output_grad->data[i] * learning_rate;
        }

        layer->biases[i] -= output_grad->data[i] * learning_rate;
    }

    return input_grad;
}