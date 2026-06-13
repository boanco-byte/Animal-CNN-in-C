#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "image.h"

Tensor *tensor_create (int width, int height, int channels) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if(!t) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        return NULL;
    }

    t->width = width;
    t->height = height;
    t->channels = channels;
    t->data = (float *)malloc(width * height * channels * sizeof(float));
    if(!t->data) {
        fprintf(stderr, "Failed to allocate memory for tensor data\n");
        free(t);
        return NULL;
    }

    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;

    if (t->data) {
        free(t->data);
    }

    free(t);
}

Tensor *image_to_tensor(Image *img) {
    if (!img || !img->data) {
        fprintf(stderr, "Invalid image for tensor conversion\n");
        return NULL;
    }

    Tensor *t = tensor_create(img->width, img->height, img->channels);
    if (!t) {
        return NULL;
    }

    int size = img->width * img->height * img->channels;
    for (int i = 0; i < size; i++) {
        t->data[i] = (float)img->data[i];
    }

    return t;
}

Tensor *flatten(Tensor *t) {
    if (!t || !t->data) {
        fprintf(stderr, "Invalid tensor for flattening\n");
        return NULL;
    }

    int new_channels = t->width * t->height * t->channels;
    Tensor *out = tensor_create(1, 1, new_channels);
    if (!out) {
        return NULL;
    }

    int size = t->width * t->height * t->channels;
    for (int i = 0; i < size; i++) {
        out->data[i] = t->data[i];
    }

    return out;
}

Tensor *flatten_backward(Tensor *output_grad, Tensor *input){
    if (!output_grad || !output_grad->data || !input) {
        fprintf(stderr, "Error: Invalid arguments to flatten_backward.\n");
        return NULL;
    }

    int total_elements = input->width * input->height * input->channels;

    int grad_elements = output_grad->width * output_grad->height * output_grad->channels;
    if (total_elements != grad_elements) {
        fprintf(stderr, "Error: Kích thước gradient không khớp với Tensor đầu vào.\n");
        return NULL;
    }

    Tensor *input_grad = tensor_create(input->width, input->height, input->channels);
    if (!input_grad) {
        fprintf(stderr, "Error: Không thể cấp phát bộ nhớ cho input_grad.\n");
        return NULL;
    }

    memcpy(input_grad->data, output_grad->data, total_elements * sizeof(float));

    return input_grad;
}

