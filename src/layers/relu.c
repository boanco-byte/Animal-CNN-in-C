#include <stdio.h>
#include <stdlib.h>
#include "relu.h"
#include "tensor.h"

void relu (Tensor *t) {
    if (!t || !t->data) {
        fprintf(stderr, "Invalid tensor for ReLU\n");
        return;
    }

    int size = t->width * t->height * t->channels;
    for (int i = 0; i < size; i++) {
        if (t->data[i] < 0) {
            t->data[i] = 0;
        }
    }
}

Tensor *relu_backward(Tensor *output_grad, Tensor *input){
    if (!output_grad || !output_grad->data || !input || !input->data) {
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
    if (!input_grad) return NULL;
    
    for (int i = 0; i < total_elements; i++) {
        if(input->data[i] > 0){
            input_grad->data[i] = output_grad->data[i];
        }
        else input_grad->data[i] = 0.0f;
    }

    return input_grad;
}