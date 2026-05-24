#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "softmax.h"
#include "tensor.h"

void softmax_forward(Tensor *input) {
    if (!input || !input->data) {
        fprintf(stderr, "Invalid input to softmax_forward\n");
        return;
    }

    int size = input->width * input->height * input->channels;
    float max_val = input->data[0];
    for(int i = 1; i < size; i++) {
        if (input->data[i] > max_val) {
            max_val = input->data[i];
        }
    }

    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        input->data[i] = expf(input->data[i] - max_val);
        sum += input->data[i];
    }

    for(int i = 0; i < size; i++) {
        input->data[i] /= sum;
    }
}

int tensor_argmax(Tensor *tensor) {
    if (!tensor || !tensor->data) {
        fprintf(stderr, "Invalid input to tensor_argmax\n");
        return -1;
    }

    int size = tensor->width * tensor->height * tensor->channels;
    int max_idx = 0;
    for(int i = 1; i < size; i++) {
        if(tensor->data[i] > tensor->data[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}