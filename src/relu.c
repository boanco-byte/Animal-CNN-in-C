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