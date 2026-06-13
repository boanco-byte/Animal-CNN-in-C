#include "loss.h"
#include "tensor.h"
#include <stdio.h>
#include <math.h>

float cross_entropy_loss(Tensor *predictions, int target_class) {
    if (!predictions || !predictions->data) {
        fprintf(stderr, "Error: Invalid predictions tensor.\n");
        return -1.0f; // Return an error value
    }

    int size = predictions->width * predictions->height * predictions->channels;
    if (target_class < 0 || target_class >= size) {
        fprintf(stderr, "Error: Target class index out of bounds.\n");
        return -1.0f; // Return an error value
    }

    float predicted_prob = predictions->data[target_class];
    if (predicted_prob <= 0.0f) {
        fprintf(stderr, "Error: Predicted probability must be greater than 0.\n");
        return -1.0f; // Return an error value
    }

    float loss = -logf(predicted_prob);

    const float epsilon = 1e-10f;
    if (loss < epsilon) {
        loss = epsilon; // Prevent loss from being too small
    }

    return loss;
}

Tensor *cross_entropy_gradient(Tensor *predictions, int target_class) {
    if (!predictions || !predictions->data) {
        fprintf(stderr, "Error: Invalid predictions tensor.\n");
        return NULL; // Return NULL on error
    }

    int size = predictions->width * predictions->height * predictions->channels;
    if (target_class < 0 || target_class >= size) {
        fprintf(stderr, "Error: Target class index out of bounds.\n");
        return NULL; // Return NULL on error
    }

    Tensor *gradient = tensor_create(predictions->width, predictions->height, predictions->channels);
    if (!gradient) {
        fprintf(stderr, "Error: Failed to create gradient tensor.\n");
        return NULL; // Return NULL on error
    }

    for (int i = 0; i < size; i++) {
        gradient->data[i] = predictions->data[i];
    }
    gradient->data[target_class] -= 1.0f;

    return gradient;
}