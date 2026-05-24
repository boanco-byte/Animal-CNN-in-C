#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "tensor.h"
#include "cnn.h"

int main() {
    // ===== Load image =====
    const char *filename = "dataset/cat.jpg";

    Image *img = image_load(filename);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        return 1;
    }

    Image *resized = image_resize(img, 32, 32);

    image_free(img);

    img = resized;

    printf("Loaded image: %s\n", filename);
    printf("Width: %d\n", img->width);
    printf("Height: %d\n", img->height);
    printf("Channels: %d\n", img->channels);

    // ===== Create CNN =====
    CNN *cnn = cnn_create();
    if (!cnn) {
        fprintf(stderr, "Failed to create CNN\n");
        image_free(img);
        return 1;
    }

    // ===== Forward pass =====
    Tensor *output = cnn_forward(cnn, img);
    if (!output) {
        fprintf(stderr, "CNN forward failed\n");

        cnn_free(cnn);
        image_free(img);

        return 1;
    }

    // ===== Print probabilities =====
    printf("\nOutput probabilities:\n");

    int size =
        output->width *
        output->height *
        output->channels;

    for (int i = 0; i < size; i++) {
        printf("Class %d: %.6f\n", i, output->data[i]);
    }

    // ===== Prediction =====
    int predicted = tensor_argmax(output);

    printf("\nPredicted class: %d\n", predicted);

    // ===== Cleanup =====
    tensor_free(output);
    cnn_free(cnn);
    image_free(img);

    return 0;
}