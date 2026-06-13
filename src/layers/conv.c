#include "conv.h"
#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"
#include "image.h"

Tensor *conv(Tensor *t, float *kernel, int kernel_size, int out_channels) {
    if (!t || !kernel || kernel_size % 2 == 0 || kernel_size <= 0 || out_channels <= 0) {
        fprintf(stderr, "Invalid input to conv\n");
        return NULL;
    }

    int w = t->width;
    int h = t->height;
    int in_c = t->channels;
    int pad = kernel_size / 2;

    Tensor *out = tensor_create(w, h, out_channels);
    if (!out) {
        fprintf(stderr, "Failed to create output feature map\n");
        return NULL;
    }

    float *in = t->data;
    float *out_data = out->data;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                float sum = 0.0f;

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = -pad; ky <= pad; ky++) {
                        for (int kx = -pad; kx <= pad; kx++) {

                            int px = x + kx;
                            int py = y + ky;

                            float val = 0.0f;

                            // SAME padding: ngoài biên = 0
                            if (px >= 0 && px < w && py >= 0 && py < h) {
                                int in_idx = (py * w + px) * in_c + ic;
                                val = in[in_idx];
                            }

                            int k_idx =
                                oc * in_c * kernel_size * kernel_size +
                                ic * kernel_size * kernel_size +
                                (ky + pad) * kernel_size +
                                (kx + pad);

                            sum += val * kernel[k_idx];
                        }
                    }
                }

                int out_idx = (y * w + x) * out_channels + oc;
                out_data[out_idx] = sum;
            }
        }
    }

    return out;
}


Tensor *conv_backward(Tensor *input, Tensor *output_grad, float *kernel, int kernel_size, int out_channels, float learning_rate){
    if (!input || !input->data ||
        !output_grad || !output_grad->data ||
        !kernel)
    {
        fprintf(stderr, "Invalid arguments to conv_backward\n");
        return NULL;
    }

    int w = input->width;
    int h = input->height;
    int in_c = input->channels;
    int pad = kernel_size / 2;

    Tensor *input_grad = tensor_create(w, h, in_c);
    if (!input_grad) return NULL;

    int input_size = w * h * in_c;

    for (int i = 0; i < input_size; i++) {
        input_grad->data[i] = 0.0f;
    }

    for (int oc = 0; oc < out_channels; oc++) {

        for (int y = 0; y < h; y++) {

            for (int x = 0; x < w; x++) {

                int out_idx = (y * w + x) * out_channels + oc;
                float grad_out = output_grad->data[out_idx];

                for (int ic = 0; ic < in_c; ic++) {

                    for (int ky = -pad; ky <= pad; ky++) {

                        for (int kx = -pad; kx <= pad; kx++) {

                            int px = x + kx;
                            int py = y + ky;

                            if (px >= 0 && px < w &&
                                py >= 0 && py < h)
                            {
                                int in_idx =
                                    (py * w + px) * in_c + ic;

                                int kernel_idx =
                                    oc * in_c * kernel_size * kernel_size +
                                    ic * kernel_size * kernel_size +
                                    (ky + pad) * kernel_size +
                                    (kx + pad);

                                /* dLoss/dInput */
                                input_grad->data[in_idx] +=
                                    grad_out * kernel[kernel_idx];

                                /* SGD update kernel */
                                kernel[kernel_idx] -=
                                    learning_rate *
                                    grad_out *
                                    input->data[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return input_grad;
}
