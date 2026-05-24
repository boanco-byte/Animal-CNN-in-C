#include "conv.h"
#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"
#include "image.h"

Tensor *conv(
    Tensor *t,
    float *kernel,
    float *bias,
    int kernel_size,
    int out_channels
) {
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
                float bias_val = bias ? bias[oc] : 0.0f;
                out_data[out_idx] = sum + bias_val;
            }
        }
    }

    return out;
}