#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "pooling.h"
#include "tensor.h"

Tensor *max_pooling(Tensor *t, int pool_size, int stride) {
    if (!t || !t->data || pool_size <= 0 || stride <= 0) {
        fprintf(stderr, "Invalid tensor or parameters for max pooling\n");
        return NULL;
    }

    int w = t->width;
    int h = t->height;
    int c = t->channels;

    int out_w = (w - pool_size) / stride + 1;
    int out_h = (h - pool_size) / stride + 1;
    if (out_w <= 0 || out_h <= 0) {
        fprintf(stderr, "Pool size/stride too large for input dimensions\n");
        return NULL;
    }

    Tensor *out = tensor_create(out_w, out_h, c);
    if (!out) {
        fprintf(stderr, "Failed to create pooled feature map\n");
        return NULL;
    }

    float *in = t->data;
    float *out_data = out->data;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            for (int ch = 0; ch < c; ch++) {
                int x0 = x * stride;
                int y0 = y * stride;

                float m = -FLT_MAX;
                for (int py = y0; py < y0 + pool_size; py++) {
                    for (int px = x0; px < x0 + pool_size; px++) {
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            int idx = (py * w + px) * c + ch;
                            if (in[idx] > m) m = in[idx];
                        }
                    }
                }

                int out_idx = (y * out_w + x) * c + ch;
                out_data[out_idx] = m;
            }
        }
    }

    return out;
}

Tensor *max_pooling_backward(Tensor *output_grad, Tensor *input, int pool_size, int stride){
    if (!output_grad || !output_grad->data ||
        !input || !input->data ||
        pool_size <= 0 || stride <= 0)
    {
        fprintf(stderr, "Invalid arguments to max_pooling_backward\n");
        return NULL;
    }

    int w = input->width;
    int h = input->height;
    int c = input->channels;

    int out_w = output_grad->width;
    int out_h = output_grad->height;

    Tensor *input_grad = tensor_create(w, h, c);
    if (!input_grad) {
        return NULL;
    }

    int input_size = w * h * c;

    /* khởi tạo toàn bộ gradient bằng 0 */
    for (int i = 0; i < input_size; i++) {
        input_grad->data[i] = 0.0f;
    }

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            for (int ch = 0; ch < c; ch++) {

                int x0 = x * stride;
                int y0 = y * stride;

                float max_val = -FLT_MAX;
                int max_idx = -1;

                for (int py = y0; py < y0 + pool_size; py++) {
                    for (int px = x0; px < x0 + pool_size; px++) {

                        if (px >= 0 && px < w &&
                            py >= 0 && py < h)
                        {
                            int idx = (py * w + px) * c + ch;

                            if (input->data[idx] > max_val) {
                                max_val = input->data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                }

                int out_idx = (y * out_w + x) * c + ch;

                if (max_idx != -1) {
                    input_grad->data[max_idx] += output_grad->data[out_idx];
                }
            }
        }
    }

    return input_grad;
}
