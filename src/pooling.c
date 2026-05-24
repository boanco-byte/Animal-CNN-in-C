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

Tensor *average_pooling(Tensor *t, int pool_size, int stride) {
    if (!t || !t->data || pool_size <= 0 || stride <= 0) {
        fprintf(stderr, "Invalid tensor or parameters for average pooling\n");
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

                float sum = 0.0f;
                int count = 0;
                for (int py = y0; py < y0 + pool_size; py++) {
                    for (int px = x0; px < x0 + pool_size; px++) {
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            int idx = (py * w + px) * c + ch;
                            sum += in[idx];
                            count++;
                        }
                    }
                }

                int out_idx = (y * out_w + x) * c + ch;
                out_data[out_idx] = (count > 0) ? (sum / count) : 0.0f;
            }
        }
    }

    return out;
}
