#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include "stb_image.h"
#include "image.h"
#include "tensor.h"
#include "conv.h"

Image *image_load(const char *filename) {
    Image *img = (Image *)malloc(sizeof(Image));
    if (!img) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        return NULL;
    }

    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 0);
    if (!img->data) {
        fprintf(stderr, "Failed to load image %s: %s\n", filename, stbi_failure_reason());
        free(img);
        return NULL;
    }

    return img;
}

int image_save_bmp(Image *img, const char *filename) {
    if (!img || !filename) return 0;
    if (img->channels < 3) return 0;

    int width = img->width;
    int height = img->height;
    int row_bytes = width * 3;
    int padding = (4 - (row_bytes % 4)) % 4;
    int stride = row_bytes + padding;
    int file_size = 14 + 40 + stride * height;

    unsigned char file_header[14] = {
        'B', 'M',
        (unsigned char)(file_size      ),
        (unsigned char)(file_size >>  8),
        (unsigned char)(file_size >> 16),
        (unsigned char)(file_size >> 24),
        0, 0, 0, 0,
        54, 0, 0, 0
    };

    unsigned char info_header[40] = {
        40, 0, 0, 0,
        (unsigned char)(width      ),
        (unsigned char)(width >>  8),
        (unsigned char)(width >> 16),
        (unsigned char)(width >> 24),
        (unsigned char)(height      ),
        (unsigned char)(height >>  8),
        (unsigned char)(height >> 16),
        (unsigned char)(height >> 24),
        1, 0,
        24, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    FILE *f = fopen(filename, "wb");
    if (!f) return 0;
    if (fwrite(file_header, 1, sizeof(file_header), f) != sizeof(file_header)) {
        fclose(f);
        return 0;
    }
    if (fwrite(info_header, 1, sizeof(info_header), f) != sizeof(info_header)) {
        fclose(f);
        return 0;
    }

    unsigned char pad[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            unsigned char *pixel = &img->data[(y * width + x) * img->channels];
            unsigned char b = pixel[2];
            unsigned char g = pixel[1];
            unsigned char r = pixel[0];
            unsigned char out[3] = {b, g, r};
            if (fwrite(out, 1, 3, f) != 3) {
                fclose(f);
                return 0;
            }
        }
        if (padding && fwrite(pad, 1, padding, f) != (size_t)padding) {
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

Image *image_create(int width, int height, int channels) {
    Image *img = (Image *)malloc(sizeof(Image));

    if(!img) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        return NULL;
    }

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (unsigned char *)malloc(width * height * channels);

    if(!img->data) {
        fprintf(stderr, "Failed to allocate memory for image data\n");
        free(img);
        return NULL;
    }

    return img;
}

void image_free(Image *img) {
    if (img) {
        if (img->data) {
            free(img->data);
        }
        free(img);
    }
}

unsigned char *image_get_pixel(Image *img, int x, int y, int channel) {
    if (x < 0 || x >= img->width || y < 0 || y >= img->height) {
        return NULL; // Out of bounds
    }

    if (channel < 0 || channel >= img->channels) {
        return NULL; // Invalid channel
    }

    return &img->data[(y * img->width + x) * img->channels + channel];
}

void image_set_pixel(Image *img, int x, int y, int channel, unsigned char value) {
    if(x < 0 || x >= img->width || y < 0 || y >= img->height) {
        return;
    }

    if(channel < 0 || channel >= img->channels) {
        return;
    }

    img->data[(y * img->width + x) * img->channels + channel] = value;
}

Image *image_resize(Image *img, int new_width, int new_height) {
    if (!img || !img->data || new_width <= 0 || new_height <= 0) {
        fprintf(stderr, "Invalid input to image_resize\n");
        return NULL;
    }

    Image *resized = image_create(
        new_width,
        new_height,
        img->channels
    );

    if (!resized) {
        fprintf(stderr, "Failed to create resized image\n");
        return NULL;
    }

    float x_ratio = (float)img->width / new_width;
    float y_ratio = (float)img->height / new_height;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {

            // nearest neighbor source pixel
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);

            // safety clamp
            if (src_x >= img->width) {
                src_x = img->width - 1;
            }

            if (src_y >= img->height) {
                src_y = img->height - 1;
            }

            for (int c = 0; c < img->channels; c++) {

                int src_idx =
                    (src_y * img->width + src_x) * img->channels + c;

                int dst_idx =
                    (y * new_width + x) * img->channels + c;

                resized->data[dst_idx] =
                    img->data[src_idx];
            }
        }
    }

    return resized;
}

void grayscale(Image *img) {
    if(img->channels < 3) {
        return;
    }

    for(int y = 0; y < img->height; y++) {
        for(int x = 0; x < img->width; x++) {
            unsigned char *r = image_get_pixel(img, x, y, 0);
            unsigned char *g = image_get_pixel(img, x, y, 1);
            unsigned char *b = image_get_pixel(img, x, y, 2);

            if(r && g && b) {
                unsigned char gray = (unsigned char)(0.299 * (*r) + 0.587 * (*g) + 0.114 * (*b));
                image_set_pixel(img, x, y, 0, gray);
                image_set_pixel(img, x, y, 1, gray);
                image_set_pixel(img, x, y, 2, gray);
            }
        }
    }
}


Image *image_convolution(Image *img, float *kernel, int kernel_size) {
    if (!img || !img->data || !kernel || kernel_size <= 0 || (kernel_size % 2) == 0) {
        fprintf(stderr, "Invalid arguments to image_convolution\n");
        return NULL;
    }

    Tensor *t = image_to_tensor(img);
    if (!t) return NULL;

    int out_channels = t->channels;
    float *bias = (float *)calloc(out_channels, sizeof(float));
    if (!bias) {
        tensor_free(t);
        fprintf(stderr, "Failed to allocate bias for image_convolution\n");
        return NULL;
    }

    Tensor *res = conv(t, kernel, bias, kernel_size, out_channels);
    free(bias);
    tensor_free(t);
    if (!res) return NULL;

    Image *out = image_create(res->width, res->height, res->channels);
    if (!out) {
        tensor_free(res);
        return NULL;
    }

    int size = res->width * res->height * res->channels;
    for (int i = 0; i < size; i++) {
        float v = res->data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        out->data[i] = (unsigned char)(v);
    }

    tensor_free(res);
    return out;
}

