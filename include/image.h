#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Image;

Image *image_load(const char *filename);

int image_save_bmp(Image *img, const char *filename);

Image *image_create(int width, int height, int channels);

void image_free(Image *img);

unsigned char *image_get_pixel(Image *img, int x, int y, int channel);

void image_set_pixel(Image *img, int x, int y, int channel, unsigned char value);

Image *image_resize(Image *img, int new_width, int new_height);

void grayscale(Image *img);

Image *image_convolution(Image *img, float *kernel, int kernel_size);

#endif