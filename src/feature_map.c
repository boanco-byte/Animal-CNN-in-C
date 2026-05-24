#include <stdio.h>
#include <stdlib.h>
#include "feature_map.h"
#include "image.h"

FeatureMap *feature_map_create (int width, int height, int channels) {
    FeatureMap *fm = (FeatureMap *)malloc(sizeof(FeatureMap));
    if(!fm) {
        fprintf(stderr, "Failed to allocate memory for feature map\n");
        return NULL;
    }

    fm->width = width;
    fm->height = height;
    fm->channels = channels;
    fm->data = (float *)malloc(width * height * channels * sizeof(float));
    if(!fm->data) {
        fprintf(stderr, "Failed to allocate memory for feature map data\n");
        free(fm);
        return NULL;
    }

    return fm;
}

void feature_map_free(FeatureMap *fm) {
    if (!fm) return;

    if (fm->data) {
        free(fm->data);
    }

    free(fm);
}

FeatureMap *image_to_feature_map(Image *img) {
    if (!img || !img->data) {
        fprintf(stderr, "Invalid image for feature map conversion\n");
        return NULL;
    }

    FeatureMap *fm = feature_map_create(img->width, img->height, img->channels);
    if (!fm) {
        return NULL;
    }

    int size = img->width * img->height * img->channels;
    for (int i = 0; i < size; i++) {
        fm->data[i] = (float)img->data[i];
    }

    return fm;
}

FeatureMap *feature_map_flatten(FeatureMap *fm) {
    if (!fm || !fm->data) {
        fprintf(stderr, "Invalid feature map for flattening\n");
        return NULL;
    }

    int new_channels = fm->width * fm->height * fm->channels;
    FeatureMap *out = feature_map_create(1, 1, new_channels);
    if (!out) {
        return NULL;
    }

    int size = fm->width * fm->height * fm->channels;
    for (int i = 0; i < size; i++) {
        out->data[i] = fm->data[i];
    }

    return out;
}