#ifndef FEATURE_MAP_H
#define FEATURE_MAP_H

#include <stdio.h>
#include <stdlib.h>
#include "image.h"

typedef struct {
    int width;
    int height;
    int channels;
    float *data;
} FeatureMap;

FeatureMap *feature_map_create(int width, int height, int channels);

void feature_map_free(FeatureMap *fm);

FeatureMap *image_to_feature_map(Image *img);

FeatureMap *feature_map_flatten(FeatureMap *fm);

#endif // FEATURE_MAP_H