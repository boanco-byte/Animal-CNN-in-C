#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include <stdio.h>
#include <math.h>

float cross_entropy_loss(Tensor *predictions, int target_class);

Tensor *cross_entropy_gradient(Tensor *predictions, int target_class);

#endif // LOSS_H