/**
  ******************************************************************************
  * @file    weights/batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const float batch_normalization_4_bias[16] = {-0x1.43a7cc0000000p-1, -0x1.95f4b80000000p-1, -0x1.d5d3fc0000000p-1, -0x1.881cc80000000p-1, -0x1.73628e0000000p-1, -0x1.9e9c7e0000000p-1, -0x1.397b7c0000000p-1, -0x1.ab8aaa0000000p-1, -0x1.9861ae0000000p-1, -0x1.6e759c0000000p-1, -0x1.7f8edc0000000p-1, -0x1.54b1160000000p-1, -0x1.928b100000000p-1, -0x1.62bf2c0000000p-1, -0x1.7231b60000000p-1, -0x1.a82ad40000000p-1}
;
const float batch_normalization_4_kernel[16] = {0x1.e8fb460000000p-1, 0x1.805bdc0000000p+0, 0x1.b6d99e0000000p+0, 0x1.8aa82e0000000p+0, 0x1.3421c60000000p+0, 0x1.17616c0000000p-1, 0x1.19f5360000000p+0, 0x1.860dd60000000p+0, 0x1.afb5cc0000000p+0, 0x1.29a7440000000p+0, 0x1.48b4de0000000p+0, 0x1.a3de760000000p-1, 0x1.68a09e0000000p-1, 0x1.3ce2ec0000000p-1, 0x1.05dab60000000p+0, 0x1.553e2e0000000p+0}
;