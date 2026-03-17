/**
  ******************************************************************************
  * @file    weights/batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const float batch_normalization_3_bias[8] = {-0x1.2e40b00000000p-1, -0x1.2168c40000000p+0, -0x1.04467e0000000p-1, -0x1.c1d9d80000000p-3, -0x1.c328f60000000p-2, -0x1.9d2a180000000p-1, -0x1.0cd1140000000p+0, -0x1.7cdbc00000000p-1}
;
const float batch_normalization_3_kernel[8] = {0x1.7aca4e0000000p+4, 0x1.f085540000000p+3, 0x1.29a0f40000000p+2, 0x1.9976ca0000000p+4, 0x1.335f880000000p+3, 0x1.c4b4aa0000000p+3, 0x1.90a8f60000000p+4, 0x1.4a20da0000000p+4}
;