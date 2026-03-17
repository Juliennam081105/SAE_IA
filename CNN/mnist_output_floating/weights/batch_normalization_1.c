/**
  ******************************************************************************
  * @file    weights/batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const float batch_normalization_1_bias[16] = {-0x1.e5a6d80000000p-1, -0x1.763fda0000000p-1, -0x1.0c83b00000000p+0, -0x1.5a14ea0000000p-1, -0x1.df0ea40000000p-1, -0x1.b7355c0000000p-1, -0x1.f531b00000000p-1, -0x1.cb81a40000000p-1, -0x1.54b0d00000000p-1, -0x1.9e4e760000000p-1, -0x1.c37ce80000000p-1, -0x1.fef8c40000000p-1, -0x1.d66dac0000000p-1, -0x1.84e3c20000000p-1, -0x1.b3a1480000000p-1, -0x1.6b1e380000000p-1}
;
const float batch_normalization_1_kernel[16] = {0x1.de61cc0000000p-1, 0x1.2db02c0000000p+0, 0x1.6d48620000000p+0, 0x1.e7e42a0000000p-1, 0x1.1840700000000p+0, 0x1.6769e40000000p+0, 0x1.216b760000000p+0, 0x1.16a7440000000p+0, 0x1.2631400000000p+0, 0x1.355e9a0000000p+0, 0x1.8de3500000000p-1, 0x1.696cda0000000p+0, 0x1.930c880000000p-1, 0x1.8369fc0000000p-1, 0x1.0588260000000p+0, 0x1.3b788a0000000p+0}
;