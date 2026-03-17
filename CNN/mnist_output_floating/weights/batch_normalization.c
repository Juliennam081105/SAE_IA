/**
  ******************************************************************************
  * @file    weights/batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const float batch_normalization_bias[8] = {-0x1.fdb0d60000000p-1, -0x1.be6dd60000000p-2, -0x1.0c33000000000p-1, -0x1.5a79500000000p+0, -0x1.2594cc0000000p-1, -0x1.73d7240000000p-2, -0x1.b473f20000000p-2, -0x1.ef0e1c0000000p-1}
;
const float batch_normalization_kernel[8] = {0x1.a30d420000000p+3, 0x1.fa3a8a0000000p+4, 0x1.2899ae0000000p+4, 0x1.a5b38a0000000p+4, 0x1.3f3e340000000p+4, 0x1.ef53b60000000p+3, 0x1.0af9d20000000p+4, 0x1.6b5bbc0000000p+4}
;