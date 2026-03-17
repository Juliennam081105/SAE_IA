/**
  ******************************************************************************
  * @file    batchnorm2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_2_H_
#define _BATCH_NORMALIZATION_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      100
#define INPUT_HEIGHT        
#define INPUT_WIDTH         1

typedef float batch_normalization_2_output_type[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];

#if 0
void batch_normalization_2(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_HEIGHT
#undef INPUT_WIDTH

#endif//_BATCH_NORMALIZATION_2_H_