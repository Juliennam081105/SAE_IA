/**
  ******************************************************************************
  * @file    batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_2.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      100
#define INPUT_HEIGHT        
#define INPUT_WIDTH         1
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void batch_normalization_2(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (size_t y = 0; y < INPUT_HEIGHT; y++) {
    for (size_t x = 0; x < INPUT_WIDTH; x++) {
      for (size_t z = 0; z < INPUT_CHANNELS; z++) {
        tmp = (LONG_NUMBER_T)input[y][x][z] * (LONG_NUMBER_T)kernel[z];

        // Scale for possible additional precision of bias
        tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        // Scale bias to match accumulator
        tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Activation function
#ifdef ACTIVATION_LINEAR
        // Linear (MEANS NONE)
        output[y][x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // ReLU
        if (tmp < 0) {
          output[y][x][z] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
          if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
            tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
          }
#endif
          output[y][x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_HEIGHT
#undef INPUT_WIDTH
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR