/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv2d_7.c"
#include "weights/conv2d_7.c" // InputLayer is excluded
#include "conv2d_8.c"
#include "weights/conv2d_8.c" // InputLayer is excluded
#include "conv2d_9.c"
#include "weights/conv2d_9.c" // InputLayer is excluded
#include "flatten_3.c" // InputLayer is excluded
#include "dense_6.c"
#include "weights/dense_6.c" // InputLayer is excluded
#include "dense_7.c"
#include "weights/dense_7.c"
#endif


void cnn(
  const input_t input,
  dense_7_output_type dense_7_output) {
  
  // Output array allocation
  static union {
    conv2d_7_output_type conv2d_7_output;
    conv2d_9_output_type conv2d_9_output;
    flatten_3_output_type flatten_3_output;
  } activations1;

  static union {
    conv2d_8_output_type conv2d_8_output;
    dense_6_output_type dense_6_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_7( // Model input is passed as model parameter
    input,
    conv2d_7_kernel,
    conv2d_7_bias,
    activations1.conv2d_7_output
    );
  
  
  conv2d_8(
    activations1.conv2d_7_output,
    conv2d_8_kernel,
    conv2d_8_bias,
    activations2.conv2d_8_output
    );
  
  
  conv2d_9(
    activations2.conv2d_8_output,
    conv2d_9_kernel,
    conv2d_9_bias,
    activations1.conv2d_9_output
    );
  
  
  flatten_3(
    activations1.conv2d_9_output,
    activations1.flatten_3_output
    );
  
  
  dense_6(
    activations1.flatten_3_output,
    dense_6_kernel,
    dense_6_bias,
    activations2.dense_6_output
    );
  
  
  dense_7(
    activations2.dense_6_output,
    dense_7_kernel,
    dense_7_bias,// Last layer uses output passed as model parameter
    dense_7_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif