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
#include "conv2d_3.c"
#include "weights/conv2d_3.c" // InputLayer is excluded
#include "conv2d_4.c"
#include "weights/conv2d_4.c" // InputLayer is excluded
#include "conv2d_5.c"
#include "weights/conv2d_5.c" // InputLayer is excluded
#include "flatten_1.c" // InputLayer is excluded
#include "dense_2.c"
#include "weights/dense_2.c" // InputLayer is excluded
#include "dense_3.c"
#include "weights/dense_3.c"
#endif


void cnn(
  const input_t input,
  dense_3_output_type dense_3_output) {
  
  // Output array allocation
  static union {
    conv2d_3_output_type conv2d_3_output;
    conv2d_5_output_type conv2d_5_output;
    flatten_1_output_type flatten_1_output;
  } activations1;

  static union {
    conv2d_4_output_type conv2d_4_output;
    dense_2_output_type dense_2_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d_3( // Model input is passed as model parameter
    input,
    conv2d_3_kernel,
    conv2d_3_bias,
    activations1.conv2d_3_output
    );
  
  
  conv2d_4(
    activations1.conv2d_3_output,
    conv2d_4_kernel,
    conv2d_4_bias,
    activations2.conv2d_4_output
    );
  
  
  conv2d_5(
    activations2.conv2d_4_output,
    conv2d_5_kernel,
    conv2d_5_bias,
    activations1.conv2d_5_output
    );
  
  
  flatten_1(
    activations1.conv2d_5_output,
    activations1.flatten_1_output
    );
  
  
  dense_2(
    activations1.flatten_1_output,
    dense_2_kernel,
    dense_2_bias,
    activations2.dense_2_output
    );
  
  
  dense_3(
    activations2.dense_2_output,
    dense_3_kernel,
    dense_3_bias,// Last layer uses output passed as model parameter
    dense_3_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif