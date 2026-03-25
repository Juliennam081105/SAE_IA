#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_H_
#define _CONV2D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_HEIGHT        28
#define INPUT_WIDTH         28
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int16_t conv2d_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_HEIGHT        28
#define INPUT_WIDTH         28
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv2d(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     1
#define CONV_FILTERS       16
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int16_t conv2d_bias[CONV_FILTERS] = {0, 0, 0, -18, -5, 18, -1, -4, 102, 1, -3, -5, -4, -5, 37, -7}
;


const int16_t conv2d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-123}
, {-127}
, {-22}
}
, {{95}
, {135}
, {31}
}
, {{67}
, {202}
, {15}
}
}
, {{{-120}
, {-101}
, {-158}
}
, {{24}
, {174}
, {16}
}
, {{139}
, {173}
, {124}
}
}
, {{{-112}
, {51}
, {-161}
}
, {{3}
, {172}
, {58}
}
, {{71}
, {134}
, {79}
}
}
, {{{45}
, {-98}
, {65}
}
, {{-68}
, {33}
, {224}
}
, {{-16}
, {-9}
, {106}
}
}
, {{{23}
, {85}
, {-75}
}
, {{108}
, {88}
, {76}
}
, {{72}
, {-9}
, {33}
}
}
, {{{124}
, {178}
, {115}
}
, {{-17}
, {-13}
, {67}
}
, {{-154}
, {-200}
, {-225}
}
}
, {{{-100}
, {45}
, {-29}
}
, {{68}
, {126}
, {170}
}
, {{-118}
, {118}
, {80}
}
}
, {{{51}
, {65}
, {-13}
}
, {{92}
, {208}
, {85}
}
, {{-33}
, {-14}
, {-62}
}
}
, {{{145}
, {88}
, {-197}
}
, {{92}
, {-223}
, {-140}
}
, {{-291}
, {-205}
, {88}
}
}
, {{{-31}
, {-161}
, {-161}
}
, {{0}
, {-55}
, {-155}
}
, {{185}
, {185}
, {206}
}
}
, {{{-79}
, {47}
, {-75}
}
, {{48}
, {166}
, {64}
}
, {{-49}
, {107}
, {115}
}
}
, {{{19}
, {38}
, {154}
}
, {{66}
, {133}
, {133}
}
, {{-199}
, {-169}
, {-12}
}
}
, {{{-210}
, {-127}
, {43}
}
, {{-161}
, {121}
, {231}
}
, {{-70}
, {41}
, {163}
}
}
, {{{-57}
, {-55}
, {-2}
}
, {{108}
, {181}
, {127}
}
, {{-90}
, {-108}
, {-130}
}
}
, {{{17}
, {190}
, {194}
}
, {{-109}
, {39}
, {10}
}
, {{-320}
, {-226}
, {-323}
}
}
, {{{92}
, {43}
, {-135}
}
, {{157}
, {218}
, {-17}
}
, {{46}
, {28}
, {-200}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_1_H_
#define _CONV2D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_HEIGHT        13
#define INPUT_WIDTH         13
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int16_t conv2d_1_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d_1(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_1_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_HEIGHT        13
#define INPUT_WIDTH         13
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv2d_1(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     16
#define CONV_FILTERS       32
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int16_t conv2d_1_bias[CONV_FILTERS] = {42, 15, 8, 22, 32, 44, -17, 21, 30, -44, -6, -37, -7, 33, -2, -13, 12, 74, -25, 41, -12, 41, 36, 22, -35, 34, -20, 4, 1, 69, 26, 12}
;


const int16_t conv2d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-24, 36, -1, -77, -36, 30, -51, -16, -31, 122, -58, -60, -113, 10, 34, 31}
, {-46, -93, -59, 47, -47, 41, -4, 17, -43, -107, -10, 9, 51, -41, 50, -49}
, {32, 63, 78, 69, 69, -82, 78, 29, -96, 37, 79, 56, 91, -93, -42, -11}
}
, {{32, 14, 30, -108, 15, -32, -68, -24, -27, 143, 23, -33, -177, -95, 11, 105}
, {-123, -167, -161, -9, -101, -67, -46, -37, 8, -152, -99, 32, -33, -174, -50, -111}
, {-8, 43, 38, 89, 79, -147, 70, 47, -180, -210, 66, 106, 98, -125, -103, 39}
}
, {{-45, 47, -44, -45, -4, 145, -108, 58, 71, 86, 6, -80, -90, 51, 31, 19}
, {-76, -105, -95, -16, -102, -27, -62, -131, 16, -94, -55, -98, -69, -193, -50, -120}
, {-51, 20, 26, 100, 5, -228, 79, 21, -142, -86, 57, 110, 65, -81, -60, 7}
}
}
, {{{-198, -264, -172, -33, -236, 45, -92, -188, -47, -21, -134, -14, -39, -39, 116, -263}
, {-160, -173, -52, 8, 24, 113, 5, 58, 7, -19, -85, 119, -167, 65, 125, -45}
, {-112, -210, -133, -98, -33, 131, 10, -13, 31, -151, -115, 118, -69, 269, 134, 7}
}
, {{-12, 23, -55, -66, -100, -29, -51, -237, 51, 65, -77, -240, -42, -58, 47, -170}
, {-21, 20, -53, -65, -123, 81, -81, -280, 25, 83, -61, -308, -27, -241, 119, -228}
, {-48, 41, -37, -42, -88, 27, -86, -131, 52, 185, -45, -237, -38, -259, 146, -184}
}
, {{63, -47, 45, 31, 45, -113, 32, 44, 55, -17, 18, 16, 82, -10, -128, 19}
, {65, 23, 88, 68, 30, 25, -8, 5, -70, 44, 46, 51, 31, 96, -3, 61}
, {108, 69, 58, 29, 108, 6, 80, 58, -66, 124, 74, 30, 77, 0, 88, 60}
}
}
, {{{5, -69, -43, 29, -18, -47, -16, -45, -8, -28, -4, 55, -19, -17, 32, -29}
, {53, 16, 3, 56, -5, 27, 13, 26, -101, -11, 57, 65, 28, -19, 78, 21}
, {52, 73, 82, 74, 93, 124, 86, 26, -82, 30, 27, 141, 12, 42, 87, 71}
}
, {{-87, -117, -156, -85, -110, -33, -29, -49, -76, 80, -65, -21, -110, -1, 13, -108}
, {-170, -216, -209, -100, -10, 33, -120, 7, -4, -153, -90, -26, -185, 75, 38, -61}
, {-36, -42, -28, 68, -9, 102, 5, -46, 14, 8, 53, 111, 6, 2, 85, -138}
}
, {{-114, -42, -87, 5, -129, -52, -82, -141, 80, -8, -63, -96, 33, 7, 86, -120}
, {-2, 50, 30, 29, -49, -203, 40, -62, 173, 60, 12, -186, 94, -36, -113, -100}
, {55, 85, 24, 40, -10, -123, 55, -6, -13, -35, 25, 15, 60, 14, -158, 5}
}
}
, {{{15, -43, -6, 28, 32, 1, -22, 45, 58, -89, 42, 40, 4, -29, 50, -40}
, {26, 11, 53, 19, 67, -39, 16, 30, -65, 86, 34, -22, -39, -10, -32, 29}
, {36, -2, -68, -98, -54, -94, -84, -55, -35, 90, -36, -69, -133, -60, -105, -46}
}
, {{-4, -40, 22, 23, 1, -13, 33, -13, 83, -91, -47, 34, -16, -67, 36, -35}
, {60, 63, 85, 44, 54, -88, 93, 47, -181, -25, 40, 8, 26, -51, -15, 50}
, {85, 41, 14, 20, 63, -12, 2, 11, -176, 36, 25, -45, -66, 147, -40, 28}
}
, {{15, -21, -64, 4, -21, -12, -32, -28, 76, 2, -11, -12, -7, 97, -42, -21}
, {15, -22, -11, 59, 31, -50, 0, 88, -98, -156, -6, 96, 54, 19, -165, -16}
, {-12, -34, 33, 1, 10, 13, -14, 21, -70, -14, 13, 23, -80, -104, -28, 36}
}
}
, {{{3, 27, 28, -12, 10, 47, -2, 64, -19, 69, -18, -4, -12, 53, -61, 18}
, {6, 29, -1, -5, -71, -17, -16, -70, -36, 80, -39, -30, 70, 97, -26, 20}
, {21, -19, -14, -45, -58, -83, 13, -56, -14, 122, -54, -79, -25, 91, 8, -24}
}
, {{17, 10, 23, -12, -3, 1, 10, -19, -57, 65, 34, 17, 20, -36, -29, 11}
, {-11, 4, 11, 4, 14, -66, -37, -52, -87, 106, -10, -57, -7, 73, -131, -26}
, {34, 37, 14, -40, 18, -27, -10, 6, 19, 74, 10, -37, -25, 70, -8, -19}
}
, {{48, 25, 9, 34, 62, -45, 27, 6, 35, 36, 11, 8, 31, 123, -25, 12}
, {73, 42, 80, 72, 62, -79, 8, -31, -104, 76, 35, -1, 31, 37, -119, 41}
, {57, 48, -4, -16, 57, -5, 58, 29, -68, 32, 10, 1, -41, 103, -24, 50}
}
}
, {{{62, 59, 56, 54, 97, 120, 21, 94, 1, 6, 87, 112, 13, 97, 49, 129}
, {-16, 5, -27, 27, 76, 51, -55, 29, -13, -94, 46, 0, -43, 43, 31, 76}
, {-81, -108, -134, -84, 29, 56, -62, 29, 50, -146, -78, -30, -7, 49, 30, 45}
}
, {{43, 63, 44, -40, 21, 64, 16, 60, 45, -31, 43, 22, 44, 169, 53, 70}
, {-127, -125, -61, -169, 3, 64, -101, -31, 25, -139, -77, -20, -125, 6, 93, -10}
, {-142, -137, -168, -99, -125, -59, -92, -72, 20, -80, -101, -54, -31, -111, -4, -80}
}
, {{-97, -109, -35, -79, 26, 110, 2, 20, 85, -196, -17, 53, -118, 112, 97, 25}
, {-114, -85, -126, -176, -102, 33, -100, -155, 107, -114, -111, -141, -114, -60, 44, -97}
, {-105, -118, -100, -62, -118, -21, -32, -131, 23, -120, -49, -9, -13, -144, 84, -157}
}
}
, {{{-98, -19, -28, -22, -29, 35, -62, -2, 31, -121, -46, -8, -73, -69, 43, -37}
, {-33, 37, -13, 34, 55, -56, 42, -22, 8, -142, 49, -1, -19, -138, -7, -65}
, {41, -14, -12, 6, 33, -66, 34, 75, -55, -35, 35, 5, -3, -98, -177, 64}
}
, {{-97, 27, -6, 4, -49, -10, -3, -66, 174, 38, 1, -93, -2, -85, -4, -113}
, {35, 29, 35, 58, 60, -152, 71, 24, -53, -93, 46, -38, 62, -43, -128, 54}
, {48, 32, 13, 18, 45, 47, 55, 28, -114, -84, 39, 71, -60, 83, -22, 57}
}
, {{61, 15, 77, 26, -8, 12, 60, 31, 12, -20, 22, -60, 44, 53, -76, -15}
, {-16, -30, -20, -1, -15, 1, -1, 77, 13, -162, 15, 23, 46, 1, -96, 64}
, {-2, -78, -16, -38, 0, -16, -5, -48, -9, -64, -20, -23, -91, -10, 46, 0}
}
}
, {{{47, 22, 56, 50, -7, 25, -26, 3, 31, -72, -17, 27, -16, -44, 63, 34}
, {27, 36, 15, -18, 50, -66, 18, 71, -75, 13, 43, 6, -28, 1, -68, 40}
, {41, 4, 4, -24, -31, -82, 5, 43, 0, -47, 27, -65, -70, -3, -1, 5}
}
, {{-76, -20, -78, 7, 29, -32, -8, -9, 120, -46, 35, 14, -48, -7, -8, -10}
, {12, 66, 63, 19, 47, -71, 62, 5, -199, 33, 82, 59, -17, -61, -3, 6}
, {-6, 46, 102, 17, -25, -58, 15, 50, -199, 6, 48, 39, 22, 34, -28, 18}
}
, {{31, 10, -24, 35, -44, 6, -44, 19, 45, 56, 8, -20, 27, 7, 6, -68}
, {-35, 9, 6, -1, 24, -21, 28, 35, -194, -109, 29, 71, 41, 32, -21, 35}
, {-8, -35, 50, -6, 1, -50, 17, 4, -121, -54, -28, 41, -37, 64, -29, -13}
}
}
, {{{38, -21, -1, -39, -23, 36, 29, 30, 43, 10, 5, 80, 7, 93, -6, 22}
, {44, 74, 6, 101, 30, 62, 72, 62, -33, 106, 43, 83, 49, 64, 61, 53}
, {64, 99, 51, 117, 48, 147, 74, 52, -80, 44, 72, 127, 85, 115, 156, 55}
}
, {{-81, -125, -151, -132, -129, 47, -99, -33, -25, -65, -70, 27, -98, 63, 112, -46}
, {-133, -169, -112, -43, -29, 100, -66, -23, 46, -69, -111, 60, -47, 150, 132, -67}
, {-81, -61, -8, -6, 33, 119, -21, 67, 76, -167, -13, 75, -39, 191, 163, -9}
}
, {{19, 82, 51, -78, -22, -33, -49, -89, -75, 108, -6, -137, -67, -28, 63, 5}
, {-1, 49, -52, -71, -102, 8, -89, -226, 40, 100, -36, -219, -12, -167, 55, -164}
, {8, -38, -52, -119, -105, 37, -69, -115, 78, 64, -29, -178, -47, -124, 75, -81}
}
}
, {{{27, 35, -20, 8, -42, -104, -46, -60, -20, 28, -32, -81, -16, -116, -128, -95}
, {-81, -33, 9, -77, -33, -59, -57, -56, 56, 94, -23, -141, -89, -140, -25, -34}
, {-31, 25, 3, -20, -36, -65, -86, -125, -37, 174, -24, -245, -7, -37, -130, -128}
}
, {{13, 50, 22, -18, 4, -4, 29, -10, -55, 25, 20, 6, 19, 114, -3, 52}
, {54, 8, 46, 31, 51, 131, 28, 60, -82, 119, 53, 109, 22, 225, 118, 4}
, {90, 45, 127, 98, 16, 75, 42, 106, -135, 15, 93, 100, 62, 229, 141, 102}
}
, {{-41, -13, -81, 31, -15, 23, -13, -21, 20, -23, -74, 25, -85, 107, -31, -9}
, {-91, -101, -48, -91, -51, 85, -33, 16, 3, -195, -59, -7, -155, 18, 76, -12}
, {-85, -164, -109, -72, -28, 11, -102, -104, 109, 10, -94, -63, -95, -94, 23, -96}
}
}
, {{{-29, -2, 44, -4, 34, 19, 3, -2, -40, 14, 59, 3, 15, 16, -78, -14}
, {-18, 21, -8, -69, -62, 8, -45, -49, 42, 73, -67, -71, -71, 120, -61, 4}
, {-92, 2, -40, 6, -69, 114, -41, -28, -26, 25, -8, 14, -40, 184, 8, -27}
}
, {{16, -14, -27, 53, 4, 45, -31, 30, -68, 92, -1, 19, -19, 65, 96, -55}
, {32, 15, 85, 10, 22, 61, -3, 72, -239, 41, 69, 43, -42, 175, 20, 38}
, {10, -14, 33, -36, 33, 1, 15, -35, -193, 93, -9, 31, -49, 86, -28, 44}
}
, {{-51, 1, -28, -52, 2, 14, 12, 34, -63, -79, -38, 69, -53, 20, 74, 18}
, {1, 3, 16, 10, 62, 82, 35, 11, -319, -95, 38, 82, -29, 202, 129, 33}
, {-3, 38, 13, -27, -34, 49, 11, -6, -147, 113, 16, 59, -51, 104, 122, -25}
}
}
, {{{-54, -106, -38, -71, -16, -10, -51, -27, -55, 67, -60, -16, -9, 58, 54, 7}
, {-132, -190, -213, -62, -52, 55, -80, -66, 36, -128, -174, 11, -92, 190, 15, -30}
, {-133, -90, -89, -56, -71, 31, -109, -80, 63, -4, -97, -22, -57, -59, 15, -149}
}
, {{-106, -61, -94, -86, -124, 20, -74, -117, 98, -51, -113, -111, -1, -56, -33, -96}
, {35, 56, -8, -24, -60, -135, -20, -67, 76, 105, -10, -157, 37, 69, -174, -137}
, {81, 65, 54, 114, 38, -67, 57, 11, 13, 135, 54, -66, 48, 120, -187, -41}
}
, {{2, 10, 12, 62, -26, -68, 22, -50, -15, -59, -8, -61, 28, 32, -67, 8}
, {78, 78, 69, 43, 75, 23, 89, 93, -57, 10, 122, 60, 86, 116, 1, 54}
, {39, 30, 36, 3, 55, 91, -13, 47, 0, -44, 49, 50, 81, 142, 81, 46}
}
}
, {{{70, 46, 30, -9, -22, -24, 14, -14, -86, 36, -12, -58, -58, -13, -110, 56}
, {-6, -37, -52, -58, -79, 1, -88, -39, -40, 9, -38, -79, -39, 44, 46, -37}
, {-124, -139, -95, -5, -59, 39, -9, -38, -48, -187, -98, 120, -25, 103, 68, -27}
}
, {{19, 52, 66, 61, -7, 14, 32, 32, -253, 25, 58, 72, -37, 174, 23, 69}
, {66, 28, 70, 26, -3, -168, 32, -18, -175, 118, 29, -2, -88, 23, -66, 25}
, {-72, -7, -37, -91, -40, -86, -132, -114, -135, 52, -92, -251, -101, -131, -34, -56}
}
, {{-39, -18, -3, 46, -23, 99, -26, 54, -149, 51, 52, 44, -13, 121, 72, -2}
, {42, 13, 48, 79, 82, 76, 65, 54, -431, 96, 64, 188, 17, 125, 102, 2}
, {25, 25, 94, 41, 51, 1, 34, 67, -281, 99, 98, 66, -93, 41, 55, 75}
}
}
, {{{-21, -35, -14, -9, -56, -26, -11, -1, -18, 13, -80, 10, -76, -71, 46, -85}
, {-18, 30, -6, 63, -13, -110, 6, 8, 111, -112, 48, 48, 51, 17, -70, 12}
, {51, 33, 46, 48, 37, -163, 19, 51, -24, 22, 28, -3, 0, -25, -264, 92}
}
, {{-119, -23, -112, -104, -122, 7, -39, -54, 72, -70, -62, -10, -66, -146, -25, -67}
, {0, -8, -4, 84, -27, -183, 22, -2, 96, -97, 23, 30, 74, -301, -155, -60}
, {59, 7, 90, -36, 67, -107, 30, 88, -102, -76, 33, 24, 32, -77, -256, 97}
}
, {{-103, -67, -49, 0, -66, 20, -40, -119, 124, -32, 7, -29, -29, -155, 14, -156}
, {28, -2, 52, 49, -26, -213, 58, 6, -8, -116, 43, 44, 18, -46, -199, -37}
, {-39, -14, -17, -24, 39, 8, -22, 48, -236, -116, 29, 42, -28, 6, -147, 93}
}
}
, {{{-96, -75, -25, 2, -2, -9, 26, -18, 13, -42, -5, -8, -29, -68, 48, -8}
, {-10, 17, 61, -10, 6, 2, 54, 43, -7, 15, 45, 50, -18, -18, 2, 53}
, {31, -4, 57, 26, 26, 39, -18, -11, -69, 44, 11, 23, -16, 25, 84, 12}
}
, {{3, -24, -15, 36, -66, -83, -19, -70, 50, -29, 18, 26, 37, -62, -57, -138}
, {31, 71, 94, 20, 25, -52, 41, 57, -89, -25, 90, 10, 44, 33, -141, 83}
, {-27, -63, -35, -130, -11, -11, -96, 18, 18, -252, -6, -88, -177, -1, -58, 75}
}
, {{70, -1, 36, 39, -47, -1, 21, -43, 52, -48, -39, 50, 30, -50, -64, -50}
, {54, -25, 9, -7, 45, 90, 7, 16, 118, -295, 8, 31, 122, 163, -113, 37}
, {-190, -209, -126, -262, -48, 11, -272, -13, 46, -373, -169, -202, -298, -208, -199, 12}
}
}
, {{{-4, -12, -12, 13, 54, -66, -3, -9, 8, -22, -29, 5, -62, -142, -67, 30}
, {35, 28, 2, 17, -41, -200, -10, -25, -108, 52, 19, -109, -55, -123, -120, -59}
, {-61, -19, -8, -52, -36, -127, -64, -89, -30, 127, -53, -118, -37, 64, 32, -35}
}
, {{12, -28, 12, 25, -2, 101, 47, 54, -93, -47, 37, 103, 12, 136, 126, 22}
, {7, 2, 65, 70, 38, 114, 66, 105, -163, 116, 3, 139, 23, 221, 229, 62}
, {54, 55, 3, 61, 17, 54, 28, 44, -172, 117, 36, 90, 19, 170, 127, 21}
}
, {{-70, -38, -83, -116, -18, -41, -37, -81, 35, -71, -99, 15, -153, -40, 26, -23}
, {-110, -84, -151, -95, -31, 36, -9, -17, -84, -21, -74, 33, -44, 16, 123, -46}
, {-34, -54, -79, -1, -35, 88, 32, 31, -35, -11, -52, 47, -49, 106, 125, 8}
}
}
, {{{17, 5, -30, 12, -34, -21, -11, -8, 42, 6, 17, 13, -16, -77, 71, -11}
, {24, 57, 11, -19, -4, -64, 4, -17, 18, 42, 8, -14, 30, -63, 2, 11}
, {64, 56, -36, 15, 62, -134, 39, -64, -70, 63, -6, -138, 23, -71, -26, -36}
}
, {{12, -53, -3, 36, 10, 21, -10, 47, 1, -46, -22, -21, -61, 74, 25, 36}
, {42, 16, 42, 48, 27, 37, 50, 35, -163, 60, 30, 60, 63, 62, 111, -36}
, {9, 70, 69, 62, 69, 11, 63, 3, -219, 61, 40, 74, 41, 157, 22, 39}
}
, {{-59, -100, -52, -46, -64, 46, -24, -53, 31, -137, -68, 44, -57, 112, 19, -2}
, {-81, -179, -135, -88, -68, 52, -79, -50, -63, -184, -113, -24, -75, -21, 83, -11}
, {-12, -51, -3, 42, 26, 47, 8, -28, -84, -18, 5, 96, 8, 183, 145, -11}
}
}
, {{{20, 41, 80, -8, 71, 15, 21, 71, -16, 25, 26, 100, -74, -194, -77, 68}
, {-31, -15, -69, -47, -5, -91, -68, -65, -1, 26, -63, -96, -164, -100, -175, 50}
, {-118, -104, -79, -65, -70, -59, -141, -112, 32, -15, -149, -96, -91, -155, 33, -86}
}
, {{16, 49, 77, 78, 65, -15, 70, 23, -23, 19, 30, 98, 42, -86, -240, 94}
, {31, 33, 8, 28, 61, -155, -5, 62, -24, 7, 35, -116, -48, -115, -167, 75}
, {-131, -196, -195, -70, -86, -105, -162, -57, 123, -94, -147, -157, 23, -126, -12, -20}
}
, {{50, 28, 49, 34, 50, -126, 33, 15, 0, -22, 45, 36, 98, -95, -153, 74}
, {4, 41, 40, 17, 83, -65, -14, 82, 5, -63, 53, -88, -22, -143, -109, 100}
, {-114, -104, -166, -9, -113, -69, -84, -128, 76, -59, -95, -130, 38, -168, -81, -109}
}
}
, {{{-69, 7, -8, -63, 29, -42, -15, -55, 33, -31, 23, -37, -48, -55, 0, 10}
, {73, -2, 40, 10, 11, -111, -11, 35, -50, 7, -4, -56, -5, 61, -86, -14}
, {40, 54, 77, 30, -1, -10, 29, -8, -24, 71, 73, 11, 86, 199, 20, -70}
}
, {{64, 25, 5, 39, 63, 12, -27, 68, -42, 5, -7, 31, 30, 67, 21, 29}
, {64, 77, 20, 40, 18, 89, -2, 56, -99, -112, 31, 58, -25, 284, 54, 92}
, {7, -77, -8, -32, 14, 123, 7, 7, -103, -30, -6, 64, -131, 172, 125, 37}
}
, {{2, -109, -67, -70, -34, -42, 11, 28, 27, -217, 3, 70, -70, 88, -34, 17}
, {-86, -60, -27, -41, -2, 53, -67, -32, -120, 56, 6, 61, -51, 52, 119, -5}
, {-14, -20, 2, -20, -2, 16, -40, 5, 1, 43, -33, 39, -74, 88, 128, -8}
}
}
, {{{7, -45, 44, 5, 30, 19, 31, 29, -57, 54, -11, 7, -34, 26, 44, -11}
, {9, -14, 23, 54, -3, 12, 40, 58, 44, -54, 42, 8, 7, -26, -49, 3}
, {10, 54, 55, -61, 4, -83, -34, -18, -48, -9, -59, -6, -35, -94, -65, 42}
}
, {{6, -47, -29, -87, 33, 54, -80, -34, -4, -20, -68, 10, -81, -126, 24, 104}
, {-46, -18, -61, 16, 5, -13, 37, 28, 16, -46, -10, 90, 51, -131, -33, -59}
, {68, 15, 76, 62, 2, -214, 34, 1, -157, -8, 60, 67, 77, -138, -188, 66}
}
, {{-95, -114, -107, -144, -18, 64, -138, -87, -18, -51, -85, -45, -73, -122, 73, 43}
, {-112, -92, -113, 29, -21, -57, -9, -31, 46, -236, -76, 67, -42, -218, 17, -121}
, {2, 30, 47, 124, 91, -251, 106, 54, -267, -31, 55, 121, 92, -224, -275, -11}
}
}
, {{{28, 31, -22, -37, 47, -19, 5, 39, -64, 140, 0, -3, 5, 33, -108, 25}
, {-44, -52, -75, -65, -32, 64, -94, -52, -33, 119, -73, -49, -129, 7, 50, -77}
, {-100, -130, -115, -15, -97, -91, -61, -71, -85, -45, -67, -77, -26, 3, -85, -16}
}
, {{58, 22, 57, -4, 82, -1, 60, 57, -181, 47, 59, 40, -17, 70, 42, 32}
, {60, 56, 7, 26, -2, -65, 37, -30, -70, 154, 44, -80, 33, 166, -92, 3}
, {85, 61, 52, 45, 2, -125, 21, -27, -89, 117, 4, -68, 72, 80, -89, 26}
}
, {{25, 48, 26, 75, 28, 38, 0, -13, -118, 53, 50, -8, -8, 104, 13, -3}
, {19, 50, 31, 15, 46, 173, 3, 17, -154, 34, 38, 26, 6, 42, 25, -12}
, {18, 23, -19, 75, 15, 48, 4, 46, -40, 22, 7, 43, -44, 117, 55, 42}
}
}
, {{{57, 21, 3, 19, 75, 80, 39, 43, 106, -31, 21, 41, 81, -84, 21, 96}
, {10, -23, 2, 4, 28, 79, 72, 60, 68, -66, -10, 11, 74, -37, 56, 13}
, {92, 2, 38, 3, 45, 73, 79, 30, 44, -163, 14, 75, 33, 134, 54, 63}
}
, {{-68, -81, -90, -7, -15, 118, 7, -18, 123, -83, -58, 79, 13, 25, 113, -16}
, {-22, -121, -17, 48, 57, 12, -1, 53, 33, -63, -9, 57, -69, 194, 28, 67}
, {-161, -235, -203, -250, -66, 84, -192, -18, 207, -30, -182, -105, -267, 65, 39, 42}
}
, {{-81, -96, -151, -167, -63, 89, -165, -96, 16, 212, -194, -12, -173, -46, 240, -139}
, {-179, -82, -177, -113, -219, 46, -190, -178, 61, 90, -142, -129, -149, -183, 89, -145}
, {11, 48, 2, -13, -90, -47, -7, -148, 153, 68, -31, -271, -11, -169, -57, -218}
}
}
, {{{9, 0, 55, -23, 56, -31, 35, -11, 31, -69, -18, -61, 47, -104, -128, 52}
, {11, 81, 12, 26, 19, -184, 28, -9, -37, 2, -17, -49, 91, -29, -144, 36}
, {12, 11, 0, -116, -41, -47, -123, -51, 56, -15, -39, -149, -103, 2, -80, 24}
}
, {{61, 8, -28, 0, 23, -188, -7, 44, -26, -81, 30, -44, 48, -20, -181, -15}
, {-5, 12, 34, 17, 66, -206, 45, 50, -44, -95, 84, -10, 71, -63, -288, 18}
, {-43, 34, -42, -29, -40, -9, -89, 4, -102, 99, 29, -138, -64, 5, -57, 30}
}
, {{46, 14, 12, 16, 38, -2, 29, -16, 62, -20, 7, -77, -26, -124, -103, 47}
, {76, 66, 89, 62, -15, -101, 72, 13, -95, -42, 54, 26, 0, -37, -94, 44}
, {44, 15, 29, -44, 51, 8, -1, 23, -32, 57, -21, -23, -87, 50, -75, 9}
}
}
, {{{63, 35, 88, -9, 35, 61, 31, 17, -62, 72, 31, 63, -46, -82, 1, 52}
, {-155, -185, -113, -138, 14, 19, -68, -47, 47, -112, -92, -7, -138, -28, 11, 15}
, {22, -39, -32, -30, -26, -84, -38, -48, 89, 89, 41, 1, 53, -75, -56, -53}
}
, {{-135, -137, -124, -200, 15, 145, -217, 97, 80, -71, -171, -16, -176, 55, 30, 147}
, {-106, -31, -44, 32, -159, -118, -16, -148, 139, 48, -33, -138, 5, -215, -134, -235}
, {97, 90, 34, 95, 25, -178, 19, -15, -3, 20, 67, -11, 110, -40, -228, 38}
}
, {{-117, -128, -157, -74, -160, 68, -103, -115, 87, -91, -66, -218, 93, -215, -66, -160}
, {98, 34, 35, 15, -28, -139, 30, 33, 67, -60, 40, 49, 49, -23, -42, -40}
, {60, 15, 24, 38, 30, 93, 24, 78, -60, -76, 12, 73, 88, 31, -43, 30}
}
}
, {{{-34, -90, -20, -61, -26, 35, -24, -11, 1, -10, -22, 32, -10, 70, -1, 3}
, {-21, -62, -59, -2, -25, 45, -61, -64, -38, 6, -10, -46, -67, -17, 57, -40}
, {-17, 49, 5, 49, -35, -137, -6, -1, -5, 17, -20, -42, 29, -99, -108, -34}
}
, {{-27, 31, 63, 68, -16, -165, -26, -63, 44, 77, 38, -83, 47, 105, -52, -11}
, {43, 73, 70, 30, 14, -139, 38, 19, -37, 63, 46, -99, 39, 177, -138, 14}
, {71, 45, 60, 15, 20, 15, 80, 45, 18, 7, 37, 52, 88, 27, -19, 38}
}
, {{72, 19, 17, 8, -7, 122, 37, 52, -29, 19, 22, 4, 8, 78, 9, 18}
, {-7, -18, -40, 22, 38, 128, -4, 46, 75, -144, -44, 45, 24, 127, -34, 29}
, {-31, -1, 63, 28, -6, 3, 38, -14, -11, -123, -2, -18, 26, -67, -19, -29}
}
}
, {{{-30, -26, -62, 28, -19, -78, 22, -12, 133, -76, -1, 62, 0, -86, -46, -103}
, {31, 23, 87, 57, 57, -204, -25, 72, -7, 26, 75, 16, 83, -23, -233, 53}
, {2, -29, -17, -56, 59, -36, -45, -3, 4, -27, -51, -48, -176, 88, -107, 30}
}
, {{-56, 12, -104, 22, -60, -33, -8, -63, 52, -149, -47, 13, 80, -189, -85, -97}
, {60, 59, 30, 78, 83, -218, 100, 60, -210, -163, 76, 116, 34, 5, -289, 64}
, {-37, -14, 14, -118, 7, -79, -74, -9, -49, 42, -75, -155, -198, -151, -126, 18}
}
, {{-53, -73, -61, 17, -62, -136, -21, -30, 23, -241, -37, 60, -14, -59, -166, -116}
, {0, 67, 80, 21, 37, -143, 68, 18, -318, -145, 35, 31, 34, -52, -54, -9}
, {26, 31, -13, 2, -22, -82, -9, -2, -147, 88, -3, -122, -98, -54, -38, 29}
}
}
, {{{38, 20, 46, 51, 74, -89, 49, 9, -51, -7, -2, 10, 2, 5, -73, 38}
, {31, 21, 68, 81, 16, -19, 2, 40, 5, -70, 24, 71, 29, 12, -26, 1}
, {45, -19, 30, 44, 20, 16, 45, 58, -51, -86, 50, 64, 72, 67, -22, 49}
}
, {{83, 43, 59, 74, 28, 53, 5, 26, 29, -26, 29, -6, 50, 20, 25, 33}
, {13, -15, -25, -29, 54, 27, 29, 48, -42, -95, 66, 63, 24, 94, -23, 37}
, {-28, -28, -82, -15, 52, 75, -34, 74, -28, -177, -9, 83, -32, 104, 4, 46}
}
, {{-103, -140, -74, -44, -38, 111, -44, -16, 11, -212, -29, 20, -136, 111, 93, 3}
, {-177, -199, -82, -129, -80, 84, -126, -49, -7, -108, -100, -22, -228, -53, 146, -28}
, {-66, -100, -45, -100, -136, -57, -80, -88, 104, -61, -124, -42, -84, -201, 77, -160}
}
}
, {{{24, 41, -5, 16, -12, -15, 45, -44, -85, -27, 36, 50, -24, -46, -29, 4}
, {27, 23, 16, 27, 73, -50, 55, 24, -61, -1, -6, 45, -13, 87, 15, 55}
, {-11, -10, 39, 68, 39, -91, 69, 41, -84, -33, 51, 21, 35, 100, -14, 21}
}
, {{-94, -66, -1, -4, -20, 24, -4, -19, -60, -82, -56, 50, -64, 76, 64, -55}
, {55, 40, 6, 58, -25, -34, 44, -5, -87, 96, 34, 67, -2, 128, 94, 43}
, {35, 28, 74, 85, -11, 48, 71, 42, -161, 42, 52, 76, 34, 161, 64, 52}
}
, {{-101, -133, -132, -157, -122, 36, -128, -139, 13, 105, -178, -30, -151, -115, 47, -121}
, {-118, -132, -166, -59, -80, 141, -68, -67, -2, 3, -113, -14, -98, 30, 206, -100}
, {-79, -91, -40, 32, -31, 47, -14, -51, 20, -101, -14, -4, -40, 72, 88, -52}
}
}
, {{{34, 38, -48, -52, -8, -12, 30, 6, 54, 25, 22, -39, -10, -17, 41, -66}
, {78, 39, -2, 24, -32, -114, 48, -6, -41, 39, -2, -67, 56, -99, -217, -48}
, {-38, -77, 48, -27, 56, 60, 19, 44, -38, -81, 11, 37, -31, 12, -4, 66}
}
, {{9, 4, 60, 82, 8, -87, 36, -46, 144, -41, 37, -66, 25, -58, -179, 23}
, {125, 51, 46, 90, 58, -30, 67, 14, 64, -116, 7, -15, 162, 5, -243, 43}
, {-26, -72, -75, -121, -5, 92, -106, 2, 62, -145, -68, -13, -198, 43, -118, 34}
}
, {{55, 64, 20, 24, 64, -70, -12, 23, 55, -151, 2, -31, 74, -45, -231, 51}
, {-18, -25, 6, -64, 46, -22, 52, 11, 11, -170, 14, 3, -29, 88, -136, 42}
, {-24, -28, -72, -41, -75, 3, -102, -79, 41, 46, -18, -167, -55, -153, -65, -15}
}
}
, {{{-17, -21, -82, -117, -107, 26, -150, -175, 70, 145, -102, -107, -90, -182, 8, -77}
, {-90, -57, -81, -38, -126, -34, -131, -185, 29, -35, -127, -65, -93, -117, 143, -131}
, {-106, -56, -97, -41, -11, 8, 0, -9, 71, -225, -78, 10, 21, -24, 57, -55}
}
, {{50, 87, 45, -7, 52, 77, 63, 30, -52, 163, 14, 80, 7, 49, 4, 72}
, {58, 78, 20, -44, -48, -99, -11, -44, 15, 47, -27, -83, -73, -234, 5, -3}
, {-63, -122, -172, -78, -98, 24, -185, -195, 34, -89, -215, -52, -180, -210, 110, -55}
}
, {{53, 79, 35, 54, 84, 20, 15, -5, -60, 77, 45, 34, 51, -68, -13, 40}
, {71, 43, 41, 96, 105, -98, 69, 74, -158, 41, 111, 71, 10, -30, -20, 36}
, {27, -15, 19, -50, 19, -208, -84, -65, -49, 73, -2, -66, -84, -169, -47, 38}
}
}
, {{{-19, -33, -16, 7, -24, 34, -23, 19, 76, -154, -7, -11, 51, 48, -24, 23}
, {28, -61, 30, -4, 36, 23, 0, 33, 51, -173, 30, 65, 106, 66, 17, 31}
, {51, 49, 49, 91, 46, 82, 77, 46, 69, -82, 73, 76, 79, 116, 104, 67}
}
, {{-14, -21, -14, -115, 4, -20, -93, -20, -6, 99, -76, -156, -78, -124, -74, -25}
, {-192, -179, -162, -101, -113, 82, -210, -189, 17, 123, -265, -35, -162, -100, 211, -109}
, {-157, -195, -125, -98, -123, 80, -71, -77, 75, -162, -75, 24, -143, 37, 103, -166}
}
, {{37, 54, 68, 57, 91, -69, 78, 48, -92, 26, 39, 65, -17, 91, -90, 52}
, {50, 107, 73, 4, 36, -263, 62, 0, -135, 196, 54, -59, -17, 252, -247, 15}
, {29, 69, -6, -48, 37, -210, 18, -59, -6, 52, 49, -154, -27, 53, -239, -23}
}
}
, {{{14, 7, 43, 57, -2, -240, 84, -36, 66, 7, 63, 17, 87, -143, -208, -3}
, {24, -24, 30, 45, 42, -41, 11, 113, 7, -129, -2, 42, 28, 18, -146, 45}
, {-98, -73, -36, -44, 4, -3, -93, -12, 85, -103, 8, -27, -85, -88, -90, 44}
}
, {{67, 53, 74, 62, 37, -132, 35, 80, -118, -136, 52, 18, 56, 73, -210, 41}
, {-69, -67, -52, -81, 24, -59, -94, 2, 131, -211, -14, -63, -137, -117, -209, 55}
, {-40, -35, -28, 32, -49, -54, -57, -59, 12, 26, -73, -36, -32, 12, 32, -8}
}
, {{10, -24, 52, -28, 31, -19, -27, 72, 1, -245, 14, 29, -18, -2, -55, 95}
, {-8, -51, -77, -64, -12, 0, -46, 1, -66, -2, -31, -57, -108, -163, 70, -55}
, {29, 34, 14, 51, -3, 38, 6, 21, -45, 114, -13, 56, 77, 56, 117, -62}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV2D_2_H_
#define _CONV2D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_HEIGHT        6
#define INPUT_WIDTH         6
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef int16_t conv2d_2_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void conv2d_2(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_CONV2D_2_H_
/**
  ******************************************************************************
  * @file    conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv2d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_HEIGHT        6
#define INPUT_WIDTH         6
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE_Y  3
#define CONV_KERNEL_SIZE_X  3
#define CONV_STRIDE_Y       2
#define CONV_STRIDE_X       2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_TOP    0
#define ZEROPADDING_BOTTOM 0
#define ZEROPADDING_LEFT   0
#define ZEROPADDING_RIGHT  0

#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv2d_2(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS / CONV_GROUPS], // IN

  const NUMBER_T bias[CONV_FILTERS],						                // IN

  NUMBER_T output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]) {               // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, pos_y, z, k; 	// loop indexes for output volume
  unsigned short x, y;
  int input_x, input_y;
  LONG_NUMBER_T	kernel_mac;
  LONG_NUMBER_T tmp;
  static LONG_NUMBER_T	output_acc[CONV_OUTHEIGHT][CONV_OUTWIDTH];

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        output_acc[pos_y][pos_x] = 0;

        for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
          kernel_mac = 0; 
            
          for (y = 0; y < CONV_KERNEL_SIZE_Y; y++) {
            input_y = pos_y * CONV_STRIDE_Y - ZEROPADDING_TOP + y;

            for (x = 0; x < CONV_KERNEL_SIZE_X; x++) {
              input_x = pos_x * CONV_STRIDE_X - ZEROPADDING_LEFT + x;

              if (input_x < 0 || input_x >= INPUT_WIDTH || input_y < 0 || input_y >= INPUT_HEIGHT) // ZeroPadding2D
                tmp = 0;
              else
                tmp = (LONG_NUMBER_T)input[input_y][input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][y][x][z];
              kernel_mac = kernel_mac + tmp;
            }
          }

          output_acc[pos_y][pos_x] = output_acc[pos_y][pos_x] + kernel_mac;

        }
      }
    }

    for (pos_y = 0; pos_y < CONV_OUTHEIGHT; pos_y++) { 
      for (pos_x = 0; pos_x < CONV_OUTWIDTH; pos_x++) { 
        // Scale for possible additional precision of bias
        output_acc[pos_y][pos_x] = scale(NUMBER_T, output_acc[pos_y][pos_x],  WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

        // Scale bias to match accumulator
        output_acc[pos_y][pos_x] += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


#ifdef ACTIVATION_LINEAR
        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
        // Activation function: ReLU
        if (output_acc[pos_y][pos_x] < 0) {
          output[pos_y][pos_x][k] = 0;
        } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc[pos_y][pos_x] > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc[pos_y][pos_x] = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
          output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc[pos_y][pos_x], INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        }
#else
#error "Unsupported activation function"
#endif
      }
    }
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNELS];
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_WIDTH, //dim_im_in_x
                                      INPUT_HEIGHT, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE_X, //dim_kernel_x
                                      CONV_KERNEL_SIZE_Y, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      ZEROPADDING_TOP, //padding_y, top and bottom must be equal
                                      CONV_STRIDE_X, //stride_x
                                      CONV_STRIDE_Y, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTWIDTH, //dim_im_out_x
                                      CONV_OUTHEIGHT, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTHEIGHT * CONV_OUTWIDTH);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS     32
#define CONV_FILTERS       64
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const int16_t conv2d_2_bias[CONV_FILTERS] = {24, 9, 24, 20, -35, 53, 44, -5, -36, 33, 35, 10, 12, -47, 74, 40, -16, -46, -9, 30, 42, 19, 61, 12, 29, -8, -25, 12, -3, -32, 1, -23, -28, -28, 61, 14, 3, -16, 24, -21, 47, -14, -1, -15, 38, 7, -17, 55, 11, -22, 4, -5, 79, 55, 42, 19, 20, 6, 42, -1, 31, 51, -12, -14}
;


const int16_t conv2d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{53, 64, 10, 36, -5, 2, 40, -4, 24, -25, 1, 76, 30, 59, -36, -88, -48, -7, -118, 68, 6, -8, -96, 93, 62, -38, -46, -7, -88, -1, -76, -67}
, {38, -12, -10, 18, 26, 90, 91, 27, -89, 6, 27, -14, -13, 112, 37, 17, 61, 109, 38, 40, -36, -41, 37, 75, -27, 38, 12, 10, 76, 39, 40, 29}
, {-38, 28, -31, -123, -196, 5, -44, -88, -52, -110, -63, -136, -97, -118, 34, -56, -47, -5, 16, -76, -127, -21, -98, -23, -97, -41, -39, -24, 7, -140, -65, -9}
}
, {{51, 35, 56, 6, 18, -13, 3, 15, -39, -29, -19, -8, -66, 24, -41, -11, 42, 123, -10, 27, -10, -54, 82, 22, 37, 72, 10, 1, -14, -98, -150, -10}
, {-31, -58, 15, -30, -27, 67, -3, -54, -11, 19, -14, -157, -62, -125, 9, -71, -161, 42, 1, -154, 10, 36, -36, 57, -50, 63, 47, -43, 68, 35, -2, 98}
, {74, 31, 17, -30, -33, -94, -69, -47, -45, -217, -142, 31, -89, -5, -88, -162, 17, -131, -188, 67, -75, -13, -44, 56, 2, -117, -126, -17, -136, -108, -40, 17}
}
, {{21, -28, 21, -17, -17, 87, 16, 1, 56, -9, 5, -96, -35, 4, -12, -33, 28, -28, 46, 24, -62, 99, 33, 38, 91, 35, 43, 37, 45, -23, -70, 29}
, {-73, 82, -12, -17, -3, -26, -79, -34, 8, 16, -18, 28, 31, -68, -36, -8, 4, 9, -25, -7, 16, 68, -14, 39, -4, 1, -32, -30, -56, -53, 45, 8}
, {32, 37, 80, 5, -4, -166, 37, 36, 54, 67, 27, 56, -22, -29, 40, -17, -10, -144, -25, 18, 13, -13, 16, 56, 71, -49, -13, 43, -21, -44, -12, -97}
}
}
, {{{-39, 9, -24, -19, -53, 102, -64, -74, -4, 22, -58, 13, 4, -26, 70, 40, 23, -67, 2, -10, 33, 27, -81, -26, -5, -7, 22, -2, 57, -18, -1, 0}
, {-33, 7, 46, -76, -15, -16, 19, 52, 40, 2, -38, 22, 25, -23, -77, -5, -35, -70, -50, -128, 14, 59, -20, -49, 58, -98, -62, -43, -95, 1, 82, -82}
, {51, 30, 0, 70, 101, -70, -8, 5, -21, 44, 68, -6, 54, 19, -24, -22, -57, -13, -6, -88, 13, -15, 15, -35, -6, 55, -23, -12, -31, 4, 49, -47}
}
, {{2, 13, -55, 34, -10, -81, 61, 2, -42, -2, 61, 68, -41, 5, -48, -2, -14, -75, 70, 41, 4, -76, 3, 53, 38, -43, 29, 3, -19, -111, -79, -44}
, {-106, -18, -103, 50, 9, -19, 47, 43, -76, 31, 35, 32, -34, -19, -6, -15, 2, -207, 43, -55, 6, -97, -25, -7, 43, -35, 8, -13, -4, -107, -186, -114}
, {-20, -101, -129, -6, 20, 10, -34, -7, -47, 13, 54, -41, 49, -163, -57, 7, 15, 23, 35, -13, 7, -25, -7, -217, -9, 6, 14, -16, 50, 22, 18, 43}
}
, {{40, -19, 52, -16, 67, -15, 0, -2, 54, -10, 12, 40, 44, 29, -69, 37, -23, -112, -15, 96, 43, -26, -20, 31, 19, -4, -11, 67, -10, -33, 19, 36}
, {-63, -86, -36, 1, 41, -91, 56, 43, 20, -10, 37, -84, 23, -5, -41, -3, 31, -21, 48, -59, -8, -17, 43, -108, -9, 26, 11, -11, 11, -14, 43, 31}
, {-33, -8, 17, 8, -2, -14, -1, 14, 52, 57, 18, 30, -40, -114, 16, 35, -10, 4, 20, -55, -70, 25, -43, -124, -14, -55, 9, 25, -7, -24, -3, -28}
}
}
, {{{-21, -131, -94, -27, 23, 0, -6, 7, 1, 29, 57, 44, 9, -11, -40, 20, 62, -25, 17, 19, -6, -20, -67, 18, 3, -41, -28, -18, -41, -109, -27, 31}
, {54, -13, 17, -15, -20, 71, 17, -13, -64, 55, 18, -43, -32, 70, 37, -52, 40, -70, -1, 6, 1, -18, -7, 10, -14, -39, 44, -44, 33, -107, -54, 85}
, {-70, -83, -66, 22, 24, 51, 63, 14, -4, -43, -53, 11, -10, 6, 88, -15, -11, -6, -18, -28, 16, 54, 3, 70, 10, 42, -4, -84, 11, -50, 35, 34}
}
, {{42, -14, 33, -28, -8, -154, -38, -20, 24, -55, -8, -17, -15, 58, -46, 15, 43, -94, 17, 45, 55, -77, -60, -18, 4, -18, 9, 98, 9, 9, -17, -105}
, {-59, -74, -102, 11, 23, 11, 18, 10, -125, 52, 42, 48, -4, -25, -28, 0, 16, -75, 47, -64, 25, -162, -17, -1, 21, -20, 20, -49, 13, -75, -38, 42}
, {-51, -36, -57, 7, 59, 3, -85, 34, -37, -23, 43, -12, 54, -91, 28, 47, -5, 8, 13, -7, 1, -61, 26, -95, -19, 103, -29, -50, 53, 53, -25, 13}
}
, {{23, -106, 18, -3, -47, -38, 26, 19, 23, -38, 3, -57, -97, 62, 20, -55, 23, -3, 18, 59, -81, 7, -15, 60, 1, 57, -52, 28, 55, -140, -17, 24}
, {-1, -99, -33, 9, 21, -114, -31, -10, 33, -77, 20, -10, 66, -69, -103, 18, 0, -139, -41, -49, 56, 4, -25, -24, -7, -55, -17, -41, -115, -21, 32, -20}
, {26, 38, -1, 60, 64, -48, 32, 50, -34, -19, 30, -40, 16, 29, -13, 29, 4, -1, 58, -3, 2, -77, 10, -157, 34, 35, 43, 32, -14, 53, 27, -24}
}
}
, {{{41, -32, 14, -6, -35, 27, 11, -17, 40, -47, -31, -122, -32, -28, 90, -43, -21, 4, 80, -46, -79, 32, 25, -18, -28, 65, 47, -28, 85, -78, -32, 27}
, {30, -24, 22, 50, -42, 7, -19, 16, 29, 16, -4, 9, -29, 25, -40, -35, 12, -39, 1, 7, -2, 10, -33, -25, 8, -24, -26, -10, -9, -88, -56, -53}
, {51, 6, 52, 10, 19, -5, -2, 14, -33, 26, -13, 17, 17, 57, -2, 8, -27, 13, 17, -16, 11, -41, -27, 110, 39, 38, -23, -2, -2, -23, -31, 2}
}
, {{-90, 39, -53, 53, 26, -11, -19, 0, -11, -44, 8, 18, 72, 31, -17, 14, -28, 5, 15, 17, 26, 20, -5, 27, -13, -43, 15, -1, -16, 70, 33, -51}
, {-6, 0, 0, -172, -56, 28, -110, -117, 37, -15, 11, -20, 72, -110, -52, 75, 65, -31, -8, -93, -40, 19, -65, -11, -32, -105, -19, 56, -30, 178, 75, -12}
, {81, -46, 24, -8, -83, 62, -15, 11, -18, 35, 28, -59, 10, 4, 7, 10, 13, 12, -22, -15, -21, 37, -45, -45, -44, 43, 9, 28, -25, -50, 82, -40}
}
, {{-30, -22, -43, 34, -22, -74, 42, 59, 30, -19, -1, -1, -6, 11, -63, -17, 0, -16, -17, 24, 26, -38, 16, -56, 14, 47, -38, -24, 0, 34, 39, 5}
, {-48, 78, -32, 6, 81, -108, -30, 44, -81, 21, 71, 66, 76, -51, 18, 48, 27, 50, -7, 6, 93, -122, 74, 6, 73, -20, -91, 6, -5, 192, 104, -57}
, {-66, 57, -65, 35, 14, 2, -48, -1, -105, 5, 23, 0, 28, -35, -82, 3, 4, 48, -36, -56, 40, 0, -24, -23, 21, 8, -53, -24, -78, 104, 76, -37}
}
}
, {{{7, -39, -12, 20, -37, -68, 60, 49, -106, 2, -5, 29, -95, 63, -57, -46, -32, -31, -64, -7, -46, -42, -20, 80, 114, 7, -89, -78, 21, -155, -80, 31}
, {-40, -29, -60, -5, 9, -84, 68, 34, -118, -24, -3, 18, 10, 16, -42, 19, -18, -73, 27, -13, 0, -109, -7, 8, 48, 17, -38, -68, 56, -35, -72, 16}
, {-114, 26, -119, -75, -26, -1, -115, -43, -3, -48, -24, 45, 117, -103, -35, 10, -117, 8, 22, -128, 56, -35, 56, -99, -23, 11, -35, -70, 34, 29, 63, 44}
}
, {{23, 12, 9, 9, -51, -100, -25, 24, 57, -47, -36, 11, -47, 35, -65, 20, 7, -145, 12, 37, -18, -21, -27, 31, -38, -61, 14, 25, -60, 6, 60, -40}
, {-38, 48, -31, 38, 62, -47, 11, 2, 42, 13, -2, 76, -50, -41, -52, 49, 26, -153, 35, -79, 43, 61, -2, -22, 25, -17, 86, 42, -95, -72, -2, -78}
, {56, -40, 19, 59, 13, -61, 52, 53, -36, -22, 0, -53, 34, -12, -38, 27, 21, 35, -16, 53, 55, -110, 46, -164, -26, 13, -38, 24, 48, 46, -137, -20}
}
, {{-14, 33, 21, -25, 8, -90, 20, 4, 21, 63, 3, 92, -6, -46, -20, -21, -17, -76, 25, -126, 2, 18, 9, 40, 21, -66, 19, 29, -2, -27, -29, -15}
, {-30, 64, -56, -10, -41, 33, 7, 21, -34, 35, 31, 76, -75, -13, 10, 9, -42, -142, 79, -15, 25, 72, -63, 29, 75, -76, 27, 3, -9, -9, -9, -12}
, {40, 11, -47, 22, 4, -37, 62, 56, -78, -68, 14, -10, -47, 39, 23, -81, -33, -6, 56, -26, -22, -33, -13, 0, 7, 11, -33, -80, 38, -130, -65, -21}
}
}
, {{{80, -45, 50, 49, -10, -25, 43, 67, 12, 1, 50, -22, 9, 64, 31, 23, 27, -42, 10, 65, 37, -43, 32, -39, 36, 33, -20, 30, -38, 32, -28, -57}
, {-60, -17, 3, 39, 46, 47, 18, 65, -79, 67, 70, 17, -28, 35, 77, 2, -24, 36, 3, 20, 59, -15, 33, 101, 55, 21, 40, -58, 96, -31, -121, -1}
, {-58, -43, -118, 0, -103, 122, 13, 29, -11, 45, -28, -3, 27, -21, 95, -19, -40, 41, 46, -122, -8, 87, -4, 16, 75, 17, -1, -53, 103, 16, -109, 56}
}
, {{73, -28, 67, 6, -8, 34, 9, -14, 57, 13, -28, 16, -50, 36, 72, 32, 49, 33, 82, 16, 16, 61, 15, -2, 26, 19, 57, 57, 17, -21, -36, -20}
, {-51, -31, -39, 4, -79, 36, -1, -19, 11, -43, -6, -118, -52, -89, 83, -21, -10, 89, 31, 49, -91, 19, -43, -92, -47, 13, 46, 7, 25, -68, 64, -6}
, {-8, -52, -18, -23, -41, -13, -126, -47, 34, -110, -50, -57, -24, -62, -114, -96, -80, -34, -160, -37, -51, 30, -25, 13, -167, -53, -80, -22, -131, 49, -26, 37}
}
, {{67, 67, 104, -36, -8, 100, 14, -40, 50, -14, -40, 1, 42, -73, 25, -102, -24, 5, -80, -17, -10, 79, -26, 55, -11, -18, -23, -43, 8, 61, 25, -51}
, {-20, -28, -29, 40, 54, -21, 17, -21, -24, 5, -19, -40, 14, 36, -42, -100, -82, 31, -136, -39, 32, 16, 6, 40, -5, 15, -164, -145, -79, 1, 20, -24}
, {-1, -3, 33, 42, 23, -59, -5, 16, 8, 18, 27, 5, 14, 34, -4, -39, -49, 2, -54, -52, -1, -20, 33, -2, -17, -2, -53, -148, -39, 41, -19, -56}
}
}
, {{{-30, 9, -32, 14, 22, 50, 4, 31, 35, -60, 47, 18, -31, 1, 21, -60, -7, 51, -3, 103, -55, 9, 27, 32, 22, 34, 0, -34, 32, 32, 60, -24}
, {74, -72, 64, -63, -73, -2, 32, -24, 22, -109, -110, -143, -107, 97, 30, -50, 26, 79, -39, 19, -140, 6, -109, 42, -56, -27, -3, 10, -48, 44, 20, 104}
, {122, 20, 53, 15, -69, -44, -2, 19, 8, -99, -6, -80, -5, -17, 10, 10, 46, -32, -8, 89, -43, -57, -12, -85, -93, 60, -44, 72, 42, -65, -16, 6}
}
, {{9, -101, -81, 63, 10, 49, 15, 9, -102, -79, -23, -14, -46, 94, 52, -69, -145, 141, -80, -22, -20, -54, 16, 20, -37, 84, 8, -85, 28, -3, -61, 53}
, {-1, 37, 7, -46, 25, -2, -80, -63, -23, -138, -124, 7, -13, 33, 55, -149, -130, -67, -166, 18, 5, -6, -48, 55, -64, -77, -55, -115, -2, -61, 39, 59}
, {-55, 28, 51, -6, 18, -26, 31, 31, -111, 78, 25, 83, 1, -7, 59, -54, -73, -61, 33, -51, 16, -191, -33, 81, 57, -49, -9, -102, -17, -13, 22, 20}
}
, {{24, -50, -45, 48, 6, 6, 27, -49, -136, -7, -14, 33, 40, 35, -8, -26, 23, 68, 29, 77, 13, -60, 3, 36, 3, 37, -6, -88, 61, -45, -91, 49}
, {-29, -73, -52, 22, 72, -130, 16, 40, -148, 20, 44, 69, -55, 27, 100, -52, -52, -186, 17, 0, 30, -187, 49, 77, 22, -4, -78, -122, 82, -125, -46, -17}
, {-9, -147, 32, -39, 41, -22, 20, 37, 56, 83, -4, 12, 12, -1, -9, 18, 32, -15, 43, -17, -2, 17, 23, 10, -12, 5, -15, 25, 46, -70, 63, 4}
}
}
, {{{-53, 29, -60, 5, 38, 37, 5, -15, 59, -31, 12, 34, 10, -21, 76, -27, -56, 16, -4, -82, 0, -4, 32, -8, -39, 3, -12, -41, 56, 41, 1, -40}
, {-2, 6, 41, -27, -54, 49, -18, -41, -10, 41, -12, -32, 25, -55, 28, -8, 14, -14, 24, 20, -28, 28, -56, 20, -11, 61, -24, 2, 35, 59, 0, 5}
, {17, -2, 52, 14, 4, -61, 38, 65, 32, 64, -8, 42, -13, 41, -68, -46, 18, -46, 32, 40, 30, -36, -41, 58, -19, -43, -46, 6, -10, -49, -30, 18}
}
, {{-186, 9, -117, -15, 17, 21, -57, -14, -44, 66, 32, -88, 35, -94, -83, 3, -60, 59, -23, -160, 35, 41, 35, -131, -49, 38, 38, -70, 21, 66, 6, 21}
, {-43, -9, -38, -121, -64, -27, -65, -86, -3, -9, -80, -48, -41, -92, -69, -7, 8, -126, -38, -148, -47, 19, -92, -49, -74, -161, -22, -3, -182, 5, 34, -130}
, {-13, 21, 22, -60, -99, 34, -12, -44, 54, 58, -35, -1, -95, -29, 22, -5, 32, -34, 65, -86, -39, 68, -118, 52, -10, -124, 52, 69, -46, -35, 25, 2}
}
, {{-9, 67, 27, -7, -7, 45, -59, -29, 61, 59, -11, 53, 40, -61, 33, 33, 15, 25, 16, -39, 45, 60, 11, 6, 5, -1, 132, 35, 58, 74, 10, 19}
, {-93, 97, -49, 9, -34, -37, -45, -66, 72, 66, 5, 69, 43, -101, -56, 29, 37, -11, -29, -47, 39, 70, -24, -36, 31, -46, 48, 58, -62, 70, 22, -126}
, {-97, 86, 45, -29, -17, -29, -80, -46, 66, 85, 10, 7, 68, -101, -95, 52, -16, -69, 11, -65, 42, 103, -70, -12, 29, -121, -24, 9, -113, 28, 51, -139}
}
}
, {{{-12, -49, 3, 27, 14, -41, 2, 17, -12, -15, -68, 86, -5, 67, 5, -42, -36, -41, -73, 43, 29, -43, -64, 52, 65, -66, 3, -39, -6, 32, -10, -15}
, {-3, 14, -92, -3, -7, 49, 3, -34, -20, 59, -13, 44, -23, -10, 51, -8, -19, -69, 38, 12, -20, -26, 23, 53, 71, -4, 52, -69, 25, -19, 8, 68}
, {59, -44, 19, -6, 76, 18, -2, 19, 53, -17, -9, -73, 62, -17, 11, -22, -5, -36, -12, 83, 42, 30, 43, -18, 25, 44, -8, 1, 4, 37, -16, 18}
}
, {{-36, -38, -14, 54, 19, -90, 27, 62, 25, 38, 47, -31, 53, 22, -94, 4, 69, -53, 24, 9, -19, -18, 56, -15, 2, 79, -31, 23, -2, -55, -12, -15}
, {-57, -55, -2, 11, 65, -87, 40, 39, -4, -9, 56, -21, 21, 28, -71, 20, 1, -18, 0, 8, 41, -77, 48, -49, 26, 7, -16, 38, -12, -41, -43, -11}
, {-10, -29, -10, 9, 3, -9, 9, 17, -138, 50, 20, 39, -59, 43, 47, -31, -23, -54, 32, 5, -2, -110, 61, 53, 61, -17, 51, -58, 18, -134, -177, 27}
}
, {{-35, 28, -43, -26, -28, 30, 34, 48, 35, -75, -42, 31, -46, -2, -6, -6, -33, -85, 7, -14, 14, 36, -50, -43, 20, -34, 18, 26, -35, -8, -3, -70}
, {27, -75, -20, -22, -46, 9, 17, 12, 30, 5, -24, -39, -127, -16, -23, 3, 54, -78, 50, -19, -68, 53, -25, 24, 3, -5, 46, 66, 26, -69, -153, -6}
, {-61, -93, -99, 7, -10, -10, 54, 24, 21, -47, 21, -21, 9, -7, -41, 33, 21, -18, 76, 17, -55, 31, -62, -47, -28, -20, 75, 55, -2, -91, 19, 45}
}
}
, {{{-67, 19, 59, 70, -10, 20, -62, -29, 19, -19, 36, 71, 61, 3, -83, -18, 8, -63, -43, 50, -3, 40, -10, 31, -6, -48, -32, 0, -40, 26, 36, 8}
, {-40, 9, 0, 3, 13, -49, -1, -7, -35, 54, -28, 54, -121, -30, 49, -9, -99, -98, 64, -31, -7, -6, -86, -4, 4, -72, 52, -109, -3, -83, -31, -27}
, {-82, -130, -50, -45, -104, 34, 42, -43, 0, 57, 6, -80, -161, -104, 111, 39, -24, -39, 65, -66, -62, 75, -57, 77, 36, -25, 50, -3, 15, -175, -84, 57}
}
, {{3, -15, 1, 23, -69, -63, 62, 32, 28, -13, 4, -11, -59, 99, 6, 8, -1, -256, 4, 83, -14, -81, -43, 62, 21, -66, -22, 36, 73, -85, -102, -104}
, {-94, -6, -180, -8, 5, -36, -3, 3, -109, 18, 62, -45, 67, -23, -71, 64, 29, 10, 64, 88, 15, -188, 18, -80, -9, -7, -24, 39, 8, 99, 68, -3}
, {-189, -42, -214, -56, 15, -40, -139, -76, 58, -95, -26, -36, 3, -166, -118, -73, -24, 20, -115, -1, 45, 79, -79, -112, -24, -8, -137, 1, -80, 55, 52, -16}
}
, {{-10, 36, 41, -45, 0, -79, 2, 16, 43, 4, -32, 47, 48, -13, -49, -5, -29, 34, 23, -110, 19, 24, -51, 19, -102, 23, -56, -7, -34, -2, 62, 32}
, {-8, -7, 88, -8, 32, -68, 39, 28, 52, 5, -9, 15, 49, -43, 36, 34, 7, 8, -26, -48, 57, -53, -5, -105, -15, -60, -17, 24, 5, 46, 43, -62}
, {76, -54, 25, 74, 15, 9, -48, 29, 13, -5, 38, -44, 30, 0, 59, 0, -51, 13, -15, 65, 10, -28, -36, -67, -18, 14, 32, 0, 40, 73, -67, 76}
}
}
, {{{56, 2, 10, -119, -91, 27, -23, -61, -50, -82, -80, -7, -114, 15, 45, -67, -45, 35, -15, 42, -101, 20, -59, 72, -18, -85, 6, -70, 68, -33, -56, 104}
, {36, -47, 11, 20, -18, 81, 10, -20, -30, -7, -71, 11, -87, 50, 73, -104, -54, 47, 4, 40, -65, 22, 11, 89, -17, 39, -14, -23, 38, -34, -89, 87}
, {54, -100, -16, 90, -14, 9, 55, 52, -20, 67, 24, 0, -20, 100, 96, -20, -75, 83, 13, -52, 4, 22, 34, 101, -14, 60, -2, -54, 56, -64, -199, 48}
}
, {{19, 28, 3, -43, -6, 89, -52, -33, -58, -103, -109, 15, -28, 10, 29, -99, -60, 110, -62, 28, -12, -17, -32, 25, 17, -56, 30, -3, 11, 49, -56, 84}
, {-72, 76, -35, 12, 24, 79, 11, -2, 8, -30, -50, -7, -56, 28, 14, -41, -44, 56, 8, -60, -57, 15, -19, 66, -2, 68, 66, -13, 39, -71, -93, 47}
, {-66, -51, -54, -92, -71, 56, -44, -111, -23, -143, -113, -41, -83, -32, 12, -84, -185, 25, -35, 32, -104, 32, -73, -23, -72, 5, 23, -44, 44, 31, -80, 79}
}
, {{6, -86, -8, 22, 49, 76, -25, -32, -47, 63, 22, 0, -14, 31, 38, 28, 9, 81, 17, 119, -7, 6, 52, 83, 43, -6, 53, -38, 34, 29, -28, -9}
, {-24, 13, 4, 3, 39, 63, -20, 6, -65, -73, 15, 11, 65, 67, -18, -89, -87, 14, -45, 34, 3, 57, 6, 89, -55, 15, -106, -73, 16, 69, 70, 27}
, {-49, 68, -37, -22, 52, -75, -37, 32, -36, -27, -4, 55, 33, -39, -147, -34, -37, -12, -33, 17, 43, -19, 14, 5, -9, -55, -159, -105, -76, 38, 60, -91}
}
}
, {{{-74, -21, -103, 34, 54, 16, 38, 24, -27, -53, 33, -49, -40, 52, 19, -2, 28, -60, 20, -4, -69, -18, 28, 20, -39, 13, 6, -51, 54, 10, -25, 49}
, {20, -68, -45, -26, 31, -16, 30, -18, -52, -31, 7, -15, -15, 44, 30, 1, -28, 64, 10, 7, -13, -71, -8, 44, -4, 1, -11, -14, 83, -61, -35, 55}
, {-41, -33, -8, -69, -7, -30, 42, -27, 16, -33, -56, -34, -27, -10, -10, -19, -63, -2, -11, 29, -12, 34, -4, -79, -35, 54, -32, -81, 20, -9, 27, 4}
}
, {{52, -74, -5, 13, 19, -14, -9, 20, -16, 17, 31, -52, 122, 26, -49, -8, 47, 57, 8, -32, -2, -18, 57, -68, -5, 51, 41, 11, 4, 60, -33, 60}
, {35, -34, -27, 2, 2, -139, 41, 46, -5, 7, 23, -46, 84, -22, -75, 37, 48, -96, 25, 15, 5, -46, 16, -65, -32, 4, 37, 23, -42, 26, -19, -111}
, {2, 46, 17, 20, 64, -60, 33, 36, 94, 57, 62, 78, 54, -20, 20, 27, 59, -128, 62, -16, 96, 3, 16, -63, 60, -44, 80, 88, 23, -31, 16, -71}
}
, {{-65, -23, 19, -29, 22, 3, -41, -8, 36, -64, 29, -74, 27, -23, 9, -41, -93, -55, -2, -2, 8, 19, 14, -93, -54, 42, -2, -15, -31, -56, 6, -85}
, {18, 82, 43, -93, -17, 26, -132, -63, 41, -40, -86, 12, -31, -55, 19, -90, -153, 13, -116, -29, -43, -27, -64, -6, -131, -37, -13, -70, -89, 18, 62, 1}
, {-35, 17, 27, -22, 49, 45, -52, -21, 24, -18, -27, 85, -18, 2, -37, -141, -116, -71, -86, -15, 29, 31, -50, 11, -17, -98, -57, -17, -115, -28, 82, -114}
}
}
, {{{59, 9, 50, -23, -14, 39, 25, 39, 31, 0, 3, -84, 62, 1, 10, 67, -11, -29, 33, 37, -8, 41, -39, -65, 12, 29, 61, -24, 32, 7, 40, 107}
, {-22, 22, 111, -6, 29, -79, -103, -67, 27, -113, -16, 1, 55, -49, -65, -47, -134, -21, -85, -25, 89, 75, 0, -12, -9, -102, -84, -129, -111, 13, 104, 28}
, {-108, 24, -15, -28, 45, -40, 53, -27, -113, 22, 19, 37, 44, -81, 72, 43, 8, -59, -12, -121, 47, -142, 29, -20, 81, -43, -4, -60, 2, 22, 28, -63}
}
, {{37, -39, -39, 44, 15, -124, 24, 37, -37, 20, 20, -16, -48, 54, 8, -2, -8, -30, 31, 18, 2, -115, 25, 19, 5, 82, -45, -34, 4, -42, -45, -11}
, {-81, -122, -98, -26, -35, -87, 35, -29, 16, 40, 17, 52, -104, -46, 38, 30, 11, -89, 90, -32, -50, 35, -41, -10, 36, -40, 48, 31, 65, -178, -32, 40}
, {6, -100, 30, -15, -50, 46, -8, -14, -1, 13, -10, -120, -68, -29, 36, -8, 27, 2, -14, -17, -69, 3, -57, 9, -36, -56, 34, 13, 17, -117, 43, -1}
}
, {{41, 21, 44, -2, 5, -141, -17, -25, 71, 4, -27, -61, 2, -7, -12, 5, 47, -46, 54, 7, 49, 14, -49, -63, -24, -38, -2, 54, -49, -3, 36, -88}
, {-14, -113, 20, -36, -30, 10, -46, -54, 72, 124, -20, -70, -37, -72, -25, -35, 35, 41, -12, 8, 34, 50, -36, -37, 39, -65, 28, 6, -35, -45, -5, -37}
, {-11, 5, 37, 2, -16, 13, -46, -7, 26, 38, 40, 11, 42, -51, -35, 8, -29, 13, 33, -41, 3, 31, 1, 39, -23, 17, -11, -5, -24, 19, 3, -14}
}
}
, {{{5, -10, 6, -25, 32, -42, -52, 31, 70, 5, 16, 5, 44, -6, 5, 23, 33, 30, 22, 0, 69, 61, 12, -123, 42, -31, 46, 78, 17, 46, 19, 2}
, {2, 6, 26, 41, 1, -121, 40, 23, 0, 45, 23, 37, -3, 11, 59, 46, 19, -68, 19, 20, 25, -23, 6, 43, 17, 25, 23, 15, 21, -40, -55, -41}
, {69, -8, -42, 63, -7, 47, -4, 46, 0, 33, 69, 33, 105, 69, -2, 21, 8, 50, 14, -41, 63, 52, 42, -15, -17, 58, 32, -6, 26, 42, -18, 42}
}
, {{17, 16, 15, -29, -59, -36, -16, 4, 57, 88, 20, 25, -19, -17, 56, 34, 50, 4, 36, -8, -17, 15, -41, -60, -9, -46, 93, 61, -29, -8, 14, -67}
, {17, -98, -18, 11, -18, 25, -4, 20, -27, -74, -4, -6, -154, 45, 79, 13, 37, -22, 12, 5, -28, 44, -2, 48, -15, -12, 49, -6, 11, -106, -86, -27}
, {-6, -79, -34, -39, -89, 13, 30, -87, -51, -52, -48, -189, -92, 3, 58, -54, -82, 41, -18, -40, -127, 10, -1, 50, -122, 2, -13, -33, 79, -113, -105, 55}
}
, {{-24, -11, 26, -28, -127, 32, 13, -20, -20, -75, -115, 19, -226, 24, 49, -130, -105, -45, -77, 17, -94, 19, -27, 36, -54, -47, -12, -77, 70, -101, -134, -34}
, {-26, -116, 33, -98, -188, 68, -35, -73, 93, -58, -111, -102, -217, 16, -7, -25, -59, -19, 12, -15, -330, 67, -110, -15, -141, -45, 98, 40, -47, -153, -103, -5}
, {-28, -42, -84, -67, -100, 42, -83, -73, 19, -63, -76, -12, -102, 6, -126, -61, -33, -107, -104, -11, -103, 87, -76, 35, -74, -27, -53, -53, -63, -191, -93, -70}
}
}
, {{{-24, 15, 60, 20, -35, -30, -58, -91, 6, 38, -63, -8, -47, 20, -118, -13, 13, -129, -80, 62, -44, 19, -23, 69, -81, -61, -99, -52, -85, -79, -77, 5}
, {-24, 35, -123, 26, 4, 26, 44, -18, -90, 19, 5, 25, -16, -11, 24, 20, 38, -54, 50, -16, -18, -35, 4, -30, 30, -35, -28, -49, 49, -13, -75, -22}
, {62, -28, 37, -59, -52, 68, -4, -21, 68, -61, 25, 36, 25, 5, -8, -25, -19, -8, 83, 14, -73, 73, -25, 61, 4, 2, 70, -26, -12, 9, 32, 92}
}
, {{-8, 6, 122, -29, -45, -28, 3, -33, -2, 1, -76, -1, -112, 38, 29, -51, -15, -8, 15, -88, -60, 47, -56, 15, -50, -90, -2, 15, -31, 32, 87, -103}
, {84, -34, 72, 12, -30, -83, -28, 19, -20, -3, -27, -73, 42, 22, -43, 70, 26, -57, 26, 61, -29, -67, -25, 9, 11, -11, -31, -21, -67, -100, 21, 17}
, {-50, -62, -32, 22, 18, 20, -28, -22, -49, 10, 4, -28, 43, 80, 7, 16, 0, 41, 4, -50, 54, -109, 4, 6, 7, 26, -51, -65, 21, 29, -1, 30}
}
, {{5, 10, -4, 13, 27, -93, 11, -15, -47, 78, 30, 10, 31, 12, 43, -36, -32, -38, 42, 34, -6, -32, -14, 34, 41, 15, -84, -131, 58, 36, 31, 18}
, {50, -118, 58, 13, 98, -120, 36, 33, -93, -93, -35, -4, -78, -6, 41, -111, -37, -74, -64, -2, 15, -140, 25, 44, 28, -8, -18, -72, 11, -52, -53, -74}
, {-31, -161, -114, -37, 29, 20, -68, -2, -85, -42, 25, -98, 21, -60, -59, -10, -92, 5, 6, -8, -11, -7, -22, 48, -58, -26, -49, -102, 2, -61, 13, 24}
}
}
, {{{10, -31, 65, 38, 16, 52, 25, -24, -4, -22, 19, -20, -31, 31, 40, -3, -5, -47, 4, 11, 14, -24, 17, 47, 43, -39, -106, -67, 47, -62, -48, 27}
, {18, 16, -34, 20, 49, -37, 4, 27, -71, 57, 51, 63, 44, 4, -26, -27, -43, 1, -2, -32, 24, -21, 42, 11, 52, 23, -52, -48, 31, 8, -12, -43}
, {16, -1, -47, -4, -3, 33, 81, 35, -31, 110, 41, 48, -11, 20, -15, 60, 112, 5, 75, -46, -11, -64, -18, 78, 65, -22, 64, 95, 81, 14, 2, 37}
}
, {{15, -9, -62, 10, -15, -40, -28, 22, -14, -40, 32, 46, -9, -18, 6, 31, -3, -67, 16, 8, -33, -29, -6, 29, 0, 32, -9, -21, 31, -37, 54, 0}
, {-63, 13, -17, -99, -61, -36, -101, -77, 72, 21, 22, -52, 98, -134, -109, 53, 11, -75, 11, -52, -48, 27, -67, -157, -60, -107, 58, 52, -61, 8, 223, 80}
, {-8, -46, 47, -33, -82, -3, -56, -14, 35, 22, 0, -116, 6, -75, -65, -1, 37, 63, 16, -19, 19, -12, -91, -139, -83, -110, -41, 38, -66, 76, 63, -101}
}
, {{-5, -58, 39, 2, -5, -39, 5, 9, -8, 11, -17, -48, -41, -17, -15, 54, -2, 48, 6, -17, 19, -34, 21, -71, -43, 25, -71, -17, 13, 13, -19, -19}
, {-25, -2, -54, 31, -9, -67, -74, 9, -146, 31, 9, 10, -4, -10, 71, -14, 7, 30, -46, -41, 27, -120, 26, 57, 8, -19, -106, -85, 11, 46, 5, -60}
, {-17, 32, -27, 33, -3, 30, 52, 26, 19, 58, 31, 21, 55, -2, 43, 90, 56, 34, 7, -19, 23, 14, 39, 74, 59, -3, 45, 78, -6, 46, -30, 17}
}
}
, {{{-50, 10, 40, 30, 43, -4, -37, -13, 39, -87, 63, 8, 54, -1, -15, 35, -19, -41, -3, -26, -7, -15, 27, -10, -49, -22, -21, 40, -33, -21, 2, -49}
, {62, -42, 12, 18, 16, 21, -50, -10, -48, -62, -1, -11, 22, 2, 39, 13, 23, 105, -48, 21, -1, -46, 22, -7, 61, 19, -48, -50, -3, 21, -23, -22}
, {35, -5, 3, -6, 25, -4, -42, 10, -30, 24, -18, 17, 11, 30, -42, -80, -34, 47, -82, 1, 73, 0, 40, 54, 20, -21, -17, 30, 23, 51, 17, -17}
}
, {{-66, -6, -54, -6, -12, 14, -4, 9, -43, -72, -22, 5, -16, -22, -24, -18, -93, -18, -25, -59, 6, -23, 31, -21, 7, -11, 17, -46, 84, -19, -36, -19}
, {3, -103, -16, -19, -79, -82, 13, 24, -24, -77, -32, -48, -61, 45, -23, -17, 9, 58, -6, 21, -20, -70, -45, 37, 11, 57, -72, -13, 16, -8, -63, 13}
, {5, 45, 40, 18, 35, -78, -6, 19, 28, 12, -6, 59, 52, -9, -3, 35, 38, -30, 22, 39, 20, 24, 52, -47, -11, 51, -2, 39, -45, 15, 45, -10}
}
, {{-147, 3, -12, 15, -6, 110, -28, -2, 18, 13, 8, 20, -3, -70, 27, -15, -41, -47, 2, -144, 11, 36, -1, -93, -4, -25, -10, 16, 22, 50, -15, -23}
, {-14, -116, 11, -60, -169, 17, -19, -96, 10, -23, -90, -168, -69, -79, -88, 7, -10, 43, -15, 9, -109, -6, -134, -74, -79, -47, 3, 12, -103, -36, -54, 35}
, {156, -66, 58, 50, 6, -103, 17, 25, 67, 76, -23, -77, 36, 64, -8, 62, 111, -69, 11, 81, 17, -53, 6, -9, -34, -7, 7, 58, -50, -39, 9, -149}
}
}
, {{{-23, -15, -2, 2, -16, -14, 59, 7, -39, 43, -20, 46, -85, 22, 36, -55, 23, -24, 43, 27, 2, -1, -12, 4, 44, -65, -15, 14, 7, -64, -53, -14}
, {27, -90, 31, 18, -2, 89, 32, 40, -11, -46, 29, 15, -56, 38, 66, -63, -59, 67, 39, -10, -32, -10, 33, 49, -9, 25, -1, -47, 32, -58, -41, 79}
, {-69, -100, -19, -36, -62, 12, 5, -12, -51, -22, -36, -44, -87, -103, -17, -66, -80, 13, -46, -77, -30, 20, -6, -88, 3, -40, -24, -88, -16, -36, -121, -2}
}
, {{6, -9, -31, -11, 28, 19, -6, -7, -71, -33, -44, 50, -79, -15, -26, -62, -41, 7, -20, 58, -18, -33, -29, 36, 41, -22, 42, -47, 43, -14, -180, 48}
, {-62, -108, -132, 42, -15, 69, 54, 41, -149, -43, -21, 9, -92, 66, 59, -98, -115, -30, 4, -69, -41, -11, -11, 1, -31, 30, -4, -114, 56, -126, -129, 53}
, {-5, 42, -44, -103, 10, 41, -141, -109, -104, -185, -58, 103, -17, -36, -37, -105, -119, -35, -103, 41, 9, 29, -11, 28, -39, -57, 18, -94, -29, -33, -53, 61}
}
, {{10, -27, -52, 30, 13, 102, 0, 48, -96, -45, -25, -22, -30, 10, -24, -60, -35, 12, 1, 21, 21, -3, 9, 41, 48, 32, 52, -38, 38, -2, -45, 92}
, {-94, 11, -74, -1, 10, -68, 62, -10, -93, 78, 52, 41, 25, -16, -108, 37, 66, -65, 37, -55, 35, 14, -5, -28, 47, -14, 25, -18, -53, -9, 133, 28}
, {-15, 165, -82, 35, 83, -199, 52, 54, -97, 107, 92, 109, 43, 28, -63, 58, 79, -68, 82, 55, 65, -153, 0, -15, 85, -16, -2, 53, -22, 61, 139, -84}
}
}
, {{{-20, 38, -15, -94, 4, 7, -32, -45, 35, -56, -54, 30, -43, -15, -81, -43, -26, -49, -83, -2, -21, 18, -101, 18, -27, -100, 32, 16, -97, -37, 42, -4}
, {15, 37, 9, 36, -30, 46, 50, 24, 12, -18, -22, 54, -35, 15, 54, -86, -98, 84, -42, -11, -51, -38, -10, 58, -15, 66, 16, -47, 60, -9, -59, 11}
, {-55, 24, -70, -98, 4, -18, 2, -17, -94, -121, -12, 5, 8, 26, 4, -73, -118, 18, 42, 18, -24, 10, 0, -14, -35, 40, -3, -136, 27, 19, -13, 0}
}
, {{-1, 4, -21, 3, -19, 17, 35, 40, -74, 0, -14, 3, -34, 17, -38, -13, 16, -7, -24, 62, 23, -22, 0, -11, 17, 38, -42, -3, -40, 43, -109, -47}
, {49, 30, -1, 5, -57, -39, -3, 44, -20, 6, -1, -43, 10, 89, 3, 3, 42, -31, -18, 31, -36, 64, 36, 68, 22, 42, 0, 34, 0, 94, -47, 31}
, {-117, 98, 29, 6, 15, -39, 43, 30, 67, 77, 1, 67, -20, 72, 0, 24, 31, -83, 41, -60, -3, -34, 21, 65, 66, 6, 35, 10, 0, -22, 13, -30}
}
, {{24, 6, 26, 33, 29, 76, 25, -29, -7, -84, -27, -19, 73, 68, 31, -34, -21, 126, -51, -10, -92, 12, 7, 15, -8, 46, 21, 3, 75, 57, 9, 28}
, {-48, 94, -4, -52, -26, -18, -151, -115, 42, 17, -24, 15, 8, -155, -61, -13, -25, -10, -49, -107, -63, 21, -34, -59, -141, -58, 27, 62, -41, 71, 49, -3}
, {-92, 36, 44, -131, -16, 53, -180, -205, 38, -93, -198, -68, -18, -79, 12, -81, -144, -33, -126, -90, -89, 42, -110, -71, -205, -162, -42, -11, -184, -1, 96, -124}
}
}
, {{{25, -32, 16, -2, -36, -16, 73, 14, -9, -33, -43, 40, -52, 86, -50, -30, 23, -49, 1, -30, -48, 17, 15, 145, 18, -23, -38, -41, 35, -91, -63, 55}
, {-59, 6, -48, 38, 1, -72, 41, 34, -37, 25, 83, 9, 11, -67, -119, 55, 80, 21, 55, -4, 22, -105, 30, 29, 2, -11, 133, 117, 38, 18, 13, 129}
, {-95, 60, -159, -14, 33, -75, -26, 8, -10, -66, 21, 38, 73, -89, -84, 70, 58, 69, -15, 1, 55, -70, 21, -114, 0, -14, -26, 18, -15, 52, 43, -50}
}
, {{-7, 94, -4, 12, 21, -84, 26, 27, 30, 7, 21, -5, 37, 4, 8, -7, -6, -118, -35, -70, 39, 29, 46, -45, -19, -28, -1, -4, -43, -8, 20, -63}
, {-39, -52, 50, -17, -18, -48, -43, -20, -18, 74, 29, -96, -11, -62, -69, 46, 3, -136, -5, 43, -5, 10, -66, -82, -43, -18, -23, 0, -80, -6, 87, -40}
, {137, -178, 120, 9, 41, 12, 15, 20, -113, -64, -25, -107, -41, 98, 66, -16, 3, 32, -75, 71, 6, -235, -6, 18, -16, -22, -103, -17, 45, -29, -176, 7}
}
, {{-37, 47, -36, 15, 21, -88, 10, 16, -32, 13, -1, 26, -21, -5, -82, 20, -7, -104, 52, 77, -1, -3, -39, -1, 26, 12, 3, 4, -20, -90, -8, -36}
, {-58, -42, -78, -43, 11, 12, 9, 21, -89, 15, 22, 31, -28, -68, -2, 12, 49, -73, 14, -43, 15, 55, -41, 38, 57, -34, 65, 15, 44, -50, -61, -23}
, {-16, -306, -93, -54, -50, 73, 7, -30, -40, -43, -28, -15, -59, -18, -58, -11, 4, 9, 27, -37, -71, 30, -36, 67, 46, -58, 16, 0, 5, -92, -35, 45}
}
}
, {{{21, 7, 29, 38, -4, 14, -9, 30, 79, 36, 3, 11, 25, -5, 23, 48, -16, 44, -30, 19, -7, 34, 18, 20, 47, 7, -18, 5, 18, 47, 21, -1}
, {36, 15, -6, -5, -42, 30, 3, -38, 75, -133, -105, -54, -124, 6, 67, -68, -16, -78, -85, 23, -56, 50, -86, 28, 11, 24, -3, -2, -67, -175, -18, 31}
, {18, 7, -7, -32, -25, 31, -33, -54, -12, -64, 8, -1, -32, -30, 31, -101, -69, 20, -28, 39, -59, 6, 40, 20, -30, -3, -56, -81, 49, 34, -9, 64}
}
, {{-62, -3, -1, -17, 45, 3, -4, -22, 68, 18, -14, 16, 10, -17, 52, -14, -26, 38, 26, -103, 30, 2, 79, -37, -36, 66, -28, 61, 28, 98, 17, 68}
, {61, 7, 14, -75, -62, 29, 17, -25, -88, -19, -50, -36, -43, 42, 47, -42, 23, 87, -72, 51, -57, -50, -63, -6, -23, -35, -25, -4, -3, 1, -22, -22}
, {-14, 36, -44, 5, -7, 6, 12, 6, -210, 33, -10, 35, -57, 31, 107, -73, -56, -22, 5, -10, 9, -113, 21, 96, -26, 39, -37, -169, 69, -60, -73, 61}
}
, {{-75, 90, -96, 55, 34, 58, 0, 45, 6, 107, 29, 16, 42, -83, 45, 93, 47, 13, 60, -9, 59, 21, -11, -161, 49, 20, 51, 7, 44, 36, 68, 16}
, {27, -66, -4, -5, -55, 78, -9, -24, 51, -84, -10, -57, -79, 47, -85, -24, -11, -59, -47, 72, -66, -21, 22, 23, 12, 11, -43, 12, -61, -18, -181, -5}
, {43, -151, -113, 41, 9, -73, 18, 38, -197, -11, 29, -56, 31, 78, -42, 32, 29, 13, 30, 62, 8, -286, 84, -17, -17, 121, -89, -18, -8, -35, -48, 31}
}
}
, {{{-24, -23, -88, -39, 19, -32, -97, -73, 3, 12, 26, -37, 36, -136, -137, 35, 1, -44, -37, -21, 16, 4, 41, -32, -61, 47, -50, -3, -32, -12, 23, 50}
, {-15, -3, -32, -17, 39, -51, 82, 20, -50, -39, 37, 19, 8, -34, -5, -13, -15, 12, -21, -10, 7, -30, 46, -29, 77, 69, -94, -42, 9, 32, 0, -53}
, {28, -60, 55, 13, 2, -11, -32, 4, -107, -29, 54, -27, 6, 61, 4, -44, -60, 78, -33, -21, 55, -149, 40, 52, 8, -18, -88, -101, -5, 22, -37, 51}
}
, {{-34, 70, 75, -27, 21, -2, 15, 13, 38, 50, 12, -15, 1, -82, 19, 45, 15, -25, 15, -50, 37, 77, -36, -81, -24, -94, 38, 40, -71, 71, 97, -133}
, {3, 29, -26, 15, -11, 47, 8, 27, -84, -68, -32, 23, -73, 13, 88, -44, -29, 46, -8, 18, -14, -134, -17, 85, 11, 14, 9, -27, 84, -128, -230, 6}
, {-26, -174, -91, -40, -70, 64, 31, -60, -93, -41, 21, -144, -24, -43, 35, -52, -55, 58, -8, 21, -27, 0, 19, 53, -23, 92, -23, -88, 77, -88, -135, 97}
}
, {{-35, -11, -43, -26, -23, 93, 35, -19, -56, 60, -5, -8, -52, 15, 42, 16, 81, -65, 33, -18, 3, 67, -47, 15, 18, 4, 55, -16, 39, 54, -14, -49}
, {-2, -238, -47, -32, -34, 4, -2, -63, 35, -141, -1, -186, 55, 8, -85, -62, -33, -16, -77, -36, -89, -78, 14, -34, -93, 25, -27, 65, -9, -8, 25, 44}
, {46, 62, -91, -1, 74, -123, 1, 20, -61, -13, 20, 72, 65, 4, -85, 53, 65, 3, -23, 61, 57, -138, 59, -148, -81, 71, -99, 86, -108, 60, 102, -24}
}
}
, {{{4, -19, 18, -6, -119, 54, 24, 6, -9, -54, -5, -45, -200, -10, 44, -2, -14, 140, 6, 33, -176, 13, -15, 45, 38, 56, -6, -36, 95, -61, -32, 2}
, {33, 45, 76, 1, -27, 87, -51, -19, 54, 48, -59, 5, 7, 15, 41, -71, 22, 30, -20, 89, -27, 39, 39, 22, -127, 61, 2, 32, -13, 27, 0, 28}
, {57, 18, 33, -11, -17, -51, -3, 10, -13, -13, 1, -13, -13, 11, 17, -32, 10, -94, 5, -18, -36, -36, -30, -21, 2, -36, -60, -59, 6, -2, -11, -13}
}
, {{70, -20, -8, -141, -128, 71, -68, -84, -22, -178, -165, -82, -75, -13, -145, -85, 8, 207, -114, 59, -79, 7, -180, 7, -2, -51, -63, 17, -74, 111, -121, 93}
, {74, 24, 35, 22, 23, 67, 54, 65, -147, -48, 13, 11, 17, 14, 15, -142, -83, 65, -32, 75, 16, -44, 73, 43, 35, 24, -142, -38, 29, -71, -164, 87}
, {-70, 11, -55, -24, -2, 41, -25, -8, -101, -5, -6, 73, -12, 27, 39, -34, -11, 2, -7, -74, 8, -27, -31, 82, 8, -28, 38, -48, 27, -14, -65, 24}
}
, {{-4, 35, -38, 42, 12, 97, -16, 56, -21, -2, 12, 31, 16, 37, 9, -50, -16, 118, -66, 31, 52, 31, -47, -3, 62, -129, -63, -56, -14, -24, -88, -19}
, {59, -230, -42, 40, 34, 5, 46, 16, -47, -129, 71, -173, 17, 40, -75, -33, 51, 17, 0, 47, -48, -132, 61, -15, -5, 55, -72, 42, 24, -18, -108, 71}
, {-122, -88, -212, 39, 5, -24, -21, -10, -127, 18, 27, -57, 31, 10, -59, -11, 17, 43, -20, -20, 27, -91, 34, -101, -3, 65, -66, -19, 20, 14, 53, 35}
}
}
, {{{-16, -93, -34, 10, -43, -11, 76, 23, -23, 27, -1, -137, -40, 26, -27, -21, 8, -64, -31, -99, -89, 7, 16, 35, -26, 4, 23, -15, 65, -95, -54, 34}
, {-70, 22, -61, -97, -43, 110, -103, -98, -7, -13, -65, -19, 24, -162, -8, 7, -65, -20, -77, -40, 0, 49, -37, -73, -145, -88, -21, 20, -82, -40, 16, -28}
, {51, 17, 33, -151, -49, 37, -72, -94, 48, -31, -78, -42, -31, -81, -70, -51, -18, -88, -18, -61, -38, 57, -85, -91, -154, -56, 16, 94, -105, 41, 3, -91}
}
, {{-36, 44, 37, 29, 42, -1, -30, 3, 31, 23, 2, -20, 21, 6, 12, 9, 7, -156, 47, -32, -26, 65, 63, 4, -39, -2, 22, 11, -21, -15, 29, -84}
, {-48, 62, 82, -12, -8, -87, 3, -1, 81, 42, 11, 73, -6, -102, -111, 40, 30, -190, 13, -43, 12, 77, -7, -86, 38, -74, 16, 47, -84, 16, 13, -185}
, {-122, 71, -66, 27, 22, -50, -66, 13, -82, 65, 45, 26, 5, -121, -24, 6, 47, -5, 13, -66, 40, -105, -1, -149, 23, -20, -86, -11, 2, 56, 10, -50}
}
, {{-58, 40, -77, -24, 2, 2, 48, 12, 4, 8, -30, -29, 2, 60, 15, 12, 8, 9, -4, 8, -5, -9, -16, -23, 22, -15, 6, -7, 14, 12, 33, 16}
, {-17, 7, 33, 14, 22, -70, 47, 44, 2, 69, -13, 20, 37, 13, 0, 11, 32, -111, -32, -68, 9, -13, 33, 27, -4, -6, -11, -64, 33, -12, -9, -46}
, {-21, -148, -38, 11, -21, 27, 42, 14, -53, 36, -24, -27, -47, -9, 16, -32, -37, 15, 22, -16, -21, -29, -9, 52, 3, 27, -12, -19, 26, -91, -94, -8}
}
}
, {{{-52, 23, -33, -21, -19, -69, -3, 3, 0, -38, 52, -5, 57, -74, -47, 9, 8, 44, -43, -35, 45, -34, 25, -17, 9, -33, -116, -1, -57, 69, -54, -32}
, {-132, 21, -28, -37, 47, 101, -69, -36, 53, -55, 35, -33, 36, -25, -6, 37, -10, 26, -18, -168, -3, 84, -36, 4, -86, 7, -31, -11, 0, 17, 58, -7}
, {53, 44, 159, 4, -103, -22, -48, -14, 107, -63, -99, -113, -220, -47, -4, 29, 60, -106, -28, 33, -88, 47, -63, 18, -97, -63, -6, 103, -67, -53, 42, -67}
}
, {{94, -25, 30, -8, 18, -77, 0, 11, -43, -57, 21, 17, 54, 3, 6, -47, 43, 121, -29, 69, -12, -14, 15, 50, 11, 45, -9, -14, 19, 59, -85, -58}
, {74, 54, -51, 71, 32, 61, 56, 3, -209, -124, 23, -55, -2, 150, 74, -45, -72, 86, -23, 53, 31, -101, 19, 15, -77, 63, -22, -54, 17, 121, -11, -7}
, {68, -17, 17, -191, -98, -16, -83, -187, 35, -161, -171, -118, -46, -41, -167, -151, -43, -35, -99, -38, -35, 33, -96, 61, -139, -55, -65, -57, -54, -25, -24, 8}
}
, {{30, 25, 36, 15, 70, 9, 31, 52, -44, -35, 53, 49, -44, 14, 55, 26, -19, 20, 7, -6, 27, -21, -17, 22, 14, -21, 0, -13, 28, -28, -8, -24}
, {132, -140, -23, 41, 8, 101, 47, 67, -29, -43, 15, -131, 46, 132, 7, 13, -42, 116, -13, 48, 11, 37, 43, 0, -48, 71, -23, -10, 94, -43, -84, 78}
, {-28, 16, -92, -47, 30, -6, -133, -66, -12, -33, -21, -79, 23, 0, -68, -82, -73, -14, -118, -84, 10, -34, -36, -29, -38, 20, -99, -131, -74, -16, 8, 9}
}
}
, {{{-82, -25, -36, 103, 36, -66, 19, 36, -14, -68, 54, -8, 39, 2, 19, 3, 2, 65, 57, 8, 18, -18, 53, -75, -8, 36, -16, -62, 0, 32, -34, -3}
, {10, -16, 34, -17, 35, -18, -15, 14, 58, 64, 37, 13, 74, -39, -42, -13, 5, 30, 14, 3, 33, 6, 3, 20, 38, -8, 34, 11, -51, 10, -10, -27}
, {-31, 5, 57, -2, 22, 19, 16, 13, -6, 20, -16, -8, -10, 66, -10, 71, 6, -59, 32, 32, -4, 1, -32, -29, 29, 8, 36, 50, 12, 2, 30, -35}
}
, {{16, -138, -121, 31, -12, 70, 22, 16, -25, -107, 15, -116, -43, 46, -39, 20, 52, -4, -48, -2, -56, -10, 27, -8, -79, 112, -14, 15, 19, -77, -116, -4}
, {19, -26, 0, 21, 46, -92, -16, 22, 41, -6, 38, -24, 27, 25, -1, 15, 1, 44, 4, 77, 43, -11, -24, 49, 14, 52, 15, -14, -3, 52, 46, -86}
, {-67, 88, 37, 15, 21, -57, -38, 34, 45, 12, 31, 63, -2, 12, 40, 28, 10, -44, 42, 3, 28, 95, 20, -44, 55, -49, 67, 39, 17, 66, 49, -22}
}
, {{-65, 42, 3, -82, 3, -99, -52, -41, 14, -23, 24, -68, 98, -113, -3, -57, -111, 21, 22, -75, -17, -25, -14, -121, -109, -28, 15, 2, -98, 63, 40, 23}
, {19, 46, 74, -91, -2, -9, -43, -96, 27, 17, -113, 5, -4, -66, 17, 16, -47, 71, -47, -49, -66, -47, -78, -44, -128, -104, 1, 14, -60, 110, 95, -41}
, {3, 41, 32, -24, -44, -34, 31, -15, -24, -65, -85, 14, -100, 15, 80, -71, -79, -4, -60, -61, -82, -56, -23, 35, -115, -67, -55, -62, -16, -12, -5, -64}
}
}
, {{{-10, -4, 19, -21, -10, -31, 35, 16, -3, 24, -21, 5, -86, -27, 5, -26, 7, -69, 40, 59, -14, 17, 8, -8, 20, 1, 35, 29, -41, -16, -25, 8}
, {-5, -23, -31, 12, 10, 25, 19, -24, -77, -62, -67, 42, 6, -30, 34, -79, -32, -32, 7, 33, -32, -11, 68, 70, -7, 32, -43, -122, 25, -45, -123, 33}
, {-28, 60, -50, 1, 41, 48, 29, 0, -7, 45, 2, 13, 24, -28, 13, 26, 13, -47, 15, 3, -21, -40, 23, -103, -4, -20, -13, 39, 18, 21, 37, 16}
}
, {{36, 7, -29, -7, -35, -75, 20, 15, -90, -19, 5, -13, 32, 20, -169, 17, -4, -9, -34, 19, 17, -14, 96, -37, 21, 108, -25, -32, 10, 18, -57, -31}
, {14, -26, -72, -1, 63, 5, 11, 38, -16, -16, 12, 12, 50, 29, -86, 7, 12, 1, -18, -29, 1, -87, 23, -48, 57, 9, -44, 81, -3, 25, -20, 31}
, {-54, 69, 1, -36, 11, -67, -3, -12, 2, 52, 53, 81, 26, -100, -83, 53, 16, -64, 26, -46, 31, -21, -29, -99, 15, -103, -39, -13, -18, -15, -4, -29}
}
, {{17, 36, -12, -27, 43, 42, 27, -4, -5, -76, 17, -14, 0, 17, -9, 14, -29, 16, 3, -16, 48, 37, -9, -6, 35, 38, 49, -37, -37, 41, -10, -13}
, {28, -18, 5, 36, -18, -13, -31, 15, 12, -3, 84, -88, 89, -1, 1, 79, 66, -51, 35, 27, -3, 34, -9, -41, -16, 21, 24, 77, -13, 3, -12, 30}
, {-121, -21, 22, 43, 4, -4, 26, 21, 62, 59, 10, -40, 71, -35, 13, 3, 15, -79, 51, -49, 15, 10, 29, -122, 35, -36, 71, 87, -31, -54, 35, -66}
}
}
, {{{50, -74, -98, 59, 29, -6, 20, 20, -59, -8, 34, -62, 74, 33, 45, 14, 8, 49, 52, -19, 9, -61, 50, -111, -36, 79, -21, 36, 6, 27, 13, 25}
, {71, -43, 20, -39, 21, -27, -46, -20, -23, -99, -49, -4, 11, 6, 1, -68, -41, 0, -94, 52, 39, 55, 5, 18, 1, 0, -3, 23, -66, -13, 4, 26}
, {139, -17, 122, -4, 9, -44, 66, 4, -59, 84, -43, 85, -45, 105, 67, -35, 10, -50, -13, 77, 31, -41, -6, 153, 48, -16, 32, -6, 58, -17, -30, 34}
}
, {{12, 73, 62, 1, 52, 17, -24, -26, -13, 32, -16, -5, -18, -43, -20, 32, 14, -53, 34, 15, -32, 58, 1, -4, -8, -24, 40, 1, 5, -8, -5, -50}
, {-38, 69, -59, -42, 3, -35, 26, 8, -69, 32, 24, 110, -70, 22, 61, 24, -10, -164, 19, 1, 33, -49, -7, 34, 87, -79, 57, -36, 58, -150, -162, 39}
, {-186, -236, -89, -27, -87, 52, 53, -8, 70, 55, 21, 64, -115, 23, -7, 60, -8, -73, 47, -125, -38, 61, -114, 13, 47, -86, 46, -11, 46, -155, -7, 10}
}
, {{12, -14, -32, 17, 35, -34, -4, -10, -14, -58, 12, 35, 6, 21, 16, -57, -12, -12, -27, 38, 41, -65, 46, -1, 19, -9, -52, -49, 7, -9, 2, 104}
, {-12, -66, -62, -36, 9, 9, 0, -29, -4, 89, 12, -45, 68, 0, -14, 51, 51, 8, -19, 2, -3, -53, 3, 15, -5, 2, -26, 33, 0, 52, 30, 42}
, {-185, -2, -66, 10, -42, 14, -170, -93, 29, 27, 41, -81, 54, -200, -79, 21, 18, -1, -55, -107, 38, 22, -25, -139, -80, -15, -51, 20, -107, 68, 36, -61}
}
}
, {{{-32, 15, -5, 30, -12, 10, -39, -23, -102, 21, -18, 27, 11, -30, 50, -24, -29, 60, -20, 1, -43, -64, 7, -4, -70, 27, -17, -92, -17, -21, -5, -25}
, {-31, -17, -3, -26, -9, -66, -85, -65, 17, 28, -23, 2, 44, 19, -89, -38, 1, -9, 15, -34, -36, 17, -24, 24, -29, -21, 23, -12, -39, 15, 36, -19}
, {78, 20, 87, 1, 7, -56, 14, -15, 35, 45, 9, 35, -18, 38, -23, 3, 56, -42, 2, 12, 47, -61, -29, -21, 63, -64, -35, 35, 20, 22, 67, -62}
}
, {{39, -72, -56, 13, -17, -20, 15, 12, -102, -85, -27, -75, -65, 19, 20, -119, -35, 57, -52, -42, -28, -93, 50, -36, -44, 62, -96, -98, 4, -41, -50, 40}
, {-23, 18, -81, -34, 56, -42, 14, -31, -67, 24, -32, 88, -12, 17, 42, -29, -56, -72, 55, -28, 12, -64, -7, 83, 18, -63, 29, -31, 16, -65, -31, -22}
, {-43, 1, -36, -61, -55, 32, 41, -30, 37, 37, 18, 59, -35, -129, -44, 21, 30, -139, 28, -67, 15, 46, -34, 23, 73, -71, 15, 9, -12, -99, 27, -14}
}
, {{-20, -27, -33, -1, 50, -132, 37, 36, -64, -19, 66, 42, 45, 34, 1, 20, 21, 15, 51, -35, 57, -71, 24, 63, 42, 9, -2, 9, 16, 55, -19, -12}
, {-3, -64, -79, -34, 22, 29, 14, -29, 17, 21, 22, -121, 71, -26, -6, 18, 33, 56, 41, -62, -48, -64, -30, -149, -42, 43, 12, 7, 8, -33, 30, 27}
, {-127, 8, -27, -57, -7, 57, -120, -27, 100, 88, 56, 49, -16, -210, -25, 54, -18, -47, -17, -212, 42, 54, -70, -38, 30, -64, 36, 70, -60, 0, 62, -107}
}
}
, {{{25, 15, 29, -33, -65, -90, -16, 13, 43, 24, -38, -31, 11, 10, -92, -17, 10, -69, 46, 7, -23, -51, -80, 12, -3, -43, -5, 66, -39, -77, 26, -14}
, {-76, 14, 37, 51, 20, -111, 48, 49, -36, 20, 40, 84, -20, 25, 64, 94, 26, -62, 44, 7, 37, -37, 27, 17, 47, 29, 44, -30, 48, -43, -29, -31}
, {-13, -17, -145, 45, 37, 41, 18, 46, -77, 16, 42, 13, 31, -11, 18, -14, 45, 42, 57, -112, 50, -25, 53, -33, 41, 9, 10, -69, -23, -25, -27, 70}
}
, {{-4, 54, 7, -23, -131, -117, -9, 40, 3, 8, 14, -10, -86, 16, 12, 47, 80, -80, 31, -5, -31, -32, -49, 0, -33, -56, -24, 41, -33, -21, -70, -121}
, {-1, -38, 8, -10, -34, -14, 26, 9, 2, 48, -36, 22, -105, 49, 35, -8, -8, 18, 26, 28, -33, 70, -5, 22, 4, -11, 71, 61, 29, -97, -11, 26}
, {25, -13, 38, 12, -71, -1, -43, -11, 3, 3, -28, -100, -13, 32, 24, 9, 38, 47, -39, 65, -21, -53, -47, -9, -102, 25, 44, 35, 17, 46, -26, 11}
}
, {{5, 55, 8, 16, 27, 21, 16, -29, 64, 17, -31, 116, 46, -2, 37, -51, -150, -54, -74, -53, 5, 31, -13, 20, 28, -26, 49, -75, 25, 49, 19, -23}
, {-50, 22, 32, -133, -109, 26, -74, -73, 32, -51, -21, -3, -74, -17, -38, -28, -84, -25, -25, -54, -42, 75, -117, 39, 27, -143, 43, -12, -18, 23, 12, 3}
, {7, -18, -30, -37, -48, -4, -37, -63, -19, -69, -44, -20, -93, -18, 96, -148, -77, -35, -10, -19, -102, 6, -21, 43, -81, -14, -51, -80, 26, -55, -75, -11}
}
}
, {{{-107, -12, -71, -40, -37, 23, -5, -12, 8, -18, -26, -75, -79, -62, 1, 28, 29, -84, 17, -108, -100, 16, -1, -134, -84, -8, 31, 48, -16, -50, 1, 80}
, {38, -62, -12, -50, -56, 46, -56, -58, -28, 11, -56, -73, 9, -40, -64, -38, 14, -45, 4, -9, -76, 14, -31, -15, -106, -19, -30, 15, -44, 2, -53, -64}
, {-141, 38, 18, -112, -10, -25, -74, -96, 23, 31, -26, -50, 32, -131, 6, 8, -41, -4, 53, -156, -47, -18, 4, -114, -105, -4, -9, 28, -23, -12, -33, -29}
}
, {{20, 0, 59, 75, 33, -63, 53, -10, -18, 117, 47, 10, -56, -2, -51, 54, -4, -93, -6, 45, 33, -42, 26, 57, 24, -58, 52, 12, -11, -32, -118, 22}
, {-51, 26, -134, 24, 13, -108, -14, 42, -87, 16, 43, 22, -19, -7, -31, 1, 19, 14, 59, -1, 46, -112, 17, -22, 49, 8, -41, -48, 6, 34, -82, -125}
, {-103, 31, -45, 12, -10, -32, -83, -16, 34, 18, 20, 23, 78, -166, -16, 31, 30, 38, -13, -82, 51, -55, -1, -165, -16, 0, -57, -56, 0, 73, -30, -19}
}
, {{-32, -9, -87, 34, 48, -52, 54, 26, -5, -25, 29, 36, 37, -5, -41, -2, 4, 68, 19, -11, 12, -35, 20, 23, 17, -20, 24, -22, -10, -48, 13, 4}
, {3, -10, -60, 15, 35, -54, 34, 20, -50, -40, 65, -19, -4, -17, -28, 50, 5, -59, 14, 14, 55, -22, 66, -19, 57, 27, -28, 35, -22, -48, 0, 27}
, {-27, 37, 20, -25, 23, -40, -21, 11, 22, -9, 23, 19, 18, -66, -33, 26, 9, -33, 39, -42, -10, -28, 32, -57, 10, -57, 2, 13, 47, 43, 44, 46}
}
}
, {{{-126, 14, -23, 23, -66, 86, 7, -3, 3, -26, -3, -9, -24, -49, 58, -35, -68, 49, -11, -133, -102, 31, -21, -94, -34, 22, 8, -42, 12, -92, 29, 0}
, {18, 52, 17, -10, -40, -82, 25, -34, -14, -72, -186, -12, -54, 8, -27, -124, -68, -154, -91, 37, -109, 21, -67, -9, -115, -51, 28, -2, -61, -51, -9, -1}
, {-80, 15, 23, -100, -7, 79, 51, -35, -7, -68, -92, -81, -55, -85, 54, -19, -59, -21, -5, -101, -154, 12, -23, -27, -72, -7, 56, 5, 57, -22, -58, 28}
}
, {{-62, 42, 12, -3, -35, 11, -2, -14, -52, -14, -16, 79, 21, 7, -94, -47, -78, -124, -50, 42, 16, 43, -33, 81, 36, -83, -82, -97, -80, 22, 42, 64}
, {-29, 1, -69, 15, 76, -133, 49, 13, -132, -92, -20, 25, 34, 60, -59, -74, -47, -60, -45, -33, 36, -96, 48, 45, -3, 21, -79, -128, 0, 2, 73, -27}
, {-152, 58, -37, -42, 3, -45, 13, -20, -73, 24, 16, -13, 67, -132, -94, 23, 34, 80, -24, -42, 2, -69, 57, -122, -27, -36, -46, 17, 14, 44, 111, -21}
}
, {{30, -92, 28, 8, 22, -100, 23, -3, -12, -23, 3, 29, -124, 57, 33, -11, 2, -67, -3, 13, -48, -64, -12, 37, 32, 6, -76, -6, 17, -139, -32, -24}
, {-20, -47, -91, -15, -54, 36, -13, -18, 14, 71, -6, 26, -12, 1, 28, 2, 12, -76, 42, -64, -54, 23, -13, -45, 28, -1, 59, -24, -3, -164, -1, 30}
, {-65, -47, 7, -81, -77, 91, -42, -84, 45, 50, -64, -6, -81, -89, 40, 18, -1, -3, 11, -15, -52, 19, -84, 1, -14, -83, 3, 7, -24, -119, -4, 40}
}
}
, {{{-36, 42, -36, -21, 7, -122, -7, -5, 24, 26, -1, 54, -24, 20, -32, 17, 8, -81, -37, -60, 25, -35, 17, -24, 28, -38, -17, 0, -41, 115, 14, -89}
, {33, -4, -63, -26, -6, -105, -10, -24, -71, -54, -62, -42, -2, 15, -39, -56, -18, -25, -62, 32, 1, -125, 4, -39, -74, 57, -54, -10, -23, -5, -52, 35}
, {-81, -66, -181, -87, 16, 3, 50, -74, -63, -136, -66, -95, 9, -45, -18, -75, -112, -17, -92, -14, 5, -76, 54, -108, -122, 27, -95, -90, -36, -7, 12, 24}
}
, {{-45, -9, -31, 13, 50, -23, 12, -12, 8, 22, -8, -19, 14, 22, -50, 20, -6, -54, 48, 39, 30, -62, 2, -2, 63, -26, -6, 8, 71, 1, -25, 46}
, {40, -76, -79, 2, 49, -39, 14, 47, 1, 11, 40, 13, 53, 13, -30, 42, 23, 3, 18, 100, 37, -65, 30, -103, 15, 1, -18, 47, 23, 38, -77, -74}
, {11, -38, -11, 26, 53, -59, 33, -1, 33, 115, 40, -15, 43, -25, -29, 73, 66, -19, 69, 33, 34, -109, -11, -51, 2, -46, -19, -15, -3, 11, 1, -14}
}
, {{27, -51, 13, -6, -77, 44, 16, 4, -28, -57, -57, -2, -46, 4, 48, -18, 0, 48, -7, 7, -37, 9, 3, -19, -28, 4, 35, 11, 15, -71, -54, 54}
, {63, -116, -22, 10, -70, 55, 13, 19, 15, -48, -56, 7, -169, 50, 46, 5, -10, 6, 0, 79, -108, 1, -32, 15, -53, 36, 29, 59, 31, -123, -126, 15}
, {18, -89, 32, -58, -40, -7, -48, -69, 75, -51, 38, -56, 65, -95, -85, 55, 48, -23, 43, -29, -37, 28, -51, -18, -79, -1, 17, 32, -54, -24, 73, -20}
}
}
, {{{51, -13, -1, -44, 22, 45, -38, 30, -7, -17, -15, -52, -2, -28, -31, -12, 0, -4, -17, 6, -13, 18, -66, -61, -62, 6, -7, 29, -10, -45, 18, 10}
, {22, 5, -23, 32, -10, -11, 26, 35, 27, 36, -9, 25, -54, -2, -8, 58, 63, -50, 41, -57, -8, 0, 28, -3, -15, 88, -31, -21, 19, -10, 6, -37}
, {57, -76, 30, 33, 16, -60, 40, 28, 7, -55, 18, -31, -6, 79, 15, -8, 23, -7, 7, 79, 22, -75, 70, -28, 23, 15, -84, 46, 14, -2, -4, 18}
}
, {{-117, 60, -28, -39, -10, -27, -12, 10, 16, 8, 21, 7, 19, -94, -10, 42, 56, -21, 36, -48, 33, 8, -13, -14, -5, -12, 9, -5, -38, 12, 18, -84}
, {-21, 67, 68, 60, 11, -68, 47, 33, 52, 53, -21, 45, -53, -8, 5, 1, -15, -98, 40, 2, 36, 95, 75, -15, 28, 15, -64, -17, 9, -184, 8, -127}
, {0, -186, 21, 36, -3, 11, 55, 11, -116, 46, 11, -34, -16, 59, 62, 2, -34, 47, 22, -92, 18, -100, 15, -10, 18, 34, -5, -94, 58, -20, -180, 28}
}
, {{55, -14, 58, -78, -73, -10, -126, -65, 10, 27, -56, 74, -68, -2, -36, -7, 33, -142, -4, 43, -68, 17, -173, 19, -40, -140, -21, 64, -150, -130, -6, -140}
, {52, -153, -12, 38, -9, -47, 39, 36, -90, -47, -21, 21, -188, 2, 64, -119, 9, -20, -28, 45, -134, -178, 12, 60, 5, 66, -27, -80, 37, -112, -131, 34}
, {-108, -134, -155, -122, -43, -12, -76, -76, -14, -115, -47, -34, 6, -40, -26, -19, -75, 7, 29, -1, -61, -17, -2, 11, -158, -16, -8, 2, 9, -107, 76, 18}
}
}
, {{{-86, 36, -58, 63, -21, 42, -4, -4, 22, -137, 23, 41, -30, 79, -3, -64, -80, 113, -41, -88, -105, 22, 17, -14, -29, 62, -55, -76, -12, 77, 63, -35}
, {54, -42, 10, 49, -94, -14, -64, -38, 31, -231, -140, -137, -145, 52, -79, -36, -24, -41, -62, 64, -173, -8, -58, -2, -117, -9, -26, 55, -110, -64, -58, 21}
, {50, -30, 61, 29, 11, -74, -11, 61, -50, -52, -3, -84, 66, -7, 4, 15, 55, 32, -46, 97, -38, -93, 39, -1, -50, 61, -21, 51, 17, -30, 8, -8}
}
, {{61, -18, -49, 22, -10, 43, 0, 14, -71, -50, -6, -43, 1, 65, 73, -83, -161, 129, -68, 23, -14, 8, 47, 17, -26, 103, 15, -41, 75, 113, -66, 91}
, {82, -31, 38, -10, -3, -11, 1, 9, -69, -189, -91, 10, -23, 91, 97, -70, -53, -49, -184, -3, -14, -5, -20, 40, -28, 43, -154, -126, -57, -48, 33, 56}
, {141, -38, 71, -25, -5, 42, 54, 9, -64, -13, -4, 30, -50, 79, 48, -82, -32, 42, -40, 19, -48, -22, -41, 102, 7, 48, -9, 28, 28, 4, -3, 44}
}
, {{57, -72, -59, 7, 53, 51, 18, -11, -153, -79, 22, 10, 29, 64, -6, -32, -55, 119, -47, 18, 12, -73, 54, 33, 21, 52, -121, -136, 84, 31, 7, 75}
, {3, -5, 5, -53, 7, 24, -3, 1, -73, 3, -12, 21, -22, 49, -20, -78, -51, -86, -69, -31, 19, -62, 29, 63, -13, 6, -84, -85, 3, 19, 15, -7}
, {27, -25, -26, -64, 37, -7, -21, -71, -59, -62, -53, 35, 3, -24, 60, -86, -99, 16, -37, -5, -2, 4, 12, 74, -29, -50, -3, -70, 6, 14, 76, -3}
}
}
, {{{-24, 5, -58, -20, 49, -9, 26, 22, -44, -130, 6, 31, -48, 65, 19, -115, -32, 26, -14, 28, -8, -38, 2, 0, -22, 46, -8, -72, 23, 57, -16, 19}
, {11, -57, -50, 34, -105, -24, -22, -29, -65, -143, -92, -154, -140, 48, -63, -41, -30, 27, -18, 142, -115, -21, -102, -52, -46, 6, -50, -22, 3, -12, -44, -34}
, {95, 67, 43, 14, 53, -91, -11, 38, -92, -174, -57, 41, -19, 68, -51, -126, -84, 37, -82, 38, 33, -62, 25, 56, 27, 25, -122, -62, -58, 21, -49, -40}
}
, {{73, -55, -89, 70, 17, 0, 43, 35, -46, -1, 59, 12, -7, 51, 56, -33, -59, 116, -13, -21, 3, -8, 86, 11, -6, 112, 31, 5, 62, 84, -32, 45}
, {84, -94, -1, -93, -26, -3, -57, -80, -64, -188, -125, -29, -19, 15, -72, -120, -56, 109, -126, 14, 3, -75, -34, 30, -45, -37, -121, -96, -55, -71, -146, 66}
, {57, -37, 27, 50, 12, 45, 60, 39, -182, -82, 14, 19, -65, 42, -9, -86, 9, 56, -44, 62, -25, -217, 14, 40, 20, 89, -50, 2, 42, 1, -171, 55}
}
, {{-13, -33, -19, -1, 39, 9, 33, 23, -90, -121, -29, -4, 5, 8, -22, -13, -35, 58, -28, 65, 14, -55, 57, -95, 2, 37, -30, -10, 6, 54, 9, 64}
, {-28, 71, -7, -35, 62, -8, 2, -5, -38, -84, -41, 44, 27, -3, 52, -102, -108, -5, -19, -10, 40, -6, 41, 46, -3, -52, -75, -79, 25, 38, 19, -13}
, {39, -155, 16, -6, -14, -17, 66, 41, 35, -18, 23, -36, -67, 47, 17, 0, 40, 16, -6, -3, -8, -9, 19, 17, 27, 14, 49, 86, 26, -2, -108, 14}
}
}
, {{{-67, 59, -15, -19, 38, -22, -18, 34, -37, 86, 55, 63, 19, 16, -30, 6, 58, -97, -5, -18, 28, -38, 32, 19, 21, -75, -15, -24, -21, 60, 64, -83}
, {-1, 71, -75, -86, -22, -25, -65, -87, -47, -70, 5, 47, -31, -26, 13, -91, -74, 52, -48, -1, 10, -41, -79, 44, -19, -59, -36, -78, -20, 98, 57, -35}
, {-131, 4, 27, -18, -67, 33, 20, -40, 15, 14, -47, -72, -125, -24, 104, -1, -14, 2, -14, -73, -83, 0, -27, -67, -14, 8, 65, -12, 58, -60, -58, 6}
}
, {{17, 40, 9, 14, -3, -60, 26, -4, 33, -8, -6, -59, -72, -36, -35, -15, -21, -178, 28, -46, -39, -27, -16, -17, -32, -35, -14, -23, 18, -68, 24, -53}
, {19, -113, -38, 31, -21, -80, -7, 45, -23, -21, 32, -65, 35, 32, -85, 11, -3, 26, -31, 57, 7, -164, 24, -91, -31, 54, -75, -23, -18, 30, -54, -23}
, {-65, -56, -174, 21, 43, -116, -48, 1, -181, -1, 34, -6, 34, -19, -91, 44, 15, 9, -58, 20, 35, -208, 43, -13, -3, 48, -84, 12, -19, -12, -48, 25}
}
, {{-48, -9, 62, -55, 35, 72, -72, -45, 4, 25, 10, 5, -33, -62, -34, -13, -30, 31, 5, -106, 57, 104, -45, -17, 3, -66, 41, 14, -46, -21, 20, 17}
, {-37, 42, 59, -55, -60, -33, -17, -46, -6, -30, -64, 37, 23, -35, -21, 81, 28, -65, 40, -36, -22, 19, -28, -48, 0, -10, 78, -14, -31, -92, 2, -12}
, {-36, 71, 43, -57, -1, -15, -42, -47, 51, 23, -65, 85, -53, -16, 41, -58, -22, -4, 30, -6, -41, 31, -24, 7, -35, -77, 8, 4, 2, 40, 59, -8}
}
}
, {{{-38, -14, -37, 31, -49, -53, 41, 55, 20, -45, 18, -106, -10, -22, 5, 59, 11, -110, 35, -77, -20, -14, -42, -80, -59, 5, -19, 30, -16, -64, -7, -82}
, {60, -86, -18, -47, -35, 13, -39, -32, 21, -113, -41, -135, -33, -67, -5, -59, -51, -6, -33, -14, -8, -18, -21, -33, -66, 18, -62, -52, 41, -46, -95, 14}
, {26, -163, -78, 21, -58, 21, 79, 6, -94, -66, -52, -93, -61, 60, 43, -26, 16, 19, -52, -5, -52, -87, 6, 14, -5, -8, -34, -15, 43, -131, -173, 53}
}
, {{-65, -8, -40, 17, 20, -85, 37, 38, -115, -13, 14, 29, 20, -12, -43, -23, -2, -14, 9, 20, 1, -138, 56, -11, 12, 4, -4, -21, 52, 32, -8, -23}
, {-122, -8, -127, -4, 57, 11, -35, -17, 6, -72, 20, 42, 31, -47, -73, -16, -52, 42, -29, -48, 35, 70, 33, -55, 7, -7, -13, -41, -47, 76, 54, 0}
, {-87, 16, -58, -117, 32, -38, -44, -126, 49, -45, -23, -42, 58, -91, -95, 2, -44, -7, -42, -111, -28, -13, 42, -90, -56, -4, -97, 0, -39, 30, 19, 89}
}
, {{2, -54, -41, -12, 2, 87, 67, -14, 21, 15, 5, -7, -74, -16, 60, -6, 20, 82, 29, -42, 29, 20, 1, -31, 42, -9, 63, 50, -12, -10, -23, 61}
, {4, -25, 23, 14, -18, -1, -8, 19, 55, 37, 11, 1, 1, -43, 20, 51, 65, -88, 31, 24, 12, 21, 17, -55, -3, 17, 4, 56, -24, -11, -43, 48}
, {71, 4, 112, 47, -5, -55, -13, 55, 56, 60, 71, -8, 72, -39, -43, 31, 64, -26, 40, 27, 26, 38, 19, -128, 73, -43, 18, 91, -39, -4, 67, -46}
}
}
, {{{-8, 48, 51, -3, -27, 38, -75, -63, 44, -19, -8, -16, 18, -5, -67, -25, -39, 143, 2, 29, -15, -7, -54, -1, -9, -23, 42, 4, -24, 49, 1, 78}
, {-21, -25, -37, 36, -39, 49, -5, -25, -24, 17, -76, -28, -120, 100, 35, -43, -13, 120, -81, -1, -61, -13, -88, -1, 11, 42, -80, 1, -29, -20, 32, 0}
, {53, 26, 54, -34, -124, -27, -17, -52, 7, -163, -138, 8, -103, 28, -9, -135, -63, -113, -83, -2, -172, 79, -60, 61, -44, -24, 0, -22, -35, -52, -42, -5}
}
, {{101, -12, 120, 17, 19, -50, 36, 1, -68, 9, -24, 57, -62, 84, 47, -69, -54, 15, -39, 104, 3, 5, 30, 83, 10, 3, -54, -36, 32, -96, -63, 44}
, {24, -25, -42, -108, -89, 71, -76, -137, -37, -83, -90, -118, -65, -86, 94, -108, -173, 21, -67, 25, -4, 33, -112, -14, -23, -66, -9, -118, 32, -6, -13, 96}
, {32, 18, 41, -51, 48, -120, 67, 6, -153, -67, -108, 65, -82, 32, 94, -209, -107, -85, -98, -14, -14, -67, -6, 34, 23, -3, -123, -100, 30, -22, -62, 11}
}
, {{9, -42, 31, 0, 72, 58, 54, -27, -20, 24, 43, -37, 47, 11, 18, 6, 7, 57, -15, -9, 27, 13, 50, -63, 101, -8, 34, -45, 10, 74, -11, 20}
, {15, 43, 12, -48, 6, -23, -73, -31, -37, -44, -29, 19, 1, -29, -67, -129, -66, -59, -99, 51, 49, 25, -59, -20, 23, -194, -186, -103, -113, -3, 26, 94}
, {-27, -73, -87, 52, 12, 77, 61, 1, -64, -33, 17, 66, -80, 36, 46, -53, -51, -25, 20, -13, -37, -93, 39, 126, 42, 50, 32, -56, 75, -23, -97, 55}
}
}
, {{{8, -20, 3, 6, -39, 68, 55, 19, -39, -23, -21, 25, -17, 9, 82, 29, 29, 91, 57, 46, 7, 38, 25, -17, -25, 61, 31, 47, -1, 15, -56, 80}
, {58, -29, -18, -6, -21, 56, 26, 36, -24, -24, 10, 7, 11, 33, 8, 30, -11, 36, -13, 31, -29, -28, -2, 52, 17, 49, -7, -6, 43, 48, 12, 84}
, {82, -89, -22, 0, -66, 60, 58, -30, 35, -45, -49, -46, -63, 14, 93, -17, -44, 86, 10, -10, -74, 33, 4, 0, -23, 41, 37, -28, 24, -4, -43, 31}
}
, {{-99, 26, -9, -20, 26, 9, -2, -46, 71, 26, -52, 52, -27, -34, 49, -48, -139, 87, 0, -109, -15, 57, -69, 37, -40, -64, 48, -76, 50, -60, 17, 16}
, {-52, 87, 36, -161, -107, 26, -73, -133, 34, -34, -125, 2, -56, -55, -18, -35, -123, -130, 12, -40, -109, 88, -130, 44, -46, -161, -8, -61, -76, -76, 69, -25}
, {-82, 83, -77, -90, -7, 48, -46, -80, -14, 1, -81, 91, -12, -47, -35, -34, -8, -27, -82, -7, -82, 28, -54, 40, -22, -105, 20, 1, -62, 74, 74, -16}
}
, {{-66, -4, -89, 27, 7, 33, 52, 4, -1, 2, -16, 3, 8, 3, -29, -67, -38, -16, -69, -15, -2, -11, 84, 45, 47, 80, -54, -23, 40, -40, -73, 7}
, {-20, 1, -68, 25, 9, -82, 36, 37, -51, 20, 29, 73, 7, 25, 8, 33, 22, -129, 21, 11, 49, -29, 38, 66, 84, -46, 12, 0, 21, -7, -3, 6}
, {38, 41, 10, 73, 57, -128, 18, 0, 48, 47, 61, 77, 48, -46, -39, 45, 86, -71, 82, 108, 75, -54, 32, -65, 14, -52, -10, 12, 9, 42, 90, -41}
}
}
, {{{-21, -17, 24, 20, 20, 23, 8, 8, 16, -21, 62, 14, 31, 13, -21, 24, 27, -10, -17, -7, 59, 11, 11, 47, 10, -37, -47, -5, -43, -71, 4, 1}
, {-41, 10, 27, -51, 40, 20, -14, -30, -17, 6, 4, -3, 26, 25, 49, -9, -109, -2, 13, -51, 40, 39, -10, 16, 30, -36, -43, -29, 64, -6, -4, 3}
, {-170, 35, -67, -161, -20, 23, 34, -55, 80, 27, -16, 55, -88, -193, 38, 58, -4, -28, 41, -160, -136, 117, -79, -51, 19, -137, 36, 40, 24, -69, 95, 44}
}
, {{-4, 3, -24, -43, -21, -55, 19, 10, -1, 0, -41, 19, -11, 11, 69, 10, 0, -169, 6, -4, -50, -28, -75, 56, -52, -16, -7, 31, 38, -52, -34, 25}
, {61, -68, -7, 25, -7, -126, 3, 19, 58, 20, 50, -32, 40, -32, 2, 40, 3, -14, 75, 34, -1, -69, -25, -39, 3, 12, 6, 43, -5, 95, 22, -60}
, {-49, -124, -99, 57, 21, -54, -107, 26, 9, -82, 51, -154, 49, -144, -115, -4, 22, 33, -31, 42, 57, -25, 22, -367, -48, 15, -123, 55, -43, 51, 88, -52}
}
, {{-119, -23, -58, -9, 3, -22, -31, 16, 18, 24, 50, 61, -9, -24, -30, 14, 1, -16, 1, -56, 43, -4, 16, -18, 9, 43, -45, -55, -6, -77, 49, 38}
, {-42, -8, 52, -23, 14, -112, 8, -14, -32, -97, 14, -20, 24, 1, -7, -48, -6, -88, -49, -65, 4, -57, 38, 55, 4, -66, -58, -57, 33, -44, 53, 2}
, {54, -130, -112, 50, -40, 38, 60, 6, -142, -42, -20, -48, -78, 92, 77, -82, -99, 47, 3, -47, 16, -45, -20, 69, 45, 24, 32, -150, 71, -37, -160, 46}
}
}
, {{{53, 25, 21, -66, -90, -17, -27, -119, 17, 74, -66, -30, 10, -98, 9, 43, 26, -143, -19, -64, -120, -29, -68, 3, -47, -60, -1, 25, -54, 14, 12, -79}
, {-21, -40, 0, -39, -97, 6, -13, -35, -16, 77, -69, -85, -49, -64, 17, 12, 32, 25, 17, -127, -61, -17, -133, -48, -42, -35, 79, 71, -76, 6, 17, -11}
, {-47, 15, 3, -106, -14, -50, -75, -112, 8, -43, -88, -33, -35, -83, -48, 16, -17, -108, 46, 21, -58, 17, -127, -66, -118, -117, 40, -7, -65, -45, 23, 41}
}
, {{-104, -39, -71, 34, 4, 36, -66, 14, -34, -121, 34, -4, -22, 80, -74, -121, -110, -23, -54, 15, 40, 47, 21, -14, 26, 12, -118, -228, -27, 53, -3, -99}
, {-31, 65, 0, -43, 2, -7, -70, -16, -76, 5, -31, -8, 24, -25, -14, -53, -16, -108, -44, -9, 13, 4, 6, 21, -8, -132, -141, -152, -82, -41, -32, -50}
, {-9, 59, 25, -19, 4, -200, 62, 57, 6, 85, 19, 31, -34, -25, -11, 29, 71, -205, 53, -73, 41, -93, -39, 13, 56, -163, 88, 0, -18, -34, -13, -105}
}
, {{-95, -18, -67, 9, -14, 46, 32, 24, -25, 47, 12, 32, -22, 34, 64, 54, 43, 109, 35, -127, 53, 5, 40, 1, 10, 16, 9, 12, 36, 50, 0, 9}
, {74, -84, 55, -27, -134, -25, 2, 9, 30, 35, 19, -57, -10, 19, -37, 32, 35, -58, 19, 59, -27, -19, -42, 20, -10, 25, 11, 91, -5, 12, -1, -16}
, {-45, 102, 20, -54, 2, 46, 16, -56, 21, 97, 4, 51, -9, -148, 92, 11, 5, -43, 30, -92, -13, 55, -55, -6, 2, -19, 44, 45, -16, -31, 48, -23}
}
}
, {{{-22, -23, 8, -13, -30, -23, 19, 9, -60, 29, -15, -35, -19, -2, 89, -38, -11, -88, 12, -4, 11, -37, 2, 44, 105, 11, -25, -16, 14, 23, -61, -78}
, {40, 20, -52, 44, 29, 32, 17, 60, -213, -33, 64, 47, 50, -12, 16, -33, -63, 77, -38, 11, 22, -109, 30, 44, -13, 30, -73, -111, 27, 9, -43, -5}
, {-73, 43, -63, -39, -15, -51, -50, -16, -15, -59, -39, 17, -65, 42, -32, -15, -11, 12, -37, -54, 7, -55, -47, -56, -44, -53, -60, -5, 13, 79, 7, 3}
}
, {{44, -29, -35, -14, -43, 67, 14, 21, 57, -55, -33, -38, -36, -19, -143, -58, 41, -62, -65, 26, -50, -47, -82, -21, 1, -19, -67, 28, -83, 23, -19, -20}
, {102, -144, -44, 9, 8, -31, 25, 44, -203, -128, -44, -38, -43, 8, 50, -20, 40, 56, 14, 79, -104, -256, 15, 21, -3, 95, -95, -69, 9, -127, -160, -26}
, {-61, -101, -118, -61, 17, -43, -3, -56, -16, 33, -40, -77, 35, -8, -24, 5, -44, 18, -13, -42, 49, -36, 15, -74, -75, -12, -1, -59, 14, -24, 32, 2}
}
, {{14, 61, 67, -52, -26, 39, -91, -47, 2, 51, 22, 45, -5, -72, -18, 13, 65, -92, -43, 26, 10, 21, -118, 0, 35, -130, -31, 58, -110, 8, 15, -21}
, {50, -161, 58, 61, 18, -83, 41, 38, 56, 4, 68, -115, 76, 12, -56, 1, 54, 17, 6, 68, -29, -89, 74, 7, 37, 56, -1, 24, -11, -5, -16, -31}
, {-133, 67, -151, -39, 31, -132, -25, -6, 12, 107, 42, 39, 53, -117, 33, 28, 48, -14, 42, -40, 53, -53, -14, -147, 53, -2, 7, 32, 15, 43, 16, -23}
}
}
, {{{22, 72, 59, -53, -80, 41, -7, 11, 1, -51, -114, 82, -45, 19, 70, -74, -8, 75, -29, 31, -52, 42, -48, 14, 5, -62, -14, 24, -12, 12, -22, -16}
, {-23, -5, 1, 6, -21, -107, 21, 37, 37, -35, -42, -25, 5, 50, 39, 8, -33, 39, 37, -27, 21, -45, 56, 22, -25, 2, 9, 29, 12, -34, -20, 45}
, {-2, -14, -54, -11, 4, -4, -65, 9, -47, -7, -28, 17, 36, -11, -20, 39, 40, 31, -39, 23, 23, -70, 12, -38, -14, -35, -32, 39, 3, 22, -18, 44}
}
, {{-32, -20, -62, 19, 7, -71, 8, 11, -30, -6, -16, 26, 44, 29, 32, 38, 55, 66, -4, -6, 15, -107, -27, 99, -2, -30, 10, 6, 68, 61, -8, 58}
, {-57, 22, 3, -3, -10, -80, -17, -31, 19, -3, -33, 61, 4, -80, -71, 32, 17, -141, 0, -43, 3, 18, -5, -146, -20, -32, 41, 48, -47, -6, 14, -39}
, {114, -154, 82, 24, -14, -67, -7, 54, -72, -103, 14, -92, 3, 11, -12, 15, 42, 30, -79, 98, -18, -124, -3, -99, -29, 13, -114, 43, -49, 6, -29, -25}
}
, {{-143, -26, 23, 39, 16, 46, -6, -28, 28, 104, -4, 1, 11, -71, 73, 55, -1, 66, 32, -70, 18, 36, 30, -121, -88, 29, 24, 25, 29, -12, -14, 10}
, {9, 18, 55, -83, -13, -58, -66, -89, 36, -85, -59, 45, -37, -14, -86, -24, -7, -124, -64, 57, -13, -2, -43, 41, -15, -116, -31, -3, -104, -46, 0, -81}
, {61, -236, -1, 20, 1, -36, 48, 33, -140, -85, -31, -112, -56, 88, 79, -96, -59, 78, 17, 36, -62, -253, 9, 50, -18, 93, -38, -11, 55, -14, -184, -1}
}
}
, {{{39, -60, -22, 63, -16, -66, 12, 14, 0, 15, 65, -46, -3, 6, 28, 40, 32, 45, 64, 78, -36, -65, 26, -1, -18, 12, -19, -33, 21, -13, 19, -40}
, {9, 0, 25, -32, 14, -83, -29, -38, 13, 41, 25, -27, 56, -17, -1, -8, -42, -73, 0, 10, 55, 54, -64, -25, 23, -61, -30, -74, -52, 0, 60, -48}
, {-7, 33, -19, -1, 64, 17, 35, -9, -22, 27, 11, 37, -71, -2, 67, 5, 39, -114, 37, -25, -17, -14, -51, 88, 77, -16, 39, -60, 3, -12, 10, 10}
}
, {{8, -141, 27, 29, 61, -112, 80, -1, -45, 7, 7, -26, 71, 7, -60, -46, -8, -29, -64, 6, 45, -68, -2, 25, 54, -16, -101, -47, -60, 9, -2, -90}
, {-162, -74, -232, 3, 83, 47, -24, 34, -96, 36, 48, 53, 25, 48, -39, 9, -4, 35, 67, 29, 45, -120, 43, 11, 29, 13, -19, -79, 81, -97, -20, 26}
, {-55, -96, -59, -21, -79, 5, -40, -20, 63, 91, 14, -61, -36, -159, -88, 39, 62, -81, 51, 109, -67, 76, -70, -38, 42, -60, 35, 28, -3, -90, 97, -13}
}
, {{18, -223, -20, -52, -29, -78, 0, -14, -37, -119, 1, -123, 53, -4, -51, 21, 57, 18, 13, 45, -87, -182, -48, -21, 14, 31, -51, 12, -12, -74, -2, 87}
, {72, -109, -94, 5, 41, -44, -32, 19, -127, -63, 18, -164, 41, -44, -19, 34, -4, 37, -37, -9, -2, -268, 25, -175, -61, 54, -122, -74, -14, -19, 20, 8}
, {-28, 44, -51, -75, 26, -1, -51, 26, -30, 50, -23, 19, 8, -29, 66, 18, 40, 55, -1, -156, -17, -53, -28, -57, -11, -17, -42, -19, 59, 11, 44, 5}
}
}
, {{{84, 25, -5, -35, -30, 18, -44, -29, 27, -47, -24, -36, -57, 36, -30, -15, -37, 14, 10, 76, -85, 25, -34, 22, -9, 34, 43, -17, 28, -90, -17, 46}
, {-38, 52, 61, 71, 24, -8, -7, 13, 87, 85, 20, 32, 43, -2, 44, -26, -16, 82, 53, 28, 40, 42, 6, 1, 40, 22, -19, -6, 52, -21, 9, -80}
, {-44, 35, 8, 21, 22, 39, -6, 51, 36, 36, 43, 24, -40, 21, 46, 69, 17, 8, 2, 18, -28, 34, -34, 13, 4, 5, 46, -5, 35, 13, 7, -17}
}
, {{36, 16, -8, -3, -24, -85, 1, -10, -26, 2, -45, 46, -19, 1, -17, 8, -5, -65, -10, -23, 25, 1, -47, 45, 28, -81, -2, -14, -39, -18, 17, -80}
, {56, -31, 64, 31, -3, 55, 25, 32, -4, 14, 13, 49, -72, 49, 32, -27, -23, 91, 36, -30, -30, 11, 14, 31, -14, 84, 32, 45, 8, 0, -113, 18}
, {71, -51, -56, -4, -64, 28, 25, 2, -82, -100, -65, -140, -24, 33, -58, -34, -75, 43, -56, 56, -65, -18, -24, -10, -67, 87, -33, -72, -29, -9, -84, 67}
}
, {{-41, 67, 8, -55, -4, 39, -16, -4, 73, 56, -3, 43, 24, -13, 63, -21, -19, 15, -5, 13, 21, 49, 7, 30, 6, -64, 3, 3, -19, 132, 53, -30}
, {25, 10, 10, 19, -87, 52, -14, -35, 47, -72, -116, -87, -117, 6, 23, -32, -20, 0, -3, 32, -142, 47, 9, -8, -40, -18, 7, -35, -15, -36, -104, 27}
, {90, -199, 45, 28, -64, 31, 27, 72, -139, -89, -38, -118, -72, 79, -59, -81, 5, -2, -109, 79, -85, -63, -34, 52, -23, 0, -126, -20, -24, -126, -281, 34}
}
}
, {{{-21, 15, -107, 30, 56, -93, -41, -30, -67, 31, 36, -19, 82, -33, -93, 24, -26, -22, -25, -19, 37, -68, 45, -81, 19, 22, -46, 10, -50, 35, 47, 17}
, {-23, -7, -59, 6, 43, -66, 9, -24, -36, -109, -15, 52, 84, 2, 43, 29, 46, -41, -37, -23, 21, -35, 43, 5, 19, -57, -44, -18, -42, 16, 43, -47}
, {49, -11, -115, -50, 3, -28, -25, 17, -34, 12, -10, -18, 32, -9, -57, -16, 12, -38, 0, -49, 49, -61, 17, 28, -15, -11, -32, -20, -5, -41, 1, -8}
}
, {{33, -28, 24, 24, -19, -76, 11, 20, 8, 51, 45, 67, -49, 62, -6, 47, 48, -176, 69, 22, 51, 9, 23, 20, 25, -47, 35, -9, -6, -134, -4, -127}
, {-48, -139, -85, 29, -49, 40, 5, 27, -92, 27, 2, 18, -129, 14, 32, 13, 21, 13, 41, 2, -60, -63, -20, -27, 32, 21, 19, -50, 28, -151, -179, 32}
, {38, -117, -13, -27, -87, 42, 3, -61, -2, 33, -44, -97, -86, -33, -43, 41, 7, 12, 30, -41, -57, 34, -84, -6, 0, -61, 49, -8, -34, -140, 30, 28}
}
, {{93, -50, 37, -25, -20, 37, 24, -47, -51, -57, -6, -32, -20, 42, 31, -38, -9, -52, -18, 81, -14, -11, -23, 34, 17, -19, -31, 14, 33, -89, -37, 71}
, {74, -151, -36, -15, -14, 44, 24, 32, -11, 95, -7, -32, 27, 51, 19, 24, 31, 46, 31, -40, -5, -56, 14, -70, -63, 52, -31, 41, 21, -38, -6, -1}
, {-8, 40, -33, -50, -49, -31, -151, -64, 47, 91, 36, -21, 50, -162, -26, -1, -41, -45, -15, -72, 14, 54, -14, -100, -11, -29, -38, 7, -84, 27, -5, -56}
}
}
, {{{8, 39, 52, -55, 50, 39, -64, -32, 36, -45, -20, 65, 14, -45, -38, 6, 0, 20, 10, -22, 30, 42, -19, 78, -66, 12, -14, -39, -48, 67, 53, -28}
, {20, 13, -11, 17, 11, -25, 29, 1, -36, 14, 28, 13, 20, 76, 36, 101, 43, 51, 3, 26, -26, 3, 2, -21, 51, -24, 35, 8, 12, 73, 46, -97}
, {32, 4, 24, -5, 22, -29, -45, 17, 25, -25, -10, 33, -6, -2, -86, 61, 28, -11, -22, 131, 11, 19, -34, 36, -33, -16, -12, 37, -26, 45, -5, -5}
}
, {{-22, 1, 5, 44, -28, -21, -48, 28, -98, 42, 8, 15, -22, 38, 39, -12, 16, -7, 4, 4, -3, -69, -32, -22, -23, 27, -98, -6, 31, -89, -63, 57}
, {20, -51, 2, -15, 51, -40, 31, -26, -92, -50, -22, 23, 19, 33, 44, -71, -70, 1, -52, 35, -4, -67, 25, 4, 46, -21, -39, -91, -8, -24, -30, 60}
, {85, -76, 13, 29, 22, -30, 69, 47, -129, 15, -14, 48, -16, 14, -9, -20, -13, 25, 13, 19, -13, -164, -22, 39, 28, 16, -61, -94, 33, -60, -72, 26}
}
, {{48, -53, -34, 15, -34, -97, 35, 23, -38, -56, -9, -5, 40, 51, -119, 3, 52, 26, 18, 107, -4, -136, -47, 36, 58, 47, -61, 38, -32, 14, -64, 12}
, {23, -203, -60, -2, 38, -99, -16, 35, -78, -83, 23, -50, 40, -24, -113, -16, 44, -97, -1, -17, 8, -133, 57, -89, 34, -21, -26, 6, -13, -48, -39, 47}
, {-47, -18, -64, 13, 36, -60, 16, 10, -14, 22, 37, 73, 53, -74, -4, -16, 15, -64, 50, -86, 44, -26, 51, -98, 11, -61, 42, 67, 18, -62, 45, 15}
}
}
, {{{47, 35, -16, -13, 27, 50, -7, 3, -23, -3, -42, 57, 51, 7, 46, -114, -47, 62, -87, 85, 11, -70, -16, 61, 23, 37, -89, -98, -64, 49, -31, -47}
, {-6, 26, -28, 11, 25, 87, 28, 10, -5, 58, -14, -24, 14, -23, 36, -9, -24, 91, 7, 10, -25, -36, 7, -7, 58, 54, -35, 2, 6, 33, 15, -3}
, {-10, 46, -58, 0, -13, -20, 63, 51, -55, 30, -31, -24, -8, 49, -6, -25, 10, -11, 4, -57, 19, -36, 31, 58, 8, 70, -18, -46, 19, -44, -5, -24}
}
, {{-29, 22, -118, 64, 61, 43, 13, 22, 37, 18, 13, -28, 60, 12, -54, 14, 2, 115, 39, -120, 46, 47, 100, -74, -22, 76, 41, 27, 30, 46, -2, 57}
, {9, 37, -3, -63, -70, 16, -28, -19, 64, 46, 44, 14, -10, -70, -93, 31, 60, -67, 56, -91, 19, 61, -69, -22, -10, -77, 27, 69, -70, -26, 27, 18}
, {-40, -88, -10, 11, -194, 81, 25, 8, 35, 91, -10, -29, -60, -82, -30, 26, 45, -87, 55, -95, -5, 91, -136, -117, 21, -135, 33, -7, 10, -149, 21, -70}
}
, {{-35, -22, 23, 53, 8, 29, -47, 25, 63, 55, 22, 0, 28, -20, 115, -7, -31, 59, -17, -27, 37, 64, 55, -154, -69, 47, 63, -20, 21, 20, 79, 2}
, {-60, -18, 44, 8, -84, 32, -127, -78, 48, 66, -34, -37, -63, -108, -137, 19, -10, 0, -61, 68, 31, 70, -55, 57, 31, -58, -38, -68, -129, -18, -27, -128}
, {-89, -2, -48, 40, 10, -52, -112, -1, -67, 41, 62, 11, 26, -27, -141, -30, 61, 12, -99, -16, 40, 19, -11, 10, 9, -83, -157, -40, -120, 3, -13, -221}
}
}
, {{{-3, -1, 28, 56, -21, -37, -6, 51, 4, -36, -2, -3, 7, 58, 74, -9, -42, 144, -25, -28, 45, 5, 53, -4, 41, 22, 12, -31, 62, 36, 61, 49}
, {59, 30, 65, -35, -41, 70, 2, 28, -7, 32, -50, 43, -87, 3, 73, -43, -52, -41, 10, 75, -53, 21, -41, -12, -53, 15, 86, 43, 32, -37, 12, 14}
, {37, -17, 29, -9, -4, 36, 83, 19, -27, 9, -41, -13, -38, 52, 45, -12, 13, -7, 2, 42, -24, 48, 45, 76, 17, 21, 52, 32, 32, -48, -13, 48}
}
, {{-26, -23, -28, -12, -41, 128, -48, -18, -26, -64, 0, -24, -4, 24, 83, -55, -3, 60, -26, -26, -8, -15, 13, 54, -6, -20, -68, -50, 64, -25, -8, 80}
, {26, 76, -77, -3, -11, -92, 25, -7, -60, 12, -10, 25, -6, 36, 29, 10, -9, -118, -22, 30, -29, -24, -4, 88, -16, -1, 5, -34, 11, 16, -55, 16}
, {-8, 98, -28, -19, 17, -25, 8, -18, 13, 42, -4, 73, -11, 67, -8, 53, 34, 13, 6, -18, -17, 48, 16, 45, 17, -20, 53, 39, -13, 30, 59, 44}
}
, {{-28, 0, 13, 3, 0, 11, 14, -28, -38, -69, -47, 23, -2, 18, -28, -88, -112, 83, -48, -55, -24, -8, -12, 28, -64, -32, 4, -78, -10, -12, 57, 17}
, {11, 60, -13, -112, -53, 34, -54, -115, 48, 88, -114, 12, -45, 15, -39, 8, -20, -40, -2, -122, -173, 9, -97, -13, -125, -29, 41, 20, -21, -7, 84, -14}
, {-36, -7, 13, -56, -131, 42, -89, -115, -10, -66, -108, -19, -139, -22, 8, -28, -39, -43, -26, -42, -156, 42, -104, -87, -200, -119, 21, 35, -97, -63, 45, -59}
}
}
, {{{-36, -34, -29, -70, -52, -50, 5, -60, 5, -122, -63, -61, -18, -21, 36, -2, -11, 30, -27, -7, -75, -60, -73, -31, -83, 13, 7, 29, -1, 48, 38, -55}
, {2, -32, -28, -81, -134, -53, 6, -47, 33, -23, -52, -78, -133, -83, 12, 20, -17, 3, -34, 63, -132, 31, -158, -42, -67, -48, 21, 19, -43, -102, -27, -20}
, {-34, 76, 31, -34, 17, -75, -14, -9, 11, -67, -20, 124, -6, 19, -74, -58, -41, -9, -35, 51, 25, 39, -9, 37, 38, -54, 29, -50, 12, -47, 43, 1}
}
, {{-23, -38, -68, 14, 5, -6, -23, 29, -26, 16, 41, -11, 55, 19, 19, -16, -50, 96, -38, -42, 66, -46, 82, -71, -27, 73, -6, -57, 44, 78, -68, -5}
, {136, -118, 7, -179, -97, 7, -107, -136, 46, -70, -128, -118, -55, 6, -178, -60, 24, 18, -108, 35, -44, -48, -146, 32, -44, -82, -103, 24, -89, 89, -94, 45}
, {66, -14, 16, 57, 3, -2, 42, 39, 34, 38, 33, 3, -34, 104, 8, 38, 70, 47, -5, 41, -15, -41, 13, 61, 20, 54, 27, 75, -17, -22, -18, 99}
}
, {{-29, 15, -5, 38, 7, 8, 31, 23, -24, 36, 44, 32, 36, 16, 40, -3, 80, 71, -5, 74, 18, -57, 44, -24, 24, 11, -3, 15, 15, 58, 27, 73}
, {-46, 78, -3, -12, 15, -66, -58, -15, -22, 83, 3, 32, 44, -61, -48, 34, -11, 23, -105, -78, 90, 0, 12, -47, -6, -58, -88, -20, -114, 146, 125, -141}
, {4, 63, 112, -53, -12, 39, -17, -42, 4, 34, -4, 69, -22, -14, 48, -17, 13, 7, -8, -13, 49, 49, -12, -8, 45, -14, 0, -32, 10, -37, 39, -34}
}
}
, {{{-62, 53, 57, -36, -35, 44, -98, -16, -3, -39, -34, 46, 33, -31, 25, 2, -45, -93, -73, -18, 5, 34, 27, -32, -107, -26, 44, 34, 10, -5, -16, 68}
, {-13, -8, -5, -27, -2, 43, 30, -19, 2, -54, -106, -28, -51, 11, -46, -27, -68, -54, -106, -22, 11, 21, 30, -50, -47, 6, -20, -28, 10, 27, -24, -144}
, {-28, 9, 12, -123, 19, -16, -33, -80, -43, -88, -64, -7, 68, -67, -10, -47, -54, 35, -48, 10, 11, 5, -21, -15, -61, -59, -46, -69, -32, 45, 52, 17}
}
, {{2, 21, -27, -20, -12, -43, 24, -9, -113, 19, 44, -2, 18, 32, -9, 31, 50, -37, -10, -26, -4, -25, -15, -19, 29, 2, -75, -18, 17, 13, 1, 29}
, {-103, 63, -129, 10, 7, -91, 28, 38, -117, 77, 19, 37, -23, -31, -31, 5, 21, -182, 31, -29, -27, -63, 17, -8, 35, -63, -26, -15, -25, -65, -9, -39}
, {-45, 74, 19, 47, 38, -86, 32, -24, 68, 103, 52, 68, 20, -17, -12, 36, 76, -55, 92, -54, 13, 29, 14, -89, 69, -26, 23, 23, -18, 12, 36, -44}
}
, {{80, -57, 20, -15, -5, 51, -6, 18, -22, -28, -9, 9, 19, 49, 64, -26, -34, 40, 10, -22, -12, -30, 36, -3, 32, 15, -26, 18, 34, 13, -20, 10}
, {6, -67, 10, -33, -27, 5, -36, -38, -1, 15, 23, -27, -2, 7, 2, 38, -14, 18, -19, 11, -88, -22, -86, -25, -15, -8, 18, 0, 11, -43, -60, 25}
, {11, 30, 27, 28, 14, -4, -4, -15, 49, 31, -17, 4, -10, -40, -51, -7, 72, -40, -9, -33, -18, 16, -26, -8, -13, 13, 5, 70, -26, -23, 46, -32}
}
}
, {{{18, -5, -2, 8, -42, 9, -5, -19, -10, -63, 12, -19, -19, -1, -62, -27, 27, 23, -29, 83, -76, 24, 6, -20, -7, 30, -84, -71, -22, -48, -7, 72}
, {-26, -37, 0, -4, 5, -44, -6, -30, -89, 72, -16, 60, -50, 30, 86, -68, -114, -34, -15, -15, 43, -50, -24, 67, 11, -18, -46, -194, 54, -44, -17, -56}
, {-88, -97, -149, 31, -75, 44, 23, -9, -20, 41, -35, -15, -144, -15, 80, -4, -29, 37, 47, -17, -15, 33, -7, 91, 27, 15, 31, -83, 103, -59, -93, 9}
}
, {{43, -64, 51, -16, 43, 43, -32, 24, -24, -25, -58, 24, 1, 58, -35, 23, 78, 67, 2, 53, 1, 7, -59, 58, 32, 39, -31, -3, -9, 53, -92, -64}
, {27, -91, -30, -2, -7, 97, -15, 2, -106, -71, 3, 6, 32, 66, 4, -45, -24, 104, -30, 40, -21, -149, 9, 37, 8, 80, -22, -23, 60, -27, -56, 138}
, {48, -67, -25, -65, 10, 12, -71, -84, 1, -64, -17, -95, 2, -128, -48, -65, -87, 14, -69, -18, -23, 51, -10, -18, -38, 46, -116, -19, -24, 50, 54, 23}
}
, {{34, 27, 18, -19, 36, -16, 24, -1, 17, -111, 36, 14, 16, 45, -6, -13, 9, 7, -31, 34, 10, 33, -7, 9, 58, 28, -37, -2, 9, 51, 17, -53}
, {80, -186, 46, 48, 69, -95, -15, 36, -107, -76, -16, -112, 4, 41, -12, -42, -5, 70, -81, 82, 7, -221, 49, 32, -52, 47, -154, 1, -12, 36, -3, 13}
, {-34, -65, -113, -1, 37, -16, -17, 52, -102, -10, 43, -58, -16, -143, 35, -22, 2, -3, -35, -82, 41, -132, -4, -44, 7, -1, -76, -72, -20, 21, -12, 22}
}
}
, {{{33, -4, 36, -28, 14, 23, 23, -31, 96, 31, -65, 7, -4, 23, 6, 19, 52, 15, 5, -60, -1, 56, -50, -4, -18, -57, -10, 45, 9, 15, -28, -90}
, {-68, 19, -48, -30, 15, -63, 9, 7, -32, 58, 35, 25, 6, 14, -91, 16, 39, 60, 35, 10, -10, -105, 37, 23, -9, -64, -16, 25, 29, -7, -18, 10}
, {-102, 70, -136, -58, 8, 8, -84, -45, 47, -1, -30, 50, 13, -75, -34, -35, -119, -29, -27, -77, -21, -49, -49, -119, 1, -57, -60, -185, 14, -27, 79, -8}
}
, {{-5, 53, -46, 6, -24, 106, -55, -39, -39, -47, -7, 28, 56, 8, 5, -23, -87, 10, -28, 34, -54, 39, -80, 50, -53, -41, -2, -44, 6, 61, 103, 38}
, {114, -138, 16, -24, -92, -1, -30, -10, 24, -51, -58, -63, -114, 26, -55, 9, 1, -75, -44, 30, -57, 22, -72, 12, -30, -62, 15, -19, -58, 18, 26, 5}
, {58, -111, 20, 37, 46, -99, -34, 46, 73, -16, 72, -54, 117, -1, -18, 15, 44, -20, 15, 53, 41, -86, 35, -113, 31, 47, -61, 32, 17, 18, -17, 1}
}
, {{59, -4, -4, 12, 30, -14, -2, -7, -56, -7, 31, -12, -2, 25, 55, -20, 60, 71, -17, 59, 16, -73, 52, -44, -3, 14, -29, 6, 74, 79, 37, -4}
, {-3, -21, -16, -19, 2, -27, -38, -34, -47, 10, 14, -17, 26, -10, -25, 7, 15, -5, -20, 20, 21, 34, 12, 11, -14, -73, 19, -31, 27, 50, -59, -32}
, {64, -98, 30, 4, -21, 1, 22, 41, 3, -25, -65, -3, -88, 59, 16, -34, -35, -53, 33, 40, -33, -86, 27, 70, -26, 4, -56, 9, -12, -66, -32, -34}
}
}
, {{{25, -15, 51, -69, -16, -17, 4, 15, 12, -15, 12, -40, -60, 58, -154, -42, 65, -62, -23, 10, -31, -8, -10, 76, 28, 2, -54, -40, -19, -66, -40, 30}
, {-47, 25, -78, 2, 13, -104, 8, 20, -163, -41, 16, 31, 30, 24, -15, 48, 41, -44, 19, 15, 26, -145, 2, 30, 46, 13, -53, -36, -5, -15, 27, 41}
, {-67, -9, -51, -48, 35, -23, 30, -18, 124, -27, -12, 39, 3, -61, -2, 38, 10, -9, 74, -131, -58, 47, -33, -77, -8, 56, 12, 24, 31, 23, 100, -4}
}
, {{-28, 17, 12, -19, 54, -3, -1, 11, 1, 2, 31, 10, -20, -19, -4, 20, -10, -84, -21, -70, 63, 45, 16, -62, 34, -57, 3, -5, 2, 38, 76, -32}
, {73, -24, 85, 0, -14, -57, 14, -12, 23, -138, -63, -128, -135, -3, 1, -66, -1, -28, -5, 50, -80, -55, 29, 6, -63, 37, -56, 7, -35, -97, 61, -41}
, {6, -178, -79, 54, 26, 61, -29, 48, -111, -121, -4, -138, -4, 95, 111, -76, -67, 32, -89, 17, 17, -114, 54, -54, -83, 22, 9, -123, 56, 26, -53, 66}
}
, {{-71, 69, -24, 47, 62, 3, -9, 25, -15, 52, 6, 27, 3, -23, -11, 45, -4, -29, 59, -6, 62, 7, -11, 49, 39, -51, 20, 9, -26, 39, 23, -55}
, {-45, 15, 10, -47, 5, -3, 44, -36, -5, -9, -2, 6, -46, 20, -2, -6, -42, 5, -7, 21, -28, -17, -9, 62, 0, 2, -11, -3, 27, -107, -23, 33}
, {-127, -183, -124, -30, 6, 38, 11, -68, -95, -168, -37, -166, -7, -42, -92, -82, -73, -28, -45, -18, -43, 31, 46, 0, -83, 32, -30, -102, 10, -73, 4, 29}
}
}
, {{{41, -21, 69, -33, 10, 0, 1, 51, 53, 80, 57, 20, 52, 35, -14, -26, 0, -24, 19, -3, 7, 22, 15, 59, 53, 8, -60, -30, -11, -16, -33, -26}
, {-9, 18, -51, -30, 42, -70, 50, 9, -87, 15, 15, -8, 14, -6, -20, 31, 13, 3, -4, -4, -21, -60, 21, 6, 5, 11, -11, 40, 8, -27, 27, 15}
, {-99, 0, -103, -83, 51, 23, -68, -56, 21, -5, 6, -16, -14, -96, -70, 9, -32, -19, 26, -101, 2, 13, 24, -98, -39, -27, -19, 7, -15, 42, 3, -8}
}
, {{30, 26, 43, 26, -19, -19, 50, 13, 31, -45, -23, -24, -135, 21, 25, -24, -37, -106, 37, 22, -55, -7, -27, 32, -71, -59, -4, -14, -21, -57, 97, -35}
, {46, -147, 60, 67, -19, -90, -5, 15, 44, -36, 26, -75, 26, 7, -21, 0, 49, -89, 26, 46, -17, -23, 17, -74, 19, -35, -5, 5, -38, -42, 8, -48}
, {14, -82, 47, 49, 15, -76, -27, 60, -16, 3, 36, -58, 26, 15, 12, 54, 61, 0, 18, 16, 16, -93, 62, -144, -11, 31, -34, 20, -4, -3, -21, -8}
}
, {{-39, 49, -7, -12, 48, 41, -5, -5, -22, 13, 61, 94, 19, -34, 10, 8, 60, -21, 9, -38, 33, -22, 63, 65, 55, -54, 14, -35, 16, 39, 38, -8}
, {14, -23, 0, -59, -48, 29, 28, -52, 14, -20, -62, 26, -47, -22, -12, -23, -34, -39, -21, -8, -51, 10, -47, 7, 20, -65, 40, 18, -1, -55, -6, 19}
, {71, -35, 19, 7, -41, -37, 30, 19, -55, -98, 36, -43, -28, 5, 32, 1, -20, -44, 14, 52, -35, -51, 16, 2, -17, -1, -32, 6, 26, -151, -75, 5}
}
}
, {{{-8, -75, 31, -28, 49, 98, 16, -4, 28, 9, -10, -44, -30, 39, -17, -25, -25, -3, -38, -62, -25, 28, -45, -54, -60, -16, -5, -5, 26, -31, -12, -28}
, {53, -16, 5, 31, -14, -24, 33, 14, 59, 35, -11, -49, 13, 14, 69, 24, 77, -42, -21, 15, -18, 18, -26, 12, -16, 67, -24, -21, 57, -140, 15, 31}
, {-21, 24, -36, -9, 49, 16, -54, -46, -25, -31, 15, 19, 71, 21, 17, 8, 6, -4, -16, 19, 5, 20, 0, -23, -45, -35, -15, -41, 41, 0, 8, -17}
}
, {{-152, -12, -47, 4, -31, 52, -42, -26, 30, 1, 19, -20, 47, -113, -79, 40, 5, -2, -6, -80, -3, 62, -11, -70, -10, 54, 12, 37, -99, 39, 49, -33}
, {-7, 58, 21, 30, 18, 8, 28, 33, -30, 15, 14, 0, -42, 9, 5, -31, -52, -49, 3, -64, -6, 46, 37, -28, 28, 55, -88, -42, -6, -154, -79, -111}
, {-23, -44, -28, -3, 38, 34, 32, 37, -41, 55, 17, 20, 26, 43, 39, 7, 3, -3, 68, -17, 46, -92, 32, 53, 49, -3, 53, -75, 43, -41, -101, 26}
}
, {{41, -66, 28, -75, -202, 43, -112, -75, 59, 72, -40, -74, -174, -36, 38, 89, 25, -87, -11, 69, -27, 42, -193, 12, -2, -104, 16, 2, -85, -177, -45, -122}
, {73, -191, -45, 19, -96, -97, 38, 10, -91, -3, -5, -106, -143, 46, 73, -83, 14, 18, -25, 78, -221, -185, -11, 19, -52, 70, -72, -15, 68, -145, -184, -56}
, {-114, -57, -119, 14, -27, -7, -62, -30, -17, 13, 10, -18, -28, 12, -21, 5, -11, 32, -18, 57, -66, -6, -42, -6, -149, 65, -39, -50, 2, -139, 29, -7}
}
}
, {{{63, 14, 29, -60, -22, 4, 28, -40, 7, 54, -46, 17, -44, -74, -52, -28, 12, -72, -36, -66, -6, -3, -3, -5, -68, 14, -7, 40, -20, 16, 17, -7}
, {-49, 15, -55, -50, -48, 75, -31, -39, 24, 22, -17, 6, -27, -24, 32, 71, 27, -101, 11, -71, -84, 52, -64, 25, -41, -48, 22, 38, -18, -33, 34, 8}
, {-145, 102, -93, -103, -114, 113, 7, -78, -21, -48, -103, -19, -98, -134, 166, -29, -68, -81, 62, -72, -140, 88, -83, -24, -78, -38, 93, 0, 56, -15, -12, 33}
}
, {{-56, 53, -19, -3, 15, -71, -39, 34, -22, 45, 39, -45, 69, -44, -52, -10, -15, -138, 81, -35, 42, 7, 17, -33, -18, -77, 5, 18, 18, -18, 8, -101}
, {-40, 85, 92, -15, 35, -25, -6, -17, 67, 46, 15, 30, -41, -56, -72, 9, -26, -24, 3, -5, 13, 73, -6, -67, 45, -43, 12, -31, -47, -15, 67, -134}
, {-79, 27, 11, -8, 43, -64, 15, 17, -26, 29, 10, -10, 71, -58, -12, -14, -120, -19, -27, -73, 37, 75, 50, -57, 15, -65, -89, -221, 3, 25, -29, -52}
}
, {{-2, 9, -14, -45, -19, 52, -12, -13, -74, -45, -6, 75, -62, 8, -97, -58, -32, -51, -47, 55, -28, 42, -35, 19, 33, 34, -65, -33, -51, -58, 32, -25}
, {-52, -20, -42, -40, 66, -45, 57, 24, -139, -121, 8, -11, -41, 61, -4, -96, -7, 35, -15, 0, 27, -152, 1, 66, 19, 8, -55, -65, 57, -81, -102, 10}
, {-106, -29, -134, 50, 27, -17, 16, 20, -110, -43, 73, 23, 14, -10, 41, -21, -62, 31, -22, 4, 11, -55, 46, 31, 5, 30, 1, -145, 45, 15, -31, -17}
}
}
, {{{105, -34, 69, -15, 25, 58, 71, 29, 43, -11, -21, 2, -28, 38, 7, -26, 22, 62, -7, 17, -30, 22, 18, 62, -6, 74, 52, 60, 77, -8, 65, 66}
, {44, -15, 35, -44, -10, 36, -44, 20, 19, 33, -32, -47, 31, -32, -86, 17, 8, 44, -32, 35, 28, 15, 32, -35, -63, 5, 32, 80, -45, 48, 51, -12}
, {-48, -53, -63, -14, 5, 53, -30, -59, -39, -96, 9, -92, 15, -14, -17, -35, 1, 52, -92, 16, -34, -93, 14, -91, -105, 26, -74, -1, 0, 2, 42, -18}
}
, {{50, 44, -14, -2, -17, 81, -9, 19, -10, -58, -10, -117, 93, -12, -70, 25, -24, 87, 6, 63, -26, 13, 45, -17, -35, 42, -17, -20, -101, 80, 7, 28}
, {-7, 108, -3, -32, -10, -42, -32, -50, 26, 24, -22, -5, 32, -32, -79, -1, -40, 17, -30, -11, 44, 73, -3, 6, -53, -22, -21, -23, -74, -23, 119, 7}
, {-1, 63, 98, 33, 40, -143, 22, 24, -8, 6, 33, -36, 95, -19, -122, 70, -21, -37, 21, 6, 47, -48, 25, -68, 32, 7, -62, 23, -88, -9, 77, -60}
}
, {{40, -1, 75, 31, 0, 28, 17, 31, 53, -71, -43, -8, 11, 22, -158, -46, -28, 41, -78, 11, -13, 62, -91, 71, -1, -31, -75, -12, -176, -39, 12, -23}
, {-18, -16, -49, 28, 33, -47, 30, 34, -200, -18, 62, 27, 21, 63, 12, -27, -2, 17, -9, 13, 4, -140, 27, 28, 22, 55, -138, -67, 40, -24, -48, 36}
, {1, -162, -56, -58, -2, -1, 7, 25, -36, 2, -16, 17, -121, 24, -21, -93, -41, -24, 34, -38, -66, -27, -49, 17, 31, 13, 17, -21, 25, -178, -34, 8}
}
}
, {{{-24, -74, -66, 10, 23, -36, -27, -39, 36, 5, 18, -54, 70, -70, -43, 13, 0, -59, 46, 21, 44, 3, -35, -51, 20, -62, -2, -15, -34, 17, 71, -13}
, {51, -103, 19, -38, -43, 40, 4, -25, 44, -37, -80, -84, -36, -40, -8, 34, 21, -11, 19, -25, -62, 28, -84, 13, -40, -37, 26, 40, 6, -68, -29, -13}
, {-40, -59, 21, 3, -62, 7, 19, -29, 37, 38, -36, -20, -105, -25, 131, 25, -7, -25, 14, -15, -104, 6, -22, 47, 16, 4, 14, 3, 46, -153, -49, -22}
}
, {{41, 30, 65, 38, 15, -79, -36, 9, 54, -2, 64, 1, 52, -70, 72, 61, 56, -19, -24, 59, 63, -36, 2, -25, 41, -39, -18, 40, -38, 99, -3, -1}
, {-147, 137, -18, -14, 49, 19, -47, -4, -59, -19, 9, 74, 66, -76, -5, -39, -61, -6, 0, -87, 37, 46, -5, -40, 9, -13, -25, -69, -12, 54, 39, 9}
, {7, 5, -88, -145, -57, 16, -106, -85, -14, -201, -128, -69, -27, -122, -62, -154, -127, -39, -100, -49, -110, 30, -63, -60, -61, -28, -62, -45, -81, 44, 34, 3}
}
, {{17, -31, -3, 51, -18, 32, 57, 65, -49, -47, -16, 7, -22, 38, 22, -18, -2, 43, 10, 42, -55, -52, 10, 60, 27, 79, 21, 8, 16, -101, -72, 89}
, {-15, -8, -58, 48, 39, -122, 30, 47, -49, -68, 46, 44, 74, -1, -7, 15, 40, -11, -8, 44, 54, -117, 26, 1, 34, 12, 44, 45, -28, 47, -55, 10}
, {44, 5, -59, 87, 55, -86, -22, 49, -197, -16, 50, 30, 66, -28, -51, 11, 80, 109, -15, 88, 69, -157, 45, -52, 13, 57, -48, 40, -42, 66, 42, -8}
}
}
, {{{-65, -29, -31, -10, -36, 30, 10, -18, 39, 36, -14, -9, -45, 32, 38, -66, 4, 60, -25, -26, -39, 13, -43, -60, -1, -4, -21, -68, -41, -17, 25, -61}
, {68, 14, 19, -28, -86, 44, -48, -90, 46, -105, -110, -94, -96, 44, -8, -11, 9, -77, 10, 45, -155, 60, -131, 15, -78, 11, -13, 13, -23, -145, -1, 4}
, {110, 5, 74, 16, 49, -63, 27, -9, 4, -22, 10, -38, 24, 26, -27, 50, 42, 32, -5, 73, -29, -67, -28, -4, -27, 37, -66, 47, 35, 51, 28, 17}
}
, {{-75, 40, -62, 74, 11, 23, -20, 2, 5, 21, 13, -7, 34, -40, 17, 61, -7, 57, 6, -114, 27, -5, 13, -131, -49, 111, -10, 5, 64, 68, -13, 48}
, {-13, 23, 17, -44, -13, 0, -22, -38, -10, -9, -28, 9, -46, -14, -2, -21, -28, -147, -44, 19, -26, 27, -68, -9, 10, -64, -50, 4, -134, -132, 37, -73}
, {-35, -108, 47, 15, 1, 35, 53, 11, -155, 61, 19, 45, -63, 44, 46, 6, -18, 17, 39, -33, 18, -194, 4, 57, 30, -21, 22, -150, 74, -60, -174, 39}
}
, {{2, -3, 69, 5, -54, 98, -34, -36, 72, 96, -23, 19, -4, -25, 120, 39, 20, 77, 4, 46, 70, 48, 8, 44, -21, 59, 63, 32, -7, -66, 14, 14}
, {37, -182, 19, -42, -46, 22, -10, 33, 32, -68, -60, -71, -272, 22, 28, -109, 13, -98, 4, -14, -116, -63, 30, 2, 8, 83, -73, 1, 11, -104, -185, -142}
, {-81, -226, -147, 14, -11, 26, 30, 44, -20, 8, 28, -180, -23, 54, 7, -26, 61, 20, -3, 37, -56, -73, 11, -90, -77, 79, -9, -37, 6, -143, -22, 29}
}
}
, {{{30, -32, -43, 71, 37, -7, 13, 34, -13, -57, 52, -39, -8, 44, 11, 22, 33, -23, -4, 71, 45, -29, 72, -90, -30, 95, -92, -40, -1, -14, -27, -11}
, {30, -59, -14, 26, -12, 27, -7, -26, 0, 23, 84, 4, 14, -34, 40, 35, 4, 18, -7, -31, 26, 18, 67, -42, 14, 133, -40, 0, 27, -22, -59, -12}
, {-70, -34, 4, -8, -19, 28, 71, 12, 4, 125, 7, 20, 16, 28, 0, 56, 59, -5, 32, -20, 54, 21, -19, 63, 85, 11, 67, 81, 13, -14, -22, 1}
}
, {{12, -47, -66, 17, 26, 32, -11, 12, 0, -12, 31, -53, 38, 40, 16, 47, 54, -91, 3, -13, -26, 18, -3, -5, -25, 34, 41, 47, -37, -27, -46, -1}
, {-83, -13, -22, 18, -69, 15, -40, -41, 66, 82, 41, -85, 23, -100, -6, 75, 46, -93, 39, -69, 36, 36, -69, -107, 16, 0, 25, 70, -56, -136, 16, 1}
, {-76, -29, -2, -53, -181, 58, -50, -63, 70, 66, -5, 13, -67, -132, 5, 13, -7, -28, -11, -131, -20, 76, -157, 17, -9, -85, 54, 17, -24, -104, 40, -49}
}
, {{-130, 25, -119, 4, 28, 41, -7, -28, 3, 47, 2, -70, 14, -14, 56, -4, -26, 80, -10, -86, -5, -33, 52, -79, -17, 75, -9, -65, 47, 29, 40, 53}
, {-19, 35, 18, 33, 12, -11, -53, 14, -4, 131, 35, 4, -5, -43, 31, 35, 45, 18, -35, 7, 27, -14, 15, 50, -6, -40, -67, -117, -78, -3, -35, -120}
, {-72, 35, -16, -47, -34, 46, -240, -62, 40, -24, -40, -70, 68, -115, -153, -58, -77, 38, -84, -20, -18, 1, -38, -59, -53, -1, -161, -184, -49, 78, 35, -114}
}
}
, {{{-50, -45, -102, -34, 13, -44, 45, -22, 32, -25, -43, -22, 7, 2, 0, -26, 12, 33, 8, -87, 2, 10, 10, -15, -1, 36, -4, -15, 37, 17, -29, 27}
, {-37, -8, -58, -22, 17, -30, -27, 19, -9, 128, 23, 57, 61, -8, 2, 32, 37, 3, 12, 28, 28, 8, -41, 32, 18, -70, -18, 41, -74, 15, 43, -42}
, {14, 54, 39, 26, -3, -59, -17, 17, 62, 75, 31, 30, 56, -57, -103, 48, 54, -96, 46, 45, 45, 30, 4, -30, -24, 13, 34, 26, -76, 23, 49, -102}
}
, {{-16, 11, -43, -40, 13, -43, -40, -36, -77, -77, 34, -19, 71, -47, 0, -21, -40, -6, -74, -3, -9, 14, 30, -17, -35, -37, -96, -90, -75, 52, -23, -22}
, {6, 10, -19, -34, 54, -72, -37, -4, -81, -32, -21, -24, 35, 0, -75, -51, -6, 37, -80, -28, -28, -68, -33, -35, 11, 45, -105, -73, -49, 105, 38, -70}
, {111, -49, -18, 11, -4, -118, -6, 27, -8, 4, 40, -25, -27, 69, 18, 2, 67, 5, -11, 35, 7, -175, -16, -21, 16, 16, -19, -15, -10, 28, 11, -42}
}
, {{-17, -40, -16, 31, 46, -75, -27, 32, -12, -59, -26, -5, 40, 0, -42, 9, 7, 30, 11, 9, 25, -67, 22, -2, 3, -28, -34, 24, 2, 44, 11, -8}
, {13, -15, -90, 30, 15, -46, -1, 54, -9, -39, 66, -13, 87, 4, -97, 6, 36, -47, -4, -2, 11, -94, 105, -38, 7, 72, -1, 41, -44, 70, 49, -37}
, {-80, -38, -47, 20, 25, -39, -6, 34, 22, 52, 40, -42, 63, -70, -40, 49, 29, -32, 26, -102, 35, 24, -19, -164, 77, -85, 22, -31, -64, -64, 35, -45}
}
}
, {{{-6, -38, -36, -38, -13, 32, -36, -26, -26, -88, -20, -48, 1, -67, -25, 24, 3, 25, -39, 10, -17, 48, -4, -94, 8, -13, 7, 8, 32, -16, -2, 9}
, {-21, 7, -55, 36, 7, -61, 19, 25, -28, 8, 14, 39, -17, 44, 34, 17, -9, 66, -28, 50, 15, -5, 12, 80, -1, 51, -2, 22, 63, -23, -72, -59}
, {-89, 0, -103, 69, 16, -3, 6, 49, -20, 76, 12, 14, 62, 5, 22, -33, -6, -33, 16, -44, 31, 31, 32, -18, 52, 51, 28, -12, 25, -25, 29, -9}
}
, {{2, -20, 54, -20, -43, 76, -58, -18, 19, 2, 2, -25, -26, 13, -29, 40, 13, -12, 1, 49, -21, 20, -102, -16, -17, -65, 2, 64, -149, -29, -35, 19}
, {36, 0, 29, 42, 1, -16, 47, 30, 22, 18, 20, -37, 24, 76, 68, 7, 7, -36, 40, -8, -21, -28, 2, 80, -30, -8, 39, 3, -22, -56, -119, 24}
, {-30, -22, 10, 36, -14, 13, 52, -25, 59, 30, 50, -13, 34, 35, 105, 52, -26, -42, 12, -96, 35, 69, -16, -148, 56, -14, 26, 29, 31, 7, -23, -2}
}
, {{-5, -109, 14, -47, -48, 11, 26, -35, 3, -42, -103, 18, -273, 26, 96, -150, -73, 27, -102, 19, -131, -17, -28, 49, -12, -35, -47, -76, 56, -148, -127, -28}
, {13, -66, -10, -15, -98, 56, -12, -9, 6, 15, -22, -31, -89, 63, -4, -14, 27, 3, 43, 5, -175, 88, -35, 35, -94, 50, 26, 77, -4, -5, -42, 46}
, {-54, 58, -44, -169, -123, 32, -143, -192, 32, -83, -130, -67, -87, -103, 18, -90, -118, 11, -92, -94, -128, 32, -132, -52, -202, -125, 14, -19, -130, -62, -15, -35}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 256

typedef int16_t flatten_output_type[OUTPUT_DIM];

#if 0
void flatten(
  const number_t input[2][2][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten.h"
#include "number.h"
#endif

#define OUTPUT_DIM 256

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

static inline void flatten(
  const NUMBER_T input[2][2][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_H_
#define _DENSE_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 256
#define FC_UNITS 128

typedef int16_t dense_output_type[FC_UNITS];

#if 0
void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 256
#define FC_UNITS 128
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 256
#define FC_UNITS 128


const int16_t dense_bias[FC_UNITS] = {-15, 50, 67, 29, 1, 18, -54, -15, 29, 43, -14, 13, -22, -10, 16, 41, -26, 44, 48, 9, 10, 67, -6, 79, 45, 68, -4, 45, -1, 13, 25, 70, 42, 65, 60, 9, 37, 59, 23, 81, -3, 30, -7, 19, -25, 7, 45, 67, -17, -27, 71, 26, -34, 33, 15, 35, 4, -12, -6, -8, 11, -13, -8, 44, -24, 21, 76, 34, 0, -11, 27, 53, 60, -1, 37, 46, 31, 68, -22, 6, 10, 66, 7, 72, -53, 65, 50, -17, 44, 55, -1, -17, 65, 77, 23, -34, -18, 4, 31, -7, 23, 87, -16, 17, -4, -20, -2, -14, -23, -31, 63, 1, -12, 37, -27, -5, -1, 27, -6, 39, 18, -10, 44, 3, 11, -13, -4, -37}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{24, -40, -28, -84, 35, 0, -17, -43, 41, 105, -35, 0, 46, 54, 10, -34, -85, 54, 19, 51, 18, -62, -34, -32, -42, 111, -12, 40, -30, -36, -23, 55, -22, 40, 4, -2, -47, -24, 36, 27, 8, 39, -88, -48, -35, -93, 4, -48, 140, 6, 34, -1, -74, 5, -25, 9, -30, -64, 27, -5, 24, 42, -89, 102, 79, -51, -7, -49, -14, 49, -19, -6, 27, 26, 21, 43, 20, -5, -33, -34, 50, -42, 45, 6, 29, 20, 32, -23, -18, 49, 10, 69, -35, -31, -27, -2, 62, 80, 19, -27, -42, -28, -8, -7, 34, 127, -25, 82, 76, 89, 62, -68, -7, -7, 48, 82, 63, 59, -11, 23, 34, 22, -109, -24, 105, 47, -32, -25, 60, -6, -100, 36, -98, 46, 11, 29, -145, -76, 56, -75, 39, 82, -59, 64, -41, -31, 21, -8, -33, 14, -49, -16, 38, -71, -108, -5, 65, -1, 67, 59, -114, -2, -29, -110, -79, -7, -34, 120, -30, 103, -80, -93, 21, 2, -55, -2, -6, 14, 54, -13, -106, -40, 24, -109, -14, 30, -5, 52, -73, 31, 122, -23, 14, -106, -44, 57, -25, -14, -35, 43, -32, 13, 44, -158, -104, -29, -45, -7, -9, 40, 83, 2, -30, -26, 2, 24, -26, -55, -38, 12, -45, -82, -2, 66, 35, -30, 88, 109, 41, -28, 125, 42, -1, -45, 88, 45, -95, -37, -103, -50, 13, 25, 49, -41, 2, -16, -1, -46, -42, 15, 45, -11, 39, -53, 70, 44}
, {137, 56, 3, -101, -95, 83, 17, -34, -38, -132, -39, -133, -28, -58, -46, -105, 86, -41, -171, -20, 51, 3, 46, 41, -213, -119, -29, 67, 60, -56, 6, -45, 31, -20, 12, -3, -33, -71, 50, -52, -111, 70, 34, 30, 121, 41, 79, 29, -104, -97, -9, -64, -140, -72, -2, -60, 53, 17, 9, -83, 45, 35, 5, -105, -81, 21, -73, 79, -67, 90, -107, 61, -71, -37, -55, 66, 60, 0, 16, 63, -64, -29, 41, -35, -129, -116, 12, 43, -78, 67, -11, 118, 69, 29, 12, 9, -58, -74, -148, -153, 30, -5, -231, 9, -2, 13, -34, 10, 5, -70, 36, -2, 78, 22, -49, 2, -9, 59, -6, -13, 2, -34, -17, -6, -44, 104, 23, 72, -179, -76, -38, -13, 56, -118, 9, -7, 23, 4, -8, 28, -82, -44, 3, -10, -7, -108, -25, 9, 57, -40, 59, -41, -133, 1, 23, -47, 3, -21, -52, 6, 34, -3, 77, -3, 8, 45, 25, -26, 74, -29, 25, 32, 1, -2, 11, -1, -5, -57, 6, 9, -60, 112, -54, 14, -13, -59, 52, -40, -13, -59, -23, -11, 122, 30, 27, 28, -4, 30, -76, 19, -91, 23, 7, -72, 9, -40, -2, 19, -13, -26, 1, -35, -66, 47, -38, -43, 68, 47, -47, -66, 38, 12, 67, -81, -69, -33, -81, -49, -43, -63, -83, -36, 50, -41, 59, -53, 56, 78, -30, -24, 76, -58, -121, 15, -20, 11, -2, -20, -15, 27, -57, 52, -68, -10, -35, -26}
, {35, -8, -29, -38, -76, -5, 65, -197, -45, -67, 3, -27, -6, 43, -3, -65, -40, -34, -108, -125, 35, 49, 112, -25, 47, -19, -100, 77, 51, 25, -36, -90, -69, 35, -40, -43, -101, -102, 82, 9, -113, 23, 88, -52, 136, 7, 128, -16, -64, -75, -42, 11, 129, -84, -18, -29, 22, -18, -2, -74, 35, -44, 38, 24, -21, -58, 49, 23, -116, 98, 97, -13, -66, 17, 105, 47, -25, 58, -7, 75, -35, 68, 79, -22, -44, -13, -26, 22, -64, 77, -82, 134, 38, 4, -13, 20, -57, -20, 59, 44, -1, 13, -45, 13, -41, -4, -89, -9, -51, -73, -60, -51, 43, 130, -43, 76, 59, -38, -93, -62, -28, 4, 30, -10, 70, 71, 4, 83, -89, -72, 32, 17, 34, -87, -29, -48, 50, 9, -36, 87, -78, -115, 64, -32, 66, -61, -53, 11, 8, 62, 65, 12, -51, -26, 66, 20, -143, -27, -93, -86, 27, -16, 24, -1, 138, 24, -13, -73, 68, -141, -8, 49, 43, 13, 65, -6, -54, -91, -39, -28, 25, 135, 38, 32, -2, -36, 67, -63, -19, 17, -3, 25, -27, -5, -3, 28, 10, -37, -37, -60, -6, 37, -57, 32, 14, 37, 18, 74, -10, -100, -78, -46, -7, 29, -108, -19, 38, 60, -6, 70, 98, -63, 29, -29, -29, 25, -140, -100, -45, -95, -83, -13, 66, -31, 4, -28, 50, -9, 49, 23, -5, -30, -127, 21, 5, -6, 18, 45, 104, 8, -86, 27, -26, 106, -25, 12}
, {-98, -5, 4, -84, -57, -70, 34, 85, 1, 44, -92, 37, -60, -77, -22, 20, -36, -55, 4, -97, 56, -40, -25, 3, -197, 26, -18, 84, -7, -20, 31, 17, -11, 42, -52, -170, 3, 99, 21, -4, -147, 81, -100, -85, 12, -108, -123, 24, -15, 17, 69, 21, -162, -22, -96, -12, 28, 26, -19, -66, -11, -16, 42, 26, -163, 75, 114, -7, 78, -19, -161, -88, 44, 38, -99, -10, -18, -69, 51, 53, -9, -93, -70, 45, -52, -60, -109, 47, 0, -12, 25, -81, -114, -55, -14, 15, -48, 24, -209, -112, 22, 65, -144, -93, 15, 19, -5, 28, 15, -101, 18, -64, -34, -39, -112, 63, 4, -3, 24, 40, 34, 113, 81, 57, -63, -26, -28, -48, 121, -125, -83, -24, -105, -69, -44, 65, -44, 30, -36, -7, -114, -39, -82, -20, 82, -73, -18, -66, -61, -69, -92, 35, -14, -18, -59, -42, -67, -22, -41, -8, -33, -127, 28, 61, -32, 2, -35, 29, -22, -7, 9, 28, -8, -41, -104, -13, -16, 20, 104, -47, -68, 34, -23, -9, -110, 6, -12, -55, -32, 9, -38, 1, 8, 52, -6, -128, -34, 15, 14, -66, 21, 53, 28, -25, 68, 48, 27, -11, -78, -112, -24, 4, 40, -5, 35, -26, 11, -54, -129, -23, 16, 68, -77, 58, -50, 53, -40, -127, -53, -131, 59, -142, -22, -62, -22, 62, -7, 19, 3, -10, -61, -5, -33, -30, 87, -83, 27, -57, 72, -54, -5, -87, 36, -64, -90, 9}
, {-12, 5, -12, -13, 8, 76, -43, -51, -74, 126, 26, -28, -7, 111, 20, 0, 64, 48, 9, 26, 5, 91, -52, -60, 175, 16, 2, -41, -61, 35, 28, -2, -34, -8, -3, -3, 48, -32, -52, -1, -37, -78, 42, 11, -46, 88, 20, -94, 1, -26, -66, -75, 130, 20, 30, 43, 10, 32, -34, 37, -2, 10, -9, 26, 81, -78, -35, -6, -85, -7, 91, -44, -63, -8, 107, -111, -72, 95, 44, -92, -64, 76, -92, -7, 23, -24, 109, -127, 63, -89, -92, 83, 21, 6, -73, -24, -78, -51, 115, 144, -72, -70, 57, 6, 21, -44, 52, -25, 21, 62, -32, 10, -65, 90, 87, -37, 45, -13, -1, -34, -72, -70, 2, -4, 10, -89, 19, 73, 66, -44, 27, -133, 24, 36, 13, 5, 36, -7, -28, 24, 10, -4, 83, -58, 30, -27, 2, 2, 60, 15, 11, -66, -25, -34, 14, 55, 88, 39, -7, -27, 14, 56, -44, 0, -25, -41, 80, -100, 59, 32, -23, -10, 71, 41, 43, 84, -3, -5, -1, 11, -2, -59, 7, -45, 162, 127, -32, -68, 41, 6, -4, 1, -1, 5, -25, -11, -58, 16, 31, -42, 29, -25, 27, 4, 35, 10, 47, 47, 77, 124, 35, -36, -23, 60, 39, -49, -10, 1, 30, 31, 45, 7, 99, 12, 6, 44, -103, -91, 12, 33, 9, 64, -69, 114, 76, -70, -32, -55, 5, 59, -39, -41, -116, 47, -37, -96, -10, -78, 23, -44, -27, 127, 12, -1, 47, 40}
, {26, -30, 37, 74, 13, -24, -42, 121, -35, 42, 30, 17, 33, 105, -73, 91, 15, 60, 25, 38, -51, 50, 3, -12, 42, 85, 34, -14, -53, 66, 26, -51, 45, -33, -20, -32, 50, 58, -80, 55, -1, 27, 10, -37, 2, 6, -30, -89, 67, 27, 53, 60, 50, 9, -83, -20, 31, -104, 13, 20, -66, 83, -40, 96, 37, -43, -54, 26, -46, 100, 58, 137, -39, -13, 13, -4, 3, 77, 46, 43, -46, 44, 0, 0, 41, -15, -93, 59, -1, -6, 0, -63, -2, 14, -47, 49, 16, 102, -72, 76, 1, 22, 48, 151, -20, 8, 12, -45, 0, 32, 86, -49, -42, 48, -1, 34, 10, -14, 19, -19, 78, 20, -9, 55, 50, 34, -15, 48, 45, 33, -59, -12, -69, 123, 96, 40, 9, -172, -94, -92, -7, -58, -1, -90, -13, -16, -49, 17, 64, 5, -23, 66, -129, -57, -54, -12, 24, -52, 14, 100, -73, -36, -16, -29, -91, -76, 51, 71, -7, 180, -22, -28, -96, 51, -81, 17, -85, -50, 17, 93, -93, 47, 13, -15, 17, 194, 17, 24, 100, -93, 41, -82, 52, 59, 13, 14, -120, -22, -49, 1, -29, 90, -10, 33, 24, -88, 52, 2, -30, 14, 109, -15, 24, -20, 67, 16, 101, 9, -36, 0, 96, 4, 82, 74, -3, 24, 33, -13, 28, 74, -2, -19, -17, -75, -44, -195, 76, 32, 26, 21, 25, -52, -58, 92, 42, -22, -18, -43, -23, -2, -14, 33, -34, 101, -5, -25}
, {-31, -103, -14, -51, 7, 28, -8, -22, -37, -11, -104, 26, 12, 48, 38, 25, 46, 61, 3, 58, -55, 28, -52, 26, 116, 6, -21, 57, -13, 11, 3, 52, -8, 43, -52, -46, -36, 28, 47, 68, -80, 41, -84, -8, -43, -1, 24, -41, -8, 28, 11, -14, -81, 13, -37, 8, 11, -23, 51, -69, -15, 56, -39, 74, 56, 28, -58, -56, -61, 32, -55, -120, 36, -9, -44, -27, -26, 70, -50, -12, -31, -66, 19, -4, 6, 47, -17, 78, 94, 2, -53, 23, 37, -46, 13, 85, -13, 78, 30, 37, -38, 27, -12, -78, 2, 68, 46, 38, 9, -26, 36, -44, -27, -54, -74, 9, 53, -73, 63, -38, 24, 47, 6, 28, 104, -3, -113, 79, 54, 22, -123, -100, -72, 101, -29, 49, -155, -118, 1, -61, -9, 96, 31, 13, 33, 24, -30, 2, 48, -92, 32, 42, -119, -74, -12, -178, -59, 20, 78, 83, -95, -32, -9, -8, -159, 17, 10, 40, -95, 123, -119, 81, -158, 27, -75, -34, 13, -58, -25, -26, -192, -78, 42, -111, 4, 151, 9, 51, 54, 2, -49, -80, -11, 10, 29, -61, -84, -21, 30, -4, -32, -88, 75, -27, -40, 64, -148, -35, 118, 101, 104, -102, 71, 54, 27, -95, 31, -113, 67, -44, -71, -4, -21, 122, 2, -37, 70, 26, 15, -8, 101, 137, -172, -16, 88, -96, -44, -21, -3, 14, 24, 24, -28, 111, -33, -38, -169, -150, 18, 37, 101, 14, 96, 41, -38, 53}
, {8, -44, -64, -14, 92, 27, 1, 15, -48, 21, 32, 17, -53, 86, -105, 7, 8, -36, 102, 6, -64, -14, -48, -93, 151, 96, 8, -40, -88, 14, -25, -21, -12, -47, 52, 6, 104, 111, 14, 35, -3, 34, -38, 100, -57, -9, -54, -18, 101, 30, 51, -79, 94, 36, 160, 77, -80, -81, -55, 63, 3, 64, -3, 60, 107, -63, -116, -6, -20, 126, 130, 120, -81, 31, 87, -24, -24, 107, 23, 15, 18, 58, 38, -14, -39, 83, 40, -28, -51, 58, -74, -28, 30, 54, -35, -28, -107, 92, 121, 145, -50, -32, 51, 151, -80, 14, -36, 19, -14, 59, -12, -39, 33, -20, 123, -48, -21, -37, -57, 46, 8, -34, 64, -3, 58, -13, 15, -1, 8, -38, -123, 16, -48, 61, -49, 36, 19, -65, -59, 20, 79, 29, -58, 25, -44, -33, -4, 52, -88, -60, -70, -44, 60, 41, -10, -70, -28, 0, 56, 47, 64, -34, -23, -43, -24, -27, -43, 39, 44, 77, -52, 38, -15, -74, -15, -13, 12, -24, 62, 24, -6, -10, -72, -73, -52, 83, 95, 107, -83, 41, 97, 13, 28, 18, 27, 11, 40, -35, -22, -78, -3, 74, 3, 15, -69, 16, 48, -12, -2, 50, -4, 5, -26, 40, 77, 2, 57, 31, 21, 12, -12, 11, 125, -29, 74, 51, -10, -53, 203, 46, 12, 28, 38, -21, 31, 29, -57, 23, -53, -34, 49, -26, 11, 18, 7, -4, -24, 9, 34, 14, 52, 59, -20, 28, -6, 30}
, {46, -75, -71, 25, -20, 13, -8, -14, -52, -51, 100, 2, -144, 22, -31, -72, -4, -17, -2, -67, 60, -12, 41, -74, 90, -8, -115, -59, -106, 4, -53, -121, -58, -8, 83, 24, -59, -10, -2, -3, -49, -101, 57, -11, 8, 33, 42, -113, -44, -19, -43, -120, 123, 1, 128, -35, 16, 82, -9, 0, 0, -41, -25, -63, 36, -53, -90, -26, -94, 69, 139, -34, -83, -17, 118, -59, -33, 36, 14, 51, 28, 107, 44, 7, -15, -28, -6, -65, -26, 35, -39, 61, 61, 101, -38, 3, -101, -31, 133, 76, -135, 16, 88, 60, 24, 55, -33, -52, -76, -43, -71, 4, -23, 121, -21, 4, 132, -47, -42, -98, -34, -108, -26, -36, -1, -25, -39, 115, 95, 57, 42, 18, -9, -24, -89, -28, 2, 4, 49, 5, 19, -101, -46, 2, 10, 27, 39, -70, 58, 39, 44, 37, -27, 16, 48, 50, 128, 33, 33, 29, 82, -11, -23, 12, 62, 17, -2, -18, 10, 84, 12, -34, -33, 6, -2, -11, -75, -16, 2, 57, 31, 69, 2, 38, 60, -14, 54, -69, -7, 26, -36, -38, -28, 16, 6, -8, 66, -86, 4, -16, -3, -25, -117, 136, 65, 5, 7, -3, 28, -43, 16, 10, -36, 25, -56, 36, 10, 24, 40, -8, -34, -32, -58, 66, 114, -43, -70, -53, 76, 41, -132, -3, 23, -74, -35, -43, 16, -28, -27, 104, -43, 38, -57, 29, -59, -14, -47, 9, 48, 3, -1, -21, -57, 39, -65, 87}
, {-55, -13, -31, 79, 18, -57, 45, -2, -48, 55, -120, -56, 4, 6, -12, -16, 12, -78, -40, 86, -18, 33, 108, 45, 21, 48, 83, -44, -23, -16, 18, -36, -44, -82, -59, -52, -55, -40, -38, 39, 57, 40, -20, 1, -21, 31, -22, 54, -46, -33, -15, 32, 4, 34, 12, 5, 12, 23, 46, -82, -16, -62, 79, -33, -64, -50, 30, -5, 5, 3, -4, -67, -67, 58, -149, -1, -54, -9, 1, 12, 75, -107, -42, 36, 95, -45, -37, -57, 51, -15, -3, -110, -9, -12, 31, 19, -2, -115, 90, 45, -1, -25, 85, -69, 2, 52, 44, 68, 18, 21, -52, 31, -66, -123, 88, 19, -38, 92, 47, -2, -128, -62, -16, 23, -78, -110, 51, -135, -43, 183, 34, -26, 16, 57, 38, -126, -7, 17, 31, -58, 79, -91, 61, -51, -48, -10, -11, -39, 82, 49, 31, -74, 28, -85, 26, 80, 173, -33, 11, 120, 17, 99, 80, -53, 21, 16, 34, -42, -72, 19, 25, -20, 64, -71, 47, 87, -155, -36, -129, 43, 30, -70, 92, -60, 76, -2, 39, -61, 46, -61, 47, 60, 11, -61, -76, 40, -11, -26, 70, 89, -33, 78, -69, 88, -12, -25, 52, -15, 72, 5, -26, -93, -30, -60, -131, -21, 24, 67, 67, 43, 100, -72, -91, -56, 21, -97, 41, -42, 11, 45, -24, 63, -46, 25, -74, -33, 0, -73, 53, 43, 17, 2, 42, 86, 0, -60, -54, 30, -131, -65, -68, 57, -25, 34, -11, -7}
, {26, -42, -57, 60, 15, 125, -2, 66, -16, -16, 116, 0, -15, 85, -19, 53, -38, 29, 11, -41, -99, 23, -55, -34, 8, 53, -51, 101, 1, 100, -58, 53, -39, 29, -43, -76, 5, -61, -4, 49, -97, 39, -8, -24, -74, 15, -64, 1, 16, 42, -47, -20, -23, -82, 26, -133, 9, -108, -59, -44, -19, 15, -79, 46, 10, -73, -33, 11, -44, 94, 63, 55, -62, 78, 70, -3, 24, 123, -43, -34, -15, 33, 43, -16, -80, 2, -6, -64, 97, -31, -98, 71, -70, 27, -94, 90, -70, 12, 25, -12, -14, 48, -13, 58, -12, -38, -28, -52, -61, -42, -5, -94, -47, 124, -27, -90, 74, -44, 28, -63, -21, -45, -48, 66, -55, 15, -94, 115, 10, -19, 1, -22, -56, -7, 0, -93, 53, -107, -4, -56, -32, 23, 76, -19, -33, -33, -16, -6, 100, -27, 14, 27, -69, -157, 3, -5, 38, -107, 47, 37, 87, -16, 5, 20, 3, -106, 45, -5, -50, 96, 69, -41, 16, -14, 8, 79, -88, -60, -30, 28, 19, -83, -32, -54, 7, 106, -44, -159, 44, -113, 85, -65, 165, 26, 1, 13, 36, -47, 35, 25, 45, 26, 38, -22, 100, -21, -34, -29, 32, 44, 67, 91, -38, 114, 31, 29, 15, -78, 107, -77, -8, -31, 87, 6, 12, -87, -1, -35, -15, -36, 7, 65, 10, 12, 58, -83, -56, -87, -38, 37, 37, -58, -77, 36, -56, -2, -27, 101, -42, 29, -9, 53, -138, 18, 18, -33}
, {30, -81, -83, -137, -16, 31, 7, 21, -21, -29, 5, -53, -124, 17, -5, 12, -90, 13, -67, -87, 57, -39, -22, -25, -6, -94, -92, -38, -42, 15, -21, -44, -97, 37, 28, -97, -43, -63, 21, 66, -78, -2, -98, -9, -39, 65, -17, -18, -57, -56, -84, -69, -36, -84, -52, -106, -24, 61, -16, -36, 28, -27, -46, -39, 57, 2, 42, 19, 24, 60, 4, -40, -92, 53, 122, -31, 23, 49, -52, 16, -5, 62, 15, -13, -51, 74, 57, 35, -9, -16, -70, 88, 10, 52, 16, -55, -62, -40, 13, 13, -95, -65, -55, -47, 15, -19, -48, 31, 4, -39, -86, -53, -2, -21, -61, -38, 44, 68, -46, -38, 16, 45, -8, 25, -48, 40, -83, 40, -74, -93, 38, -114, 11, -63, -129, -40, 78, 44, -70, -66, 13, 37, -29, -72, 13, 64, 13, 5, 17, 57, -8, 65, -3, 33, 16, 34, 8, -6, -28, 51, 27, -108, -48, 22, 11, -6, -23, -79, -15, 61, 13, 11, -23, -14, 18, 69, -11, 11, -47, 14, 51, -1, 34, 29, -55, -128, 30, -99, -8, -68, -92, 23, -30, 66, 23, -45, 21, -33, 11, -115, 2, -64, -7, 77, 23, 69, -101, 14, 78, -31, 3, -60, 11, 92, -94, 55, -2, 35, 11, 42, -21, 40, 48, 19, -12, 11, -124, -50, -24, 7, -133, -77, -2, -48, 23, 31, 7, 16, 81, -90, -42, -59, -32, 3, 2, -12, 22, 62, 41, -15, -16, 38, 36, -52, 13, 64}
, {11, -5, -48, -55, -84, 58, -45, -32, -24, -9, 48, -56, -73, 2, 61, -123, 26, 28, -67, -108, 75, 44, 74, -32, 77, -64, -17, -73, 64, -64, -9, 28, 37, -52, -13, 108, 77, -52, 52, 46, -41, 49, 67, 13, 97, -14, -10, 101, -123, 26, 7, -45, -40, 0, -27, -10, 40, 88, 36, 119, 20, 38, 66, -91, -104, 21, -39, 61, -77, -17, 53, 46, -63, 27, 50, 56, -8, -58, -69, 45, 128, -44, 0, -36, 83, -139, 30, -4, -65, 1, 50, 0, 3, -41, 42, 6, -10, -113, 39, 13, -70, -6, -85, 18, -17, 66, -51, -62, -36, 56, -55, 21, 31, 106, -15, 16, 24, 26, -90, 17, -22, -67, -93, -66, 48, 42, -69, 9, -109, 1, -7, 18, 43, -11, -89, 31, 29, -25, -38, 45, -27, 98, -165, 75, 26, -30, 88, -36, -70, 5, -62, 82, 36, 2, 16, -21, -28, 18, 20, 191, 10, -67, -33, -55, 99, 45, -115, 55, 43, 8, -112, -34, -87, -40, -46, -85, 74, 93, -57, -15, -67, 24, -80, -73, -143, 30, -7, 78, -65, 20, -72, 115, 23, -84, -112, -6, -9, 0, -2, 13, -1, -109, 5, 8, -33, 55, 1, -52, 108, -1, 101, -81, 181, 45, -109, -14, 64, -2, 4, 51, -143, 34, -8, -14, 126, 70, 20, 42, 7, 63, 24, 15, -51, -53, -36, -16, -89, 23, 52, -103, 24, 41, 36, 29, -74, 24, 16, 16, -24, -37, -30, -17, 130, 35, 9, 84}
, {59, 33, -25, 25, -31, -42, -32, -23, 60, -80, -3, 80, -20, -21, 15, -49, 71, -6, 22, 27, 9, -22, 3, -4, -181, 33, 9, 9, 2, -31, 14, -72, 47, -79, 44, 18, 15, -70, 40, 35, -51, 16, 118, -22, 13, 6, 26, 36, 76, 71, 86, 16, 22, 53, -3, -48, -28, 45, -3, 4, -21, 70, -6, -21, 100, -39, -77, 27, -201, 168, 43, 92, -62, -42, 33, 28, -29, 76, -51, -12, 1, -11, 40, -122, -90, -17, 5, 17, -105, 39, 43, 29, 40, -49, 24, -27, -54, -13, -53, -79, -70, 40, 50, 107, -72, 63, 14, -180, -4, 27, 35, -45, 29, 23, 32, 39, -31, -32, -101, -9, 41, -91, 19, -64, -62, 73, -73, 74, -13, -15, -43, 29, -10, 61, -113, 56, -79, 22, 23, 58, 36, -7, -88, 73, -34, 16, 59, 25, -224, -50, -122, -52, -98, 36, 5, -25, -10, 21, 1, 62, -84, -84, 18, -15, -26, 47, -21, 76, -46, 27, -76, -78, -94, -121, -86, -67, 53, 65, 94, 35, -140, -21, -53, -70, -30, 9, -44, 53, -121, 33, 75, 4, -16, -122, -44, 12, 47, 30, -124, 38, -62, 76, 23, -22, -150, 5, -57, -28, -32, 34, 4, -23, -16, -119, 91, -73, 63, 64, -5, -232, -31, -83, 26, -48, -51, -124, 6, 18, -15, 17, -44, -2, 43, -17, 118, -40, -9, 22, -114, -28, 50, -83, 14, -101, -1, 18, -27, 37, -134, 58, 4, 47, -111, -14, -33, -73}
, {-51, -68, 12, -2, -35, 63, 29, -26, -29, 92, -56, 23, 38, 64, -25, 74, 5, 0, -24, -12, -145, -64, -105, -10, 113, 5, -8, 22, -28, 15, 7, 7, -51, 41, -42, -139, -25, -23, -4, 43, 29, -26, -171, 64, -69, -46, 69, -35, 8, -17, 64, 21, 20, -54, -53, 32, -6, -8, 29, -26, -66, -59, -137, -5, 128, 32, -6, -11, -2, -45, -25, -36, 17, -61, -60, 20, 50, -77, 28, -96, 30, 25, -65, 51, -32, 110, 77, 45, 21, -56, -24, 2, 0, 35, 63, 11, 10, 48, -48, -66, 7, -19, 152, -84, -23, 2, -30, 44, -58, -7, 30, 65, -158, 25, -75, 58, 55, -41, 53, -5, -29, -17, 15, 5, -17, -37, 19, -20, 52, 6, -41, 39, 0, 56, -107, 37, -139, 114, 15, 29, 3, -51, -76, 7, -82, 1, 51, -58, -123, -111, -25, -148, 92, 29, 10, -3, -7, 33, -29, -41, -47, -91, -67, -21, -54, -11, -3, -44, -65, 15, 26, -35, -3, -160, -68, 78, -6, 44, 122, -10, 66, 23, -106, -17, -9, -37, 49, 104, -109, -10, 17, 23, -95, -113, 31, 33, 56, -31, 22, 4, 20, 100, -127, 61, -47, -14, 45, -11, -145, -161, -117, 12, -105, -86, -46, 40, 0, 28, -50, -76, -23, 4, -118, 22, -51, -47, -21, 11, 16, -65, 12, 5, 14, 23, -29, 110, -87, 15, -14, -65, 22, 26, 82, -89, -30, -65, -9, -15, -48, -23, 37, -73, -36, 30, 96, -101}
, {52, -36, -35, -1, -68, 55, 6, 32, -22, -124, 28, -64, 97, 106, -42, 24, 73, -37, -23, -56, -12, 64, -38, 57, -140, -25, -8, 36, 26, 46, 38, 86, -10, 15, -41, -60, -35, -23, 8, -54, -56, 52, -70, -122, 16, 3, -84, -31, 4, 24, 27, 26, -77, -41, -126, 8, 16, -49, 18, -84, -4, 4, 2, -30, -129, -21, 3, 4, 80, -45, -145, -82, -29, 70, -105, -80, 2, -6, 21, 60, -12, -56, -81, 109, 8, -76, -97, 66, 70, -57, -5, -164, -87, -12, 7, -4, -9, -27, -175, -94, 44, -5, -152, -79, 75, -40, 38, 87, -50, 14, -37, 29, -104, -146, -37, -55, -68, 91, 89, 29, -133, 87, 15, -2, -194, -113, 23, -78, -43, 96, -3, 44, -82, -144, 67, -38, 22, 112, -36, 20, -9, -7, -35, -9, -37, -69, -71, -34, -13, 36, -9, 12, -30, -40, -1, -2, 98, -81, -18, -22, 1, 35, -5, -69, 90, -30, -27, -24, -42, 42, -5, -56, 38, -86, -46, 38, -48, -6, -17, 0, -62, 21, -29, 27, 40, -70, 19, -17, 0, -49, 16, -18, 24, 8, 64, 19, 1, 94, -134, -22, -62, 91, 48, 4, 70, -63, 78, 46, 74, 17, -83, -18, 83, 125, -82, -115, -7, -46, -38, -4, -37, 21, -19, -96, -55, 10, -113, -123, -47, 26, -63, 39, 24, 28, -54, 109, 54, 0, 35, -4, -40, -101, -47, -8, 62, -45, 27, 65, -29, -30, -120, 26, 17, 22, -117, -53}
, {-170, -18, 27, 69, 30, -111, -79, 59, -1, 68, -42, 127, 25, -42, -35, 93, 18, -12, 5, -34, -83, -23, -19, 28, 117, 132, -16, -80, -179, 86, 30, -16, 17, -45, 42, -82, 50, 89, -48, -3, 105, -2, -112, -24, -141, 2, -88, 1, 99, 47, 35, 12, 90, 2, -6, 61, -76, -25, -96, 14, -93, 8, -45, 27, -70, -12, -53, -3, 10, -120, 8, -60, 3, 12, 12, -93, -52, -91, 89, -58, -27, 79, -182, 12, 40, 82, -5, -30, 92, -27, -37, -71, -59, 13, 39, -34, 68, 118, 59, 67, -48, -32, 15, -122, -32, -23, 28, 29, -54, -40, 9, 15, -201, -175, 69, -47, -34, -2, 106, -15, 66, -10, -20, -30, 49, -111, 21, -83, 25, 40, 13, 23, -58, -79, -61, 9, 16, -76, -13, 11, 133, 78, -100, 54, -32, -45, 19, -22, -36, -32, -63, -14, -40, 55, -28, -21, 14, 65, 16, 19, -23, -88, -64, 19, 13, 55, -38, -19, -18, 60, -41, -8, -79, 61, -75, -35, 1, 30, 73, 65, -4, -45, -97, -32, -24, -75, 41, 3, -116, -16, -22, -18, 28, 18, 31, -50, 36, -31, -39, -91, 69, 25, -51, 141, 29, -72, 20, 43, -23, -110, 73, -1, -9, -58, -135, -28, -37, -68, 36, 13, -96, -4, 13, 70, 110, 32, -51, -68, 37, 6, 8, 58, -24, -93, -18, -36, 55, -94, 59, -17, -43, 35, -68, 139, -25, 23, -7, -17, 68, 52, 68, 75, 13, -5, -48, 60}
, {-52, 46, -52, -71, -59, 48, 32, -52, -20, -177, -85, -59, 58, 23, -63, 3, 22, -69, -10, -93, 25, 4, 40, -9, -233, 10, 1, 74, 64, -3, 32, -22, 12, -21, -11, -55, -69, -32, 15, 35, -93, -5, -20, -66, 89, 14, 34, -3, -16, -69, -16, 11, -171, -2, -21, -23, 48, -17, 34, -66, 44, -57, 21, -10, -164, 80, -16, 40, 14, -21, -120, -16, -37, 28, -77, 54, 20, 49, -16, 8, -109, -89, 94, 76, -131, -77, -30, 30, -31, -35, 2, 91, 21, -48, 53, 57, -47, -34, -97, -134, 26, 30, -210, -25, 44, -16, -4, -15, -24, -139, 56, 28, -15, 124, -92, 29, -71, 34, 67, 27, -36, 62, -29, -15, -97, 56, -35, -1, -155, -14, -31, 127, 61, -79, -35, 17, -14, -12, -50, 19, -90, -37, 48, 6, -52, -140, 44, -110, -140, -151, -86, -26, -73, 48, 25, -17, -52, -5, -107, 18, 1, -74, 42, 3, 115, 73, -19, 7, 34, -18, -70, -25, -94, -6, -103, 5, 53, -2, 85, -81, -107, 5, -89, 42, -59, -77, 61, -3, -84, -4, 32, 5, 24, -9, 22, -45, 34, 53, -172, 3, -18, 60, -12, -38, 54, 13, 41, 34, 28, -37, -79, 35, -7, -61, -88, 21, 81, -70, -49, -25, 5, 9, -72, -152, -83, -40, -144, -61, -24, -13, -91, -115, 75, -121, -46, 48, -5, 34, -58, -121, 8, -64, -32, -109, 38, 17, 100, -36, -58, 48, -107, -23, -28, 12, -1, 8}
, {-5, 16, -57, -89, -28, -31, -7, 43, -34, -83, -163, 72, 9, -57, 3, 91, 18, -91, -31, -122, -35, 12, -28, 59, -158, -17, 35, 21, 27, -57, 34, -13, -44, 24, -3, -90, 45, -70, -18, 35, -119, 64, -137, 3, 50, -41, -36, -56, 70, -26, -12, 14, -220, -30, -143, -48, 22, -1, 4, -150, 3, -35, -76, 45, -83, 4, -10, -64, 24, -117, -230, -136, 15, 81, -94, -59, -21, 33, 3, -44, -56, -54, -132, 110, -105, 100, -95, 38, 105, -34, -73, -44, -112, -62, 27, 6, -7, -28, -186, -143, -20, 37, -171, -44, 74, -56, 15, 64, -58, -90, 14, 58, -114, -22, -118, -28, -21, 18, 90, 68, 78, 68, 0, -27, -87, -30, -42, 38, -30, -142, -14, 20, -45, -14, 21, 83, -45, -38, -62, -21, -54, -61, -84, -18, 32, -67, -17, 3, -100, -38, -54, -9, 9, 23, -23, -9, -48, -87, -30, 13, 14, -58, 20, 74, 45, 80, 34, 38, -4, 0, -49, -12, -48, 9, -117, -76, -30, 50, 88, 0, -90, 54, -36, 27, -86, -127, 3, 29, -63, 8, -19, -57, -3, 63, -53, -15, -43, 33, -129, -156, -62, 64, -7, -10, 30, 27, 60, 47, -87, -132, -46, 45, -67, -20, -52, -73, 50, -64, -107, -19, -52, 32, -88, -44, -81, 38, -121, -68, -95, -61, -67, -99, 34, -105, 37, 65, -8, 59, 10, -82, 28, -10, -33, -60, 19, -42, 8, 17, 25, 4, -36, -82, 2, -51, -28, 36}
, {-8, -37, -85, -25, -52, 55, 64, -58, -63, -22, -49, -85, -19, 7, -53, 46, -11, 50, -60, -21, 21, -75, -27, 13, 201, -138, -143, 0, -65, 14, -3, -41, -77, 32, -13, -21, -49, -56, 17, 22, -48, -46, 5, 40, 50, 46, 60, 42, -13, -23, -3, -110, 87, -71, 114, -66, -58, 50, -28, 30, 11, 4, 24, -24, 17, -78, -43, 1, -48, 105, 67, -5, -74, -36, 139, -46, -41, 111, -35, -18, -23, 24, -31, -89, 117, 11, 103, -43, 41, -29, -16, 90, 30, 30, -27, -60, -19, -30, 111, 169, -36, -24, 122, 21, -68, -58, -17, -92, -77, 37, -33, -29, -82, -13, 9, -66, 58, 14, 28, -107, -23, -77, -65, -15, 49, 11, -87, 2, 50, 15, 43, -54, 46, -3, -13, -133, -4, 24, 17, 35, 61, -4, -7, 1, 31, 46, 42, 17, 85, 119, -3, 46, 2, -52, 54, 43, 53, 6, -18, -18, 12, -15, -41, 26, 37, -2, -67, -163, -83, -40, 85, -14, 27, -20, 73, -34, -20, -60, -60, 4, 59, 7, -26, -19, -9, -58, -44, -138, 30, -43, 21, -39, -33, 11, -21, -24, 37, -87, 21, 5, 85, 19, -170, 53, 4, 44, -52, -40, 30, -78, 22, 7, 13, 21, -51, 23, -62, 33, 71, 19, 84, -35, 27, 65, 169, 10, -25, -81, -23, 20, -14, 38, -13, -34, 2, -27, 16, -48, 18, 38, -26, 36, -87, 42, -155, -24, -21, 11, 44, -1, 26, -42, 19, -8, -2, 84}
, {120, 19, -12, -21, -65, 109, 3, -52, 5, 41, 21, 1, -65, 133, -73, -68, -163, 27, -25, 12, 26, -5, -2, -4, 185, -110, -125, 14, -22, -18, -83, 84, -19, 92, -24, -29, -103, -48, 37, 43, -119, -73, 61, -4, 34, 37, 80, 4, 30, -45, -92, -47, 116, -101, 82, -39, 29, 59, 36, -5, -29, -32, -58, 75, 54, 9, -50, -32, -7, 41, 31, 34, -64, -3, 115, -34, -3, 3, -53, -38, 124, 43, 21, -18, 103, 67, 44, -26, 31, -68, -69, 10, -54, 65, -64, -42, 27, -17, 43, 126, -22, 16, 71, 34, 3, -78, -31, 23, -100, 49, -4, -26, -45, 48, -53, -24, 85, -2, -85, 21, -29, -54, 19, 107, -30, -3, -55, 69, 133, -6, 76, -44, -42, 13, -93, -112, 74, 86, 2, -76, -19, -68, 18, -80, 39, 120, -54, -62, 62, 52, 15, 25, 52, -47, 13, 10, 60, -94, -29, -17, 14, 27, -93, 50, 42, -40, -63, -64, -77, -64, 4, -49, 43, 28, 96, -13, -136, -11, -24, -17, 96, -13, -49, 44, 31, -52, -202, 69, -6, -43, -55, 16, -43, -14, -51, 28, -36, -202, 71, 16, 111, -25, -13, -19, 17, -19, -8, 0, -92, -22, -98, 36, -3, -67, -56, 22, -59, -58, 43, 21, 60, -154, 12, 57, 46, -23, 19, 6, -34, -25, 60, 23, -12, -48, 17, 28, -25, -40, -75, 104, -77, 16, 3, -55, -37, 1, -33, -28, 45, -32, 10, -93, 57, -123, 35, 105}
, {10, -23, -31, 6, 17, -37, -32, 9, -41, 51, -29, -56, 53, 10, 20, -44, 49, -17, -26, 45, -87, -3, -11, -27, 73, -11, -21, -51, -99, 6, -16, -82, -36, -48, 15, -2, -69, -52, 6, -30, 85, -46, -18, 62, -52, 6, -4, -100, 7, 24, -31, -6, 86, 1, 8, 11, -62, -47, -1, -80, -85, -35, -51, -131, 71, 20, -24, -70, 15, -110, 69, 3, 32, -24, -92, -96, -15, -78, 34, -59, -90, 63, -95, 104, 52, 101, -16, -69, 43, -106, -83, -15, 21, -66, -62, -17, -5, -11, 59, 98, 6, -137, 88, -88, -25, -110, 36, -44, -102, 28, -3, 5, -177, -155, 124, -32, -33, -5, 77, 20, -65, -41, 27, -56, -5, -79, 24, -20, 55, 4, -59, -32, 13, -84, 27, -51, -12, 64, -23, 79, 56, -48, -71, 18, -67, -8, -20, -27, -55, -15, 28, -36, -20, -33, -10, 26, -5, 95, -8, -3, 42, 13, 51, -85, 14, 21, 2, 62, 38, -16, -57, 81, 34, 11, 2, -27, 39, 14, 82, -65, -16, 28, 80, 59, 47, -22, 128, 5, -17, -7, -10, 50, -69, -13, 58, 3, 73, 38, -53, 11, -3, 59, -93, 45, -20, -16, 11, 48, -90, -78, -67, 45, -90, -16, -34, -48, 13, -24, -28, 29, -98, 36, -24, 70, -32, -49, -9, 13, -41, -26, -149, -37, 105, 44, -5, 121, -52, 60, -42, -37, -26, -20, -49, -55, 13, 42, -21, 25, 67, -10, 43, 7, 51, 4, 71, 71}
, {17, -35, -15, 60, 16, 134, 7, -10, -50, -128, 39, -36, 8, 133, -103, 38, 25, 70, 7, -46, -64, 37, -64, -31, -43, -5, 27, -10, 19, -2, 15, -46, -33, -48, -17, -63, 0, -113, -59, 75, -84, 49, -27, -21, -100, -10, -68, -81, 20, -28, 61, 28, -19, -4, 14, -61, -3, -42, -21, 8, -3, 67, 72, 49, -94, 25, -17, 12, 25, -15, -53, -31, 33, 110, -121, -56, -3, -34, -30, -12, 85, -66, -15, 47, -64, 65, -72, -29, 77, -23, -33, -4, -58, 93, -21, -24, -12, -44, -64, 6, 18, -38, -160, -71, 29, -64, -65, 55, -21, 18, -3, 5, 13, -2, -11, -57, 47, -4, -7, 43, -81, -21, -70, 47, -172, -74, 15, -68, -55, 24, 15, -58, -76, 30, -17, -3, 60, -70, 60, -88, 72, -61, 64, 9, -68, -48, 45, -33, -9, -41, 0, 9, -26, -118, -8, -9, 31, -64, 75, 40, 8, 5, -42, 20, 21, -59, 41, -28, -76, 5, -58, -53, 18, -16, 53, 6, -43, -143, -9, 48, 57, -67, -63, -49, 56, 79, 66, 0, -12, -7, 63, -90, 59, 10, 59, 25, -8, -34, 26, 24, 77, -21, 52, -13, -10, -149, 28, -1, -26, 58, -81, 72, -1, 60, 11, 52, -159, -87, 18, 34, 2, -138, -2, 24, 87, -11, 120, 81, 67, 10, 148, 74, -49, 34, -111, 9, -5, -85, -29, 52, -64, -41, -65, 9, 25, -23, -60, 23, -50, 1, 3, 57, 53, -42, -65, -87}
, {-92, -52, 31, -89, 13, -5, -10, -56, 33, -154, -80, -48, 101, -67, -59, 49, 11, -31, -57, 29, 25, 14, 77, 3, -163, 26, 42, -1, 48, 38, -82, -46, 0, -24, -8, -42, -63, -73, 17, 36, -46, 61, 70, -1, -6, 87, -14, -53, -25, 8, -43, 16, -66, 3, -83, 29, 38, 19, -9, -54, 12, -3, 6, -58, -156, 26, 10, -44, -31, -35, -82, -47, 45, -41, -121, 76, 59, 12, 40, 65, 4, -53, -4, 61, -135, -43, -150, 8, -17, 6, -17, 63, 22, -84, 24, 94, 23, -46, -130, -201, 1, -57, -208, -50, 29, 58, 27, 2, -51, -96, -4, 26, 0, 90, -149, 25, -93, -22, 36, 31, -37, -4, 10, -25, -125, -22, 12, -78, -179, -75, 1, 58, 2, -86, -35, 28, -26, -16, -8, 45, -67, -103, 0, 50, -10, -42, 14, -29, -71, -60, -22, -88, -51, 21, -47, -1, -31, -47, -50, -30, -33, 60, 82, -32, 88, 26, -28, -9, 75, -34, -116, -19, -10, -85, -99, -2, 4, 17, 45, -18, -144, 24, -14, 27, 24, -92, 105, -22, -34, 61, 28, 70, -10, 15, -54, 13, 121, 48, -91, -4, 21, 91, -2, -14, 25, -4, 161, -12, 50, -103, -69, -12, 27, -33, -109, -34, 81, -8, -82, -32, 18, 62, 7, -103, -112, -21, -10, 36, -25, -35, -108, -36, 49, -70, 12, 62, -24, 14, 37, -108, 54, -15, -21, -71, 25, -19, 55, 148, -49, 70, -40, 73, -46, 31, -4, -39}
, {38, -2, -127, -80, -114, 13, 41, -135, -94, -64, 24, -123, -83, -9, 31, -127, -9, 4, -5, -47, -17, 1, -59, 8, 73, -158, -155, 36, 46, -78, -116, 33, -29, 99, 32, 13, -31, -93, 76, 41, -52, -12, 25, 62, 86, 38, 61, -62, -80, -57, -90, -55, 41, -44, -18, -96, 34, 60, 57, -93, 39, -147, -95, 79, 32, 35, 65, -22, -10, 69, 80, -17, -57, -18, 138, -36, -4, 38, -28, 6, 0, 78, 35, -68, 10, 83, 45, -5, -101, -54, -35, 159, 49, -30, -57, 16, -55, 69, 76, 7, 34, -84, 7, 63, -20, -54, -25, 1, -9, -42, -12, -11, 60, 51, -69, -1, 6, -34, -140, -67, 42, 61, -9, -16, 21, 49, -34, 103, -1, -65, 1, 51, -61, -60, -98, -17, 50, 65, -57, 56, -56, -30, 67, -24, 69, 57, -27, -80, -16, -28, 23, 9, 34, 20, 48, -44, -76, 2, -137, -11, 124, -30, 10, 26, 79, 30, 46, -82, 85, -98, 23, -16, -44, -34, -29, -82, -17, 26, 99, -42, 35, 115, 12, 85, 44, -111, -12, -28, 3, -5, -12, -11, -6, 71, 15, 27, -4, -82, -67, -59, 58, -8, 34, 57, 85, 8, 19, 55, 4, -105, -39, 19, -51, 23, -49, 61, 7, 25, 47, 25, 9, -57, 30, 9, -12, 62, -92, -99, -23, 9, -182, -43, 97, -108, 16, 58, -5, 22, 37, -34, -33, -39, -61, -49, -54, -25, 70, -18, 170, 29, -78, -59, 66, -57, -14, 51}
, {99, 30, -23, -45, -1, -14, 19, -71, -7, -21, -65, -124, -58, -98, 25, -119, -12, -96, -17, 48, 49, 43, 5, -29, 45, -137, -8, -3, -14, -147, 21, -74, 28, -52, -1, 64, -18, -40, 53, -22, 30, -83, -2, -4, 108, 79, 71, 29, -128, -93, -21, -78, 40, 10, -13, 75, 8, -4, 6, -62, -4, -122, -54, -102, 26, -45, -33, -15, 7, -108, 26, -39, -42, 35, -10, -5, -13, -144, 55, -83, 26, -43, -138, 53, 71, 20, 11, -82, 22, -104, -130, 4, 38, -45, -78, -4, -83, 4, 75, 127, 24, -175, 48, -88, -3, -68, 57, 70, -21, 63, -97, 93, -85, -88, 82, -72, -22, 77, 11, 52, -29, -122, 75, -118, 95, -138, 60, -12, 35, 48, 55, -20, 77, -50, 40, 29, -17, 80, -18, 63, 66, -25, 84, -27, 7, 49, 42, 63, 16, 112, 22, -27, -81, 21, -31, 61, 116, -29, -75, 28, 51, 99, 81, -27, -4, 69, 19, -75, 120, -47, 11, 98, 94, 29, 69, 12, -87, -31, -21, 5, 10, 77, 37, -15, 24, -75, 76, -50, -3, 41, -104, 11, -8, -56, 61, 18, -76, 62, 68, 18, -61, 50, -5, 64, 50, -16, 17, 21, 88, -40, 42, -73, 86, -59, 5, -107, -14, 35, -28, -12, 26, 10, -81, 61, 19, 26, 2, -26, -32, 31, -37, -33, -25, 37, 12, 88, -24, 44, 58, -62, 94, -35, 93, -61, 64, 46, 4, -68, -6, -26, 22, -53, 119, 32, -105, -33}
, {-83, -48, 1, -23, -73, -77, -11, -7, -2, -40, -7, 51, -74, -163, 58, -33, -22, 56, -21, 2, 23, -29, -30, -54, -9, -31, -27, -45, -28, -38, 58, -41, 18, -50, 11, -49, 24, 0, 24, 6, 48, 22, -19, -33, 5, -3, -48, 62, 65, 12, -8, 28, -34, 39, 69, 9, -35, 49, -2, 94, -14, 41, -31, -72, 29, 38, 13, -14, 9, -67, 16, -25, -55, 59, 68, -30, -123, 15, 67, -47, -23, 36, -48, -57, -10, 15, 14, 46, -9, 3, -23, -33, -46, -52, 41, -34, -56, 92, 11, 2, -77, 6, 64, 30, 13, 10, -60, 34, -6, 23, 28, 8, 5, -68, 62, -30, -3, -35, -6, -104, 31, -12, -34, -14, 57, 39, -16, -9, 13, -40, -26, -75, 29, 60, 11, -47, -13, 20, -90, 83, -28, 106, -15, 2, -11, -51, 26, 7, -30, -54, -119, 83, 10, 31, -70, -83, -126, 83, -98, 45, -59, 1, 53, -18, 5, -36, 46, -55, 61, -72, -82, 30, -50, 4, -34, -41, 82, 91, -57, 27, -53, 1, -18, -30, -87, -36, 19, -45, -75, 36, -123, 88, -10, -45, -92, -200, -98, -8, 20, -62, 20, -60, -28, 43, 27, 112, -98, -17, 51, 32, 106, -125, 13, 16, 87, -141, 19, -58, 40, -1, -66, 26, -89, 71, 62, 129, -46, 7, -19, 26, 59, -2, -126, -24, 60, 60, -51, 79, 91, -85, 14, 59, -25, 37, -15, -49, -59, -81, 119, -76, 97, -52, 99, 19, -34, 85}
, {28, 19, -59, -137, -68, 8, 42, -147, 25, -14, 2, -134, -21, -55, 27, -217, -15, 27, -22, -98, 56, 85, 31, 3, 29, -181, -3, 18, -8, -76, 3, -53, 8, 10, 71, 41, -104, -60, 116, 18, -25, 23, 31, 38, 83, 27, 43, 4, -196, -119, -35, -64, 40, -32, 34, -7, 46, 62, 32, -27, 11, -137, 36, 0, -18, -11, -28, 58, -75, 113, 78, -8, -85, 19, 99, 35, 40, 34, -76, 109, -48, -64, 97, -7, -57, 9, -5, 40, -98, -8, -35, 102, 49, 76, -12, 5, -55, -72, 5, 39, 2, -36, -50, 51, -25, -72, -68, -60, -2, -91, 27, -21, 98, 69, -139, -5, -57, 19, -56, -19, -16, 22, 0, -47, -53, 90, 3, 113, -99, -88, 23, 19, 77, -74, -61, -49, 13, 13, -61, 25, -15, -78, 32, -4, 57, -6, -23, -55, 1, 60, -11, 24, 18, 49, 44, 64, -42, 24, -25, 83, 14, 59, 45, 3, 84, 24, -10, 10, 55, -163, -47, 56, -21, 16, -25, -77, 13, -48, -46, -32, 18, 90, -15, 37, -18, -116, 69, -28, 2, 35, -71, -10, 19, 81, 7, -14, -14, 70, -36, -91, -7, -83, -12, -1, 29, -2, 28, 38, 76, -8, -24, -64, 43, 83, 30, 10, 82, -33, -12, -35, 11, -26, 21, -116, 36, -3, -114, -81, -106, -18, -109, -40, 26, -52, 43, -4, 109, 93, 41, -12, 14, -77, -78, 25, -63, -1, 32, 6, 129, 40, -86, 41, -29, 62, -6, 34}
, {61, 12, 11, 35, 7, 26, -29, 56, -68, -69, 17, 55, 45, -47, 32, 59, -61, -14, -106, 86, -20, -50, -1, 18, -106, -44, -31, -65, -69, 4, -30, -42, 18, 86, 65, -9, -57, -48, 20, -11, 20, -21, -99, 53, 22, -12, 5, 5, 58, -46, 57, -21, -93, 23, 79, 17, -36, 5, 6, 43, -51, -16, -31, -37, 37, -9, -12, -20, -83, -61, 16, 51, 31, -68, -117, 19, 70, 14, -51, -41, 44, 25, 14, 16, -104, 24, 53, -11, -37, 3, 67, 80, 53, 20, 16, 30, 20, 31, 17, -80, -8, 51, 15, 28, -70, 68, 41, -115, -5, -25, 51, -4, 61, -7, 19, 5, -54, -3, -3, -10, 53, 43, -37, 35, -7, 10, 78, 44, 46, 47, -6, 46, 11, 33, -65, 20, -106, 67, -61, -5, 3, -71, -74, -30, -14, 20, 72, -37, -124, -128, -67, -91, 185, -30, 23, 6, 13, 34, 61, -8, 30, -137, -66, -55, -21, 14, 44, -4, 5, -26, 27, -24, -12, -22, -77, 8, 52, 4, 71, 21, 22, -43, -37, -43, -89, 17, -45, 29, -107, 26, 47, -12, 30, -182, -87, 30, -5, -11, -74, 99, 2, 1, -47, 23, -152, 29, 50, 73, -164, -53, -123, 88, -93, -52, -53, 25, 64, 32, -13, -124, 11, 55, -87, 30, -45, -35, 96, 6, 91, -65, 74, -10, 63, 21, -58, 9, -63, -19, -122, -71, 21, 23, 62, -78, -47, 42, 16, 39, -102, 47, 66, -66, -70, 32, 70, -77}
, {-75, 4, 41, 11, 3, -46, -6, 18, -3, -78, -34, -7, 35, -104, 105, 20, 35, 0, 64, -32, -18, 55, 25, -29, -7, 10, 8, -60, 4, -53, -50, -40, -31, 59, -24, -65, 73, -6, 35, 32, 55, -39, 24, -6, -22, 10, -60, -6, 45, 32, 50, 9, -58, 6, 72, 16, -18, -41, 76, -23, -15, 3, -71, 16, 47, 10, 36, 1, -17, -17, -50, -2, 51, -10, -19, 38, -12, 22, 63, -4, -68, 12, 26, 53, -80, -41, -2, 35, 23, -10, 49, 107, -19, 27, 46, 32, 19, 57, -18, 29, 5, 71, -10, 39, -16, 80, 33, 9, 5, -72, 58, -8, -26, 51, -63, 21, -41, -13, -21, -7, 56, 39, 28, -11, 60, 22, -92, 28, 55, -121, -55, 47, -26, 49, -49, 90, -80, -9, -73, 32, -1, 63, -126, 72, 39, -95, 72, -77, -123, -116, -187, 59, -129, 109, -47, 14, -155, 25, -114, -42, -30, -133, 31, 10, 11, -4, -44, -40, -30, -36, -129, -52, -135, -71, -163, -176, 63, 66, 199, 37, -67, -47, -143, 6, -138, -50, -59, -23, -91, 135, -29, 22, -53, 53, -35, -11, 41, 80, -94, -66, -6, -10, 23, -30, 5, 89, 83, 1, -56, -78, -76, 36, -5, -51, -54, 33, 93, -36, -92, 35, -8, 66, -80, 58, -45, 85, -94, 36, -50, -69, 17, -19, 1, -86, -61, 108, -22, 79, 58, -158, -37, -16, 59, -29, -34, 27, 42, 28, 81, 14, 15, -88, 50, 30, -95, 52}
, {-43, 7, -11, -32, -46, 15, 0, -25, -9, -32, -37, 21, -24, -106, 8, 57, 0, -3, 55, 55, 29, 17, -16, 5, 139, -35, 20, -16, 49, -15, 67, 41, 7, 52, 1, -20, 55, 55, 18, 6, 21, -22, 30, 79, 31, 19, -47, 53, -11, -48, 12, -54, 193, -15, 25, 12, 16, 42, -8, -12, 49, -11, -6, 35, 25, 1, 72, 9, -33, 78, 72, -50, -67, 31, 65, 2, -71, 50, 37, -80, 0, -8, -125, -3, 45, 49, 23, 43, 39, -20, -43, -10, -7, -133, 82, -57, -10, -30, 46, 87, -75, -47, 26, 14, 10, 25, -18, 40, 28, -48, -17, 14, -134, -42, 0, 3, 15, -9, -57, -50, 9, 4, -17, 44, 53, -64, -8, -10, 7, -118, -31, 30, -38, 39, -61, 15, -20, 36, -23, 29, -12, 40, -124, -19, 41, 10, 40, -32, -31, -42, -140, 9, 33, 42, -11, -51, -23, 9, -123, 18, -20, -46, -83, -20, 46, 31, -24, -38, -16, -53, -42, -23, -115, -36, -62, -100, 6, 48, 56, -25, 45, 33, 56, 16, -106, -94, -42, -43, -118, 40, -153, 2, -74, 45, -86, -99, -10, -5, 25, -151, 46, -62, -83, -53, 20, 130, -4, -30, -8, -123, -98, -88, 36, 68, -63, 3, -72, -43, 25, 34, 39, 17, -49, 29, 93, 64, -28, -24, -83, -88, -17, -48, 6, -76, -47, 93, -81, 65, 64, -41, -35, 18, -48, 14, -28, -62, -30, -24, 157, -77, 26, -129, 102, 40, 3, 72}
, {-9, 85, -35, -69, 20, 20, 14, -19, 2, -29, -74, -17, 105, -40, -23, -53, -9, -53, -17, -101, 59, 132, 47, -20, -78, -3, 44, 37, 40, -24, 12, -48, -49, -25, 19, -42, -44, -92, 11, -41, -78, 34, 37, -52, 59, 78, -5, -13, -77, -24, 18, 11, 28, -9, -9, -49, -3, 2, -9, -71, 5, -39, -31, -72, -59, 53, 19, -38, -41, 84, -5, -117, -11, -7, -62, 28, 61, 33, 11, -11, -56, 35, -31, 57, -55, -56, -27, 29, -30, -32, 46, 55, 3, 28, 8, -27, -14, -20, 66, -10, 46, -2, -77, -65, 32, -62, 4, 19, -30, -61, -44, 47, 46, -40, -87, 1, -66, 39, 26, 46, -64, -19, 36, -15, -69, -37, -17, -91, -34, -95, -24, 106, 29, -25, -39, 54, 32, 115, -1, 82, -14, -19, -79, 62, -58, -1, 58, 45, -91, -23, -76, -29, -113, 34, -28, -9, -12, -13, -169, -60, 8, 30, 91, -31, 88, 32, 35, -30, 61, -18, -50, 28, 45, -59, 13, -61, 44, -2, 18, -39, -133, 94, -46, 20, 20, -70, 52, -72, -13, -30, 71, 66, 12, -31, 33, -24, 68, 46, -110, 0, -1, 71, -56, -19, 47, -37, 41, 31, 40, -90, -109, -22, 72, 13, -121, -73, 45, 36, -70, 33, 28, 29, -25, -44, -96, -59, -97, -119, -50, -7, -36, -49, 62, -26, -36, 95, 49, 25, 11, -125, 31, -61, -13, -70, 61, 16, 55, 122, 9, 20, -152, -14, 18, 55, -81, -28}
, {-76, -43, -22, -51, -10, 106, 2, -6, 12, -54, -83, -22, 61, 53, -4, 99, 73, 23, 4, -15, -17, -3, -6, 21, -129, -7, 1, -31, -6, -35, -21, -16, -24, 45, -22, -107, 12, -2, 27, 9, -26, 53, -143, -91, -12, 25, -73, -23, 6, -53, -49, 62, -136, 0, -45, -29, 65, -10, 2, -80, 48, -64, -69, -5, -51, 54, 58, -34, 13, -69, -222, -146, -44, 56, -41, -105, -39, -4, 0, 26, -141, -62, -122, 56, -27, 18, -141, 56, 43, -38, -36, -55, -51, -17, 9, -42, -44, -45, -203, -134, -4, 34, -210, -119, 66, -79, 17, 108, -69, -115, 18, 51, -113, -78, -154, 16, -59, 72, 9, 29, -110, 54, -18, 6, -166, -116, -53, -153, -98, 14, 41, 60, -124, -61, -42, 45, 21, 58, -74, 25, -63, -8, -41, 3, 78, -156, 11, -16, -79, -56, -6, -71, 43, -2, -23, 6, -36, -25, -3, -11, -40, -50, 18, -38, 14, -7, -47, 4, -150, 29, -54, -30, -38, 9, -154, 12, 7, 48, 49, 22, -15, 23, -149, -15, -36, -170, -35, 4, 5, 5, 45, -12, 37, 38, -8, 41, 20, 76, -96, -28, -3, 32, 52, -25, 107, -4, 55, 54, -30, -51, -173, 71, -82, 14, -85, -42, -8, -32, -91, 8, -47, 39, -134, -202, -115, 19, -149, -10, -21, -36, -60, -53, 44, 35, -37, 50, -17, -35, 28, -142, -2, -71, 16, -188, 90, -47, 30, 18, 3, 24, -85, 53, -66, -62, -53, 18}
, {89, -16, -74, 20, -78, 47, 21, -89, -6, 85, -13, -86, -6, 17, -5, -74, 2, -38, -72, -61, -18, 19, 5, -23, 179, -25, 24, 19, 53, -2, -55, -62, -58, 27, -35, 16, -104, -115, 21, -28, 1, -20, 10, -16, 55, 43, 4, 5, -1, -56, -6, -28, 111, 3, 67, 31, -13, 18, -27, 22, 33, -51, 35, -17, 70, 4, -60, -31, -162, 52, 111, -26, 20, -17, 154, -87, 16, 22, -15, -42, 18, 97, -43, 43, 78, 62, -46, -49, -24, -44, 12, 46, 44, 51, -2, -27, -11, -99, 59, 105, 6, 0, 159, 32, -64, -5, -17, -4, 2, 28, 36, 73, -108, -4, 8, -41, 48, -36, -38, -72, -67, -67, 50, -6, 26, -36, 26, -6, 76, 116, 76, 1, 60, 35, 94, -151, -20, -6, -7, -1, 42, -7, 39, 30, 18, -4, -31, 12, 83, 54, 43, 50, 19, -1, 5, -23, 13, 65, -1, -36, -12, -5, -14, -40, -7, 4, -39, -99, 46, -39, 21, 65, 118, 13, 115, 57, 41, -18, -27, 115, -37, 56, 76, 47, 109, -75, -7, -87, 20, -29, 9, 48, -110, 2, 121, 67, -46, -22, -16, -23, -9, 80, -126, 153, 52, -28, 2, 44, -145, -116, -45, 0, -95, -2, 16, -35, -57, 34, 95, 25, 92, -4, 28, 59, 107, -20, -53, -38, -19, -31, -37, -12, 51, -7, -95, -4, -43, -67, -17, 42, -6, 24, 45, 58, 30, 68, -16, 15, 21, 56, 37, 2, -58, 59, -22, 57}
, {27, 11, 5, -16, 4, -75, -74, 23, -5, 22, -19, -32, 10, 33, -35, -84, 8, -43, -102, 16, 38, -14, 71, 20, 62, 48, 49, -34, -1, -85, 41, 8, -25, 7, 47, 46, -47, -32, 10, -40, 50, 12, 81, 7, 58, 112, -44, -14, -162, -113, 49, -46, -9, 80, 18, -1, 8, 60, -4, -61, 20, -60, -49, -49, -53, 41, 44, -39, -41, -23, 17, -64, 39, 30, -144, 49, 31, -12, -53, 34, -7, -4, 39, 5, -108, -26, -8, -38, -15, -42, 27, -19, 9, -29, 29, -14, 31, -77, -10, 9, 32, -76, -112, -95, -4, 25, 18, -77, 36, 5, -35, -19, -30, -78, 17, 78, -94, 48, -41, 41, -83, -80, 37, -26, -104, -8, 32, -77, -30, 75, 41, -27, 25, 95, 100, -76, -104, 84, 91, -69, 92, -134, 34, -140, -157, -2, -98, 11, 30, 25, -24, 7, 38, -116, -73, 65, 118, -88, -16, 23, -56, 75, 62, -27, -81, -21, 38, -45, 42, 17, 36, -50, 54, -28, 22, 10, -88, -93, -106, 28, 79, -10, 73, -82, 67, -78, -24, -73, 60, -28, -89, 84, -20, -81, -112, 3, -53, 20, 94, 73, -128, 31, 28, 17, 45, -51, 8, 9, -8, -73, -29, -62, 20, -96, -73, -8, -11, -7, -56, 79, 132, -34, -115, -32, -83, -111, 91, -27, -166, -28, -40, -49, 42, 73, -112, -10, 54, 32, 62, -27, -24, -22, 59, -34, -10, 34, -22, -87, -53, -30, -65, -135, 76, 43, -76, -44}
, {23, 13, 6, -55, -18, -79, 38, 8, 29, 16, -106, -30, -32, 1, -44, 42, 94, -33, 27, 8, -31, 24, -32, 35, 32, -53, -11, 15, -12, 25, -28, -66, -47, -31, -48, -29, -59, -70, -11, 41, -28, 5, -82, -32, -36, 42, 30, 15, 49, -17, 23, 72, -80, -66, -57, -6, -4, -19, 41, -22, -20, 11, -8, 12, 35, 13, 19, -80, 13, -49, -2, -143, -5, 39, 36, -40, -19, 19, 40, -110, -103, -5, -104, 66, -9, 60, 17, 30, 99, -63, -98, -20, -48, -63, -8, 14, -15, 19, 98, -57, -27, -61, -51, -111, 62, -107, 18, 60, 24, 28, -48, -13, -141, -174, -30, -51, 11, 27, 70, -38, 27, 25, -26, 21, -2, -146, -11, -42, -10, -72, -15, 109, -50, -77, -29, 82, -75, 76, -45, 80, -48, -6, -61, 24, 72, 22, 53, -39, -103, -5, -40, -78, 44, 47, -43, 32, -93, 2, -115, -50, 14, -38, -8, 93, 66, -64, -50, -62, -78, -36, -88, -88, -120, -83, -75, -25, 12, 17, 92, 12, -85, -12, -80, -37, -51, -126, -102, 47, -66, 50, 21, 80, -22, -32, -52, -74, 51, 24, -71, -31, 55, 65, 10, -45, -16, 74, 69, -41, -16, -6, -59, 9, -20, -5, -56, -27, 86, 1, -69, 10, -89, 50, -114, -161, -70, -8, -99, -58, -89, -23, 51, 19, 15, -72, 70, 95, 24, 15, 56, -143, -66, 4, -9, -116, 84, 30, -36, -17, 7, -36, -18, -77, -36, 12, -96, 41}
, {-53, -48, 38, 23, 18, -51, -13, 11, -44, 20, 19, -6, 76, 20, -59, 57, 40, 24, 29, 68, -44, -109, -78, 67, -26, 42, 3, -93, -42, -5, -35, 20, 5, -42, -34, -1, 24, 76, -44, -1, 77, 37, -42, 27, -10, -10, -46, -1, -8, 30, -32, 45, 24, -62, 31, 11, -4, 58, -39, 57, -58, -14, -44, -112, 128, 82, 42, 31, -35, -7, 64, 43, 39, -55, 59, -23, 68, 80, 8, 30, -67, 90, -23, -21, -58, 99, 16, 30, 13, -117, 1, 95, 20, -30, 44, -61, -92, 110, -33, -49, -22, -7, 62, -3, -112, -33, 46, -49, -28, -3, 72, 76, -56, -75, -52, 7, 42, -25, 45, -75, 10, 31, 4, -41, 4, -20, 48, -2, -71, -25, -76, -20, -43, -64, -40, 19, -37, 55, 37, 2, -10, 19, -36, -1, 35, -107, -41, 69, -23, -30, -9, -80, 23, 26, -7, -1, -19, 37, 37, -103, 17, -35, -53, -50, -80, 62, -158, -29, -19, 14, -80, 93, -8, -61, -94, -44, 28, -11, 74, -11, 64, 39, -15, 13, -25, -11, 17, 44, -10, 21, 40, 24, -222, -18, 30, 51, 87, -34, -49, 8, 41, 29, -148, 44, 12, 3, 2, 116, -236, -172, -96, 73, -80, -140, -98, 49, 1, -7, -41, -131, -112, 15, 13, 21, -103, -17, -54, -14, 18, -120, -187, -89, 90, -13, -52, 26, -4, 27, -54, -13, -29, -5, 3, -67, -61, 72, 28, -6, -73, 36, 82, -23, -30, 27, -4, 16}
, {51, 36, -59, -126, -18, -13, -21, -74, 4, 5, -48, -140, -99, -116, 29, -146, 89, -78, -128, -61, 31, 53, 69, 6, -45, -53, 69, 31, -13, -174, -75, -157, -64, -96, 26, 117, -49, -45, 43, 31, -104, -28, 106, 3, 123, 66, 18, -1, -49, -26, 50, -36, -79, 60, -32, -40, 8, 82, 21, 51, -9, -95, 45, -210, -25, -27, -31, 18, -139, 28, 68, 5, -73, 25, -74, 101, -23, -149, -20, 56, -47, -8, 83, -19, -77, -114, 91, -3, -31, -27, 90, 9, 56, -67, 41, 75, -64, -62, 63, -20, -54, -31, -81, 47, -31, 10, -21, -95, 33, -3, -39, 5, 57, 31, 28, 39, -13, 34, -61, -64, 10, -10, 44, -23, -50, 26, 27, 15, -120, -32, -80, 0, 44, 2, 29, 54, -117, 12, -22, 49, -21, -100, -35, 19, -74, -88, 7, -30, -88, -10, 7, -27, -26, -49, 3, 9, 85, 32, 12, 55, -68, -7, 46, -18, 42, 37, 81, 47, 56, -26, -74, 135, -11, -58, -50, 30, 111, -42, -57, 56, -107, 24, -93, -39, -38, 59, 114, 28, 2, 20, 75, 49, -7, 26, -71, -56, -56, 68, -88, 37, -44, -20, 51, -107, -26, 66, 24, -10, 23, 37, 71, 4, 10, 52, 38, -81, 122, 39, -25, -10, 9, 23, 0, -99, -6, -3, -33, -114, -60, -78, 28, -89, -13, -45, 74, -25, 73, 24, 108, -85, 74, -29, -13, -20, 38, -5, 11, -29, -20, -14, -36, 47, 52, 103, 8, -30}
, {48, -39, -15, -13, -7, 63, 73, 13, -47, -25, 25, -30, 7, 109, 50, -22, 12, -18, -25, -58, 10, 15, 77, 62, -138, 9, 15, 76, -17, -52, -6, 22, 54, 14, 24, 26, -29, -8, 22, 30, -100, -14, 24, -52, 68, 43, -8, -28, -22, -27, -9, -25, 59, 35, -17, -32, 49, 2, 13, -25, 17, 25, -36, 104, -58, -34, 31, 28, -109, 15, 16, 95, -80, 2, -17, 58, 80, 29, -67, 41, 46, 26, 96, -138, -41, -72, -21, 7, -41, -2, 20, 51, 76, 7, -9, -11, -41, -86, 1, -98, 37, 72, -178, 56, -36, -29, 14, -56, 85, -63, -24, -96, 144, 52, -73, 25, 56, 14, -135, 3, -9, -14, -7, 61, -45, 105, 35, 94, -80, 51, 70, -29, -54, -75, -28, 5, 51, -45, 43, -24, 43, -122, 34, -25, -43, -15, -3, -19, 71, 106, 173, -15, -108, -104, 39, 85, 67, -132, 6, 82, 5, 44, 4, -48, 74, -69, 21, 2, -70, 55, -27, -106, 55, -56, -6, 70, -107, -64, -31, -6, -7, -26, -13, -7, 70, -13, -10, -78, -50, -21, 34, -27, 2, 7, 5, 41, -41, -44, -43, 53, 53, 52, -51, 32, -7, -9, 6, -14, 57, -75, 18, -25, -91, -79, -125, 27, 136, 49, 25, 3, 120, -65, -38, -81, 33, -168, -66, -88, 33, 23, -67, -35, 52, 51, -13, -45, 99, 31, -64, 2, 28, -44, -82, 26, 20, -8, 25, 17, -141, -35, -47, 145, -65, 31, 25, -7}
, {45, 2, 15, -116, 8, 54, 22, -1, -48, -78, 4, -122, -5, -107, 30, -95, 10, -38, -121, 4, 32, -20, 13, -9, 34, -64, -25, -10, 96, -111, 1, 12, -97, 28, 19, 44, -85, -123, 55, -57, -51, -1, -44, -15, 66, -22, 25, -7, -68, -47, -14, 0, -19, -40, -31, -64, -12, -4, 64, -100, -26, -91, 20, -72, 15, 33, 7, 3, -39, 2, 19, -147, -29, -1, -27, -112, -33, -27, 86, 4, 36, -31, -86, 18, -6, 13, -32, -33, 47, -31, -62, -104, -46, -55, -51, -1, -25, -39, 71, 31, 9, -45, 46, -58, 58, -111, 3, 82, 2, 49, -16, 67, -106, -190, 65, -85, -28, -10, 64, -43, -28, -16, 33, -46, 69, -102, 2, 12, -44, -39, 28, -2, 53, -97, 58, 10, 52, 39, -143, 17, -21, -51, 48, -8, 6, -70, 7, -11, -112, 13, 29, -73, -50, 4, 27, 63, 56, 68, -129, -40, 47, 30, 17, -7, 93, -18, 14, -78, 4, -97, 43, 79, -5, 20, -60, 30, 6, 58, -21, 15, -71, 85, 15, 80, -7, -120, 75, -3, 44, 5, -87, -20, -41, -50, 55, -41, -27, 74, -130, -39, -44, -6, -40, 25, 9, 9, 45, 20, 12, -96, 13, -95, 79, 22, -60, -86, 49, -28, -42, 19, -44, 68, -85, -117, 0, -1, -173, -131, -79, -31, -75, -144, 39, 84, 22, 146, 52, 63, -13, -49, -42, -73, -63, -44, 67, -25, -34, -23, 51, -46, -92, 39, 26, 11, -88, -60}
, {9, -17, -48, 41, -14, -68, 19, 55, -2, -27, -75, 8, -78, -68, 22, 2, 10, -14, -32, 18, -52, -13, -40, 21, 19, 20, 42, -53, -81, -62, -50, -4, 21, -42, -15, -23, 54, -6, 15, 18, -36, 24, -129, 70, -23, -30, -115, -31, 72, 31, 53, -24, 16, 8, -52, -2, -39, -11, 9, 42, -5, 40, -41, -77, 41, 35, 44, -18, 63, -44, 33, 25, 9, 31, 40, -5, -30, 56, 61, 20, -24, 1, 34, -7, -51, 51, -10, 32, 33, -38, 28, -16, 22, -115, 49, 22, 14, 100, 48, -7, -7, 17, -69, -4, -10, 51, -70, -14, 30, 11, -52, 24, -43, -32, -22, 32, 42, -90, -27, -49, 63, -5, -52, -4, 115, -20, 13, 68, -7, -110, -47, -122, 13, 18, -21, 40, -113, -70, -109, 36, -35, 88, -67, 8, 74, -60, 34, -11, -127, -98, -187, 51, 6, 77, -86, -60, -150, 28, -28, -14, -77, -57, 8, 30, -31, -61, 15, -23, 34, 3, -79, -3, -168, 24, -80, -166, 57, 86, 91, 16, -61, 48, -8, -18, -71, -37, -18, 36, 4, 63, -156, -7, 0, 12, -77, -235, -99, 91, -15, -89, 10, -99, -4, 2, 8, 106, -77, -98, -75, -4, -18, -93, 49, -12, 73, -46, 19, -69, 56, 7, 49, 76, -4, 49, 57, 102, 17, -51, -6, -100, 68, -64, -81, -48, 62, -2, -23, 85, 30, -87, 28, 31, -1, 31, -9, -93, -10, -47, 143, -4, 70, -105, 12, 12, -47, 107}
, {41, 17, 85, -44, 3, -61, -5, -13, 47, -76, -64, 64, 130, -15, -59, 35, 91, -118, -31, -82, -37, 29, 10, 27, -180, 48, 48, 26, -31, 144, 15, 31, 42, -7, -41, -53, -31, 17, 18, -2, -36, 92, -96, 8, -21, -11, -41, -7, 62, 69, 52, 10, -124, 19, -43, 7, 84, -22, -20, -2, -24, 74, -29, 41, -181, -31, 87, 21, 30, -49, -162, 95, -81, 56, -115, 3, 53, -106, 8, 88, -69, -50, 53, 78, -48, -100, -144, 72, 9, 42, 37, -3, 9, -11, 37, 50, 17, -93, -176, -234, 110, 22, -253, -32, 73, 61, -59, 60, -31, -124, -9, -30, 46, 34, -132, 95, -154, 31, -58, 44, -50, 72, 19, 14, -130, 3, 13, 36, -111, -44, -10, 21, -2, 13, 31, 35, -13, -124, -36, 29, 22, 13, -9, -45, -1, -94, 46, -19, -74, -39, -21, -27, -21, 19, -49, 4, 57, -18, 91, 73, 8, -63, -97, 47, -74, 47, 0, 87, -13, 32, -124, 32, -64, 46, -47, -22, -9, 24, 51, 8, -125, 28, -39, -24, -77, 64, 4, -43, 7, -21, 8, -15, 39, 38, -60, -36, 58, -24, -24, -69, 22, -66, 57, 24, 44, 50, 28, -16, 116, -86, 51, 40, 80, 12, -18, -68, 22, -46, -45, 57, 46, 29, -22, -8, -52, 1, -32, -82, 62, 11, -10, -80, -55, -116, -95, -17, 19, 28, 29, -87, 6, 18, -122, 15, -21, 40, 87, 58, 24, 26, -62, 8, -13, -18, -47, -14}
, {104, 30, 39, 36, -48, 79, 31, -50, -28, -31, 51, -12, -75, -46, -37, -38, 24, 17, -27, -71, 50, -40, 2, -6, -62, 9, -3, 26, 5, -80, 5, 32, 87, 61, -3, 27, 57, 50, 32, 66, 38, -5, 52, 35, -4, -84, 33, 60, -2, -38, 85, 32, -91, 1, -30, 30, 48, 6, 27, 99, 51, 18, 55, 43, -132, -30, 28, 14, -90, -2, -44, 84, -60, 31, -65, 26, 35, -72, -75, 59, 50, -77, 82, -93, 24, -121, 13, 66, -177, 52, 74, 76, 35, -36, 23, 44, 26, -55, -114, -169, -11, 62, -121, 81, -48, 49, -72, 5, 18, -10, 32, -56, 51, 30, -58, 107, -25, 59, -144, -21, 109, 27, -84, 38, 0, 22, -24, 61, 56, -7, -90, -3, -9, 33, -81, 32, -88, 27, 47, 31, 15, 21, -103, 5, 24, 9, 28, -47, -13, -108, -78, -85, 150, 16, -12, -48, -11, -8, 40, 56, -52, -25, 27, -2, -60, 44, 14, 13, -20, -49, -34, -23, -12, 76, 21, -104, 9, -30, 17, -55, -43, -49, -62, -55, -77, -3, -89, 30, -84, 32, 66, 36, 70, -80, -139, -48, -4, 1, -4, 85, -78, -20, 41, -80, -168, 49, 1, -57, -37, 11, 40, 4, -21, -32, 50, 22, 56, -24, -70, -83, -25, 34, -7, 82, -52, -28, 126, 113, 24, 34, 115, 3, -48, 82, 18, -3, -170, 20, -71, -65, -48, 59, 103, -97, -41, 27, -7, -14, -27, -54, 55, -70, 45, -36, -7, 40}
, {-1, -104, -5, 61, 100, 58, -23, 54, 4, 20, -4, 4, -50, -61, -39, 74, 33, -9, 36, 108, -29, -81, -33, 5, 60, 89, -111, -122, -107, -6, -66, -99, -50, 0, -19, 22, -20, -56, 10, 8, 129, -21, -1, 46, -65, 46, -53, -47, 6, -16, 70, -60, 96, 54, 40, -44, -89, -60, -38, 73, -14, -37, -81, -85, 11, 26, -29, 23, -33, -9, 10, 56, 38, -3, -24, 3, 79, -37, 76, -31, 37, 50, -108, 40, 32, 55, 131, -64, -39, 31, -70, 2, -16, 60, -13, -5, 25, 26, 76, 25, -14, -36, 103, -9, -5, 43, 67, -5, -43, 39, -36, -62, 11, 6, 54, -64, 9, 43, 85, 33, -31, -76, -76, -93, 39, -16, 15, -3, -145, 30, -18, 69, 36, 31, 6, 1, -17, 8, -23, 50, 39, 43, -46, -25, -42, -129, 15, -32, -160, -124, -64, -74, 60, 38, 66, -31, 4, 24, 97, 27, 4, 17, -82, -87, 17, 72, -62, 55, 54, 89, -61, -60, -53, -10, -52, -11, 51, -4, 56, 73, -91, 46, -68, 9, -61, 179, 71, 63, -111, 19, 53, 34, -30, -72, 15, -24, 30, -44, -111, -32, 12, 6, -82, 93, -54, -55, -20, 44, 0, -71, -39, -6, 51, -31, -33, 26, 29, -12, 47, -37, -126, 1, 4, 67, 70, -24, -49, -29, 95, 48, -34, 56, -1, -112, -26, 63, 33, -42, 9, -103, -30, 5, -21, 1, -107, 8, -19, 45, -33, 43, 82, -2, -31, -20, -23, -10}
, {-58, 82, -6, 71, 78, 44, -25, 2, 18, 58, 165, 16, -27, 60, 38, -2, -59, 26, 39, -10, 0, -24, 20, 11, 120, 40, 20, -30, 0, 63, -2, 20, 9, 28, -39, -81, 22, 7, 14, -28, -43, -39, -11, -64, 19, -49, 61, 34, 43, 28, -87, 63, 65, 32, 98, 6, 40, 74, 22, 22, -11, 20, -17, 59, -171, -21, -54, 19, -49, -80, 68, 152, 30, -38, -62, -5, 10, -43, -13, 72, 14, 25, -75, -66, 26, -27, -5, -8, -50, 9, 5, -58, -105, 9, 22, -18, 57, 47, -77, 62, -31, -30, 43, 143, 25, 147, -80, -14, -13, -76, 53, -72, -50, 4, -94, 33, -86, -60, -67, 6, -23, 78, 21, -6, 5, 44, 66, -19, 34, 7, -214, -39, -24, 63, -27, -21, -123, -87, 24, -62, -31, 128, -30, -10, 14, 24, -29, -34, 49, -57, -140, 55, -7, 9, -95, -78, -125, 35, 6, -13, -124, -18, -98, 43, -138, 31, 36, 116, -19, 82, -128, 33, -101, 78, -103, -94, -51, 6, -17, 67, -151, -46, 63, -70, -19, 101, 52, 60, -79, -2, -89, 31, -67, 17, -47, -74, -49, -39, 57, 9, 73, -155, 33, -110, -15, 11, 4, -36, -55, 3, 67, 0, 44, -64, 38, -85, 20, -31, 9, -35, -7, 4, -14, 192, 60, 51, 151, -16, 28, -2, 18, -51, -146, -84, -23, -89, -127, -12, -98, 93, -21, 94, -53, 24, -65, -61, -28, -107, 79, -78, -11, -78, 60, -8, 21, -11}
, {21, -29, -62, -89, -62, 67, -28, -75, 16, -180, -59, -26, 22, 36, 16, -41, 21, -12, -60, -34, 73, -26, 12, -18, -155, -135, -65, 55, 44, -48, -38, -33, -26, -25, 9, -63, -95, -144, -4, 12, -101, 31, -37, -12, 70, 4, 80, 11, -111, -17, -49, 4, -108, -32, -13, -104, -3, -29, 13, -67, -43, -110, -12, -33, -47, 16, 57, 28, 2, 16, -38, -13, -26, 60, 59, -31, 78, -102, 0, 48, -50, -86, 94, -17, -135, -8, -63, 8, -29, -18, -3, 105, -50, 55, 18, 71, -22, -103, -81, -146, 39, 28, -173, 25, 0, -33, -4, -21, 6, -85, 26, 48, -21, 112, -72, 6, -14, 0, -110, 6, -60, 63, 40, -23, -143, 112, -136, 70, 9, -144, 36, 21, -39, -73, -183, 117, 64, 113, 57, 9, -126, -39, -37, 21, 30, 44, 5, -134, -72, -24, -30, -109, -1, 44, 91, -84, -110, -29, 33, -65, 30, -84, 11, 29, 29, 20, -5, -71, -66, -86, 104, -86, 22, -58, -131, 9, -26, 31, 97, -53, 114, 5, -85, 60, -25, -164, -32, 35, -78, -26, 42, -70, 1, 29, -83, 52, 86, -9, 10, 83, 102, -62, 13, -132, 41, 49, 41, -42, 68, -146, -64, 105, -38, 55, -80, 131, -18, -41, -79, -50, 14, 45, -20, -79, -42, 41, -68, 51, -58, -80, 82, -90, 59, 97, -19, 19, -60, 14, 16, -29, 7, -17, 9, -41, 34, 20, 95, 4, -23, 53, -26, -47, -3, -83, 84, 60}
, {15, 26, -1, 44, 128, 28, -49, 164, 36, 30, 64, 57, 7, 41, -12, 150, 50, -85, 49, 90, -55, 51, -7, -28, -121, 53, 22, 8, -21, 121, 22, -30, 1, 2, -43, -68, 43, 22, -105, 25, 46, 11, -51, 42, -54, -7, -45, -51, 122, 41, 51, 5, -67, 20, -56, 69, -4, -145, -61, -141, -36, 48, -47, 23, 71, 17, 9, -11, 66, 53, -88, 13, 13, -18, -106, -50, -7, 41, 68, 11, -32, -51, -47, 75, -127, -24, -53, 21, 41, 9, 27, -13, 24, 44, -3, -75, -13, 97, -22, 63, 47, -16, 10, -15, 10, -42, 96, 89, 41, -49, 58, -18, -37, -79, 89, -35, -94, 28, 94, 33, -63, -13, 18, -1, -73, -2, -37, -105, -83, 90, -91, 23, -21, -53, 96, -60, -34, -134, 70, 18, 38, -147, 13, -51, -38, -67, 14, 16, 59, -4, 46, 12, -122, -14, -22, -13, 113, -45, 104, 51, -51, 14, 73, -58, -57, -28, 18, 68, 1, 153, -79, -13, 37, -87, 55, 24, -27, 73, 48, 13, -36, -1, 2, -23, 64, 93, 150, -19, 14, -58, 127, -59, 25, 89, 57, 9, 4, 6, -67, 22, -38, 60, -29, 16, 15, -144, 12, 58, 0, 19, -7, -42, -71, -19, -3, -25, -75, 93, -3, 14, 61, -88, 41, -6, 1, -39, -17, -9, 98, 26, -89, 7, 39, 32, -61, -41, 135, 21, 72, 32, 67, -82, -62, 52, 72, -8, -46, 28, -40, 34, -18, 96, -109, 102, 33, -94}
, {24, 9, 2, 70, -27, 33, 7, 20, -11, -42, -12, -12, 5, 58, -54, -31, 34, 9, 18, -19, -6, -10, 33, -5, -153, 52, -28, 5, 12, 47, -51, -24, 29, -74, -18, 21, -2, -19, 44, -38, -57, -14, 17, -21, 37, 18, -8, -15, -39, -7, 55, -14, -3, -21, -19, -16, 40, -22, 23, 8, 10, -8, 37, 27, -87, 62, -26, 11, 10, 44, -56, -22, -5, -6, -43, 21, 78, -58, -7, 69, 44, -64, -14, 49, -28, -69, -50, 4, -166, 21, 68, 71, 117, 119, 20, 22, 50, -66, -4, 33, 70, 6, -59, 23, -20, 23, -12, 32, 63, -20, 60, 0, 33, 21, 55, 66, -142, 48, -20, 64, -7, -35, -54, -20, -176, 5, 84, -51, -117, 69, -9, 31, 4, -36, -23, -77, -83, -39, 88, -7, 4, -77, -19, -24, -65, -115, 18, 38, -73, 51, 64, -33, 62, -28, 52, 65, 170, 15, 13, -47, -95, 36, 0, -42, 23, -35, -85, -45, 59, 17, -33, 4, 94, -66, 56, 108, -79, -63, -36, 70, 38, 38, 29, 7, 62, 23, 71, -33, 14, -32, 116, 15, -33, -10, 49, 71, -39, 14, 35, 105, -85, 114, 18, 37, -25, -138, 73, 71, -95, -90, -140, 19, -77, 8, -43, 54, 33, 90, -90, 39, 40, -89, 27, -146, -46, -52, 26, 95, -3, -22, -17, 67, 29, -8, -63, 50, 37, 2, -19, 23, -28, -81, 65, -27, 50, 45, 11, -12, -78, 83, 58, 71, -61, 4, -4, -146}
, {-86, 33, 42, -37, 126, 83, -8, 41, 9, 34, -35, 134, 57, 56, -40, 47, -50, 3, 7, 26, -22, -36, -52, 61, -125, 138, -27, 46, -44, 60, 35, 52, 5, -5, -45, -80, 52, 15, -50, 13, 45, 24, -122, 16, -59, -35, -7, -51, 139, -19, -7, 4, -81, -70, 34, 86, 45, -99, -37, -58, 19, 55, -74, 89, -47, 6, 58, -55, 27, -57, -194, -144, 0, 56, -87, -33, -36, 27, 0, -58, 0, -125, -54, 84, -43, -41, -10, 51, 156, -58, -40, -78, -96, 55, 7, 5, 71, 100, -89, -33, -1, 8, -139, -137, 43, 13, -37, 94, -24, 66, 47, 47, -64, -73, 26, 35, 42, 81, 30, 105, 16, 113, -37, 84, -51, -102, -57, -24, 30, 82, 43, 22, -103, 46, 1, -1, -66, -74, 44, -59, 59, -35, -5, 19, -19, 25, 1, -64, 57, 12, 77, 11, 36, -24, 12, 14, 9, -14, 13, -61, -22, -54, -1, 26, -100, 11, 47, -62, -67, 127, -2, 13, -20, 50, -15, -29, -26, 34, 81, 41, 38, -68, -3, -21, 4, -20, -14, -29, 11, 14, -10, -110, 56, 28, -83, 33, 7, -49, 26, -55, 55, -32, 0, 8, -40, 2, -7, 14, -77, -4, 93, 41, -53, -11, 58, -34, -128, -64, -6, 6, 8, 32, -39, 95, 31, 7, 78, 46, 64, 25, 100, 80, -98, 74, -51, -72, -68, 50, 19, 14, 3, 59, -13, 10, -35, -57, 45, -86, -60, 47, 43, -88, 29, -93, -7, 76}
, {-56, -45, 61, -13, 89, 48, -138, 108, 58, 44, 18, 45, 57, 17, -140, 17, -21, 8, 95, 24, -100, -37, -120, -16, -35, 105, 22, -42, -89, 44, 14, 18, 16, -41, -91, -69, 140, 6, -161, 58, 116, 18, -145, 46, -150, -73, -21, -72, 85, -20, -3, -9, -84, -52, 89, 55, -21, -129, -109, -10, -86, 40, -116, 83, 74, -31, 13, -7, 15, -15, -66, 4, -56, 30, 76, -9, -1, 78, -9, -97, -35, 27, -90, 37, -49, 80, -83, 55, 68, -53, -44, -59, -63, 40, -33, -48, -50, 206, -8, 6, 2, -49, 60, -17, -31, 3, 23, 34, -5, -50, 27, 12, -120, -84, 45, -71, 33, -77, 34, 33, 53, 129, 15, 38, 88, -105, -5, 85, -7, 15, -103, 19, -115, -3, -24, 53, -61, -52, 24, -38, 67, -16, -76, 9, 58, -18, 17, 48, -40, -100, -134, 1, -74, 9, 12, -34, 12, -12, -21, 51, 9, -159, -42, 17, -31, 5, 3, 55, -16, 71, -59, -28, -78, -101, -31, -97, 17, 33, 82, 59, -69, -47, -150, -69, -88, 48, 5, -12, -58, 41, 67, -74, 59, 34, -18, 42, 27, 13, 12, -34, 23, 75, 14, 32, 56, 3, 14, 38, 58, 125, 84, 84, -24, -32, 118, 27, 9, 4, 93, 15, -23, 20, 118, 79, -14, 36, 68, 18, 116, 80, 16, 24, 4, 0, 11, -122, 49, -21, 52, 48, -8, 14, 16, 74, 33, -21, 18, 1, 65, 53, 57, -6, -73, 1, -27, 42}
, {29, 9, 6, -22, 83, -17, -109, 22, 25, -48, 52, -72, 34, -10, -12, 37, 66, 1, 14, 63, -31, 33, -49, -63, -114, 21, 31, -40, 5, -63, -97, -43, 26, -39, -26, 5, -3, -53, -55, -24, 55, -9, -14, 76, -27, 64, 18, -36, -18, -10, 7, -14, -58, 24, -27, -43, 37, -62, -73, -128, -8, 11, -5, -40, 101, 53, -18, 21, 52, 69, 17, 53, 29, 23, -63, -6, 19, 3, 65, 61, -61, -7, -25, -15, -124, 24, 5, -6, -12, -8, 8, 38, 73, -6, 46, -4, -17, -2, -32, -51, 31, -105, -23, -40, -16, -7, 68, 10, -44, -48, 42, 17, -4, -62, -50, -2, -68, -14, 32, -22, 16, 9, 35, -57, -51, -37, 45, -38, -116, 69, -100, 37, 33, 6, -43, 52, -36, -56, 62, 86, 29, -69, -89, -5, -78, -23, 34, 37, -128, -11, -4, -29, -94, -27, -62, 57, 35, 45, 53, -64, -36, -41, 51, 16, -99, -16, -51, 17, -56, 13, -73, -5, -98, -88, -100, 36, 73, -3, 117, -22, -6, 45, -64, -39, -48, -15, 88, 109, -24, -3, 66, -5, -71, -55, 40, -24, 60, 14, -44, 24, -27, 120, -47, 22, -5, -81, 72, 21, -81, -113, -137, 82, -158, -131, 23, 52, 29, 97, -3, -66, -118, 40, 29, -87, -100, -83, 9, 48, 26, -29, -107, -67, 62, 3, -67, 56, -19, 15, -14, -17, 20, -82, -58, -85, 99, 62, -13, 37, -126, 73, 57, 47, -30, 60, 65, -57}
, {0, 21, 10, 11, 132, -64, -61, 142, 6, 121, 43, 103, 50, 28, -104, 31, -38, -95, -2, 86, -58, 3, 4, -39, -79, 121, 44, -7, -84, 155, 21, 5, 46, -17, -13, -42, 8, 113, -147, 26, 88, 51, -63, -47, -129, 49, -123, -33, 148, -5, 56, 51, -16, 19, -50, 18, -3, -64, -73, -69, -77, 25, -46, 43, 59, 6, 22, 48, 22, -31, -118, -42, 6, 18, -41, -34, 3, -89, 16, -6, 6, -59, -97, 93, -6, 19, -104, 49, 34, -36, 19, -59, -7, 15, 25, -10, -4, 62, 3, -23, 14, -44, -40, -6, 35, -37, 103, 49, -42, -68, 88, 40, -72, -105, 28, 0, 5, 51, 112, -4, 87, 45, 60, 20, 12, -13, 62, -59, -23, 72, -62, -38, -110, 14, 94, -9, -57, -102, 23, -47, 58, -55, 40, 13, -59, -38, 9, 79, 42, -21, 2, -28, -118, -77, -60, 9, 45, -9, 28, 20, -50, 17, 20, -35, -89, 32, -12, 11, -85, 117, -73, -30, -16, -88, 41, 20, -71, 46, -14, 57, -86, -68, 9, -11, 54, 189, 78, -24, 12, -35, -43, 3, 11, 103, 102, 36, 7, 12, -120, -80, -51, 120, -14, 66, 90, -113, -26, 39, 14, -14, 52, 26, -80, -95, 100, -25, 3, 29, 30, 62, -17, -18, 66, 28, -47, 0, -46, -117, 40, 77, -91, 45, 9, 29, -90, -63, 115, -11, 16, -67, 9, -101, -32, 55, 112, -3, -18, -23, -30, -38, 29, 15, -141, 29, -32, -26}
, {-59, -31, 62, 0, 54, 107, -16, -29, -42, -38, -16, 113, 28, 67, -26, 103, -17, 1, 71, 3, -41, -12, 20, 63, -100, 18, -55, -82, -29, 47, 1, 69, 78, 0, -43, -93, 17, 52, -123, 57, 26, -55, -100, 65, -74, -59, 52, -139, 96, 29, -3, -19, -117, -3, 108, -35, -32, 20, 12, 15, -62, 49, -72, 51, 11, -2, -33, 46, 4, -76, -14, 58, -37, 3, -30, -18, 40, -27, -37, -42, 30, 37, 47, 21, -54, -83, -41, -24, -2, 43, -9, 14, 11, -10, -4, 17, 66, -59, -113, -105, -30, 101, 20, 4, -29, 51, 10, -64, -8, -104, -26, -39, 40, 67, 89, 8, 11, -59, -68, 4, -39, 38, -21, 43, -108, 101, -41, -27, -56, -49, -30, -7, 8, -21, -54, 89, 13, -58, 12, 4, 62, 17, -118, 52, -2, -111, 12, -53, 25, -90, -48, -81, -154, 4, 2, -31, -39, 12, -51, 8, 81, -41, -14, 35, -16, 4, 61, 31, -169, 91, 3, -52, -82, -84, -87, -36, 60, 47, 128, 46, -70, -70, -106, -33, -80, -48, -116, -106, 13, 24, -10, 51, 68, -79, -95, 44, 12, -10, -10, 53, 18, 63, 36, -5, 63, -86, 6, 31, -13, -24, 4, 56, -58, -199, -113, 93, 22, 24, 76, -112, -42, -5, 22, -13, 60, -111, 26, 56, 89, 4, 12, 72, 39, -36, -87, -67, -107, -43, -81, -47, 47, 7, 3, 41, -3, 12, 19, 41, -133, 84, 62, -12, -122, 64, 30, -78}
, {150, 4, -48, -93, -57, 96, -2, 2, -49, -28, 31, -118, -61, -62, -59, -142, -11, 68, -43, -22, 44, 48, 29, -30, -41, -108, -13, -19, 49, -55, -26, -69, -39, -76, 82, 56, -88, -30, 103, -18, -109, -90, 104, -18, 25, 66, 74, 58, -110, 13, 32, -3, 107, 57, 40, -59, -49, -54, -74, -85, -16, -65, -28, -70, 109, 43, -50, 0, -73, 67, 128, 94, -79, -26, 54, 44, 71, -9, -86, 141, -32, 27, 63, -62, -60, -7, 42, -35, -35, 13, -3, 52, 89, -53, -23, 4, -86, -19, -26, -14, -66, 3, -48, 121, -93, 49, -17, -159, -11, -33, 15, -27, 37, 52, -38, 42, 4, -58, -59, -67, 32, -14, 103, -44, 24, 105, 38, 89, 5, -81, -2, 0, 41, -90, 7, 6, -37, 29, -23, -56, -14, -95, 3, 3, 15, 13, -62, 74, 45, -45, 21, -69, -77, 34, 4, -7, 15, -111, 17, -66, -51, 2, -1, 61, 1, 47, 71, 64, -56, 38, -35, 59, -54, -10, -54, 43, -25, -54, 78, 2, -64, -38, 56, 63, 8, -13, -13, -32, -7, -50, 62, -4, 41, 60, -40, -21, 24, 15, -42, -29, -60, 26, -4, -125, -80, -77, 67, 6, -87, -57, 62, 14, -70, -56, 61, 26, 118, 17, -1, -162, 66, 14, 104, 14, -92, -119, -35, -16, 24, -4, -30, -92, 118, -71, 36, -88, 21, 36, -89, 43, 8, -113, -84, -72, 7, 48, -33, -47, -57, 26, 65, 59, -88, 35, 45, -82}
, {-32, -31, -58, -55, -63, -36, -28, 46, 49, -19, -122, 39, -128, -154, -5, 7, 50, -13, -70, -30, -15, 19, -9, 22, -83, -52, -13, -17, 53, -104, -67, 36, 28, 55, 20, -87, -36, 51, 7, 21, -118, 36, -125, -15, -34, -61, -27, 17, -102, -28, -1, -32, -222, -2, -10, -50, 35, 17, -8, -49, 3, -124, -5, -26, -74, 31, 81, -65, 61, -135, -199, -85, 6, 73, -107, -22, -7, -2, 75, -63, -8, -128, -51, 55, -19, -1, 10, 107, 92, -71, -15, -82, -66, -2, 31, 46, -6, 56, -122, -92, 26, 14, -143, -104, 65, -58, -18, 73, 25, 83, -44, 41, -84, -91, -147, 38, -25, 78, 23, 30, 36, 27, 29, 65, 1, -137, -43, -57, 53, -33, 34, 34, -36, 7, -44, 15, -50, -25, 23, -18, -34, 16, 17, -19, 69, -11, -5, -53, 4, -32, 3, -61, 26, -40, -10, -23, -35, 10, -37, -69, -19, -29, 103, 87, -16, -25, -2, -53, -73, -18, 42, 27, -11, 38, -53, -32, 14, -51, 14, -44, 59, -24, 13, -53, -4, -87, -58, 21, 20, -13, -35, -69, 4, -35, -65, -44, 2, -14, -21, 9, 11, 10, 35, -80, 4, 28, 69, 23, -62, -78, 2, 59, -52, -50, 14, -75, -24, -85, -131, -8, -33, 24, -87, -44, -117, 4, -9, 70, -92, -67, 3, -21, 8, 54, -46, 31, -10, 60, -40, -8, 61, 62, 44, -63, 77, -33, 17, 1, 26, -41, 22, -137, 13, -23, -16, -1}
, {105, -63, -56, -152, -39, 97, -12, -122, -135, -59, 12, -89, -94, 6, 54, -244, -84, -3, -30, -28, 19, -32, 28, -20, 87, -146, -115, -31, -18, -119, -7, -115, -107, 87, 24, 59, -67, -37, 47, 20, 35, -167, 52, 63, 23, 59, -14, 24, 2, -2, 23, -207, 150, 5, 78, -129, -2, -26, 49, 6, -10, -150, -33, 16, 94, -23, -73, -81, -172, 19, 184, -25, -68, -67, 132, -57, 4, 40, -59, -74, -10, -13, 50, -68, 130, -1, 112, -93, -18, -110, -28, 35, 106, 27, -48, 24, -124, -14, 72, 93, -87, -61, 82, -2, -95, -23, -2, 23, -6, -12, -19, -67, -3, 88, 49, -43, 13, -133, -41, -110, -16, 11, 97, -6, 122, -35, 31, 115, 94, 8, -42, 29, -16, -4, -46, 6, 46, 22, 20, 13, 14, -66, 29, -38, 63, 44, 4, 11, 30, -22, 40, 9, 64, 1, 3, -35, 77, -66, -27, -41, 50, 49, 6, 63, -8, 77, 32, -58, 16, 23, 11, 58, 61, 31, 57, -17, 21, -24, 75, -27, 41, 64, 105, 21, -19, -19, -6, -16, 96, -8, -30, -35, 55, 60, 10, -50, 7, -32, -49, -62, 44, -28, 49, -59, 20, 23, -15, 28, -52, -36, 46, 27, -15, -32, 5, 24, 95, -53, 60, 15, 6, 15, -3, 78, -25, 8, -68, -12, 16, 11, -9, -93, -50, 6, 8, -62, 12, -38, 41, 13, 20, 29, 15, -77, 20, -11, -3, -21, 5, -76, 6, -38, 47, 5, 61, 63}
, {-38, -42, -35, 82, -79, 1, 0, 14, -27, 20, -56, -13, -103, -61, 48, 61, -25, 46, -2, -59, -27, -72, -5, -34, -67, 54, 10, -39, 2, -40, -20, -4, 51, 88, 26, -61, 71, 37, 49, 45, -22, 33, -13, -62, -13, -38, -39, 19, -6, -24, -5, -18, -79, -56, 0, -23, -1, -17, 9, 50, 35, 66, -15, 15, -121, 66, -12, 79, 8, -55, -168, 64, -1, 19, 2, 53, -36, -35, 10, 9, 27, 17, -48, 45, 16, -34, -89, 47, -2, -31, 27, -20, -41, -103, 36, 6, 14, 48, -208, -170, -32, 54, -120, 57, 36, -20, -106, 35, -53, -67, -68, -56, 4, -24, -32, 41, -30, 15, -29, 50, 28, 24, 26, 28, -8, 52, 4, -24, 53, -132, -108, -147, -63, -1, -68, -53, -34, -42, -17, -1, -47, 80, -144, 1, 8, -58, -20, -21, -61, -44, -173, 5, 11, 30, -17, -99, -149, 16, -40, -46, -50, -123, -6, -31, 7, -19, -8, 24, 41, -20, 48, -9, -102, -32, -77, -118, 2, 24, -29, -47, -51, 2, -53, 7, -120, -40, -19, -17, -60, -33, -105, -38, 20, 14, -60, -202, -28, -13, -10, -85, 45, -75, 35, -29, 24, 47, -44, -135, -64, -45, 95, 13, -1, 62, -10, -36, 11, -79, -128, 30, -20, 74, -34, 47, -8, 95, 96, -16, -68, -94, 3, -16, -145, -29, 24, -83, -89, 103, -29, -32, -54, 53, -44, 26, -53, -4, -58, -56, 82, -39, -27, -188, 58, -66, -22, 86}
, {44, 9, 85, 31, -14, 21, -90, 31, -2, 75, -28, -1, -72, -90, 5, 30, -47, 104, -16, -16, 17, -96, -69, 46, 122, -1, 53, -63, -121, -105, 53, -24, 37, 56, 78, -81, 21, -12, 61, 43, -17, -3, -34, 33, -131, -100, -144, -11, 21, 1, 65, -20, 23, 44, 11, -23, -50, -22, 6, 34, -28, 13, -15, 44, 18, 21, 6, -50, 56, -37, 5, 4, 59, -13, 38, 66, -91, -20, 43, -73, 14, 71, -54, -54, 47, 18, -39, 11, -18, 14, -6, 24, -33, -61, 11, 76, -18, 67, 6, -13, -87, 17, 139, 26, -79, 14, -28, -14, -42, 70, 47, -9, -75, -19, 18, 19, -9, -128, -41, -57, 105, 16, -1, -27, 104, -5, 25, 35, 72, 40, -75, -108, 1, 12, 44, -6, -120, -101, -2, -27, -17, 87, -17, 29, 33, -41, 10, 13, 52, 27, -33, 97, 116, 86, -129, 4, -68, 30, 6, 18, -88, 75, 51, 55, -65, -18, 46, -57, 62, 13, -136, 8, -33, 86, -1, -89, 61, 38, 42, 66, 65, 40, 69, -41, 53, -15, 9, 0, 4, 34, -40, -73, -40, 8, 27, -146, -38, 11, 67, -44, -43, -44, 41, 43, -36, 54, -93, -99, -85, 8, 73, -86, 79, -81, 77, -115, -4, -76, -26, -40, -51, 24, -35, 69, 72, 77, 97, 44, -63, -60, 56, -39, -127, -28, 23, -50, -79, 32, -29, -145, 67, 45, 50, 16, -38, -5, -128, -102, 16, -31, 44, -31, 81, 29, -33, -19}
, {20, 39, -15, 59, -51, -5, 27, 46, -61, 150, 2, 26, -115, -11, -11, 55, -12, 47, 30, 21, 13, -67, -15, 50, -25, -2, -48, -3, 22, -5, 50, -1, 51, 91, -19, -8, -22, 123, -8, 37, 28, 24, -8, 64, -134, -73, -64, 21, 3, -18, 50, -44, -30, 48, -3, -42, 85, 36, -21, 104, -11, -64, -21, 97, -62, -41, 17, -22, 0, 16, 59, 8, 11, -1, -11, 39, -85, 17, -17, -35, -24, 73, 44, 3, 73, 10, -20, 32, -7, -10, 8, 30, -33, -52, -25, 18, 52, -79, 12, -5, -18, 60, -25, 84, -46, 48, -62, 30, -43, 1, -12, -26, 60, 9, 58, 41, 40, 11, -31, 0, -55, -11, 68, 49, 0, 68, 23, -63, 36, 49, -28, -93, -36, 75, -21, -9, -101, -16, 34, -154, -78, 87, 33, -9, -5, -53, -5, -5, -1, 68, 74, -75, 60, -42, -110, 0, 22, -11, -42, -108, -38, 87, -39, 14, -102, -54, 89, -47, -42, -53, -10, 5, 113, 38, -19, -12, -43, -7, -57, -129, 78, 8, 93, -112, 102, 34, -29, -35, -26, 18, -44, -30, -20, -114, -38, 14, 19, -25, 6, 16, -31, 10, 30, -46, -51, 65, -56, -20, -137, -44, -21, 23, -70, -68, 39, -45, -38, 22, -55, 19, 15, 25, -61, 64, 2, -47, 54, 57, -86, -31, 95, 29, 4, 46, -121, 46, -73, 6, -40, 39, -191, 32, 16, -110, -56, -20, 9, -68, -53, -60, 51, -135, 50, -113, -14, 31}
, {-21, -2, -5, -14, 9, 74, -4, 57, -66, 42, 39, -20, 19, 120, 63, 7, -63, 56, 3, -12, 44, -56, 48, 11, 193, -19, 16, -21, -19, -40, 18, 73, 56, -5, -9, -19, 32, 17, 97, -51, -38, 9, -8, 19, -65, 84, 9, -18, 37, 38, -34, -34, 86, 13, 70, -69, -18, 29, 17, -48, -13, -4, -14, 28, 21, -38, -9, 25, -10, 56, 108, 53, -13, 42, 83, -80, 9, 46, 24, 11, 74, 72, 24, -65, 48, -24, 49, 8, 40, 46, -47, 123, 21, 117, -76, -8, 23, 0, 42, 38, -34, 10, 58, -11, 7, 8, -43, 27, -45, 2, -48, -54, 24, 58, 18, -11, 28, 88, -5, -2, 55, 16, -51, 59, 65, 0, -60, -25, -57, 22, 58, -82, -56, 22, -113, -69, 49, 27, -51, -23, 70, 59, 15, -41, 44, 101, 29, -87, 34, 59, 17, 45, 60, -21, 46, 6, 17, -92, -48, 37, -43, -13, -60, 73, 15, 45, -80, -97, -6, -5, 48, -74, 18, -36, 42, -6, -12, 0, -47, -8, 75, -15, -47, -57, -27, -81, -38, -47, 80, -6, -58, 31, -40, 3, 19, -46, 5, -113, 111, -8, 168, -76, -96, -39, -52, 109, -76, -12, 86, -72, -18, -40, 94, 55, -77, 0, -71, -40, 81, 3, 36, -98, -49, 127, 101, 43, 64, -15, 23, 55, 32, 19, -44, 5, 34, 29, -55, -19, 43, 33, -124, -16, 9, 75, -40, -36, -59, 30, 100, -70, 6, -78, 205, -140, -3, 138}
, {-13, -152, -38, 44, 72, 132, 23, 17, -104, 62, 44, -26, -129, -68, -31, 5, -26, -13, -7, 55, 32, 34, -94, -15, 136, -44, -111, -53, -120, -14, -118, -100, 30, -53, 29, 52, -13, -17, 27, -5, 51, -55, 36, 83, -41, 47, -111, -29, -50, -16, -58, -105, 147, -16, 130, -23, -15, -12, -46, 3, -75, 71, 2, 36, 85, -24, -17, 1, -25, 48, 41, 31, -34, -23, 197, -156, -64, 156, 27, -87, -113, 91, -36, 16, 32, 53, 13, -95, 58, -6, -102, 39, 49, 70, -82, -6, -143, -4, 130, 97, -124, -28, 73, -23, 7, -38, 6, -67, -42, 92, -8, -55, -24, 50, 7, -62, 59, -79, -16, -48, 79, -55, 25, -9, 30, 28, 33, 75, 8, 18, 37, 41, 33, 7, -4, 16, -22, 84, -52, 20, 21, -22, -16, -21, 20, 19, 31, -12, 51, -16, 8, -37, 36, 24, 38, 30, -3, 31, 16, -22, 48, -7, 19, 39, -26, 27, 68, -70, 28, 22, 1, -17, -64, -11, 44, 56, 2, 65, -34, -1, 61, 36, -32, 26, 79, 94, -42, -12, 27, 36, 37, 3, 78, -34, 18, -7, -15, -20, -53, -26, -5, 9, -29, -16, -5, 9, -31, -19, 45, 12, 89, 73, -41, 45, 80, 28, -8, 0, 124, 25, 14, -28, 109, 45, 80, -20, -2, -6, 130, 126, 42, 95, -56, 67, 15, 0, -48, -57, 15, 103, 19, 46, 71, -29, -47, -14, -20, -14, -2, 74, -5, 29, 2, 36, -1, 24}
, {-43, -30, 65, 71, -17, 40, -57, 23, 48, 35, 62, -20, 27, -57, -24, 44, -40, 64, -126, 59, 4, -105, -115, 7, -6, -11, 32, -76, -28, -86, -7, 52, 50, -3, -20, -47, -32, 101, 20, 20, 10, 13, -66, 43, -161, -43, -36, -38, -94, -30, -24, 29, 76, 39, 92, 21, 10, 33, -12, 79, -102, -76, -32, -135, -21, -69, -23, 32, 51, -35, 99, 21, 80, -33, 47, -28, -40, 11, -39, -88, -28, 114, -68, 47, 31, 137, 3, 36, 16, -38, 15, -18, -14, -4, 7, -2, 11, 53, -13, 26, -24, 19, 44, -7, -75, 7, -4, 11, -27, 3, 8, 84, -81, 23, 92, 40, 5, -45, 27, 18, -29, -25, -4, 75, 23, -48, -14, 35, 73, 46, -26, -52, -36, 8, -26, -27, 11, 45, 43, -108, -32, 111, -66, 25, -33, -75, -67, -73, -17, -6, 24, -114, 87, -23, 18, 36, 34, -7, 15, -13, 50, 9, -49, 33, -75, 13, -74, -31, -19, 54, -3, -52, 49, -6, -2, 31, -1, -1, 10, -58, 136, 26, 56, -26, -43, -23, -84, -58, 7, 89, -75, -56, -168, -32, -52, -25, 71, -71, -27, 4, 131, -42, 2, -13, 9, 64, 75, 35, -55, -43, -83, 111, -49, -108, 45, 18, -86, -88, -8, 60, 76, -1, -37, 35, -12, 63, 43, 4, -68, -55, 47, 38, 23, 6, -101, 24, -60, -34, -147, -38, -147, 69, -47, -88, -51, 29, 32, -84, 4, -47, 52, -71, 32, -167, 120, -29}
, {58, -47, 44, -74, -137, 92, 0, -12, -22, 32, 84, -13, -46, -29, -68, -164, -9, 65, -27, -45, -52, -38, -38, -94, 109, -62, -87, 2, -16, -143, -146, -55, -17, 44, 58, 71, -63, 75, -11, 53, -101, -69, 21, 26, 55, 10, 64, 34, 19, 55, -49, -72, 121, -82, 41, -120, -35, -108, -59, -52, 22, -92, -9, 44, 58, -28, -25, -33, -82, 23, 127, 127, 57, 38, 83, -101, -79, 58, -25, -42, 12, 87, 20, -42, -26, 5, 25, -14, -7, -18, 10, 64, 59, 33, -52, 27, -74, 24, -49, 16, -92, 46, 4, 48, -53, 5, -37, -75, 52, -62, -38, -21, 77, 73, -66, 0, 49, -188, -32, -154, 52, 68, 54, 7, 84, 72, -44, 91, 27, -36, -18, -66, -59, -63, 13, -38, -35, -24, 38, -83, -30, 26, -55, -55, 73, -15, -81, 44, 12, 12, 19, 49, 55, -42, -38, -119, -72, -74, -39, -59, 41, 25, 11, 105, -55, -33, -3, -74, -10, -10, 32, -19, 27, 48, -75, -25, -1, -3, 42, -37, 67, -26, -56, -41, 48, 45, -87, -19, -8, 28, 50, -132, 28, 49, -12, -62, 7, -29, 5, -1, -16, 1, 53, -77, -5, 62, -43, -4, -48, -30, 57, 87, -7, -49, 0, 26, 4, -26, 55, -89, 24, -11, 136, 39, -52, -2, -5, -26, -43, -99, 61, -57, 7, 1, 85, -83, 5, 27, -13, -9, 32, 21, -105, 63, 36, -9, -10, -87, 40, -10, 95, 10, -104, -30, 84, 34}
, {-101, 43, 73, 15, 53, -61, -31, -63, 60, -69, -37, -26, -13, -61, -36, -60, 36, -20, -13, -41, -39, -45, 11, 77, -214, 43, 16, 30, -4, -39, 23, 7, 21, 9, -2, -44, 4, -28, 28, -39, 54, 56, 7, -70, 50, 21, -39, 29, 8, 36, -23, 69, -151, -17, -75, 37, 45, 45, 48, -83, 22, -15, 50, -11, -71, 33, 96, -60, -70, -53, -136, -66, 50, -19, -53, 20, -2, -2, -71, -13, -57, 8, 5, 32, -179, 32, -55, 55, -47, -14, 27, 33, -1, -10, 15, 56, 33, 15, -136, -110, 45, 34, -173, -40, 37, 161, -59, -16, -37, -62, 41, 28, 29, 42, -113, 81, -100, 40, -18, -7, 28, 35, 11, -60, -23, -7, 29, 12, -42, -52, -51, 48, -80, -91, 25, 67, -17, -11, 22, 84, -5, -12, -92, 18, 75, -78, -18, 25, -98, -151, -87, 16, 1, 75, -16, -10, -107, -23, -124, -44, -32, -34, 8, -36, 43, 36, -27, 60, 23, 11, -43, 11, -27, -35, -76, -52, 53, 33, 44, 17, -95, 65, -22, 10, -34, -156, 18, -136, -1, -17, -52, 22, -28, 5, -26, -47, 112, -18, -115, -41, -72, 88, -10, -70, 7, 22, 64, -21, -9, -64, -29, 72, 41, -39, -86, -11, -3, -26, -82, -69, -69, 27, -65, -4, -111, -40, -63, 15, -72, -17, -13, -70, -8, -140, -26, 35, 22, 86, 25, -41, -24, -57, -33, -75, 7, 13, 56, 76, 22, -34, 12, 3, -129, -20, -106, -71}
, {-29, 27, -64, -4, 51, 18, -3, -34, -11, -150, 36, -28, 75, 65, -44, 36, -17, -85, 22, -20, 6, 15, 10, 20, 44, 49, 24, -35, 42, 54, -18, -12, 7, -23, -57, 20, -51, -57, -3, 22, -23, 8, -43, 21, -13, 2, 21, 54, -7, 52, -1, -33, -67, -24, 51, -24, -19, 12, -54, -5, -37, 39, 22, 19, -123, 38, -65, -58, 1, 18, -126, -58, -11, 15, -173, -7, 69, -19, -28, 6, 86, -176, -47, 4, -27, -31, -92, -72, 100, 46, -13, -97, -5, 68, -59, -77, 46, -43, 46, -25, 15, -25, -48, -105, -28, -100, -15, 6, 50, 53, -13, -70, 22, -139, 28, -67, -3, 37, 39, 2, -107, 0, -92, 93, -91, -58, -37, -68, -74, 108, 94, -52, 13, 15, 79, -125, -50, 9, 7, -136, 42, -53, 56, -58, -49, -60, -182, 0, 27, 52, 97, 16, -35, -156, -16, 26, 74, -86, 27, 7, 12, -7, -76, -21, -131, -69, 11, 3, 11, -20, -23, -4, 93, -12, 44, 22, -61, -145, -112, -18, 8, -101, -91, -35, 90, 72, 83, -64, 30, -33, -33, -60, 57, -74, -13, 71, -126, -30, 54, 9, 88, -78, 33, -90, -51, -75, -37, -64, 5, 24, 38, -73, -5, 55, 21, -24, -87, -149, 46, 27, -38, -56, -12, -14, 53, -1, 56, -15, 42, 35, 107, 62, -196, -58, 4, -27, -76, -188, -49, 20, -68, -40, -17, 8, -39, -95, -29, -54, -45, 14, -21, 26, 85, 20, -48, -9}
, {-32, 2, -18, -35, 20, -52, 45, -132, -31, -43, -51, -43, 43, 37, 52, 57, -42, -6, -44, -24, 17, 9, -25, 21, 39, -46, -109, -72, -73, 13, -32, -1, -9, 17, 22, -34, -52, -111, -8, -53, 101, -13, -2, -37, -2, 33, 19, -44, -2, -108, -55, -67, 9, 18, 44, 8, 54, 52, 79, -33, -21, -75, -32, -48, 13, -21, -33, -11, -38, -16, 15, -149, -70, 7, -41, -162, -58, -49, 1, -124, -69, -2, -105, 12, -8, 121, 34, 3, 13, -134, -90, 8, 46, -90, 34, 12, -32, -4, 88, 19, -11, -89, 38, -73, 55, -90, 68, 18, -65, 53, -41, 70, -166, -5, 9, 22, 95, 61, 20, 64, -84, 46, -3, -11, -27, -127, -43, -69, -39, -92, 43, 125, 19, -34, -101, 31, 16, 101, -92, 46, -41, -47, -3, 38, 5, -132, 43, -86, -173, -40, -183, -107, 9, 38, -32, 45, -53, 26, -91, 14, 4, 16, -2, 16, 65, -17, -17, 15, 93, -124, -61, -13, -83, -105, -82, -93, 36, -13, 50, -61, -125, 49, 35, 14, -110, -122, 57, 100, -50, 83, -60, 74, -2, -38, 1, -29, 19, 27, -95, -122, 24, 19, -54, 69, 65, 66, 55, -38, 59, -199, -32, -15, 11, 16, -138, 10, -35, -34, 59, 46, -52, 66, -79, -79, -16, 28, -82, -11, -16, 23, 7, -79, 9, -108, 45, 113, -75, 70, -16, -173, 11, 9, -83, -7, -48, 106, 62, 26, 49, -47, -16, -49, 90, 36, -72, 58}
, {10, -40, 27, 34, -40, 39, 12, -2, -63, -29, 6, -44, -2, 66, -3, 13, 44, -33, -43, 24, 41, 31, 39, 51, 9, 17, -12, 68, 4, 41, 16, -34, -31, -57, 8, -39, -10, 23, 22, -10, -41, 62, -6, -8, 53, 42, 8, -54, -3, -35, -9, 25, 56, -2, -46, -66, -11, 8, -4, -29, -9, -67, 83, 35, -31, -40, 2, 45, -55, 69, -10, -61, -36, 32, 66, -20, 47, 104, 16, 20, -36, 118, -4, 70, 17, -38, -114, 7, 66, -52, -37, 8, 26, 23, -58, -54, -39, -5, -54, 49, -9, 29, -10, 14, -6, -94, -11, 12, -34, 50, 14, -40, -2, -8, -35, 9, 13, -44, -14, -33, -91, -85, -25, 69, -127, -65, -42, 34, 10, 10, -14, -136, 73, -31, 29, -202, 36, -55, 1, -126, -37, -65, 121, -163, -66, -102, -92, 85, 37, 94, 36, 104, -136, -164, -20, 2, 68, -44, -35, 9, 76, 93, -24, -45, -23, -38, 18, -64, 25, 18, -65, -20, -29, -37, 81, 28, -133, -81, -101, -27, -58, 21, 50, 26, 112, 57, 60, -137, 37, -130, -19, -72, -40, 59, 66, 1, -9, 2, -27, 67, 15, 54, -93, 45, 99, -119, 6, 64, 9, 45, -81, 48, -93, -7, -89, -15, -50, 34, 46, 69, 115, -64, 10, 4, -29, -86, -41, -113, -60, 66, -67, 22, 59, 76, -121, -72, 35, -19, -27, 34, -40, -32, -5, 7, 62, 6, 22, -1, -27, -77, -86, 77, -97, 23, -38, -70}
, {71, -8, 15, -59, -101, -64, 74, 28, 36, 96, -39, 26, -118, -129, -21, 55, -13, -33, -73, -25, 36, 28, 35, 39, -86, 68, -25, -56, -44, -66, 77, 54, 51, -1, -41, -50, 90, 68, 45, -20, -51, 99, -61, 20, 54, -57, -168, -72, -87, 18, 21, 24, -159, -22, -94, 26, 48, 55, 3, -84, 78, -67, -65, 82, -132, 27, 47, -42, 83, -1, -147, -58, -105, 51, 1, 19, -15, 62, -13, 74, -40, -49, -45, -48, -21, -51, -4, 103, -46, 27, 63, 28, -77, -26, 92, -3, 41, 22, -140, -161, -26, -9, -70, -28, 102, 53, 15, -20, 53, -46, -49, -110, -110, -33, -39, -1, 22, -5, -1, -47, 22, 25, 28, 34, -4, -52, 48, -2, 69, 10, 8, 16, -31, 4, 68, -50, 24, -106, -13, -5, -62, 38, -35, -96, -28, 19, 17, -88, -43, -61, 20, 4, -23, -1, -12, -105, -8, -72, -81, -58, -34, 11, -20, -104, -41, 32, -36, 26, -28, 38, -3, -84, 8, -25, -147, 54, -20, 41, -12, -5, -134, -22, 9, -91, 11, 62, 18, -60, 20, -40, 21, -66, -83, 47, -111, -8, 12, -36, -56, -115, -43, 60, 44, -47, 73, 11, 32, 41, -107, -29, -65, -20, 77, -33, 23, -99, -8, -74, -136, 19, 65, -3, -122, 16, -103, 44, -5, -35, -99, -30, -93, -99, 17, -94, -53, -12, 88, 92, -2, -66, -42, -14, -49, -74, 36, -57, 36, -46, 60, -65, -45, -134, -34, 13, -68, -18}
, {74, -36, -24, -63, -72, 34, 32, -86, -75, 69, 9, -26, -111, 84, 104, -47, -76, 125, -14, -98, 84, 32, -33, 25, 30, -77, 24, -17, 29, -78, -15, -31, 35, 123, 27, 52, 17, 35, 99, 5, -16, -45, 69, -19, 53, -51, 39, 67, -47, 8, -28, -19, 91, 29, 47, -131, 54, -41, 51, 17, 27, -135, 24, 43, -30, -66, 30, 12, -57, 34, 51, 71, 24, 16, 117, -5, 62, -16, -124, 15, 48, 72, 40, -55, 45, -42, 25, 58, -76, 55, 96, 45, -24, 17, -28, 83, 19, -52, -106, 41, -65, 66, -14, 150, -71, 46, -85, -98, -72, -22, 7, -86, 22, 84, -58, 91, 18, -61, -130, -47, -43, -33, 17, 6, -8, 108, -47, 70, 48, 32, 31, -43, -33, 49, -143, 27, -36, 3, 68, -143, 27, 56, -8, -1, 26, 29, -56, -72, 61, -6, 25, -9, 19, -31, -47, -12, -17, 24, -38, -153, 20, -21, 22, 58, -58, 1, 19, -34, -80, -99, 28, -19, 17, -10, 40, 60, 20, 19, -12, -38, 2, -89, 102, -3, 0, -31, -222, 10, 5, 32, -31, -50, -69, -40, -73, 12, -37, -13, 6, 12, 26, 7, -27, -116, -81, 52, -57, 12, -95, -53, -148, 23, -72, -33, 28, 129, -65, 45, 42, 20, 59, 54, -45, 115, 85, 7, 63, 23, 5, -36, -77, -8, 27, -2, -110, -13, -82, -55, -110, 27, -43, 67, -14, -93, -33, -6, 6, 24, -57, 0, 55, 25, -63, -110, -15, 64}
, {20, -153, -6, 46, 53, 45, 2, -9, -61, -37, 14, -46, -148, -103, 19, -68, 43, -47, -154, -12, 45, -26, -15, 34, 2, -113, -61, -135, -74, -80, -60, -115, -40, -58, -29, 42, 26, -68, 28, 14, 21, -22, 31, -23, -55, -30, -25, 22, -138, 7, -25, -87, -6, 45, -11, -20, -61, 16, 59, -36, -32, -57, 41, -68, 73, 31, 0, -20, 22, -39, 11, -123, -49, 1, -39, -67, -35, -57, 49, -141, 67, -62, -192, -1, -18, -36, 74, -47, 69, -39, -45, -56, 6, -27, 41, -4, -66, -17, 43, 83, -101, -83, -55, -113, -11, -65, 42, 14, 23, 18, -78, 18, -96, -145, 92, -90, 36, -37, 60, 15, 14, 32, -51, -46, 49, -137, 23, 6, -94, 78, -16, -44, 47, -73, 1, 42, 60, -10, -106, 35, 21, -4, 19, -12, -2, 36, 76, -29, -34, 68, 30, -64, -51, 53, 71, 52, -14, 13, 13, 100, 45, -34, 7, 53, 21, -6, 19, -107, -22, -32, 71, 64, -167, 16, -63, 39, -22, 62, -31, -9, -9, -32, 34, -16, -52, -65, 38, -58, 27, -12, -42, 62, 16, -27, 40, -155, -25, 17, -7, -49, 16, -93, -62, 72, -44, 51, -44, -65, 40, 138, 60, -118, 99, 77, -46, -117, -21, 21, 38, 5, -56, 56, -71, 42, 95, 7, -152, -27, 58, 77, 2, 69, -123, -5, 129, 9, -53, 6, 77, -15, 57, 40, 2, 3, 14, -96, -90, -70, -6, -87, -42, -20, 49, 43, -130, 12}
, {-49, 98, -48, 38, 84, 4, 24, 62, -45, 55, -71, 86, 46, 92, -81, 126, -4, -47, 85, -37, 1, 100, 12, -52, -56, 45, 18, 34, -11, 187, 38, -13, 39, -12, -9, -102, 64, 19, -18, -29, -23, -11, -53, -5, -51, 21, -18, -8, 66, 21, 40, 37, -65, 98, -39, 74, 61, -17, -14, 19, 21, 7, -2, -20, -76, 11, -61, -34, 97, -67, -121, -140, -49, 13, -131, -13, -44, -53, 64, 13, -11, -69, -82, 68, -69, 1, -94, 24, 17, -44, -8, -103, -88, 70, -2, -70, 81, 22, -86, -1, 82, -55, 30, -159, 53, -11, 55, 110, -90, -6, 0, 34, -115, -121, -14, -3, -17, 53, 10, 95, 28, 39, 35, -13, -106, -113, -28, -112, -3, 106, -1, -94, 20, 91, 52, 5, -91, -57, 79, -48, 42, -14, 6, -53, -24, -90, -23, 87, 19, 13, 19, -30, -116, -90, -6, 87, 106, 6, 12, -38, -100, 48, 20, -48, -67, 37, -11, 24, 27, 72, -30, -8, 135, -9, 4, -6, -79, 22, 11, 16, 56, 21, 75, 9, 46, 20, 8, -73, 71, -121, 35, -75, -41, 63, 99, 40, 32, 43, 65, 25, -47, 91, 23, 35, 9, -41, 60, 69, 36, -57, -10, -29, -19, -2, -22, -99, -17, -35, 21, 90, 132, -50, -96, 54, 8, -95, 36, 40, 18, 42, 42, 0, -19, 60, -87, -23, 46, 12, 54, 21, -21, 44, 68, 69, 16, 77, -29, 7, 8, -46, 28, -28, -19, 46, -115, 13}
, {1, 28, -17, 40, 65, -9, -15, 106, 33, -38, 106, -3, 10, 43, -45, 103, 0, -65, -41, 6, -18, 1, -43, 96, -105, 9, 20, 34, 22, 121, 21, 34, -13, 19, -44, -75, 36, -32, -71, 88, 41, 23, -23, -15, -41, 77, -34, -60, -36, 49, 37, -28, -20, -6, -9, 44, 15, 72, 11, 27, 41, -11, 35, -15, -67, 35, -31, 9, 47, -5, -99, 23, -49, 40, -72, -35, 85, 51, 35, 58, -49, -20, 36, 54, -68, 23, -54, 15, 46, 20, 27, 24, 16, 7, -9, -70, 25, -88, -30, -20, 50, 19, -77, 24, -14, -11, 82, 63, 47, -29, 5, -25, 10, -79, -2, -1, -85, -11, 76, 6, -61, -63, -34, -11, -134, 8, -33, 28, -28, 106, -26, -69, -28, 83, -8, -97, 23, -26, -20, -56, 75, -68, 73, -13, -12, -151, -40, 100, -30, 18, 53, 6, -31, -58, -73, -36, 128, 39, 60, -19, -31, 0, 5, -67, -26, -67, 27, -36, -25, 105, 22, -19, 65, 43, 3, 62, -11, -3, -2, 24, -24, -21, 40, -35, 34, -36, 58, -137, 51, -22, 107, 11, -22, 46, 94, 93, -3, 5, -51, 15, -50, 132, -27, 66, 33, -177, 85, 68, 1, -20, -117, -36, -96, -14, -52, -36, -52, 16, 31, 12, 45, -103, 16, -121, -55, -163, -46, -22, 116, 68, -95, -38, 75, 54, -125, -34, 94, -91, -61, -2, 49, -12, -12, -46, 60, 42, -51, 42, -184, 29, -41, 71, -191, 48, -27, -107}
, {-37, -15, 22, 48, -133, 6, 73, -54, -21, -20, -13, -43, 73, -97, -33, -37, -61, -30, -100, -76, 73, 33, 1, 11, -28, -75, 0, 63, 68, -126, 71, 35, -63, -28, -70, -31, -104, 28, 32, 8, 38, 27, 12, -58, 71, -159, 25, 43, -44, -28, -34, 55, -29, -51, -50, -55, 31, 35, 19, 39, 61, -2, 18, -39, -37, -39, -9, 5, -74, -40, -34, -77, -2, -6, -32, 64, -42, 37, 44, -67, 43, -64, 17, 42, 2, -44, -16, 2, 30, -26, -62, 40, 34, -27, -26, 9, -12, -56, 92, 81, 30, -19, 52, -94, -30, -18, -74, 72, -2, 58, -43, 15, -15, -6, 62, 3, 15, 25, 90, 6, -91, -3, -54, -1, -17, -54, -33, -55, -115, -74, 57, -29, 95, -76, -26, -33, -37, 28, -40, 31, -60, -128, -4, 10, 6, -70, -45, 33, -76, 5, -67, 14, -10, 56, 24, -106, -44, 45, -96, 56, -33, 56, 31, -43, 105, -2, -93, -82, 212, -167, 43, 108, -100, -22, -12, -66, 93, -27, -154, -9, -77, 160, 121, 112, 48, -5, 147, -103, -62, 10, -6, 8, -49, -97, 12, -46, -69, -14, -98, -66, -21, 7, -56, 16, 24, 0, 71, -49, 69, -71, 59, -50, 15, 90, -168, -96, -49, -53, -77, 57, 20, -42, -58, -9, -2, 18, -79, -59, -7, -52, -81, 21, -30, -53, -7, 39, 1, -48, 40, -54, -2, -1, -25, 51, 30, -52, 1, 6, 132, 13, -109, -46, 27, 113, -142, 0}
, {77, 24, -25, -53, -73, -77, 6, -149, -25, -74, -65, -76, -51, -85, -43, -71, 39, -32, -124, -40, 13, -101, -29, 57, 57, -6, -13, -83, -80, -126, 38, -22, 13, 13, -42, 38, -4, 11, 54, 32, -20, -23, 35, 0, -25, -35, 45, 48, -126, -107, -15, -23, -81, 40, 3, 33, -8, 24, 52, 109, 74, -96, 73, 32, -27, -63, -54, 10, -105, -127, 50, -30, 34, -62, -193, -3, 56, 2, -129, -43, 34, -69, 9, -97, 74, -95, 48, 15, -199, -22, 61, -19, 11, -101, 56, 60, 48, -152, 52, 73, 20, 4, 105, 49, -74, 135, -3, -19, 16, -6, -57, -8, -16, 4, 93, -20, -69, 40, -103, -4, -44, 28, -31, -16, 48, -27, 64, 23, -47, 58, -36, 3, -21, 76, -38, 4, -36, 45, 48, -72, 82, -21, 44, -8, -65, 5, -5, -16, -28, -38, 37, -31, 54, -76, 37, 123, 58, -19, -57, 49, -17, -32, 15, 19, -7, -41, -7, 8, 109, -35, 29, -17, 101, -42, 43, -24, -2, -35, -70, 6, -39, 15, 47, -89, 35, -1, 49, 29, 19, -10, 42, 3, 36, -73, -69, -1, -60, 45, 12, 71, -15, 15, -48, -6, -45, -7, 38, 3, -18, -94, -15, -24, -28, 4, -33, -52, 36, 0, -52, 20, 98, -21, -61, 36, -32, -45, -22, -25, -90, 64, 51, -40, -31, 24, -99, -23, 70, -6, 19, -105, 52, 76, -22, 18, 15, -46, 24, -28, -8, 27, -86, -67, 4, 81, -25, -39}
, {93, 25, -70, -19, 103, 48, 45, -35, 60, -28, 112, -56, 69, 95, -70, -23, -72, -76, 23, 141, 28, 8, -51, -69, 0, -12, -33, 55, 43, 37, -63, -66, 3, -23, -6, 63, -100, -77, 11, 50, 27, 35, -121, 55, -63, 23, 15, 40, 42, 19, -21, -52, -123, -49, 49, 61, 24, 25, 3, -69, 31, -22, -38, 59, 31, -30, -31, -44, 70, -44, -38, -171, 20, 1, 11, -64, -30, 48, 106, -80, 36, -4, -109, 29, -57, 80, 70, -84, 1, -11, -91, -19, -39, 65, -73, -17, -12, 2, 88, 46, -23, -108, -32, -100, 69, -128, 8, 127, 42, 52, 57, 13, -13, -99, 38, -33, 24, -110, 78, -46, -11, -65, -80, 27, 74, -47, 55, -9, 34, -8, -90, 4, 49, 43, 28, -1, 18, -95, 41, 38, -5, -1, 135, -7, -57, -31, -112, 6, 61, -100, 70, 72, -145, -5, -58, -13, -8, -11, 35, 14, 30, 114, 22, 4, -126, -77, -4, 25, 64, 27, -131, 82, -65, -5, -81, 15, 50, -70, 10, -73, -74, 82, 16, 1, 133, 42, 91, -34, 7, 46, -49, 21, 17, 68, 74, 47, 23, 30, -10, 14, -53, 108, 4, -51, -15, -71, 30, 25, 0, 16, 13, -16, -49, -62, 94, -60, -32, -115, -43, 7, -101, -46, -6, -59, -64, -27, 40, 28, -25, 24, 44, 96, -12, -14, -64, 50, 81, -44, -60, -6, 67, -65, -12, -42, -31, -18, -25, -40, -46, 11, 69, 53, 7, 51, -62, -51}
, {54, -60, 3, -52, 42, 31, -30, -112, -53, -44, 82, -11, 21, 28, 17, -38, -22, -18, -79, -36, 55, 17, -10, 25, -92, 0, 29, -12, 32, -58, 26, -40, -38, 25, 8, 40, -104, -61, 8, 18, -16, -26, 102, 36, 71, -1, 68, 38, -48, 20, 49, 2, -15, 42, 1, -13, 26, 4, 3, -19, 58, -91, 9, 68, -28, 23, -37, 28, -88, 60, 94, 73, -24, 27, 67, -18, 51, -29, -98, 16, -39, -11, 24, -47, -82, -1, 31, 22, -87, 64, 60, 90, 31, 65, 13, -22, 21, -13, -65, -60, -22, -30, -67, 21, -55, 98, -4, -114, 58, -77, -3, -31, 32, 63, -43, 53, -17, -5, -36, -24, 47, -21, 33, -57, -35, 71, 32, 139, -45, -27, 42, -2, 34, -41, 6, -67, -4, -4, 60, -59, 32, -18, -1, -28, 27, -48, -72, 51, 72, 10, 31, -5, 18, -9, 15, 54, -54, -16, -34, -63, -38, 68, 69, -3, -32, -68, 56, -3, 22, -58, -36, 89, -33, 0, 50, 15, -63, -69, -43, -37, 12, 58, 37, 70, 109, -22, -45, 36, 75, -48, 42, -1, -84, -14, 22, 13, 30, 3, -33, 23, -124, 68, -51, -19, -55, -180, 20, 64, -121, -162, -156, -7, -64, -133, -76, 14, 66, 11, -59, -75, 51, -39, 22, -144, -1, -160, 137, 92, 28, -36, -9, -105, 56, 29, -81, -45, 103, -45, -80, 68, -30, -120, -15, -98, -8, 64, 27, 23, -144, 5, -13, 81, -74, 93, 19, -257}
, {96, 11, 17, -77, 9, -18, -46, -32, -78, -47, -24, -143, -80, -103, -12, -134, -28, -45, -107, 31, 0, -89, 23, 7, 79, -108, -63, -59, 41, -94, -31, 13, 0, 24, 47, 79, -106, -42, 13, 16, 23, -81, 29, 69, 87, 25, 22, -14, -153, -98, 17, -89, 57, 83, 42, -24, 3, 54, 28, 20, 22, -72, -10, -6, -24, -2, -18, -78, 2, -68, 42, -164, -5, 24, -107, -40, -66, -83, 3, -64, 48, -94, -15, 31, 87, -15, 32, -106, 11, -59, -37, -24, 21, 32, -24, 25, -15, 0, 30, 36, 77, -67, 27, -181, 12, -22, -6, 48, 17, 67, -43, 50, -27, -168, 120, -62, 4, 76, -12, 6, 33, -6, 30, 24, 107, -73, 18, -64, 75, 71, 9, -107, 30, -14, -16, -26, -8, 12, 31, -54, -21, -131, 55, -49, 7, 84, -10, -17, 39, 28, 33, -49, 72, -11, -45, 55, 105, -90, -33, 14, -67, 49, 58, 4, -17, -11, -3, -76, 35, -77, 7, 85, 22, 9, 69, -13, -104, -59, 5, -13, 73, 96, 20, 80, 29, 10, -54, 11, 49, -26, -86, 29, 14, -104, -40, -17, 8, 68, 3, 25, 23, -86, 18, 21, -38, -33, -43, -16, -66, -41, 61, 18, 5, -109, -87, -29, 36, -35, 1, -6, 16, 22, -61, -8, -29, 46, 45, 54, 4, 5, 3, -57, -73, -78, -4, 130, -42, 53, -23, -40, 43, 36, 34, -44, 23, 83, -16, -62, -33, -32, 39, -121, 114, 21, -33, -2}
, {-19, 91, -37, 16, -75, 0, 41, 27, -31, -9, -25, -3, 10, 125, 14, 19, -52, 29, -54, -71, 23, 22, -4, 25, -114, -38, -3, 8, 13, 52, -4, -50, 57, 10, 25, -68, -52, 34, 66, 44, -74, 1, -52, -62, 3, -98, 70, -34, -18, 44, -21, 63, -97, 15, -15, 21, 42, 92, 22, 58, 12, -34, -33, -1, -38, 15, -12, 36, 36, 15, -35, 32, -77, 64, -21, 24, 7, -72, 22, 76, 19, -23, 27, 21, -39, 0, -75, -8, 3, -55, 18, 40, -19, 64, 35, 20, -43, -61, -36, -88, 54, 68, -163, 35, 64, -49, 5, -45, 23, -101, 39, -84, -21, 46, -48, 62, 21, 40, -39, -16, -93, 40, 14, 47, -138, 66, 35, 8, -76, -35, 20, -16, 4, -37, 8, -52, 55, 10, 73, 7, -9, -93, 5, -81, -21, 12, -10, 26, 68, 25, 100, 34, -21, -50, 54, 81, 40, 21, -24, -109, -6, 64, 49, -10, 37, -11, -56, -47, -12, 43, 59, -38, 52, -23, -18, 86, -45, -49, -34, 3, 69, 46, 16, 63, 61, -44, -15, -35, 55, -83, -13, -42, 1, 17, 8, 115, 8, -10, 13, 34, -55, 92, -28, -32, -15, -78, 75, 122, -65, -56, -129, 55, -162, 48, -133, 75, -34, 54, -13, 8, 28, -50, -68, -102, -45, -83, -77, 55, -12, -3, -6, 1, 151, 110, -79, 14, 5, -15, 23, 90, -51, -45, 47, -58, 14, 3, -25, 24, -66, 77, -104, 63, -124, -20, 22, -78}
, {-72, 12, -52, 34, 55, 24, -77, -2, -51, 23, 32, 86, -11, -9, -20, 37, 12, -19, 41, 54, -83, -112, -85, 6, 111, 55, 38, -59, -9, 88, 10, 32, 73, 22, -10, -84, 65, 73, -136, 60, 170, 12, -150, 11, -85, -8, -78, -71, 52, -32, 28, -3, 72, 3, 7, 108, 14, -54, -28, 92, -63, 53, -52, 12, 52, -70, -6, 1, 38, -104, -7, 20, 27, -7, -2, -47, 35, 72, 53, -93, 14, 102, -61, 14, 10, 99, -12, 76, 42, -16, -27, -66, 11, -27, -50, -5, -2, 115, 44, 50, 17, 30, 212, 89, -1, 64, -2, 45, -51, 1, 10, -40, -104, -76, 93, -10, -8, -66, 31, -5, 65, 34, 26, 42, 20, -60, 32, -15, 123, -54, -152, 26, -35, 112, -8, 30, -151, -213, 33, 0, 51, 66, -134, -6, 55, -45, 4, -6, -152, -158, -165, -110, -95, -38, -138, -71, 13, -13, -45, -143, -136, -78, -53, -6, -99, -77, -64, 72, -47, 94, -65, -73, -98, 12, -96, -61, -31, 88, 133, -5, -99, -23, -13, -131, -96, 57, -52, -75, -94, 22, 7, -70, 19, 56, -68, -5, 34, 2, 3, -15, -65, 107, 47, 62, 91, -32, 24, -11, -65, 40, 60, 13, -20, -66, 158, 16, -15, 13, 4, 39, 31, 50, -2, 88, 10, 14, 19, -7, 61, 20, 26, 14, -9, -42, -9, -32, -8, 29, 40, -100, 45, 29, -15, 55, 38, -18, 8, 53, 47, 72, 49, -80, -22, 4, -46, 20}
, {39, -32, -18, 68, -44, 103, -51, 2, -40, 56, 106, 63, -7, 68, -59, 33, 23, 40, 18, 55, -29, 40, -2, 54, 149, 45, -9, -2, 1, 63, 14, 2, 15, 22, -7, 10, 47, 69, -25, 12, 29, -19, 50, -5, -132, 70, -92, -15, 11, 90, -57, -56, 198, 1, 87, -73, 51, -72, -29, 33, -7, 11, -12, -35, -16, -34, -5, -3, -56, 25, 51, -20, -10, -30, 154, -133, -98, 78, 52, -28, -15, 91, -47, -46, 41, 34, 30, -8, 29, 21, -81, 36, -9, 79, -5, -41, 9, 63, 104, 53, -73, -53, 147, 85, 30, -1, 2, -74, 32, 2, -13, -37, -35, 73, 43, -51, 52, -24, 41, 6, 70, 1, -21, 18, 26, -51, -31, 7, 5, 24, -23, -167, -48, 75, -47, -50, 15, 25, 0, -71, 83, 59, -22, -67, -6, 17, -14, 30, 78, 58, 87, -14, -16, 25, -27, 12, -20, -13, 13, 45, 56, 14, -13, 54, 23, 25, 48, -148, 48, -48, 2, 12, 16, -13, 47, -51, 4, 47, -8, 51, 45, 10, -14, -27, 31, -41, -74, -136, 32, -3, -33, -63, -43, 30, -49, -99, -36, -135, -4, -23, 57, 36, -179, 64, 47, 31, -18, -48, 24, 37, 37, 40, -56, -45, -34, 2, -74, -67, 91, 1, 55, -17, 35, 216, 144, 5, -81, 9, 79, 52, -92, -5, -72, 19, 4, -46, -10, -107, 77, 21, -55, -58, -49, 100, -107, -50, -37, -8, -6, -18, 62, -30, -16, -34, 15, 120}
, {-40, -52, -10, -22, 30, -23, -36, 26, 65, 7, -51, 12, -12, -36, 5, 14, -36, -22, 26, 42, -101, -21, -70, -25, -114, 56, -24, -52, -6, -17, 7, -33, -66, -12, 35, 26, -14, -63, -3, 0, 3, 4, -68, 39, 17, 19, 23, 14, 37, 62, 7, 74, -80, -17, 73, 16, 34, -73, -37, -44, -53, 20, -45, 34, 121, 42, -8, 5, -36, 111, 34, 61, 26, 5, 69, 18, 7, 29, 37, -29, -28, 15, 42, -16, -68, 36, -6, 37, 3, 10, 56, 43, 55, 6, -18, -42, 3, 81, -22, -8, -21, -11, 60, 7, -6, -1, -19, 73, -3, 45, 52, -23, 14, -48, 50, 13, -5, -55, 27, 4, 21, 40, 3, -8, -10, 71, -39, 52, -8, -39, -127, 68, -2, -21, -131, 137, -119, 30, 39, 65, -30, 42, -111, 82, -87, -30, -6, 0, -156, -231, -151, -137, 36, 82, -14, -28, 33, 2, -24, -29, -104, -121, 51, -21, -26, -10, -56, 67, -79, 86, -143, -166, -140, -189, -84, -33, 28, 103, 150, -25, -53, 14, -121, -63, -96, 23, -8, 21, -179, 81, 100, 105, -32, -93, -28, 37, 34, 53, -173, 40, -50, 83, 35, 31, 19, -6, 60, 5, -71, 3, -76, 4, -98, -156, 100, 32, 55, 20, 8, -173, -55, 60, 102, -53, -115, -63, -56, -36, 19, -3, 4, -7, 36, -10, 33, 51, 67, 47, 13, -51, 41, -141, -51, -100, 50, 85, 4, 5, -132, 130, 44, 57, -95, -29, 66, -18}
, {62, 65, -67, -18, 102, 77, 34, 29, -5, -102, 19, -35, 21, 59, -66, 145, -8, -64, -60, 13, -42, 42, -17, -67, -100, 33, 58, -10, 19, 19, -60, 51, -42, 44, -1, 7, -11, -32, -58, 15, -27, 43, -71, -16, -73, 34, 3, -70, 3, 3, 12, 8, -152, -68, -50, 22, 52, 7, -84, -43, -4, 59, -39, -45, 30, 10, -41, 10, 32, 26, -29, -101, -35, 28, -92, -66, -26, 23, 30, -17, 0, 18, -156, 152, -19, 28, -65, -4, 49, -7, -39, -88, -34, 23, -31, -30, -59, -8, -58, -13, -2, 3, -32, -97, 27, -100, 106, 128, 33, 75, 36, 64, -70, -190, -61, -27, 17, -53, 46, 26, -46, 14, -41, -12, -34, -165, 2, -80, -64, 64, -55, -58, 44, -20, -33, -24, 25, -70, -50, 9, -29, -81, 7, 17, -13, -20, -14, 98, 14, -53, -1, -28, -95, -17, -46, 51, 53, -16, 68, 46, 3, 19, -23, -10, -20, -44, -56, -3, 45, 91, -52, 40, 71, 24, -30, 42, -20, -41, 28, 42, -37, 8, -14, -30, 39, 43, 44, 13, 11, -86, 67, -12, 43, -19, 134, 46, -14, 42, -123, -13, -73, 62, -12, 18, 79, -207, 55, 29, 49, 42, -69, 1, -100, -46, 53, -32, -24, -35, -72, 43, -33, -2, -20, -110, -53, -36, -81, -94, -19, 5, -7, 8, 65, 95, -24, 57, 78, -41, -38, 65, 23, -164, -62, -44, 68, -5, -28, 68, -76, 21, -19, 78, -128, -9, -35, -188}
, {-2, 8, -43, -73, -60, -155, 3, -28, -31, -68, -89, -78, -52, 37, 15, -116, -132, -20, -14, -60, 30, -18, -18, 62, -41, -144, 5, -18, 38, -5, 35, 16, -65, -23, -13, -21, -66, -27, 20, 72, -31, 6, 26, -6, 45, 14, 65, 75, -99, -49, -79, 1, -79, -100, -6, -50, -2, 35, -3, -40, 74, -68, 45, 3, -32, 24, 8, -29, 27, -134, -122, -106, -45, -8, -81, 9, -53, -1, 16, -4, 24, -104, 3, -14, 6, -48, -44, -3, 55, -5, 8, -13, -1, -83, -48, 7, 17, -82, -73, -61, 19, 4, -43, -95, 45, -56, -30, 39, -30, -11, -33, 36, -44, -23, 8, -10, -10, 63, 13, 23, -62, 50, -56, 40, -19, -79, -48, -37, -8, -72, 107, 8, -3, -56, -140, 47, 35, 100, 11, 58, -66, -44, 13, 3, 54, 125, 54, -76, 55, 15, 67, -37, 44, 39, 72, -45, -74, -59, -98, -38, -8, -3, 67, -21, 47, -1, -154, -74, -49, -80, 121, -37, 19, -110, -22, -67, 23, -5, 27, -82, 87, 0, -128, 55, 23, -123, 16, 62, 1, 41, -92, -4, -8, -51, -68, 18, -24, 52, -14, 55, 27, -113, -34, -68, -9, 110, 66, -32, 39, -33, -8, -38, 43, 85, -86, 22, -29, -53, -53, -5, -58, -27, -98, -124, -8, -11, -41, 59, -114, -53, 138, 44, -62, 25, 29, 44, -125, 87, 32, -35, -105, 15, 33, 2, 19, 21, 21, -25, 27, -34, -52, -112, 111, -13, -1, 14}
, {-111, 14, 53, 61, -83, -129, 19, -33, 2, 28, 82, -11, 2, -60, 27, 13, 93, 17, -101, -101, 32, -5, 62, 28, 41, 51, 104, -17, 28, -22, 67, -15, -2, -17, -36, -59, -57, 8, -16, -3, -24, 92, 127, -62, 45, -8, 30, -7, -21, -15, 8, -7, 167, 66, -64, -7, 3, 48, -31, 15, -42, -11, 87, -92, -106, -95, -50, 8, -30, 6, 14, 55, 23, -70, -31, 21, 35, -106, 2, -10, 60, 33, 0, 20, 48, -33, -39, 18, -43, -16, -30, -19, -17, -5, 20, 34, 20, -37, 33, -12, 34, -5, 46, 87, -25, 128, -31, -3, -39, -38, -12, 7, -45, 60, 87, -13, -97, -30, 5, -11, -80, -6, -55, -44, -22, 9, -10, -121, -163, -117, -87, -60, 111, 11, 40, -98, -39, -17, -10, -3, -41, -5, -5, 65, 6, -96, -123, 56, 5, -82, 22, -16, -27, 39, -99, -47, -89, -18, -13, 27, -82, 110, 68, -2, 21, -76, 19, -35, 211, -81, 14, 108, -20, 74, 27, -66, 21, -92, -56, -108, -12, 102, 125, 25, 47, 46, 131, -125, 8, -81, -45, -39, -23, -60, 47, 6, -56, 11, -54, -6, -130, 39, -1, 72, -67, -25, 47, 82, -3, -64, -36, -59, -32, -14, -74, -61, -3, 65, -47, 27, 119, -17, -42, 4, 27, -82, -27, -132, 27, -24, -104, -35, 12, 38, -49, -37, 88, -5, 4, -29, 0, 7, -11, 65, 19, -15, -24, 73, 17, 36, -40, -7, -73, 73, -16, -45}
, {65, -28, -25, 68, -71, -21, -37, 15, -79, 42, -58, -22, -45, -60, -56, -83, -38, 42, 41, 46, -36, -139, -28, 13, -131, 25, -20, 44, -35, -95, 41, 30, -5, -78, -19, -24, 26, 37, 31, 28, -63, 1, -131, 69, -43, -28, -45, -1, 15, 3, -24, 65, -171, -27, -23, -54, 0, 31, -64, 7, -2, 44, -29, -70, 81, 58, -8, -104, 35, 57, -10, 11, -3, 56, -65, 56, -91, -35, 48, -77, 71, -95, 73, -93, 82, 2, 102, 2, 49, 30, 51, -37, 32, -105, 19, 124, 53, 48, 16, -32, 5, 45, -198, 19, -19, -11, -23, 43, 71, 8, 12, -16, -27, -65, 31, 86, 63, -38, 48, -72, 17, -32, -27, -65, 70, -15, -69, 70, 64, -32, -42, 21, 12, 80, -58, 25, -30, -8, 21, -3, -13, -99, -47, 66, -35, 36, -31, -18, 10, -83, -28, -1, 12, 30, 63, -54, 51, -88, -52, -18, 36, -8, 92, 1, 20, -13, 67, 14, 17, 53, 32, 16, 83, 7, 58, -23, -47, -81, 51, -24, -78, -63, -69, 19, 86, -25, -21, 45, 10, -39, 7, -114, -37, -68, -75, -100, -70, 60, -92, 99, -102, -115, 54, -62, -103, 134, -81, -66, 39, 107, 84, -142, -31, 16, 38, 6, -11, -22, 36, -44, 49, -3, 31, -18, -96, -94, 110, 99, -115, 25, 20, 16, -69, 115, 91, -91, -23, -1, -53, 36, 23, 45, 23, 39, 15, -108, -66, -143, -78, -81, 37, -41, 1, -36, -3, 5}
, {-10, 91, 84, 3, 36, -41, -28, -29, 12, -33, -75, -36, 78, -19, -13, -11, 58, -63, 8, -65, 2, 46, 24, 9, -187, 84, 46, 33, 14, 110, 45, 20, 108, -9, -14, -91, 22, 53, -21, -2, -23, 72, 55, -51, 44, 13, -34, 48, -74, -57, 28, 25, -192, 29, -73, 85, 50, -42, -5, -100, 23, 96, 50, 51, -273, -2, 85, -39, 60, 38, -112, 62, 4, 62, -110, 73, 51, -110, -4, 1, 49, -82, 6, 29, -113, -27, -105, 52, -86, 38, 32, 9, -25, -18, 33, 0, 84, -74, -210, -122, 111, 126, -155, -62, 66, 106, -60, -22, 8, -17, -30, 60, 11, 43, -115, 98, -85, 92, 11, 165, 2, 64, 24, -18, -85, 47, 40, -32, -34, 10, -82, -46, -62, 5, 111, -1, -61, -39, 94, 5, -25, -37, 20, 18, -15, -66, 52, -52, -6, 11, -26, 7, 88, -5, -14, 20, 13, -5, -73, 67, -36, 20, -14, -44, -17, 7, 19, 106, 49, 34, -83, 1, 91, -2, 7, -28, -39, 77, -20, -3, -11, 70, 6, -52, 1, 39, 34, -39, -27, -13, 21, -18, 0, -21, -28, -63, -3, -5, 12, 17, -61, 67, 58, -41, 15, -38, 36, -42, 15, -41, -33, -9, 26, 34, 8, -39, 18, -54, -38, 82, 57, -7, -50, 56, -55, -18, 83, 35, -23, -22, 23, -3, 5, -27, -91, 20, 15, -2, 10, -32, 22, 54, 3, -56, -10, -13, 106, -18, 12, 32, -50, -30, 15, -9, -17, 17}
, {-14, -21, -17, -67, -85, -45, -8, -31, -19, 46, -2, -109, -46, 96, 48, -12, 34, -63, -6, -19, 37, -24, -3, -11, 86, -18, -38, 83, -6, 61, -84, -12, 8, 55, -9, 97, -62, 83, 36, 6, -46, 27, 5, -39, 84, -30, 21, 1, 6, 3, -13, -26, 31, -50, -33, -11, 25, 30, 21, -65, 20, 16, 43, 20, 32, 0, 37, 33, -35, 75, 107, -91, -32, -17, 156, 8, 29, 63, 9, 24, -76, 7, -36, -24, -37, 1, -72, -80, 119, -60, -69, 43, 12, 32, -53, -97, -96, 26, 42, 106, -22, -117, 46, 0, 35, -94, 64, -12, -7, 31, 8, 57, -48, -40, 21, -33, -23, 22, 17, -81, -30, -39, -82, -43, 9, 7, -17, 19, 37, -14, 2, -50, -22, 4, 125, -72, 24, 26, -50, -131, -29, 4, 62, -19, 76, 23, -96, 95, 6, -21, 25, 64, -111, -57, 82, 29, 0, -18, -23, 49, -15, 52, 1, -45, -34, 44, 83, 48, 29, 21, 45, 175, 92, 25, 57, -14, -27, -115, -141, 23, 0, -35, 68, 49, 57, 60, 90, -61, 40, -60, 6, -12, -8, -1, 43, -50, 13, 73, -28, -73, -95, 30, 20, 95, 43, -34, -46, 53, 24, 140, -8, -58, -30, 163, 18, -58, -6, 43, 49, -32, 74, -26, 62, 68, -10, -44, -121, -75, -63, -12, -98, 10, -28, 37, 68, -57, 71, -29, 42, -28, 37, -65, -104, -3, 50, -68, -18, -26, 112, -109, -125, 53, -50, 52, -21, 81}
, {-90, -31, 42, -50, -71, -38, 1, 21, 35, -17, -48, -41, 11, 7, -3, 43, 7, -41, -36, 42, -44, -56, -56, -43, -110, -56, -38, 8, -11, -32, -59, -15, -10, 7, 1, 2, -32, -72, -7, -25, -10, -18, 1, 51, 71, -9, 12, 47, -27, 42, 18, 22, -47, -41, 31, 18, 47, -108, -49, -50, -52, -5, 15, 1, 11, 21, -14, -41, -22, -53, -98, -4, 43, 16, -72, -3, 22, -21, -63, 27, -2, 5, 51, -12, -90, 14, 17, 5, -22, 46, -35, 38, 17, -4, 59, 45, 49, 75, -16, -46, 16, -23, -53, -104, 5, -1, 2, -107, 53, 68, 68, -6, 87, 70, 35, -1, -5, -30, -16, 56, 31, -53, -13, -47, -95, 1, -82, -12, -22, -129, -16, 100, -19, -7, -122, 93, -123, 60, -18, 120, -7, -23, -47, 49, -17, -4, 124, -80, -153, -165, -66, -46, 137, -14, -6, -50, 44, -20, 3, 21, -62, -55, 12, -18, 12, 138, -86, 24, 11, -111, 11, -103, -61, -96, -114, 34, 98, -27, 152, -70, -9, -72, -123, -70, -56, 0, 50, 127, -162, 77, 54, -8, 12, -154, -71, 57, 13, -6, -18, 28, -65, -28, -15, 5, -177, 56, 38, -18, -33, -100, -89, 62, -60, -87, -41, 14, 80, -31, -90, -162, -39, 39, -61, -108, -3, 19, -10, 37, 12, -76, 46, 62, 40, 20, -48, 95, -60, 59, -92, -83, 3, 8, 96, -78, -57, 24, 52, -14, -55, 65, 39, -23, -60, -33, 69, 3}
, {56, 10, 12, -15, -5, 17, 11, -152, 7, 30, 8, -81, -13, 126, 25, 12, -19, 1, 3, -59, -23, -11, -14, 2, 81, -68, -73, -54, 14, 32, -44, -114, -55, -14, 28, -2, -66, -90, 50, -65, 11, -99, 92, 6, 39, 20, 22, -78, -11, -164, -10, -15, 98, -82, 11, 44, -17, -7, -11, -10, -11, -66, -43, -2, 77, -88, -54, -13, -38, -38, 101, -55, 16, -42, 30, -205, -58, 106, 53, -29, 32, -7, -100, -19, 59, 55, 17, -50, 10, -96, -70, 64, 83, -3, -43, -67, -26, 0, 187, 59, -29, -105, 75, 29, -79, -9, 1, -24, -84, -9, 0, 71, -166, -14, 64, -59, 26, -98, 64, -34, 15, -51, 13, 38, 80, -129, -8, 31, 87, 3, 48, 58, 68, -15, -83, -27, -11, 46, -10, 27, 30, -43, -46, 5, 67, 29, -18, 84, -18, 84, -36, -73, 32, 6, -36, 14, -42, 37, -121, 47, 3, 3, 20, -9, 54, 51, -82, -80, 63, -137, 3, 97, 15, -12, -31, 15, 63, 2, -51, 14, 114, 53, 105, 92, -60, -68, 74, 0, 27, -1, -3, -25, -49, 20, 89, -40, -7, -36, 33, -51, -13, 25, -57, 4, 27, -8, 62, -4, 34, 24, -15, 13, -42, 8, -37, -16, -81, -85, 3, 26, -15, -81, -11, 73, 116, -14, -95, -72, 32, -53, -65, -3, 104, -14, 7, 66, -52, -109, 1, -18, -69, -45, -131, -11, -18, 10, -29, -29, 37, 48, -20, 14, -30, 15, 21, 47}
, {-83, -56, 10, 21, -70, -77, -4, -9, -21, 63, -29, 65, -12, -65, 14, -38, -9, -65, -38, -17, 52, 71, 71, 43, -58, 36, 8, 7, -9, -24, 47, 41, 12, 1, 19, -17, 31, 65, -62, -52, 6, -3, -66, -58, -1, -92, -28, 15, 37, 46, -23, 4, -55, -42, -9, 4, 21, 63, 117, 5, 1, -5, 26, -38, -14, 4, 41, -2, -23, -29, 24, -144, 41, 28, -43, -32, -18, 2, -46, 16, -18, 46, -110, -16, -23, -23, -51, 30, -49, -21, -46, 31, -8, -58, 3, -40, 43, 5, 24, -16, 22, -44, 76, -118, -31, 9, -4, -16, 30, -20, 43, 79, -97, -101, 121, 44, -47, 75, 37, -3, 31, -27, -40, 2, 72, -87, -27, -83, -6, -134, -28, -118, 114, -79, -4, -44, -59, 24, -87, 14, -86, -75, 81, 14, 34, -78, 4, 44, -23, -49, -97, 10, 105, 42, -2, -34, -76, 66, -41, -2, 6, 39, 33, -82, -9, 2, -88, 19, 120, -59, 18, 122, -53, 18, -34, -85, 19, 26, -67, -21, 5, 132, 149, 50, 23, -36, 109, -61, 43, -53, -76, -21, -110, 18, 28, -80, -119, -40, 39, -40, -24, -15, -66, 56, -6, 24, -12, 25, -53, -91, 32, -140, -56, 23, -57, -112, -65, -57, 8, 106, 78, -21, -56, 64, 63, 129, -29, -50, -29, -58, -75, -51, 50, 5, -47, -22, 68, 1, 38, -17, -12, 8, -79, 15, -41, -88, -69, -81, 93, -47, -40, -71, 50, 95, -73, 6}
, {63, 45, 38, 91, 0, -11, -30, 17, -11, 35, 138, -34, -115, -56, 101, 87, -24, 9, -49, -32, 14, 3, 18, -12, 78, 54, 3, -20, -30, -81, 43, 33, 71, 75, -44, 49, 15, 40, 38, -20, 39, 12, 71, 24, -68, 15, -95, -80, -1, -39, 61, -60, 12, 100, -25, 21, 82, -39, 25, 75, 16, -4, -69, 14, -81, -47, 36, -16, -4, -64, 22, 141, -30, 74, -10, -44, -72, -73, -34, -39, -15, 9, 25, -36, 17, -103, -63, 28, -68, 54, 21, -50, 0, -52, 23, -21, 37, -17, -90, 10, -18, 42, 74, 143, -38, 77, -100, 56, -104, -105, -90, -31, 36, 25, 57, 59, -45, -57, -93, -64, -35, -22, 17, 50, 44, 32, -7, -36, 88, -22, -92, -115, -50, -22, 41, -38, -147, -110, 5, -97, 21, 140, -6, -22, -101, -23, -67, 35, -19, -78, -187, -8, 33, -3, -208, -52, -49, 32, -119, 18, -41, -98, -25, -93, -86, -10, 4, 54, -12, -100, -107, 52, -115, 46, -7, -115, 17, -17, -79, -94, 8, -27, 36, -123, 2, 47, -65, -82, -96, -31, -56, 45, -9, -87, -92, -108, -57, -40, -2, 10, -45, -13, 38, -131, -44, 40, -44, -19, -67, -6, -10, -33, 23, -38, 72, -107, -22, -54, -81, 35, -19, 32, -106, 92, -87, -8, 47, 18, -44, -60, 57, -21, -68, -84, -83, 17, -66, 61, -120, -23, 5, 52, 58, -11, -47, -104, -47, -40, -4, -16, -53, -153, 41, -26, -61, 41}
, {-10, -15, -40, 80, -24, -4, 0, 64, 6, -72, -33, -45, 12, 125, -19, 10, -95, -24, 8, -24, -94, 36, -5, -67, -36, -88, -25, 11, 59, 41, -3, 0, -29, -25, -65, 27, -67, -150, 54, 11, 27, -20, -35, 16, -26, -62, -1, -19, -5, -24, -2, -10, -25, 5, 6, 24, -74, 18, 48, 64, -14, -23, 37, 30, -9, -3, -24, -55, 10, 8, -54, -85, -25, 41, -119, -5, -42, -7, -20, 18, 71, -162, -86, -10, 31, -10, 55, -44, 32, 28, -26, -48, -104, -43, -27, 31, 72, -36, 140, 81, 7, -77, 10, -188, 22, 11, -70, 79, -73, 95, 17, 39, -99, -53, 124, 20, 32, 92, 52, 71, -77, -57, -132, 54, -32, -55, -124, -56, 3, 12, 16, -18, -14, 21, 45, -136, 53, -18, 36, -61, 105, -2, 61, -20, -30, 63, 21, 28, 28, 14, 4, 43, -38, -85, 61, 54, 52, -122, -40, 88, 35, 34, -53, 5, 61, -35, 48, -55, 32, 10, 21, -172, 21, 10, 49, -7, -56, -89, -145, 86, 0, -149, -79, -53, 32, 54, 32, -21, 53, 5, -93, 39, 62, -40, 60, -1, -25, -123, 65, 13, -3, -31, 68, -89, -11, -24, -30, -63, 29, 102, -47, -22, 63, 34, -6, -42, -128, -112, 11, 38, -61, -108, -8, -75, 67, 4, 59, 44, 37, 33, 38, 73, -135, 0, -104, 13, -36, -131, 13, 30, -46, -48, 54, -23, 22, -10, -7, 22, -8, -79, 12, 21, 40, -92, -20, -114}
, {129, 29, 16, 35, -45, -96, -47, 0, -78, 118, -61, -63, 18, -132, 9, -35, 25, -26, -68, 46, 13, -13, 20, 25, 85, -84, 25, -40, 27, -67, 69, -63, 4, -16, -31, -5, -93, 119, 13, -52, 50, 23, 132, 58, 36, 17, -17, -17, -31, -47, 79, -12, 92, 20, 16, -2, 20, 1, 46, 17, 19, -116, 25, -109, 10, -7, -23, -35, -64, -25, 54, -10, 13, -33, -82, -5, -9, -68, 24, -7, 19, 23, -46, 29, 134, -17, 50, 71, 4, -30, 60, -43, 60, -70, -6, 61, 50, 16, 20, 51, -4, 20, 27, 6, -45, 58, -14, 16, -57, -4, 13, 31, -53, -80, 51, -16, -82, -32, 21, 6, 69, -39, 86, -2, 30, -104, 60, 7, 78, -1, -10, -96, 7, 92, 42, -85, -79, -18, 41, -33, -81, 135, 31, -109, -49, -46, -115, 110, -13, 58, -29, -32, 46, -47, -85, 46, 33, -1, -60, 48, -2, 46, 3, -14, -77, -8, 2, -46, 76, -70, -3, 24, 58, 60, 80, -57, -59, -25, -57, 9, -109, 72, 79, 1, 70, -52, 55, -127, 85, -52, -34, -21, 10, -52, 55, -36, -12, 26, -5, -7, -154, 35, 1, 1, 2, -83, -24, 47, -102, -106, 56, -53, -44, -93, 140, -44, 8, 76, -118, 9, 9, 7, 6, 55, 0, 2, -35, -19, 45, -63, -52, -106, 21, -3, -41, -5, 78, 18, -11, -40, 25, 8, -1, 1, 93, 72, -55, 50, 88, -46, -4, 35, 33, 11, -152, -84}
, {69, -15, 44, 111, -46, -58, -48, 51, -54, 119, -58, -7, -17, -127, -50, 27, 86, 0, -81, 28, -7, 80, 8, -38, 45, 6, 0, -22, 3, -65, 68, -34, -66, -39, 60, 15, -48, 9, -78, -85, 5, 9, 58, 46, 35, 163, -37, 30, -63, -39, 121, -20, 33, 63, -38, 53, -24, -79, 68, -96, -43, -13, 70, -106, 10, -53, 23, -65, 9, -35, -75, 81, 10, -14, -60, 2, -15, -86, -48, 62, -21, -26, 42, 5, -56, -35, -30, 33, -40, 12, 33, 25, 12, -68, -9, 1, -22, -13, -57, 1, 55, 71, -37, 90, -8, 107, 0, -62, -19, -75, 10, -18, 4, -21, 38, 84, -29, 14, -37, -46, 124, 13, 48, -30, 81, 24, 23, -46, 46, -19, -47, -109, 55, 90, 52, -77, -15, -57, -24, -114, -90, 48, 78, -45, -40, -98, -171, 18, -23, -15, -60, 55, -30, -33, -112, -1, 29, -62, -39, -38, -68, 79, 50, 22, -48, 43, 56, -33, 56, -21, -57, 85, -9, 62, 1, 31, -38, -113, -90, -50, -60, 37, 20, -32, 56, 44, 77, -154, 36, -47, 17, 6, -37, -73, -17, -38, -6, -4, -38, 6, -95, 19, -4, 26, -27, -96, 14, 70, -62, -66, 49, -58, -62, -24, 120, -45, 83, 15, -60, 37, 68, 1, -83, 71, -62, 0, 36, -84, 37, -105, 31, -153, 64, 48, -78, 9, 93, 112, 12, -96, 50, 51, -26, 29, 69, 95, 9, -76, 4, -92, -9, -45, -53, -5, -100, -43}
, {43, -15, -117, -75, 21, 40, 38, -112, -56, -117, -18, -103, -77, 15, 62, -132, -79, 8, -136, -12, 45, -21, 52, -3, 20, -105, -73, 14, -6, -150, -52, -40, -83, 51, 41, -41, -151, -94, 61, -24, -9, 1, -57, 57, -30, 76, -7, 60, -104, -138, -62, -134, 43, -50, 41, -38, -6, -28, 2, -19, 0, -130, -4, 1, 36, -12, -72, -49, -11, 2, 94, 20, -52, 32, 56, -111, 18, -45, 14, -47, 85, 41, -70, -6, -13, 97, 70, -91, 70, -59, -53, 53, 42, 82, -44, -46, -105, 45, 117, 107, -42, -96, 132, -11, 10, -128, -25, 22, -87, 8, -24, 26, -40, -70, 1, -32, 72, -11, 12, -107, 39, 17, 59, 5, 127, -56, 29, 6, 41, 4, 35, -154, 72, -1, -20, -28, 13, -23, -30, 13, -27, -36, 17, -62, 42, 101, -59, -22, 70, 43, 36, -41, 54, -78, 20, 7, -92, 4, -16, -4, 43, 126, -26, 119, -59, -38, -16, -139, -13, -111, 41, 97, 41, 4, 109, 16, -48, 27, -42, 76, 37, 16, 81, 105, 48, 8, -8, -46, 60, -6, -119, -5, -63, 2, 11, 27, 27, -37, 33, -20, 16, -21, 3, 27, -38, 18, -5, 19, -78, -50, -62, 27, 34, -32, 0, -13, 51, 21, -14, 13, 33, -20, -29, -23, -7, -25, 36, 50, -41, 23, 71, -64, -20, -133, -66, 63, 34, -10, -17, 16, 47, -36, 9, -9, 26, 22, -43, -46, 32, -92, -3, -106, -11, 23, 56, 22}
, {71, 12, -16, 68, 41, -87, 3, 79, 51, 190, 28, -66, -55, -106, -10, -49, -9, 62, -63, -19, -23, -2, -31, 17, 41, 97, 11, 9, 40, -66, 21, 68, 85, -55, 79, -4, 50, 51, -11, 6, 101, -22, -36, 73, 12, 15, -90, -19, 43, -45, 47, 27, 46, 4, 20, 41, -52, -56, 2, 107, 12, -9, 33, 11, 79, -67, -29, -76, -3, 18, 38, -20, -88, 2, 2, -53, -160, 24, 44, -108, 15, 17, -32, -12, 72, 95, 115, -73, -1, -1, -42, -33, 21, -134, -23, -20, -61, -19, 59, 7, -42, -22, 88, -3, -57, -18, 20, 48, 2, 45, 5, 86, -69, -116, 135, 0, 39, -105, 87, -14, 63, -20, -96, -72, 94, -129, -34, -4, 29, 45, -70, 9, 15, 61, -1, 75, -45, -53, -3, 0, 4, 121, -128, 21, -6, 68, -20, 2, -146, -77, -89, -42, -23, 31, 1, -37, -57, 113, -38, -27, 24, -48, 6, -2, -67, -4, -43, -10, 116, -19, -82, 83, -207, 160, -60, -94, 48, 35, 78, 4, -58, 17, 73, -39, -115, 86, 2, -10, -8, 30, -7, 30, -99, -20, 80, -73, -75, 17, 60, -150, 20, -8, 67, 45, -69, 23, -80, -75, 10, 49, 72, -96, 49, -35, 38, -73, 34, -65, 40, 24, 8, -79, 82, 80, 26, 10, -32, -38, 32, 1, 20, 46, -8, 10, 66, -21, 91, -32, -21, 63, -28, -52, -53, 17, 3, -64, -86, -47, 40, -4, 7, -1, 65, 18, -38, 81}
, {-43, -45, -16, 73, -38, 72, 1, 30, -99, 11, 35, 14, -62, 122, -85, 45, -79, 17, 65, 72, -14, 18, -58, 5, 17, -16, -67, 83, 34, 33, -19, 0, 16, 22, -5, -66, 22, -21, -7, 26, -63, 55, -32, 17, 7, 35, 11, -105, -28, 5, 18, -13, -4, -49, -103, -59, -10, -26, -15, -128, -52, 61, -5, 24, 23, -40, -69, 24, -75, 32, -1, -14, -87, 29, 164, -62, 56, 16, 46, 41, 42, 96, 7, -16, -61, -6, -93, -127, 61, -12, -40, -22, -55, 47, -52, -79, -61, -56, 24, -14, 0, 4, 32, 60, 6, -111, 0, -13, -14, -8, -54, -26, 7, 162, -87, -127, 56, -43, 55, 3, -115, -32, -11, 44, -128, 20, -10, -41, 13, 38, 3, -120, -42, 10, 55, -89, 37, -63, -25, -18, 71, 48, -26, -144, -41, 9, -24, -65, 52, 60, 75, 31, -132, -62, 11, 100, 7, -46, -27, -4, 26, -13, -45, -26, -60, -79, 35, -69, -97, 65, 1, -51, 31, 6, 24, 3, -113, -21, -73, -23, 23, -107, -46, -88, -6, 63, 36, -60, 29, -115, -6, -52, 74, 71, 5, -17, 29, -118, -14, -5, 76, 51, -4, -9, 40, -97, -61, -1, 77, 60, -16, -32, -13, 94, -11, 55, -108, 26, 21, -50, 4, -61, 84, -36, 41, -117, -40, -5, 87, 79, -26, 45, 7, 11, -61, -142, -40, -97, 38, 66, 54, -8, -52, 29, -5, -5, -14, 5, -92, -18, -76, 26, -88, 24, 4, -37}
, {47, -70, -29, 3, 6, 25, -73, 82, -1, -84, 113, -15, -87, -92, -48, -147, -47, -66, 13, 49, -72, 23, -44, -95, 19, -95, -25, -69, -67, -128, -178, -148, -112, -132, 1, 89, -127, -49, 72, 39, -110, -40, -121, 70, -66, 112, -69, 10, 50, 27, -10, -118, -76, -18, 60, -54, -36, -35, 23, -77, 8, -9, -13, -164, 31, -44, -78, -53, -52, 31, -13, 4, -101, -15, -30, -71, -90, 3, 21, -92, -33, -70, 12, -71, -34, -17, 84, -43, 26, -19, -45, 47, -10, -89, -94, 72, -50, -1, 48, -13, -13, -75, 8, -40, 32, -12, 9, 18, 14, 80, 24, 26, 8, -91, 57, -103, 20, -148, -1, -130, 116, -31, 70, 16, 70, 28, 0, 19, 41, -65, -25, -29, -71, 90, -28, 7, -56, -90, 60, -57, -38, 26, -131, -106, -36, -9, -25, -37, -29, -133, -13, -46, 52, -38, -5, 7, 22, -31, 1, -122, -33, -137, 46, 11, -127, -114, -3, -7, -31, 7, -50, 35, -71, -59, 9, -73, 4, 4, 16, -52, -109, -19, -30, -18, -23, 31, -12, -53, -6, -12, -100, 21, -79, -33, -33, -108, -107, 24, -78, -79, -100, 29, 92, -140, -123, -27, -22, -7, -32, -16, 32, -94, -33, -60, 15, -169, 53, -89, -18, -34, 1, 27, 3, -8, -64, 62, -9, -25, -42, -77, 53, -24, -13, -64, 76, -84, 60, 1, -23, -3, 124, 22, 2, 7, 17, -93, -64, -174, 0, -98, 105, -60, -22, -71, -43, 28}
, {16, 34, 74, -13, -38, -109, -16, -78, -64, 39, 5, 2, -100, -80, 7, -143, 33, 3, -67, -14, -10, -22, 13, 46, 120, -49, 59, -91, -60, -131, 46, 4, -14, 34, -7, 20, 62, 24, 81, 21, -44, -40, 83, -9, 16, 73, 21, 5, -7, 0, 31, -19, 46, 53, 41, 44, -18, -14, -10, 45, -35, 13, -12, -37, 18, -33, 21, 1, -23, -99, 34, -69, 34, -21, 41, -104, -18, -43, 29, -64, 21, 10, -68, 59, 73, -20, -24, -23, 13, -33, -80, 23, 18, -96, 4, -23, -51, -21, 103, 110, -54, -126, 115, -48, -39, -2, 11, 27, -12, 55, -41, 66, -105, -86, 17, -32, 12, -38, 30, -81, 134, 11, -12, -35, 160, -132, -24, 1, 32, -44, 18, -84, 12, 55, 21, -5, -55, -39, -15, 26, -7, 77, 4, 28, 19, -123, -22, 111, 24, -52, -3, 49, 104, 16, -4, -33, -45, 22, -61, -23, 31, 93, 20, -68, 48, 60, 38, -93, 78, -88, -126, 103, -81, 66, -53, -83, -32, -38, -64, -11, -148, -13, 45, -21, 36, -37, 40, -49, 104, 29, -86, -21, 21, -32, 2, -120, -35, 66, -98, -93, -28, 36, -25, 75, -44, -64, 51, 5, -36, -5, 25, -121, -57, 56, 73, -139, 22, 9, -35, 75, -21, 43, 31, 120, 84, 86, -32, -98, -34, -14, -43, -104, 26, -68, -38, -19, 110, 24, -9, -67, 16, 79, -14, 1, 12, -23, -98, -47, 58, -66, 5, 25, 57, 112, -40, 30}
, {-54, -85, 3, 14, 57, 10, -81, 80, -68, 74, 34, 30, 8, -46, -105, 31, -9, -13, 7, 104, -140, -73, -162, -31, 107, 79, 27, -73, -165, 9, -39, -82, 5, -39, -20, -42, -17, 77, -144, 23, 120, -5, -118, 99, -107, 43, -73, -130, 70, -18, 66, 12, 52, -58, 67, 41, -98, -43, -116, 43, -165, -16, -70, -65, 26, -34, -89, 45, 91, -75, -34, -44, 32, 17, -25, -45, -62, 75, 32, -74, -89, 25, -46, 107, 33, 112, 56, -39, 22, -59, -71, -90, -32, 8, -23, -15, -5, 89, 103, 105, -18, -61, 95, -136, 0, -113, 44, -13, -130, 49, 13, 0, -130, -71, 120, -20, -13, -52, 116, -50, 31, -13, -13, -67, 44, -74, 64, 38, 23, 60, -5, -34, -72, -21, -48, -7, -48, 46, -2, 40, 80, 54, -97, 54, 22, -39, -48, 27, -6, -52, -5, -41, -29, 19, 4, -43, 17, -20, -21, -34, -31, -129, -9, 31, -6, -44, -97, 35, -61, -86, 33, -23, -21, -98, 17, -19, 9, 40, 87, 2, 107, 37, 20, 4, -40, 13, 84, 52, -67, 42, -26, -1, 53, 20, -32, 27, 30, 53, 8, 50, 34, 1, -66, 26, -2, 87, -8, 17, -59, 6, 33, -10, -45, 8, 6, 64, 22, -15, 18, -156, -31, -12, 12, -88, -117, -7, -40, -32, 0, 33, 24, 39, 44, 97, -16, 62, 17, 51, 10, 4, 18, -34, 45, -81, -1, 30, 5, -30, -21, 62, 16, -15, -6, 12, 42, -3}
, {-1, -65, -80, -70, 20, 80, 1, -19, -26, -80, 95, -9, 1, -33, 10, 84, -13, 13, -49, 76, -19, -39, -60, -43, -18, -15, -120, -68, -28, -34, -94, -4, -37, 53, -17, -11, 41, -37, 15, 18, 30, -17, 48, 56, 18, -58, -8, -59, -49, -16, -31, 40, 12, -25, 114, -91, -38, -60, -50, -9, -27, -45, -57, -58, 30, -4, 2, -1, 13, 73, 63, 32, 24, 14, 82, -22, 26, -31, -67, 14, -21, 46, 9, 18, -30, 85, 16, 16, -24, 45, -17, 23, 15, 10, -57, -12, -46, 56, 16, -43, -38, 21, 74, 86, -68, 65, 27, -106, -30, -74, 45, -37, 50, 116, -87, 18, -22, -76, -39, 3, 31, -14, -1, -24, 9, 49, -49, -4, -59, -25, -58, 81, -52, -7, -77, 49, 58, -8, 48, 15, -39, -132, 4, 54, 16, -81, -10, -33, 35, -59, 16, 28, 59, -9, 73, -28, -23, -7, 72, -47, 24, -87, -72, -36, 79, 76, 27, -6, 26, 11, -3, -68, -56, -91, -90, 54, -3, -112, 7, 9, 37, 12, 38, -50, -19, 73, -68, 104, -44, 14, 40, 27, 28, 14, -50, 77, 16, -96, -45, 4, 8, 9, -33, 5, -18, -11, 36, 7, -91, -117, -46, 90, -80, -103, -44, 91, 85, 47, -56, -178, 14, -27, 54, -65, 26, -102, 17, -10, 95, -113, -65, -43, 109, -122, -43, -23, -95, 16, -53, 18, -26, -58, -13, -4, -108, 48, 87, 12, -59, 26, 66, 74, -106, 5, 24, -126}
, {-11, -46, -2, -20, 40, 62, 11, 42, 5, -82, -23, 26, 43, 14, -12, -34, 10, -60, -52, -24, -35, -49, -19, 37, -163, 15, 17, -17, -25, -4, -6, -36, 1, -29, 38, -43, -37, -61, -30, 5, -35, -26, 47, -6, 69, 18, 12, -51, -51, -22, -51, 21, -35, -35, 20, 51, -22, -6, 12, -17, 4, -47, -3, -113, -36, 91, 78, -63, -67, 17, -128, -38, -12, -41, -34, 3, 71, 24, -4, 43, -65, 58, 21, 64, -150, -1, -61, 113, -29, -47, -18, 166, 60, -20, -22, 37, -16, -13, -82, -84, 10, -25, -103, -18, -5, 72, 25, -45, 17, -89, 37, 17, 35, 82, -142, 24, -31, 4, 34, -50, -41, 66, 5, -53, -104, 43, 59, 4, -105, 42, -57, 65, 3, -17, -4, 13, -62, 16, 51, 51, -72, 19, -53, -42, -36, -98, -9, -37, -35, -105, -10, -58, -42, -32, -5, 54, -18, 35, -34, -70, 19, -13, 33, -76, 19, 9, 15, 105, 66, 45, 18, -51, -35, -20, -46, -4, 25, -29, 11, -24, -101, 78, 54, -21, 35, -48, 79, 9, 7, -69, 77, -22, -66, 39, 49, 0, 99, 22, -116, 5, 8, 44, -51, 74, 21, -43, 104, 109, -17, -139, -102, 82, -130, -130, -92, 88, 79, 91, -57, -37, 33, 43, -19, -33, -57, -68, 18, 2, -18, -69, -43, -164, 122, -1, -122, -3, 38, 18, 11, -40, -3, -71, -40, -107, -17, 67, 98, 64, -2, 33, -17, 10, -104, -9, -17, -54}
, {-123, -48, -31, -44, -8, 146, -132, -3, -9, 29, 48, 30, -1, 53, -95, 17, -48, 11, 8, 28, -113, 55, -50, 13, 35, 86, -15, 35, -60, 24, -71, -36, 0, 7, -60, -46, -5, 24, -118, -101, 14, 66, 3, -46, -67, 20, 26, -63, 36, -27, 22, -16, 26, -48, 16, 2, -4, -209, -53, -146, -19, 63, -8, 46, 7, -71, -100, -87, -50, 41, 4, -56, -37, 3, 75, -21, 13, 137, 3, -7, 7, 3, -22, 24, -20, -7, -47, 55, 70, 60, -9, -14, -4, 57, -41, 29, -5, 41, 50, -12, -28, 12, -66, 0, -41, 59, 9, -20, 60, 12, 89, -80, -48, 100, -19, -18, -36, -29, 39, 4, 81, 44, -46, 30, 63, -11, -89, 96, -15, 12, -36, -109, -20, 150, 59, -105, -26, -53, -34, -61, 94, 13, 82, -13, 9, -41, -62, 8, 16, 1, 3, 125, -131, -97, -11, -59, -5, -79, 82, 59, -9, 11, -83, -7, -27, -25, 22, 95, 14, 91, -56, 28, -43, -32, -45, -36, -6, -63, -42, 87, -39, -11, -104, -8, 1, 212, 41, 24, 65, -38, 32, -118, 135, 59, 58, 13, -77, -19, 55, 14, 31, -37, 44, -38, -1, 61, -42, 25, 48, 47, 120, -14, 28, 87, 42, -33, -65, -96, 115, -12, -27, 7, 85, 107, 39, 104, 13, -12, 12, -11, 122, 49, -107, 93, 71, -113, -153, -72, 14, 70, -28, 11, 24, 96, -46, -135, -71, 12, 47, -81, -28, 51, -9, 8, 53, 30}
, {116, 53, -2, -196, 17, 98, -32, -72, -20, -60, 103, -173, -118, -41, 59, -138, -164, -39, -100, 16, 18, 25, -32, -51, -46, -142, -34, 68, 5, -134, -51, -48, -31, 57, 8, 32, -84, -2, 68, 3, -14, -70, -45, -64, -12, 59, 65, -6, -29, -21, -77, -103, -21, -86, 0, -113, 1, -15, -41, -13, 23, -76, -82, -53, 30, 74, 1, -19, 15, 57, 80, -53, -22, -14, 101, -44, 9, -17, -34, 66, 13, 23, 42, -72, -88, 6, 84, -53, 66, 14, 39, 2, 38, 73, -43, 22, -44, 2, -5, 67, 27, -41, 13, -7, 56, -58, 33, -58, 25, 47, -33, 14, 58, -7, -82, 0, 44, -13, -52, -25, 71, 0, 8, 58, 19, 11, -11, 105, 28, -7, 18, -79, -55, 53, -26, -42, 10, 43, -18, -100, -22, -53, 71, -73, 1, 38, -177, -40, 89, -13, 85, -67, -11, -165, -43, 55, 71, -64, -4, -90, -14, 48, 4, 32, -132, -168, -40, -104, -16, 67, 27, -5, 102, 26, 6, 38, -131, -73, -28, -93, 87, 12, 67, 60, 67, 18, -26, -43, 107, 8, -46, -4, 9, -33, -52, -23, -46, -47, 72, -31, -6, -57, -10, -26, 54, 30, -54, 66, -125, -36, 7, 87, -68, -126, -50, 47, -56, -11, -48, -28, -9, 15, 35, -72, -83, -64, 82, 71, -75, -119, 22, 10, 7, -17, -92, 32, -9, 6, 8, 52, 8, 44, 72, -82, 15, -8, 8, -70, -90, 1, 71, -49, -64, -31, 60, -28}
, {145, 18, -11, -19, -79, 46, 42, -175, -12, -38, 40, -47, -56, 96, 1, -108, -130, 15, -43, -58, 56, -10, 7, -89, 114, -120, -82, 39, 32, -131, -12, 45, -20, 76, -17, -26, -71, -53, 103, -47, -78, -64, 44, 18, -9, 71, 37, -14, -20, -86, -34, -54, 102, -94, 38, -34, -13, 0, -1, 28, 8, -119, -29, 58, 109, -5, 71, -36, -8, 55, 121, 1, -42, 43, 83, -37, 30, -70, -65, -3, 36, 14, 3, -35, 12, 0, 96, -8, -31, -6, -40, 66, -64, 49, -17, -35, 61, -34, 33, 76, -46, 6, 20, 69, -7, -1, -87, 13, -59, 47, -40, -1, 5, -25, 18, 31, 71, 66, -111, 7, 41, -26, 34, 80, 80, 6, -9, 21, 87, -58, 87, -47, -58, 27, -147, -52, 34, 37, 43, -124, -16, -61, 22, -21, -10, 64, -155, -58, 15, 38, -17, -64, 76, -143, 36, -10, -8, -92, -127, -45, 96, 26, -12, 112, -2, -30, -23, -96, -86, -86, 16, -16, 14, -31, -17, 19, -189, -56, 46, -36, 74, -79, 52, 53, 42, -83, -246, -18, -1, -37, -104, -14, -59, -15, 13, -11, -44, -167, 43, 47, 20, 26, -92, 37, -42, 79, -36, -2, -118, -53, -152, 26, -55, -124, -53, 100, -120, -36, 16, -5, -7, -78, -20, 17, 41, -27, 88, 20, -18, -13, 26, 13, -15, -20, -25, 7, -24, -117, 9, 72, -100, 38, 36, -3, -188, 5, -78, -44, -84, 35, 56, -62, -11, -95, 74, 75}
, {10, -27, -8, -22, -49, -14, 6, 98, 37, 42, -57, -84, -53, -49, -63, 13, 23, 87, 21, -5, 63, 65, -33, 29, -68, 10, 44, -32, 34, 53, -2, 25, 13, -12, -96, 3, 40, -42, 29, 44, -37, -21, 30, -42, 1, -41, -87, 0, 40, -35, -22, -41, -109, 30, -38, -25, -56, 30, 2, 13, -14, 9, 27, -4, -41, -2, -26, 16, 27, -50, 23, -28, 18, -49, -46, 13, -101, 47, 109, -79, -4, -116, -106, -19, -67, -25, 28, -1, 90, -77, -53, -47, 11, -38, 23, 23, -9, 27, 12, -24, -24, 26, -38, 14, 44, 17, -41, 60, 44, -47, 0, 34, -105, -114, 26, -45, 40, -15, 38, 5, 87, -7, -62, 5, 48, -42, -14, -23, -66, 12, 32, 2, 9, -1, -28, 46, -21, -16, -49, 8, 22, 139, -70, 59, 11, 63, 55, -2, 65, 17, -27, -55, 6, 36, -5, -22, 13, 23, 25, 36, 27, -18, -9, -73, 15, -17, 28, 87, 5, 2, 0, 34, -85, 80, -24, -71, 114, 39, -75, 33, -56, -20, -67, 41, -68, 92, -57, 48, -15, 86, -106, 62, 44, 7, -71, -66, -24, 36, -14, -30, 8, -153, -3, 66, -21, 28, -48, -148, 74, 199, 147, -100, 171, 18, 117, -143, -9, -1, 66, 9, -58, 13, -14, 48, 35, 99, -37, -106, -26, 66, 57, 29, -153, -41, 110, 10, -44, 3, 50, 0, 0, 45, -63, 87, -11, -63, -42, -43, 98, -88, -70, -13, 16, -30, -102, 155}
, {50, -23, 25, -6, -73, -60, 29, 33, 2, -55, -9, 21, -99, -79, -17, -92, -13, 26, -34, -11, -27, -22, 24, -21, 29, -63, -47, -50, -31, -54, 28, -32, 5, 9, 14, -69, 85, 62, 75, 6, -40, 32, 35, -32, 22, -73, -103, 34, 7, -6, -9, 3, -49, 30, -13, 57, -22, 18, 11, -48, -59, -19, 44, 32, -117, 27, 52, 25, 22, -58, 3, -21, 26, -14, -20, 40, -89, 74, 70, -32, 36, -109, 59, -38, 24, 12, 19, 27, 5, -71, -30, 22, 20, -118, 34, 83, -5, 13, 51, -13, -50, 62, -17, -3, -42, 40, 6, 12, 35, -13, -12, -16, -9, -8, 13, 2, -73, 19, -90, -77, 79, 23, -13, 33, 66, -1, 39, 78, -36, -159, -78, -94, 37, -90, 54, -23, -24, 44, -83, 35, -61, 88, -54, 20, 110, -88, 8, -11, 44, -59, -88, 83, 85, 28, -51, -37, -185, 31, -95, -5, 1, -20, 29, 10, 61, -62, 39, -54, 26, -49, -114, 70, -148, -3, -38, -130, 100, -9, -20, 11, -20, 8, -69, -57, -45, -20, 14, 11, 5, 21, -122, 46, 96, 17, -62, -219, -66, 26, -13, -83, 9, -121, 53, -39, -21, 140, -45, -100, -46, 15, 105, -106, 116, 76, 49, -22, 31, -26, -15, 40, -61, 60, 13, -30, 31, 127, -41, -10, -102, -97, 6, -51, -64, -107, 15, 38, -118, 75, 65, -73, -1, 40, 38, 29, -44, -118, -24, 5, 136, -17, -2, -122, 109, 28, -21, 147}
, {13, -32, -49, -31, -62, 26, -19, -55, -70, 84, 12, -64, -72, -164, 29, -70, -38, -24, -14, 44, -32, -85, 9, 22, 198, -115, -90, -31, -86, -36, -45, -45, -33, -30, -19, 75, -47, -47, -6, -28, 10, -98, 30, -12, -44, 6, -72, -16, -16, -118, -1, -37, 122, -74, 19, -15, -51, 11, -29, -9, -7, -47, -2, -112, 113, -27, -58, -96, -28, -36, 37, -25, 31, -29, 69, -138, -69, 77, 83, -183, 42, 32, -81, -56, 36, 74, 91, -76, -28, -48, -47, 15, -32, 8, -15, -44, -16, 83, 103, 83, -106, 2, 120, -88, -67, 37, 43, 14, -25, 42, 16, 54, -65, -38, 75, -69, 70, -18, 77, -33, 0, -50, -43, -36, 86, -107, -5, 25, 11, 40, -24, 80, 9, 57, -18, -6, 8, 54, 20, 67, 54, -41, -7, 99, -30, 51, 77, 37, -38, 42, -23, -102, 56, -23, -49, -13, 66, 38, -79, 31, 44, 110, -47, 81, -29, 41, -13, -58, 26, -19, -44, 25, 52, 0, 13, 114, 11, 13, -34, 23, 75, -4, 15, -6, 46, 85, -10, 76, 81, -19, -7, 25, 2, -20, 23, -52, 28, -56, 44, -9, 109, -76, -76, -9, -59, 29, -78, -13, 33, 76, 64, -41, 11, -50, 21, -14, 5, -57, 136, 10, 32, -4, 63, 48, 28, 22, 43, 32, 20, 60, 38, 27, -17, 111, 49, 79, -42, -48, 23, 14, 26, 79, 38, 77, -20, -61, -44, -33, -22, -87, 30, -48, 28, 8, 81, 46}
, {13, 43, 103, -43, 113, 80, 17, 21, 16, -54, 26, 23, 6, 152, -81, -12, 20, -10, 5, -6, -55, -49, -43, -45, -92, -2, 23, 29, -48, 16, 58, 36, -34, -25, -101, -8, -49, -31, 9, 86, 71, 6, -67, 1, -76, -30, 27, -43, 4, 1, -24, 34, -69, 28, -1, -20, -64, -63, 10, 13, 2, 15, 5, -12, -54, -42, -48, 52, 30, -27, -129, 31, -12, 32, -83, -67, 24, -10, 27, -18, 91, -170, -65, 66, -63, 8, -1, -44, 137, 5, -52, -131, -72, 25, 37, 22, 16, -120, 71, 29, 10, 46, -31, -62, 36, 56, 21, 34, -8, 47, -30, -36, 38, -12, 97, 19, 14, 96, 32, 78, -103, -13, -49, 98, -138, -30, 22, -62, -68, 46, -33, -17, -38, 15, 10, -58, 9, 12, -24, -45, 107, -37, -15, 26, -56, -54, 57, 28, 57, -9, 87, 17, 31, -133, 22, 28, 115, -89, 37, 100, 25, 2, 33, -62, 32, -20, 24, -40, 8, -18, -19, -44, -10, -75, 44, 26, 6, 8, -89, 79, 44, -166, -38, -89, 4, 31, 42, 53, 36, -87, 51, -20, 30, -65, 31, 75, -90, -34, -12, 45, 24, -59, 51, 11, -65, -122, 8, -49, 70, -23, 54, 31, 17, 97, -16, 48, -92, 4, -44, 73, 21, -85, -15, -1, 65, -105, 33, 39, 60, 9, 131, 68, -180, 17, -107, -54, -52, -115, 6, 45, -5, 70, 14, 8, -28, -33, -52, 43, -161, -36, -72, 43, 25, -74, -66, -105}
, {-25, 48, 30, -75, 38, 61, 76, 25, 27, -56, 37, 22, 49, 101, -80, 16, -1, -49, -7, -53, -59, 89, -41, -94, 21, -47, 1, -16, 27, 28, -20, 20, -21, 13, -110, -84, -53, -67, -108, 23, -29, 38, 4, -121, 55, -81, 12, -12, 53, 30, -71, -25, -29, 19, 37, -24, 5, -165, -135, -57, -26, 3, -38, 20, -71, -9, -29, -52, -17, 40, -21, -41, 36, 10, -54, 18, 48, 38, 5, -20, -6, -59, 24, 49, 86, -38, 35, -32, 40, 42, -13, 26, -53, 17, -26, -23, -17, 69, 110, 32, 15, 49, -8, -61, 3, 70, -43, 58, 17, 64, 37, 2, 77, 26, 22, -68, -49, -27, 42, -53, 48, -29, -119, 57, -39, -52, -86, 18, -67, 50, -48, 18, -15, 80, 39, -18, -9, -105, 13, -150, -67, -46, 35, 38, -113, -8, -174, -35, -25, -93, -27, 59, -133, -152, 5, -49, -29, -105, 115, 48, 46, 46, 4, -59, -145, 86, 27, 108, 30, 65, -139, -67, -140, -63, -94, 55, 35, -227, -70, 62, -141, -120, -17, -140, -13, 89, 53, 33, 23, -41, 22, -75, 55, -8, 83, 46, -9, 22, 8, -31, 24, -12, 66, -137, -54, -6, -35, -9, -34, 105, 44, -94, 93, 50, 53, 75, -77, -120, 1, 54, -67, 33, 125, 35, 56, 12, 30, 39, -45, 110, 60, 65, -46, -59, -1, -56, -132, -97, -118, 7, 40, 16, -2, 29, -20, -157, -86, -106, 54, 71, -37, 28, 74, 69, -29, -76}
, {-12, 17, 9, -46, -28, 19, 27, -63, -99, 3, 16, -69, 88, 12, 34, -17, -84, -11, -20, -112, 63, 42, 48, 57, 86, -90, 27, 23, 75, 30, 3, -39, 60, 51, 36, -45, -95, -45, 17, 12, -55, 38, 63, -39, 31, -10, 89, 15, -36, -15, -64, 0, 8, -81, 50, -83, 67, 15, 90, -12, 62, -8, 2, 50, 40, -27, -30, -13, -14, 62, 81, -51, -64, -20, -13, -22, -10, 29, 33, 1, -90, 54, -80, 18, 53, -27, 40, 28, 47, -7, -89, -14, -25, -33, 8, -38, -16, -90, 90, 77, 14, -2, 26, -63, 10, -29, -37, 54, 66, 69, -42, 9, -62, -79, 20, -18, 63, -30, 26, -31, -73, 29, 15, 45, -12, -2, -16, -23, -51, -98, 83, -47, 28, -61, -104, -7, 10, 87, -103, 66, -109, -64, -30, 13, 49, 78, -7, 30, 23, -19, -3, 61, -86, 37, 67, -57, -24, 3, -94, -35, 19, 30, -22, 2, 100, 52, -160, -61, 85, -137, 16, 75, 38, -15, -57, -70, 36, -50, -56, -51, -15, 85, 9, 10, -13, -53, 132, -53, -44, 43, -22, 66, 24, 87, -33, 5, -21, -29, -76, -98, 14, -51, -70, -86, -18, 46, 39, -43, 80, -80, -28, -87, 70, 100, -120, -95, 36, -61, 33, 56, 37, -6, -59, -59, 38, 50, -72, -60, -36, -41, -108, -54, -60, -44, 61, 0, 56, -17, 53, -93, -36, -26, -7, -8, -39, -11, -22, 17, 80, -12, -192, -14, 51, 79, -109, 44}
, {16, 96, -59, -9, -34, -144, 27, -95, -36, 15, -67, -58, -1, 13, 80, -61, 13, 12, 46, -37, 24, -56, 72, 50, -82, -48, 3, 0, 12, -72, 93, 27, 93, 25, 42, 33, -4, -8, 71, 4, 43, -65, 66, 7, 38, -32, 67, 81, -135, -9, -96, -2, -73, 63, 11, -41, 0, 53, 43, 173, 35, 8, 0, -16, -40, 15, -1, 17, -45, -85, 19, 47, -8, 10, -29, 83, -1, -116, -17, -25, 65, -120, 66, -93, 18, -45, 132, -16, -22, -28, 21, -11, -42, -21, 24, 38, 16, -88, 17, -22, 31, 22, 59, -9, -35, 146, -36, 29, -60, -20, -15, -5, 28, 64, 36, 71, 63, 111, 13, 37, 19, -23, -122, -25, 94, 52, 18, 50, -19, -12, 8, 31, 16, 48, -171, 37, 28, 32, 25, 90, -44, 76, -25, 27, 2, -17, 13, -120, -11, 24, -27, -8, 70, 54, 15, -3, -103, -20, 21, 113, 9, -7, -17, 31, 17, -20, -118, -19, -15, -77, 55, -53, 11, -73, -4, -3, 43, 40, 27, -19, 30, -38, -40, -2, 21, -138, -113, -22, -28, 90, -122, 39, 2, -40, -190, -59, -35, 18, 47, -2, 1, -130, -53, -67, -105, 68, -38, -59, 41, -133, -22, 19, 115, 28, -148, -12, 55, -14, -30, -2, -38, -28, -88, 64, 70, 60, 22, 23, -57, 38, 17, 1, -44, -18, 13, 42, -166, 14, 34, -98, 39, 55, 109, 13, -85, -6, 14, -25, 83, -6, 0, -140, 79, -21, -16, 114}
, {-85, 2, -57, -3, -67, -74, 30, 35, 9, -68, -62, 40, 22, 26, 10, 50, 88, -34, 38, 30, 16, -11, -9, -22, -13, -45, 40, -39, 22, -21, 20, -50, -76, -14, -45, 20, -9, -136, 20, -20, 56, -15, 24, -59, -25, -75, 83, -22, 16, 43, 23, 8, -55, -18, -85, -16, -17, 33, 33, 59, 15, -106, 24, -89, 4, 12, 42, -1, 44, -42, 21, -68, 31, 29, -35, 34, -16, -39, 63, -26, 58, -51, 69, -37, 15, 20, 104, 11, -3, -12, 2, 21, 7, 57, 70, 17, -15, -6, 75, -48, -7, 24, -39, -72, 23, -29, -10, 21, 33, -12, -41, 5, 37, 76, 42, -4, -27, -24, -26, -13, -48, -18, -30, -79, 44, 54, -41, 3, -97, -60, 56, 120, -20, -22, -72, 146, 35, 10, -21, 109, -105, -27, -135, 31, 13, -79, 115, -61, -126, -69, -60, -54, 122, 20, 82, -15, -5, 3, 111, 93, 35, -100, -9, 6, 126, 57, -105, -3, 50, -28, -73, -91, -104, -77, -155, -56, 70, 78, 32, -39, -70, -43, -128, -52, -69, -75, -25, 90, -101, 68, -10, 82, -35, -5, -79, 10, 28, -41, -99, -17, 43, -28, -68, -28, 21, 96, 58, -104, 90, -107, -10, 13, 139, 16, -232, 103, 42, 29, -136, 1, -69, 58, -128, -160, -9, 65, -54, 58, -34, 11, 23, 7, 18, -43, -37, 138, -42, 33, 29, -67, -79, 16, -74, -69, -42, 52, 61, 25, -73, 29, -59, -64, 63, -75, 4, 37}
, {-129, 27, 48, -1, -13, -138, -39, -9, -16, 167, 29, 107, -7, 7, 26, 66, -5, -34, 44, -60, -1, -16, 13, 9, -60, 62, 37, 39, -35, 5, 62, 81, -29, 43, 35, -34, 40, 64, -75, -42, 24, 23, -9, -80, -6, 5, 14, -14, 85, 31, 4, 33, 123, -60, -27, 45, -32, -75, 61, 49, -22, -32, -71, 19, 19, -2, 1, 40, -46, 50, 38, 28, 4, -26, -38, 6, 15, -124, -16, -12, -43, 74, -48, 21, 128, 59, -143, -6, -44, 42, 13, 23, 5, 12, -28, -10, 51, 91, -4, -17, 35, -62, 64, 38, 1, 53, -9, 45, -4, -98, 24, -3, -11, 18, -24, 34, -47, 45, 9, -14, 23, 67, 10, 26, 16, -67, 1, -41, 97, -113, -36, -92, -8, 98, 71, -25, -87, -149, 56, -78, -36, 11, 12, -55, 59, -66, -27, 105, 40, -7, 27, 0, -10, -99, -112, -20, -134, 17, -72, -109, -56, 96, -45, 9, -68, -11, -37, 67, -57, 0, -71, 30, -113, 66, -30, -18, -94, -7, 24, -15, -65, 9, -22, 7, -20, -32, 9, -76, 67, -121, -30, -108, -25, 25, 41, -30, -23, -44, -6, -69, -85, 149, 2, -5, 62, -103, 99, 86, -154, -91, 27, -17, -70, -39, -40, 28, -39, -26, 17, 71, 16, -46, 3, 72, 22, -3, 15, -47, -30, 2, -102, -35, 51, -129, -92, -9, 100, -11, 6, 25, -37, 19, -44, 87, 48, 35, 5, -75, 79, -53, -45, -32, -75, -14, -12, 15}
, {32, -50, -16, 23, 18, 25, 36, -14, -7, 85, 103, 29, 18, 15, 54, 113, -95, 44, -15, -55, 0, 6, 21, 63, 15, -29, -6, 85, 36, -12, -30, 74, -35, 40, -11, 14, -18, 58, 73, -14, 1, 2, -8, 42, 34, -23, -28, 43, 29, -73, -94, -61, 22, 56, 45, -18, 113, 34, 81, 2, 0, -5, -63, 74, -49, -4, -6, 36, 46, -55, 9, 61, -33, 52, 18, -8, -24, -27, -76, 74, -39, 19, 58, -36, -5, -53, -6, 89, -41, 9, 27, 26, -53, 55, 40, 14, -19, -27, -39, -23, 26, 37, -85, 83, 22, -17, -63, 43, 48, 15, -15, -62, 36, -33, -32, 14, -39, 92, -32, -1, -32, -40, -37, -30, 4, 34, 3, -42, 48, 37, -15, -102, -31, 35, -146, -28, 55, -7, 24, -87, -25, 52, -36, -36, 28, 10, -26, -40, 51, 76, 7, -1, 18, -32, -40, -9, -47, 6, -55, -87, -34, -22, -99, 46, -29, -53, 10, -118, -42, -20, 74, -56, 135, 59, 16, -22, -15, -38, -59, 31, 65, -75, -18, -91, -11, -20, -122, -43, 3, -42, -36, 34, -4, 14, -71, -50, -131, -120, 77, 66, 72, -138, 30, -77, -55, 123, -93, -46, -84, -25, -100, -1, 6, -85, 102, 81, -104, -100, 12, -24, 24, -70, -39, -6, 61, 9, 73, 48, -6, -49, 122, 2, -86, 19, -43, -37, -82, 23, -58, 115, -139, 37, 83, 21, -37, 3, 14, -90, -18, -93, 33, -139, 44, -171, 79, 46}
, {110, -55, -11, -80, -69, 97, -18, 21, 9, -144, -23, 32, -31, 3, 40, -104, 10, 23, -7, -51, 20, 22, -23, -50, -45, -53, -35, 13, 55, -91, -56, -43, -13, 63, -6, 97, 34, -30, 52, 25, -84, 29, 82, -33, 12, 51, 70, -15, -13, 25, -12, -27, 23, 8, 26, -26, 46, 28, -22, -103, -3, -46, 24, 54, 63, -20, -10, 41, -105, 105, 36, 81, -54, 17, 56, 63, -23, -16, -106, 22, -34, -22, 85, -145, 30, -15, 81, 11, -59, 45, 15, 72, 14, 28, -5, 24, -64, -36, -19, -67, -54, 27, 2, 75, -42, 94, -23, -207, 35, -38, 29, -76, 58, 99, 29, 25, 26, -98, -131, -56, 47, -71, 61, -36, 8, 104, 12, 111, 22, -90, -20, 28, -35, -6, -62, 72, 34, 10, -23, 51, -82, -8, -68, -14, 12, 20, -8, -62, -24, 23, -47, 3, 34, -10, 80, -15, -16, -22, -13, -42, 48, -2, -1, 75, -17, 39, 2, -40, 21, 43, 18, -48, -48, 20, 26, -28, -73, -47, 134, -56, 38, -6, 22, -59, 13, 38, -139, 21, -32, -20, 6, -29, 50, 8, -31, 48, 36, -31, -68, 53, 0, 17, -30, -92, -51, -30, 28, 10, -47, 14, 54, 3, -145, -16, 128, -3, 120, 4, 17, -146, 27, -78, 76, -16, -26, -108, 31, 39, 58, -73, 40, -47, 50, -9, 119, -81, -8, 38, -102, 72, -22, -67, 43, -56, 17, 13, -12, 40, -93, -14, 54, -2, -121, -103, 42, 14}
, {-57, 13, 14, 17, 123, 85, -92, -17, -27, -135, 58, 34, -17, -13, -8, 64, 10, -55, 9, 77, -69, -17, 22, 24, -68, 36, 16, -40, -44, 1, -35, -7, -43, -81, -51, 10, 18, -33, -52, 53, 36, 22, -53, 67, -69, 22, 20, -59, 23, -21, 30, 9, -30, 20, 82, 40, -23, 18, 43, -80, -8, -2, -42, 20, 22, -14, -19, -32, 21, 8, -10, 1, 51, 14, -86, 64, 8, -102, 5, -1, -39, -90, 39, 0, -122, 3, 45, 28, -92, 20, 36, 55, -12, -1, -1, 20, 37, 4, -79, -149, 2, -34, 9, -91, -28, 65, -51, -145, 7, -21, 25, 11, 61, 59, 26, 3, -63, -39, 41, -27, -18, 16, -7, -27, -124, 78, 16, 0, -214, 4, -25, 56, 24, -28, -41, 70, -8, -23, 26, -4, -24, 1, -140, 34, -49, -108, 64, -41, -164, -73, -83, -76, 45, 19, 41, -38, -10, -18, 16, -18, 3, -93, -39, -30, -10, 94, -121, 6, -77, 12, -86, -103, -101, -125, -156, -19, 88, 27, 162, -25, -20, -43, -82, -79, -30, -26, 46, 83, -111, 74, 70, -11, 1, -111, -55, 52, 84, -25, -89, -5, -18, 64, 34, -22, -24, -50, 35, -5, -65, -3, 12, 58, -48, -62, -99, 16, 20, 2, -60, -212, -34, 4, 63, -120, -91, -133, -15, -3, 91, 30, -79, -17, 43, -6, 0, 19, -35, 2, -97, -73, 12, -113, 2, -113, -29, 0, 46, 18, -83, 25, 41, 68, -91, -60, 35, -77}
, {-80, -24, -22, -48, -28, -25, -5, -17, 6, 34, -127, 1, 4, 79, -25, 101, 59, -51, -43, -55, -10, 29, 4, 65, -109, -69, 8, 32, 5, -21, 19, -14, 9, -36, -76, -5, -4, 30, 27, -4, -76, 11, -30, -80, -15, 7, 28, -13, 8, -49, 9, 62, -96, -55, -62, 48, -4, 21, 10, 29, 20, 53, 3, -65, -82, 62, 1, 55, -11, -113, -113, 15, 9, 31, -119, 18, 24, -7, 29, 49, -30, -45, -17, 19, -21, -95, -160, 63, -2, 22, -44, -70, -52, 27, 26, -30, 20, -26, -210, -89, 17, 41, -120, -53, 72, 57, -27, 57, 91, -65, 0, -27, -12, -73, -23, 35, -84, 102, 18, 40, -4, -1, 1, -18, -67, 14, -44, -187, -18, 30, -13, -108, 47, -38, 86, -79, -65, 39, -51, -35, -62, -13, 29, -37, -11, -2, -58, 43, -5, -81, -7, 43, 20, -65, -19, -123, -44, -16, 72, 21, -37, 37, 44, -73, -11, -5, 56, 95, 36, -39, -45, 94, -70, 43, -89, -16, 12, -60, -121, 17, -108, 6, 18, -9, 40, 45, 130, 19, 58, -41, -49, 6, 0, -7, 66, -72, -97, -9, -51, -98, 15, -60, -35, 14, 23, 30, -18, -17, 30, 72, 56, -70, 75, 90, 49, -67, -48, 34, -7, 98, 35, -35, -41, 110, 32, 63, 25, -160, 37, 43, 13, 33, -78, -32, -30, -5, 36, 8, 55, 50, 28, 9, -70, 80, -31, -168, -5, -42, 113, -18, -97, -37, 82, 91, -116, 78}
, {-107, 48, 42, 13, 25, 10, -53, 35, 28, -21, -2, 61, 10, 120, -35, 47, -28, -17, -24, -12, 42, -22, 3, 83, -62, -23, -20, -38, 33, 84, -30, 34, -40, -24, 57, -6, 41, 37, -42, 9, 73, 60, -9, -70, -47, -5, -17, -42, -3, -6, 28, 41, -63, 12, 39, 28, 54, -18, 23, 28, -23, 116, 10, 48, -121, -50, -26, 25, 42, -93, -80, -57, -65, 81, -80, -75, -77, -95, 30, 26, -34, -83, -108, 22, 12, -50, -62, 11, 65, 27, -8, -149, -81, 40, 29, -56, 20, -13, 0, 52, 22, 57, -90, -80, 99, -6, -84, 78, 24, -9, -23, -17, -58, -38, 40, -63, 37, 22, 12, 28, -36, 90, -64, 59, -49, -60, -39, -62, -30, 20, 13, 5, -59, 61, -33, -37, 21, -14, -57, -29, 7, -6, 9, 76, -58, 2, 24, -67, 30, 44, -19, 98, -60, 36, 1, 78, 74, 8, 21, 46, 27, -86, -57, -81, -1, -55, 44, -12, 3, 53, -46, -54, -3, -56, 49, 27, -19, -58, -57, 24, -35, -32, -85, -48, -13, 51, 82, 8, 17, 18, -44, -34, -13, 48, 34, -10, -50, -23, -33, -38, 76, -26, 25, 40, 28, -9, 35, -99, 29, -9, 23, 8, 76, 147, -28, -110, -76, -4, -6, 27, -37, -24, 39, 102, 116, 82, -89, -84, 7, 53, 8, -48, -140, -19, -21, 4, -88, -7, -1, -11, 22, 70, -40, -28, -27, -11, 6, 13, 80, -34, -94, -20, 53, -51, -92, 80}
, {-32, 81, -37, -75, -103, 7, 32, -4, 11, -68, -37, -80, 62, 35, 35, -37, -8, -41, 3, -69, 40, 25, 86, -31, -81, -92, 63, 54, -9, 62, -35, -41, -4, -30, 1, 31, -102, -74, 74, -27, -137, 10, 87, -42, 133, -44, 117, 31, -107, 6, -46, 24, -55, 42, -106, 23, 30, 41, 12, -84, 47, 14, 32, -16, -182, 45, 17, 107, 85, 24, -63, -51, -60, 64, -41, 73, 32, 6, -13, 112, 37, -52, 91, 32, -66, -83, -134, 10, 18, 69, 19, 37, -51, 128, -31, 24, 1, -50, -89, -56, 23, -26, -153, 29, -14, -82, 20, -70, 46, -31, 27, -91, 16, 142, -118, 98, -68, 81, 14, 35, -127, 38, 24, 7, -165, 186, -6, 7, -70, -49, 18, -23, -49, -36, 91, -66, -1, 3, 70, -24, -21, -32, 110, -45, 32, 13, -2, 87, 93, 62, -15, 26, -50, -31, -30, 33, 15, -40, -9, -44, -11, 9, 14, -43, -5, -70, 16, 56, 88, -65, -3, -7, 58, -55, -5, -21, 12, -87, -151, -8, 31, 73, 50, 3, 75, -28, 40, -51, 55, -10, -66, 21, 9, -26, 2, 58, -20, -26, 21, -22, -13, -44, 19, -28, -9, -7, 0, 33, 24, 12, -159, -58, -64, 69, 28, 34, -50, 5, -38, 37, -11, -66, 12, -139, -8, 48, 33, 37, -6, 7, 13, 38, 41, 53, 7, -65, -77, -52, 43, 94, -60, -115, 88, -66, -5, 23, -2, 13, 48, -6, -26, 142, -14, -1, 36, -38}
, {22, 3, 27, 41, 5, -14, -47, 104, -55, 52, -52, -57, -5, 71, -16, -14, 9, -19, -3, 24, -74, 8, -16, 25, 144, 63, 34, 18, 59, 85, 46, 12, -95, 14, 12, -13, -50, 86, -39, -25, -26, 68, -75, -9, 9, 37, 14, 22, -36, -32, 47, 41, -65, 57, -8, -23, -53, -105, -50, 40, -39, 27, -10, -53, 58, 23, -77, 74, -31, 85, -63, 28, -7, -22, 91, 2, 9, 76, -31, 50, 6, 3, 34, 8, -71, -61, -17, -38, 92, 19, 30, -125, 31, 37, -110, -35, -34, -29, -71, 17, -2, -84, 23, 78, 47, -14, 90, -14, -24, 12, 79, 45, 39, 11, 77, 22, -30, 62, 9, -20, -9, -48, -10, 53, 8, 38, -46, 40, -11, 67, 41, -193, -33, 72, 84, -37, 46, -131, -85, -113, -11, -3, 63, -77, -12, -36, -87, 88, 133, -40, 48, 107, -156, -52, 34, 71, 74, -41, 53, 71, 50, 39, 17, 41, -130, -4, 21, 14, -29, 93, -56, 54, 12, -16, 25, 91, -24, -46, -149, -30, -46, -9, -4, 4, 57, 170, 103, -31, 50, -79, -30, -26, 26, 48, 10, -33, -46, 33, -27, -61, -51, -24, -2, 53, 25, -66, -41, -9, 35, 36, 34, -66, -120, 97, -19, 64, -12, 89, 55, -8, 71, 19, 42, 36, 18, -18, -14, -49, 72, 76, -26, 32, -74, 45, -42, -73, 118, -64, 27, 99, 24, -11, -28, 73, 14, 6, 11, 7, -88, -46, -17, 137, -29, 48, -56, 37}
, {21, -183, -1, 27, 17, 19, -15, 31, -79, -4, 64, -25, -69, 52, -31, -40, 51, -18, -24, 41, -58, -21, 5, -40, 24, -16, -41, -49, -46, 73, -59, -38, -44, -76, -28, 24, -35, -30, -31, -58, 84, -56, -21, 29, -79, 59, -122, -51, 47, -44, 5, -92, 111, -12, 94, -73, -113, 2, -61, -28, -110, 11, -83, 0, 45, -59, -47, 21, 1, 25, -65, -7, -93, -26, 87, -89, -90, 85, 48, -66, -84, 5, -119, -12, -25, 27, 114, -90, 54, -25, -47, 10, 0, 44, -97, 21, -48, -1, 56, 114, -87, -54, 21, -36, 0, -50, 10, 32, 49, 56, -77, -29, 43, -2, 121, -121, -20, -19, 41, -2, -44, -41, -39, -25, -33, -110, -20, -15, -57, 57, -51, -35, 29, 47, 33, 14, 23, -2, -46, 60, 107, 40, 0, 7, -25, -52, 60, -49, 73, -25, -16, -13, -118, 26, 31, -11, 48, -24, -36, 42, -7, -26, -39, -55, 42, -35, -6, 2, 21, 48, 23, -27, -23, 40, -24, -7, -4, 35, -57, -11, -24, -85, -37, -34, 13, 43, -50, -60, 11, 19, -20, 20, 74, 30, 15, 5, 4, -42, 73, 79, 48, -5, -31, 30, -27, -20, -81, -8, 136, 90, 39, -154, 0, 130, 42, -83, -39, -24, 104, -58, -47, 24, 18, 10, 68, -64, -39, -8, 81, 115, 127, 103, -78, -10, 90, -19, -51, -61, 35, 34, 22, 15, -64, 7, -6, -43, -138, 8, -35, -154, -8, 65, 7, 35, -79, 51}
, {9, 19, 10, 3, -52, -56, -28, -76, -16, 1, -116, 36, -5, -31, -3, 14, -25, -27, 61, 27, 19, 22, 60, 64, -129, -93, 4, 63, 1, -49, 36, 72, 32, 8, -38, -18, -25, 18, 12, 8, -32, 32, -13, -35, 27, 3, 83, 4, 41, 30, 23, 7, -95, -2, -17, -1, 53, 37, 37, -42, -12, 8, -1, 34, -183, 16, 89, -12, 40, -22, -167, -10, -75, 117, -89, -6, -5, -9, -26, 87, 80, -141, -3, 95, -39, -45, -114, 88, -68, 6, 35, -80, -96, 15, 3, 6, -41, -48, -85, -88, 85, -5, -139, -185, 55, -31, -13, 17, -41, -64, -8, -43, 27, -96, -1, 24, -45, 74, -14, 61, -55, 35, 107, 121, -131, 1, -23, -100, 23, 40, -31, -23, 19, -73, 57, -54, 30, 61, -36, -22, -29, -70, 45, -6, 3, 77, -30, 3, -27, 20, 17, 10, 24, -10, 19, 52, 66, -71, -68, -19, 3, 62, 94, -16, 39, 2, 54, -23, 14, -7, 73, 22, 66, -34, 53, 5, 7, -33, -113, -79, 44, 17, -4, 27, -37, -77, 97, 32, -32, -45, -19, -51, -21, -32, -4, 2, -81, 16, 3, 20, -11, -60, 67, -29, 26, 60, 47, -27, 58, -47, -71, -6, -49, 40, -20, -93, -52, -69, -121, 48, 61, 19, -154, -24, -36, -5, 22, -14, -142, -82, 27, 35, -77, 114, -74, 43, 50, 75, 48, -33, -57, 41, -30, -21, -3, -49, 12, -56, 14, -93, -87, -123, 23, -10, -53, 24}
, {-5, -34, 36, -106, 21, -34, -7, -76, 67, -144, -117, 45, -21, 28, 46, -73, 3, -61, 20, -31, 30, 16, 26, -31, -96, -65, 32, 44, 47, -36, 48, -37, 47, 32, -60, -67, 31, -27, 59, -7, -2, 47, -74, -80, 66, 17, 56, -26, -2, 80, -58, 60, -138, -78, -62, 46, -10, -15, -22, -56, -5, 35, 32, 15, -147, 37, 86, 53, 28, -4, -83, 36, 26, 88, -53, 0, 33, -40, -61, 8, -50, -10, 118, -47, -61, -38, -9, 66, -69, 58, -2, 64, 42, -11, 66, 24, 77, -45, -201, -184, 81, 147, -196, -58, 35, 103, -87, -26, -45, -43, -15, -46, 29, 74, -172, 24, -80, 51, -67, 80, -2, 103, 4, 1, -45, 59, -10, -13, -74, -73, -58, 32, -48, -81, -11, 52, 11, 68, 41, 42, -48, -63, -72, -1, 70, 5, 23, -27, -64, -53, 24, 49, 92, 46, 88, -78, -64, -54, 19, -39, -21, -44, 2, -114, 32, 35, -61, 1, -61, 15, -38, -81, -12, -82, -147, -13, 32, 33, 76, -26, 65, 53, -93, -37, -33, -110, -113, -6, -106, 33, 66, 26, 35, -72, -83, 7, 26, -18, 21, 68, -10, -46, 37, -107, -13, 9, 35, -17, 42, -86, -9, 110, -22, -5, -9, 75, 21, -62, -74, -20, -3, 38, -94, -77, -124, 36, 35, 106, -92, -41, -15, 31, 27, 102, -2, -30, -58, 107, 41, 8, 6, 80, 82, -133, -23, 59, 60, -12, -26, 64, -37, -118, 3, -95, 6, -27}
, {93, 39, -53, -7, -103, 120, -4, -3, 18, -52, 87, -26, -59, 23, 103, -62, 41, 56, -40, -89, 37, -72, 46, 23, 7, -4, 8, 12, -30, -88, -28, 77, 11, 40, 45, 106, -10, 45, 57, 7, -13, -28, 70, -62, 37, -15, -3, -16, 11, -84, 58, 14, 95, -4, 46, -18, 78, 122, 48, 43, 54, 10, -20, 43, -144, -10, 40, 56, -153, -9, 26, 94, -30, -33, -17, -7, 17, 32, -151, 63, 38, -7, 13, -119, -2, -114, 86, 54, -223, 49, 36, 39, -16, 30, 6, 17, -14, 13, -82, -46, -26, 62, -72, 97, -53, 124, -39, -61, 11, -81, 33, -60, 63, 113, -69, 41, -43, -13, -127, -9, 11, -20, 1, -92, 74, 84, 27, 51, -79, -56, 28, 33, -31, 94, -56, -14, -23, -40, 84, -26, -1, 97, -18, 36, 37, -115, 24, -20, 58, 44, 16, -65, 2, 27, -11, 23, -70, 22, -55, -2, -44, -49, -11, 19, -47, 75, 11, -19, -8, -27, 16, -69, -18, 78, -45, 40, -15, 33, 5, 58, 44, -25, 59, -92, -11, -64, -114, 30, -58, 15, 14, 19, -26, -82, -111, 14, -12, -56, 73, 15, -5, 62, 11, -93, -182, -21, -21, -42, -91, -101, -23, -18, 52, -140, 14, 3, 53, -39, -17, -62, -15, -23, 30, 116, -62, -18, 129, 90, 31, -10, 8, 39, 2, -133, -60, 6, -49, 2, -179, 20, 41, -7, 42, -67, -14, 26, -13, 8, -1, -42, 70, 75, -29, -45, 27, -4}
, {78, -56, -86, -91, -4, 0, 39, -57, -72, -37, 81, -23, -116, -28, 6, -142, 11, 48, -42, -49, 41, 67, 44, -24, 3, -57, -59, 11, -11, -167, -29, -4, 36, -19, 34, 127, -23, -62, 38, 59, 6, -6, 69, 56, 11, -1, -45, 74, -19, -15, 34, -60, 51, 14, 34, -13, 23, 57, 50, -18, -19, -65, -36, -87, 12, -50, -70, 43, -130, -3, 68, -28, 48, 12, 42, 39, -65, -50, -81, 73, 2, -77, 50, -104, -3, -108, 28, 6, -53, 76, -12, 96, 81, -58, 2, 46, 38, 43, -28, 14, -70, -33, 27, 77, -76, 123, -50, -19, 74, 7, 30, 12, 54, 34, 24, -32, 10, -45, -87, -9, 49, -41, 78, -116, 99, 152, 39, -38, 26, -90, 12, -15, 38, 11, -80, 60, -8, 74, -37, 23, 12, 74, -72, 37, 68, 40, 5, 25, 4, 10, -95, 12, 14, -18, 62, -50, -77, -14, -6, 13, 61, -7, 62, 52, 42, 31, 41, 71, 52, 7, -5, -2, -52, 35, -12, -82, 63, 51, 42, -20, -51, -61, 13, -35, 12, 5, -61, -56, -38, 37, -26, 5, 38, -91, 18, -71, -26, 12, 2, -73, -107, -14, 8, -99, -136, 46, -53, -64, -21, 14, 121, -31, 157, -44, 92, -98, 35, 27, -46, -155, -74, 21, -30, 27, -45, 49, -5, -13, -23, -20, 31, -9, 15, -146, 67, 16, 0, -7, -9, -60, 108, -21, -50, -32, 5, -20, -17, 16, 105, 0, 36, 19, 26, 42, -36, 55}
, {-169, -27, 0, 5, 39, 17, -130, 68, 12, 7, 43, 76, 27, 129, -72, 123, -28, -138, 46, 8, -162, 25, -87, 6, -55, 100, 0, -3, -95, 66, -27, -24, 20, -65, -71, -123, 42, 118, -174, 23, 48, -33, -117, -12, -176, -13, -82, -106, 118, 85, 21, 10, -55, -44, 65, 59, -88, -155, -111, -55, -76, 12, -68, 62, -35, 23, 9, 24, 44, 3, -126, -81, -4, -8, -62, -65, 37, 92, 100, -8, -41, 63, -77, 102, 45, 75, -101, 23, 101, -23, -45, -33, -57, 90, 12, -8, -47, 103, -45, -4, 28, -8, 30, -94, -16, -67, -3, 38, -32, -18, 25, 6, -100, -36, 17, -27, -91, 5, 4, 3, 69, 127, -1, 70, -112, -85, -33, -10, -70, -12, -9, 36, -47, -60, -8, 41, -30, -65, 26, -27, 74, -63, -26, 62, -75, -42, -26, 36, 29, -63, -65, -11, -81, 45, 17, 10, 13, -83, 72, -7, -25, -52, -42, -42, 19, 11, 0, -37, -68, 77, 31, -18, 15, -60, -13, 34, 35, 71, 30, 3, -5, -9, -90, -10, -33, 31, 20, -28, -74, 3, 9, 17, 39, 68, 15, 50, 16, 21, -1, 19, 56, 24, 30, 84, 94, -40, 10, 22, -9, 39, 91, 3, -8, 25, -49, 2, -20, -39, 68, -51, -52, 10, 73, 1, 45, -66, -69, -8, 43, 22, 2, 26, -16, 98, -42, -120, 21, -19, 33, -40, 27, -48, -94, 38, 0, -37, -6, 46, -6, 83, -8, 110, -89, -4, -7, 11}
, {4, 15, -19, -37, 19, 155, 23, -3, -21, -76, 8, 20, 6, 89, -94, 24, -72, 41, 50, -63, -22, 23, -52, -78, 28, -18, -66, 27, 74, -4, -14, 96, -65, 50, 21, -60, -18, -27, 26, 62, -55, 28, -95, 1, -137, -20, 1, -88, 39, 1, 11, -28, 2, -21, 73, -37, -13, -142, -6, 27, 39, 65, 16, 43, 35, -65, -3, -37, 51, 76, -11, -1, -40, 66, 163, -69, 34, 35, 6, 6, 35, 73, 69, 33, 2, -7, -58, -82, 69, -9, -45, -23, -97, 55, -103, -21, -31, 19, -15, -19, -32, 13, -9, 11, -24, -91, -62, 54, -34, 92, -24, -21, -20, 42, 10, -37, -41, 64, -21, -24, -106, -104, -20, 42, -101, -13, -56, 50, 147, 27, 29, 32, -69, 99, 43, 23, 53, -2, -19, -140, 64, -87, -19, -8, -80, 60, -41, 20, 81, 17, 47, -31, -8, -114, 30, -45, 41, -110, -31, 50, 54, 52, -9, 47, 17, -75, -5, -39, -52, 79, 26, -90, 5, 13, 31, 58, -149, -30, 1, 7, 75, -169, -126, -31, 30, 46, 2, 57, 3, -66, 38, -116, 39, -8, 56, 49, -132, -54, 46, 52, 145, 27, 47, -66, -22, -175, -118, -28, 91, 105, -4, -19, -11, 24, 14, 53, -79, -25, 84, 27, 32, -69, -4, -31, 41, -35, 60, 52, 64, 91, 101, 22, -54, 85, 47, -49, -29, -125, -123, 73, -75, 7, 28, 60, -28, -82, -63, 18, -128, -26, -9, 76, 79, -49, 2, -110}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_1_H_
#define _DENSE_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 10

typedef int16_t dense_1_output_type[FC_UNITS];

#if 0
void dense_1(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_1_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 10
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define BIASES_SCALE_FACTOR 9
#define TMP_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_1(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 128
#define FC_UNITS 10


const int16_t dense_1_bias[FC_UNITS] = {25, 31, -86, -29, -56, 23, -57, -77, 158, 25}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{93, -93, -35, -37, 28, -173, -33, -143, 21, 122, -99, -35, 39, 15, 72, -113, -142, -89, -136, 46, 87, -167, 59, -176, -38, 62, -172, -126, 46, -112, -77, -212, -118, 48, 100, -24, -125, -143, 65, -136, -301, -105, 127, -153, -56, 49, -106, 96, 100, -152, -169, -181, 22, -108, 62, 14, -69, 11, 87, 97, -14, 71, -36, 21, 114, -163, -78, -77, 69, 60, 96, 4, -77, 112, 53, 68, 118, 79, -167, -51, -95, -97, 72, -172, 86, 43, -93, 64, 25, -47, -87, 85, -36, -117, 84, -185, -161, -166, -99, -116, -29, -111, -144, 116, 97, -23, -137, 70, 89, -86, -114, 74, -11, -133, 71, 46, -76, -38, -19, 71, -111, -51, 101, 114, 38, -105, -207, 52}
, {117, 78, -128, -29, 95, 38, 90, 93, -161, -43, 83, -177, -120, 159, -165, -63, 9, -125, -6, -181, -148, -22, 11, -39, -183, 74, 74, 44, -171, -3, -70, -123, -148, -71, -74, 90, -190, 71, -77, 89, 33, -121, -16, -14, -67, -202, 62, -65, 36, 132, 18, 27, -104, 68, 39, 104, -21, 109, -108, -99, 92, -199, 129, -144, -54, -218, -113, -74, -144, 91, -81, -12, -179, -70, 84, 33, 146, -93, 41, -114, 118, 71, -214, -120, 79, -180, 107, -23, -45, -180, -60, -72, 91, 77, 82, 99, -18, 188, 113, 91, -118, -104, 73, 98, -126, 92, 99, 90, -108, 114, -197, -152, -94, -43, -158, 89, 25, -115, -143, -91, 99, 104, -132, -192, -167, 73, 58, 27}
, {101, -78, -74, -44, 116, 54, 83, 60, 60, 90, 80, 48, 115, -93, -142, 85, 100, -128, -104, 38, -67, -108, 130, -7, -209, -75, 27, -107, -143, -123, -101, -8, 2, 5, -58, -158, -227, -105, 36, -57, -117, 66, 62, 112, 109, -85, 78, 94, 80, 91, -68, 47, 106, -255, -53, -112, -24, -125, -71, 63, 75, -103, -82, -77, 134, -79, 68, -9, -121, 88, 92, 75, -18, -60, 90, -148, -148, 28, 15, 87, -146, 86, -171, -92, -3, 31, 97, -15, -80, -57, -106, 96, -97, -123, -103, 23, 129, -34, -47, 67, -75, -111, 79, -93, -154, 48, -89, 30, 109, 118, -63, -52, 60, 62, 7, -113, 2, 71, 59, 57, 118, 146, 17, 64, -127, -157, 90, 26}
, {61, -60, -124, 128, -114, 1, -51, 77, -57, -13, -18, -99, -101, 54, 96, 64, 100, 67, 57, -121, -114, 55, 66, 100, -78, -78, -27, -72, 59, 88, -10, 81, 98, -58, -155, 74, 63, -135, -53, 52, -20, 63, -49, 102, -42, 23, 72, 67, 60, 119, 62, 49, 108, -80, 29, -191, -19, -138, -76, -81, 23, 124, -81, 79, -104, 18, -80, 63, -165, -95, 76, 44, -93, -107, 40, -13, -47, 35, 47, -136, 84, 87, -96, -89, -141, 7, -96, 75, 29, -35, -56, -95, -130, -100, -5, 18, -88, -102, -169, 117, 84, 108, -33, -37, -114, -36, -104, 34, 21, -41, -150, -102, 101, 71, -109, -80, 119, -49, 16, -96, -33, -86, -21, 68, -148, -142, 63, -50}
, {-60, -81, 120, -81, 83, -152, -78, 47, 114, 84, -107, 41, 95, 19, 60, -30, 116, -126, -128, 60, 62, 70, 4, 6, 93, 73, 101, 104, -177, 66, 120, 67, -149, 88, -67, 66, -136, 47, -51, 80, 35, -182, -109, 99, -3, -112, -130, -182, -157, -75, -102, -133, -158, -88, -191, 94, -77, 16, -108, 87, 78, -33, -43, -39, 88, 82, -126, -204, -52, 97, -60, -154, 103, 32, -106, -163, 94, -119, -75, 70, -35, -165, 46, 84, -143, -150, 93, -39, 125, 86, -43, 108, 67, -63, 60, 77, -7, -49, 117, 28, -58, -128, -164, -80, 29, 81, 69, 92, -20, -175, 71, 45, 104, -12, -97, -108, -98, 8, 19, -110, -81, 92, -72, -56, -142, 19, -118, -3}
, {82, 102, 101, -152, -144, 20, -9, 78, 82, -10, -4, -79, 73, 128, 5, -236, -123, 86, -198, -70, -127, -91, -50, 111, 63, -241, -54, 88, 48, 71, -59, 84, -136, 27, -15, -114, 15, 69, 74, -123, -70, 53, 111, 89, 88, 22, -9, 98, -105, -24, 45, -83, 94, 87, -43, 26, -141, -4, -55, -11, 4, -52, 58, 109, -6, -53, -56, -6, 17, -174, -175, 41, 5, 83, -161, 115, -105, 58, -91, -122, 85, -103, -75, 70, 47, 32, -64, 89, -75, -61, 5, -136, -7, 63, -120, -154, -86, -31, -108, -111, 86, 110, -70, -5, -46, -69, -30, -12, -12, -16, -155, 45, 99, -104, -105, 93, 134, -162, -201, 47, -58, -37, -158, 85, 52, 76, -138, -96}
, {-102, 103, 108, -196, 93, -3, -33, -22, 127, -47, 125, 67, -45, -73, -32, -186, -169, -42, -199, 95, 94, -75, 45, -53, 102, -127, -200, 89, -13, -241, 91, -46, -190, 105, 1, -96, 28, -145, 70, -130, -230, -178, 22, -57, 64, 8, -108, 73, -154, 12, -42, -141, 27, 70, -187, 131, -258, -72, 55, 74, 99, 41, 154, -139, -56, -110, 57, -218, 43, -164, -159, 43, -124, -41, -48, 102, -19, 69, -185, 110, -151, -10, -96, -69, -104, -147, 90, -176, 102, -108, -55, -103, -13, -80, 71, -109, 84, -47, -16, -50, 73, 94, 29, 118, 118, -205, -122, 48, -164, -100, 64, -115, -123, 36, 38, 103, -134, -209, -228, 60, 36, 26, -47, 45, 37, -11, -69, 30}
, {64, -180, -68, 111, -7, 47, 106, 29, -86, -85, -106, -166, 48, -7, -152, -33, 128, -116, 33, -21, -102, -169, -145, -72, -140, -159, 102, -40, -15, 58, 111, -134, -69, 34, -89, -51, -14, -40, -194, -126, 57, 70, 111, 21, 137, -196, 73, -129, 86, 109, -166, 46, 20, -101, 40, -195, 130, 109, 109, 75, -81, 74, 5, 71, -81, -104, -109, 68, 72, -89, 80, -70, 24, 71, -92, -109, -95, -127, 35, 108, -58, -70, -58, 48, 26, 22, 48, -152, -115, 77, 175, -96, 102, 83, -223, 56, -132, -23, 83, 25, -165, -80, 93, -218, -10, 79, 82, -83, -53, 42, -28, 52, 10, 101, 55, -111, -140, 66, 54, -62, 50, -111, 66, 70, 51, 11, 67, -101}
, {-187, 99, 87, 40, 35, 25, -117, -80, 63, 109, -69, -47, -91, -112, -87, 80, -8, 63, -32, -75, -129, 74, -72, 99, 25, 51, -94, 78, -58, -101, -110, 77, 37, 95, 112, -104, -15, 49, 54, 100, -166, 42, -94, -17, -54, -84, 60, 84, -34, -35, 38, 39, -84, 50, 5, -20, -135, -75, -11, -116, -46, -104, -106, 62, -49, -83, 46, 60, -97, -114, 79, 46, 100, 100, 34, 91, 49, 60, -127, -89, -138, 69, -66, 91, -74, 37, 90, -139, 96, 105, -81, -63, 98, 102, 22, 23, -15, 49, 79, 21, -45, 126, -98, 38, -96, -70, -94, -60, -47, -225, 44, -92, -93, 107, -137, -56, -84, 71, -40, 59, 82, -112, 82, 2, -60, -38, -36, -153}
, {-151, 69, 99, 104, -83, -141, -27, -170, -27, -90, -87, 91, 63, 4, -130, 86, -45, 82, 63, -126, 55, -60, -84, 95, 90, 24, 102, 103, -124, 72, 100, 91, 81, -153, -39, 61, -165, 55, -22, 67, 43, 65, 36, -116, -11, 38, -91, -28, 9, -88, -4, -111, -48, -39, 36, -58, 110, -6, -98, 80, -149, -113, -73, 85, -13, 59, -218, 54, -63, 59, -17, -157, 100, -57, -188, -121, -7, 18, -159, -169, -1, -57, 90, 33, -5, 19, 68, 48, -6, 76, -122, 94, -81, -28, -12, -33, -128, 9, 58, -161, -98, 64, -168, -103, 11, 72, 86, -147, -53, -92, 64, 69, 101, -139, 1, -36, -16, 58, 46, 35, -145, -152, 96, 106, -111, 22, -108, 33}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv2d.h" // InputLayer is excluded
#include "conv2d_1.h" // InputLayer is excluded
#include "conv2d_2.h" // InputLayer is excluded
#include "flatten.h" // InputLayer is excluded
#include "dense.h" // InputLayer is excluded
#include "dense_1.h"
#endif


#define MODEL_INPUT_DIM_0 28
#define MODEL_INPUT_DIM_1 28
#define MODEL_INPUT_DIM_2 1
#define MODEL_INPUT_DIMS 28 * 28 * 1

#define MODEL_OUTPUT_SAMPLES 10

#define MODEL_INPUT_SCALE_FACTOR 9 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

#define MODEL_OUTPUT_SCALE_FACTOR 9 // scale factor of last layer
#define MODEL_OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_OUTPUT_NUMBER_T int16_t
#define MODEL_OUTPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[28][28][1];
typedef int16_t input_t[28][28][1];
typedef dense_1_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
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
#include "conv2d.c"
#include "weights/conv2d.c" // InputLayer is excluded
#include "conv2d_1.c"
#include "weights/conv2d_1.c" // InputLayer is excluded
#include "conv2d_2.c"
#include "weights/conv2d_2.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif


void cnn(
  const input_t input,
  dense_1_output_type dense_1_output) {
  
  // Output array allocation
  static union {
    conv2d_output_type conv2d_output;
    conv2d_2_output_type conv2d_2_output;
    flatten_output_type flatten_output;
  } activations1;

  static union {
    conv2d_1_output_type conv2d_1_output;
    dense_output_type dense_output;
  } activations2;


// Model layers call chain 
  
  
  conv2d( // Model input is passed as model parameter
    input,
    conv2d_kernel,
    conv2d_bias,
    activations1.conv2d_output
    );
  
  
  conv2d_1(
    activations1.conv2d_output,
    conv2d_1_kernel,
    conv2d_1_bias,
    activations2.conv2d_1_output
    );
  
  
  conv2d_2(
    activations2.conv2d_1_output,
    conv2d_2_kernel,
    conv2d_2_bias,
    activations1.conv2d_2_output
    );
  
  
  flatten(
    activations1.conv2d_2_output,
    activations1.flatten_output
    );
  
  
  dense(
    activations1.flatten_output,
    dense_kernel,
    dense_bias,
    activations2.dense_output
    );
  
  
  dense_1(
    activations2.dense_output,
    dense_1_kernel,
    dense_1_bias,// Last layer uses output passed as model parameter
    dense_1_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
