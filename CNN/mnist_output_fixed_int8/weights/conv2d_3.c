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


const int8_t conv2d_3_bias[CONV_FILTERS] = {-1, -3, -1, -2, -5, -1, -2, -1, -2, -2, 11, 0, -1, -1, -1, -1}
;


const int8_t conv2d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-11}
, {-35}
, {-30}
}
, {{6}
, {7}
, {-2}
}
, {{18}
, {14}
, {24}
}
}
, {{{-9}
, {11}
, {-15}
}
, {{11}
, {24}
, {18}
}
, {{-12}
, {18}
, {-18}
}
}
, {{{-18}
, {8}
, {-9}
}
, {{17}
, {21}
, {21}
}
, {{-2}
, {0}
, {4}
}
}
, {{{-8}
, {15}
, {15}
}
, {{16}
, {7}
, {9}
}
, {{-14}
, {6}
, {-8}
}
}
, {{{-28}
, {9}
, {18}
}
, {{-18}
, {10}
, {5}
}
, {{-1}
, {16}
, {5}
}
}
, {{{1}
, {-6}
, {1}
}
, {{-6}
, {27}
, {23}
}
, {{-8}
, {-7}
, {6}
}
}
, {{{-13}
, {3}
, {-9}
}
, {{-4}
, {24}
, {6}
}
, {{11}
, {16}
, {5}
}
}
, {{{-13}
, {-12}
, {2}
}
, {{-13}
, {-4}
, {14}
}
, {{0}
, {17}
, {22}
}
}
, {{{7}
, {-13}
, {-3}
}
, {{-11}
, {14}
, {29}
}
, {{-4}
, {-6}
, {11}
}
}
, {{{6}
, {2}
, {9}
}
, {{12}
, {24}
, {1}
}
, {{-16}
, {1}
, {4}
}
}
, {{{-32}
, {-21}
, {-10}
}
, {{7}
, {-7}
, {-30}
}
, {{13}
, {18}
, {7}
}
}
, {{{-18}
, {-22}
, {-1}
}
, {{-38}
, {-12}
, {10}
}
, {{-14}
, {22}
, {33}
}
}
, {{{-6}
, {-10}
, {-13}
}
, {{-3}
, {4}
, {-7}
}
, {{25}
, {28}
, {11}
}
}
, {{{19}
, {-13}
, {-24}
}
, {{28}
, {3}
, {-27}
}
, {{7}
, {11}
, {-8}
}
}
, {{{-3}
, {5}
, {-20}
}
, {{-1}
, {-4}
, {-26}
}
, {{23}
, {35}
, {3}
}
}
, {{{-13}
, {4}
, {-3}
}
, {{6}
, {7}
, {5}
}
, {{10}
, {16}
, {10}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS