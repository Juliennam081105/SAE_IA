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
#define CONV_FILTERS       8
#define CONV_KERNEL_SIZE_Y 3
#define CONV_KERNEL_SIZE_X 3
#define CONV_GROUPS        1


const float conv2d_10_bias[CONV_FILTERS] = {-0x1.54c26c0000000p-10, 0x1.eaf07a0000000p-7, -0x1.ddaede0000000p-11, -0x1.bf95d40000000p-11, -0x1.db3bc80000000p-9, -0x1.2545d40000000p-8, -0x1.b8793a0000000p-10, 0x1.461cb60000000p-5}
;


const float conv2d_10_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.3d4f440000000p-2}
, {0x1.72a0a60000000p-2}
, {0x1.f652cc0000000p-2}
}
, {{-0x1.53cc1a0000000p-4}
, {-0x1.4f156a0000000p-4}
, {0x1.3f8ece0000000p-3}
}
, {{-0x1.6063380000000p-1}
, {-0x1.25903c0000000p-2}
, {-0x1.0ebfea0000000p-9}
}
}
, {{{0x1.ec7cc20000000p-7}
, {0x1.617e660000000p-2}
, {0x1.4cb5a00000000p-2}
}
, {{-0x1.0bce2c0000000p-1}
, {-0x1.7c1d360000000p-1}
, {-0x1.1b4ace0000000p-1}
}
, {{0x1.c6d88c0000000p-2}
, {0x1.15d9900000000p-2}
, {0x1.f878fe0000000p-4}
}
}
, {{{0x1.0b7c0c0000000p-2}
, {-0x1.0e4ef60000000p-3}
, {0x1.cc364e0000000p-4}
}
, {{0x1.2859460000000p-5}
, {0x1.af26fc0000000p-2}
, {0x1.a16f320000000p-2}
}
, {{0x1.174c5c0000000p-2}
, {0x1.f369400000000p-4}
, {-0x1.47f45c0000000p-3}
}
}
, {{{-0x1.952f540000000p-5}
, {0x1.9fe3720000000p-4}
, {-0x1.a5ad4e0000000p-3}
}
, {{0x1.36bac40000000p-2}
, {-0x1.cb506c0000000p-7}
, {0x1.d43e0c0000000p-5}
}
, {{0x1.304f880000000p-2}
, {0x1.94ada00000000p-2}
, {0x1.3c20a00000000p-3}
}
}
, {{{0x1.03f5340000000p-2}
, {0x1.417bc80000000p-3}
, {0x1.3bd43e0000000p-4}
}
, {{0x1.1d83200000000p-2}
, {0x1.28b2260000000p-2}
, {0x1.4e04380000000p-2}
}
, {{0x1.36a5c60000000p-2}
, {-0x1.066ef60000000p-3}
, {-0x1.e04dd60000000p-3}
}
}
, {{{-0x1.dd4db60000000p-8}
, {0x1.2370a80000000p-2}
, {0x1.2972740000000p-3}
}
, {{0x1.cb4db40000000p-5}
, {0x1.b757200000000p-3}
, {-0x1.b7eba60000000p-5}
}
, {{0x1.225fe00000000p-3}
, {0x1.ebd5620000000p-3}
, {0x1.25d7d80000000p-2}
}
}
, {{{0x1.318d280000000p-6}
, {-0x1.0a025a0000000p-4}
, {-0x1.b4bf980000000p-3}
}
, {{0x1.c551600000000p-2}
, {0x1.c74de40000000p-3}
, {0x1.e9cd3c0000000p-3}
}
, {{-0x1.22a7560000000p-4}
, {0x1.fafb8c0000000p-3}
, {0x1.146ce20000000p-2}
}
}
, {{{0x1.346b400000000p-2}
, {0x1.5f645a0000000p-2}
, {0x1.03305e0000000p-2}
}
, {{-0x1.77d4220000000p-4}
, {0x1.2e48ca0000000p-2}
, {0x1.b46af80000000p-3}
}
, {{-0x1.2a4c180000000p-1}
, {-0x1.5eac9a0000000p-1}
, {-0x1.e66cda0000000p-2}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS