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


const float conv2d_4_bias[CONV_FILTERS] = {0x1.24202a0000000p-6, 0x1.b2d3ce0000000p-5, 0x1.92e8600000000p-9, -0x1.a7564c0000000p-4, -0x1.b928c40000000p-6, 0x1.59f1d80000000p-5, 0x1.98eda20000000p-5, 0x1.09aa920000000p-5}
;


const float conv2d_4_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.25a3320000000p-4}
, {-0x1.f88d0c0000000p-3}
, {0x1.5b9d6c0000000p-6}
}
, {{-0x1.cb790c0000000p-3}
, {0x1.3334be0000000p-5}
, {0x1.0a218a0000000p-2}
}
, {{-0x1.083b3a0000000p-2}
, {-0x1.12f8560000000p-3}
, {-0x1.0a90260000000p-4}
}
}
, {{{-0x1.9bd5c40000000p-6}
, {-0x1.eb213c0000000p-5}
, {-0x1.99ee340000000p-3}
}
, {{-0x1.e088c20000000p-5}
, {-0x1.1e09e60000000p-4}
, {0x1.be046a0000000p-7}
}
, {{0x1.54774c0000000p-3}
, {0x1.71b7040000000p-4}
, {0x1.ae3b820000000p-4}
}
}
, {{{-0x1.1479300000000p-2}
, {0x1.1a7aac0000000p-3}
, {-0x1.3ecc1a0000000p-4}
}
, {{0x1.d1125e0000000p-4}
, {0x1.077eb60000000p-2}
, {0x1.188bc60000000p-5}
}
, {{0x1.b7ac6c0000000p-3}
, {0x1.5e0c280000000p-5}
, {0x1.76e0340000000p-3}
}
}
, {{{0x1.ab27540000000p-7}
, {-0x1.745f720000000p-5}
, {-0x1.351d8a0000000p-4}
}
, {{0x1.b8d1fe0000000p-4}
, {0x1.450f780000000p-3}
, {0x1.77ea6a0000000p-3}
}
, {{-0x1.5a6b4e0000000p-4}
, {-0x1.21f8340000000p-3}
, {-0x1.6a57b40000000p-7}
}
}
, {{{0x1.9e8d6e0000000p-4}
, {-0x1.630b6a0000000p-2}
, {-0x1.1f77f20000000p-5}
}
, {{0x1.4791440000000p-3}
, {0x1.609a0e0000000p-3}
, {-0x1.9815b60000000p-4}
}
, {{0x1.1c623e0000000p-5}
, {0x1.7d7b520000000p-4}
, {0x1.54aa2a0000000p-3}
}
}
, {{{0x1.d7c9ba0000000p-4}
, {0x1.7d4db20000000p-4}
, {0x1.fbc9700000000p-4}
}
, {{0x1.75de400000000p-5}
, {0x1.18997e0000000p-3}
, {-0x1.72e1a00000000p-3}
}
, {{-0x1.0000400000000p-2}
, {-0x1.d2042c0000000p-3}
, {-0x1.3d679a0000000p-3}
}
}
, {{{-0x1.d6a4e00000000p-5}
, {0x1.efdfae0000000p-4}
, {0x1.f730ba0000000p-6}
}
, {{-0x1.ffe6dc0000000p-3}
, {0x1.80cd4a0000000p-3}
, {-0x1.3fd3c40000000p-4}
}
, {{-0x1.44dcf00000000p-3}
, {-0x1.605e6e0000000p-2}
, {-0x1.e7b6580000000p-3}
}
}
, {{{-0x1.3e3eb40000000p-5}
, {0x1.9fed360000000p-4}
, {0x1.14aaa20000000p-3}
}
, {{-0x1.5d66a80000000p-3}
, {0x1.e7cf3a0000000p-6}
, {0x1.70c3140000000p-4}
}
, {{-0x1.de94d80000000p-4}
, {-0x1.9c03580000000p-3}
, {-0x1.9664980000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS