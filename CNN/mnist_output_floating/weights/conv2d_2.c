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


const float conv2d_2_bias[CONV_FILTERS] = {0x1.6efa460000000p-5, 0x1.104f880000000p-6, -0x1.8b6e5a0000000p-8, 0x1.1055ac0000000p-4, 0x1.aceb860000000p-6, -0x1.adecfc0000000p-4, -0x1.416d360000000p-3, 0x1.3f4cfe0000000p-5}
;


const float conv2d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-0x1.45575c0000000p-7}
, {0x1.00a9c20000000p-3}
, {0x1.e7121a0000000p-3}
}
, {{-0x1.7abb320000000p-7}
, {-0x1.31a4800000000p-4}
, {0x1.57541c0000000p-3}
}
, {{-0x1.0620520000000p-2}
, {-0x1.37577a0000000p-3}
, {-0x1.0534ec0000000p-2}
}
}
, {{{-0x1.05d88e0000000p-6}
, {-0x1.4878860000000p-6}
, {0x1.1e4a900000000p-6}
}
, {{-0x1.1c393c0000000p-3}
, {-0x1.4ca4f00000000p-2}
, {0x1.2b2c880000000p-3}
}
, {{0x1.306f720000000p-3}
, {-0x1.9a74820000000p-5}
, {-0x1.4744b00000000p-3}
}
}
, {{{-0x1.169af80000000p-4}
, {-0x1.0e516a0000000p-3}
, {0x1.51ee200000000p-3}
}
, {{-0x1.6af8760000000p-6}
, {0x1.a541900000000p-3}
, {0x1.e30a200000000p-5}
}
, {{0x1.442b800000000p-4}
, {-0x1.bdc2940000000p-6}
, {-0x1.15c7160000000p-2}
}
}
, {{{-0x1.086b3c0000000p-3}
, {-0x1.be384c0000000p-2}
, {-0x1.fb84ec0000000p-3}
}
, {{-0x1.26cdea0000000p-3}
, {0x1.1fef640000000p-3}
, {-0x1.233b720000000p-3}
}
, {{-0x1.4b1ee60000000p-4}
, {0x1.69d6a40000000p-3}
, {-0x1.cb90080000000p-5}
}
}
, {{{-0x1.0ed2d40000000p-2}
, {-0x1.83f0e60000000p-3}
, {-0x1.3cd3300000000p-2}
}
, {{0x1.8872ba0000000p-4}
, {0x1.c6084e0000000p-4}
, {-0x1.d85b240000000p-3}
}
, {{0x1.0de88c0000000p-2}
, {-0x1.0b58060000000p-4}
, {-0x1.36e9600000000p-3}
}
}
, {{{0x1.3442da0000000p-5}
, {0x1.139e680000000p-3}
, {-0x1.f975f80000000p-4}
}
, {{0x1.69a29a0000000p-3}
, {-0x1.17aea20000000p-3}
, {-0x1.ef16660000000p-6}
}
, {{0x1.90678a0000000p-3}
, {-0x1.a1cd940000000p-5}
, {-0x1.f051620000000p-4}
}
}
, {{{0x1.63fe420000000p-7}
, {-0x1.934ee00000000p-3}
, {-0x1.26fede0000000p-3}
}
, {{0x1.83ea740000000p-3}
, {0x1.b0c6aa0000000p-3}
, {0x1.f546f20000000p-3}
}
, {{-0x1.031f720000000p-4}
, {-0x1.7684100000000p-5}
, {0x1.a46bc60000000p-10}
}
}
, {{{0x1.301fdc0000000p-5}
, {0x1.2bb4540000000p-6}
, {-0x1.1113200000000p-3}
}
, {{-0x1.ecf4360000000p-7}
, {-0x1.cc6f5e0000000p-3}
, {-0x1.e9b0700000000p-4}
}
, {{-0x1.18ab740000000p-2}
, {-0x1.81d2640000000p-5}
, {0x1.e1ba480000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS