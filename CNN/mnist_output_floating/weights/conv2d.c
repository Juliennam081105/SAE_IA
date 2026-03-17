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


const float conv2d_bias[CONV_FILTERS] = {0x1.8b02b80000000p-4, -0x1.6ec7da0000000p-10, -0x1.a8a0b00000000p-10, 0x1.d606d00000000p-13, 0x1.06599c0000000p-9, -0x1.50807e0000000p-9, -0x1.333e6c0000000p-10, -0x1.003c5a0000000p-11}
;


const float conv2d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.0302f40000000p-2}
, {0x1.46d4c00000000p-1}
, {0x1.418b320000000p-2}
}
, {{-0x1.82263c0000000p-2}
, {-0x1.3896f80000000p-3}
, {-0x1.436a2e0000000p-2}
}
, {{-0x1.d09a4e0000000p-2}
, {-0x1.7f521e0000000p-2}
, {-0x1.c4615a0000000p-2}
}
}
, {{{-0x1.ece4540000000p-4}
, {0x1.f3d06e0000000p-3}
, {0x1.1f32100000000p-4}
}
, {{0x1.8962240000000p-2}
, {0x1.7b10ac0000000p-2}
, {0x1.9afa580000000p-2}
}
, {{-0x1.9c3d700000000p-5}
, {-0x1.01057e0000000p-4}
, {-0x1.6021560000000p-4}
}
}
, {{{-0x1.aa453a0000000p-2}
, {0x1.7fea780000000p-5}
, {-0x1.ab401a0000000p-6}
}
, {{0x1.41dc700000000p-5}
, {0x1.84dac20000000p-3}
, {0x1.6dc5da0000000p-5}
}
, {{0x1.8b19840000000p-3}
, {0x1.1c36460000000p-1}
, {0x1.2c5a580000000p-2}
}
}
, {{{0x1.935ce80000000p-3}
, {0x1.d747360000000p-4}
, {-0x1.be45600000000p-6}
}
, {{0x1.3ce3760000000p-2}
, {0x1.fc1d3a0000000p-5}
, {0x1.c51fc60000000p-3}
}
, {{0x1.cf1a4a0000000p-3}
, {0x1.64c5540000000p-4}
, {0x1.775c480000000p-3}
}
}
, {{{0x1.b481ce0000000p-4}
, {-0x1.db2af40000000p-8}
, {-0x1.61b8c20000000p-2}
}
, {{0x1.1601bc0000000p-2}
, {0x1.2728c40000000p-3}
, {-0x1.a455c80000000p-2}
}
, {{0x1.b1314c0000000p-2}
, {0x1.bb460e0000000p-3}
, {-0x1.0837780000000p-5}
}
}
, {{{0x1.285f480000000p-3}
, {-0x1.6c28d00000000p-7}
, {-0x1.a59eb00000000p-3}
}
, {{0x1.2fc4fa0000000p-2}
, {0x1.2467f00000000p-2}
, {0x1.2b32e60000000p-3}
}
, {{0x1.e605fc0000000p-3}
, {0x1.d192260000000p-3}
, {0x1.3009aa0000000p-4}
}
}
, {{{-0x1.fad3fe0000000p-3}
, {-0x1.b13fd80000000p-2}
, {-0x1.e143100000000p-2}
}
, {{0x1.1312fa0000000p-5}
, {-0x1.32e9220000000p-3}
, {-0x1.65b3960000000p-3}
}
, {{0x1.78d1f40000000p-2}
, {0x1.bdc32c0000000p-2}
, {0x1.f8970e0000000p-2}
}
}
, {{{0x1.59bf660000000p-2}
, {0x1.3cfdc40000000p-5}
, {-0x1.1020ea0000000p-6}
}
, {{0x1.c11ef60000000p-2}
, {0x1.5564c40000000p-4}
, {0x1.bcfa160000000p-3}
}
, {{-0x1.529c040000000p-4}
, {0x1.cec6ec0000000p-3}
, {0x1.ebf2a00000000p-3}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS