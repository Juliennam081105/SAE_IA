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


const float conv2d_bias[CONV_FILTERS] = {-0x1.64da4c0000000p-3, -0x1.0693f40000000p-2, -0x1.b2f8740000000p-10, -0x1.a908cc0000000p-3, -0x1.aa6f6c0000000p-4, -0x1.1d4d300000000p-4, -0x1.f14aa20000000p-3, -0x1.083c620000000p-5}
;


const float conv2d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{0x1.1bb0700000000p-3}
, {-0x1.51314a0000000p-4}
, {-0x1.c57a800000000p-3}
}
, {{0x1.7930d40000000p-2}
, {0x1.be11e40000000p-2}
, {0x1.c3a7220000000p-4}
}
, {{0x1.f2a3740000000p-3}
, {0x1.56915a0000000p-2}
, {0x1.0b08c20000000p-4}
}
}
, {{{-0x1.6891ee0000000p-2}
, {0x1.d5916a0000000p-5}
, {0x1.1a03b20000000p-2}
}
, {{-0x1.03b1300000000p-3}
, {0x1.47ed2e0000000p-2}
, {0x1.d4d8020000000p-2}
}
, {{0x1.246b300000000p-3}
, {0x1.34d0100000000p-2}
, {0x1.f3fae00000000p-4}
}
}
, {{{-0x1.6f67220000000p-2}
, {0x1.088ea80000000p-6}
, {-0x1.05bccc0000000p-2}
}
, {{0x1.2a51c20000000p-4}
, {0x1.d9bde20000000p-5}
, {-0x1.0a71100000000p-7}
}
, {{0x1.ff96740000000p-3}
, {0x1.0383020000000p-1}
, {0x1.c59ba60000000p-2}
}
}
, {{{0x1.6a20500000000p-4}
, {0x1.6046040000000p-3}
, {-0x1.3a342c0000000p-2}
}
, {{0x1.bfbf360000000p-2}
, {0x1.89e9bc0000000p-4}
, {-0x1.3984960000000p-1}
}
, {{0x1.7bf2a00000000p-2}
, {0x1.e494500000000p-2}
, {-0x1.658c720000000p-5}
}
}
, {{{0x1.360de80000000p-2}
, {0x1.07bb8a0000000p-3}
, {0x1.5573720000000p-2}
}
, {{0x1.a544220000000p-5}
, {0x1.7bb0160000000p-2}
, {0x1.a27bda0000000p-2}
}
, {{-0x1.4335440000000p-5}
, {-0x1.9264d60000000p-4}
, {-0x1.14e8120000000p-2}
}
}
, {{{-0x1.1ce4940000000p-1}
, {-0x1.bf5dce0000000p-3}
, {0x1.52fb620000000p-3}
}
, {{-0x1.8a06ec0000000p-2}
, {-0x1.2de03e0000000p-8}
, {0x1.9cdb580000000p-2}
}
, {{-0x1.4cb3280000000p-2}
, {-0x1.e963aa0000000p-5}
, {0x1.b772fc0000000p-2}
}
}
, {{{-0x1.224c360000000p-5}
, {0x1.44b7000000000p-3}
, {0x1.bd9c700000000p-3}
}
, {{-0x1.9aa8d20000000p-10}
, {0x1.777c1c0000000p-4}
, {0x1.48e4620000000p-2}
}
, {{0x1.c2a8280000000p-4}
, {0x1.fbc5ae0000000p-3}
, {0x1.19acc40000000p-2}
}
}
, {{{0x1.06547c0000000p-8}
, {-0x1.4cfe760000000p-2}
, {-0x1.8acdee0000000p-1}
}
, {{0x1.b16af80000000p-3}
, {0x1.5cb16c0000000p-2}
, {-0x1.bde51e0000000p-4}
}
, {{-0x1.72aa400000000p-4}
, {0x1.d93a000000000p-2}
, {0x1.5c96ba0000000p-2}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS