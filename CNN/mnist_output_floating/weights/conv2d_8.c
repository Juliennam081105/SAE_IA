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


const float conv2d_8_bias[CONV_FILTERS] = {-0x1.c2d69c0000000p-4, 0x1.bd15200000000p-7, -0x1.f57a280000000p-4, -0x1.6cf35a0000000p-4, -0x1.6f4de80000000p-8, -0x1.e5f3040000000p-4, -0x1.880e900000000p-6, 0x1.2b21f00000000p-6}
;


const float conv2d_8_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-0x1.8294820000000p-5}
, {0x1.059f6e0000000p-3}
, {0x1.451c540000000p-2}
}
, {{0x1.7fea740000000p-4}
, {0x1.70cae60000000p-2}
, {0x1.da90f00000000p-3}
}
, {{0x1.9c7cac0000000p-3}
, {-0x1.d9a2b40000000p-4}
, {-0x1.16a23e0000000p-2}
}
}
, {{{0x1.0239940000000p-2}
, {-0x1.de22e60000000p-4}
, {-0x1.bb2f200000000p-2}
}
, {{0x1.19bca00000000p-2}
, {0x1.25bffa0000000p-4}
, {-0x1.0077ee0000000p-1}
}
, {{0x1.f1619c0000000p-2}
, {0x1.b1e4540000000p-4}
, {-0x1.23d2ec0000000p-2}
}
}
, {{{-0x1.a13ec20000000p-8}
, {-0x1.7cffd40000000p-6}
, {0x1.830bb80000000p-3}
}
, {{-0x1.ff744c0000000p-3}
, {0x1.e434980000000p-4}
, {0x1.3ae18a0000000p-3}
}
, {{-0x1.6292bc0000000p-2}
, {0x1.8527280000000p-3}
, {0x1.3519360000000p-2}
}
}
, {{{-0x1.1571b80000000p-3}
, {0x1.63a4920000000p-2}
, {0x1.ea6d0e0000000p-3}
}
, {{0x1.f82fa20000000p-3}
, {0x1.9c1f1c0000000p-2}
, {0x1.296a9e0000000p-7}
}
, {{-0x1.0827a80000000p-2}
, {-0x1.2fb4340000000p-2}
, {0x1.7b60280000000p-3}
}
}
, {{{-0x1.d63ae60000000p-3}
, {-0x1.e757f20000000p-2}
, {-0x1.ad083a0000000p-2}
}
, {{0x1.8acce80000000p-3}
, {0x1.28de060000000p-5}
, {-0x1.454b440000000p-5}
}
, {{0x1.b1b8ec0000000p-2}
, {0x1.ce243a0000000p-2}
, {0x1.4b973a0000000p-2}
}
}
, {{{0x1.df07640000000p-3}
, {-0x1.7d0bfa0000000p-4}
, {0x1.12de100000000p-7}
}
, {{0x1.6e89960000000p-2}
, {0x1.b7425c0000000p-4}
, {0x1.9b32aa0000000p-2}
}
, {{-0x1.1c60420000000p-3}
, {0x1.bc83600000000p-3}
, {0x1.2e8d620000000p-3}
}
}
, {{{0x1.adc4fc0000000p-2}
, {0x1.c00c9c0000000p-2}
, {0x1.4ed92c0000000p-3}
}
, {{-0x1.52852e0000000p-2}
, {-0x1.b1975a0000000p-5}
, {0x1.cf24960000000p-4}
}
, {{-0x1.30870a0000000p-2}
, {-0x1.5a6c3e0000000p-2}
, {-0x1.d722ca0000000p-7}
}
}
, {{{0x1.12b5b80000000p-2}
, {0x1.3102060000000p-4}
, {0x1.f8ae040000000p-2}
}
, {{-0x1.36a5fc0000000p-3}
, {0x1.0c46880000000p-5}
, {0x1.4bbe540000000p-4}
}
, {{-0x1.dce8b60000000p-2}
, {-0x1.ca88820000000p-3}
, {-0x1.27ea540000000p-1}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS