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


const float conv2d_bias[CONV_FILTERS] = {-0x1.aea37e0000000p-6, -0x1.652afc0000000p-6, -0x1.87eea20000000p-4, -0x1.4d83e20000000p-9, -0x1.1c89f20000000p-5, -0x1.7f6f040000000p-4, -0x1.e19b160000000p-4, 0x1.57d3500000000p-5}
;


const float conv2d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE_Y][CONV_KERNEL_SIZE_X][INPUT_CHANNELS / CONV_GROUPS] = {{{{-0x1.c924e60000000p-2}
, {-0x1.1498740000000p-2}
, {-0x1.1c62500000000p-3}
}
, {{-0x1.9616b60000000p-7}
, {0x1.5584780000000p-5}
, {0x1.e07c780000000p-2}
}
, {{0x1.469f160000000p-3}
, {0x1.0417380000000p-2}
, {0x1.22ab600000000p-2}
}
}
, {{{-0x1.35fc4e0000000p-3}
, {0x1.e5716e0000000p-4}
, {-0x1.08f1520000000p-3}
}
, {{-0x1.507a180000000p-3}
, {0x1.79278e0000000p-5}
, {0x1.7956420000000p-3}
}
, {{0x1.88b28a0000000p-2}
, {0x1.d507f00000000p-2}
, {-0x1.8b24820000000p-6}
}
}
, {{{-0x1.d9f92a0000000p-3}
, {-0x1.8806500000000p-3}
, {0x1.4b89160000000p-2}
}
, {{-0x1.23f7540000000p-2}
, {0x1.6e9cce0000000p-3}
, {0x1.335d680000000p-2}
}
, {{0x1.501b060000000p-3}
, {0x1.605bf60000000p-3}
, {0x1.28431e0000000p-2}
}
}
, {{{0x1.f7dc3e0000000p-3}
, {0x1.216efe0000000p-2}
, {0x1.a023440000000p-3}
}
, {{-0x1.cba6260000000p-5}
, {0x1.a967a20000000p-3}
, {0x1.49fe3a0000000p-3}
}
, {{-0x1.ed99020000000p-2}
, {-0x1.e5928e0000000p-2}
, {-0x1.6015d20000000p-2}
}
}
, {{{-0x1.d0c4420000000p-2}
, {-0x1.74652c0000000p-2}
, {-0x1.03749e0000000p-3}
}
, {{0x1.6807ee0000000p-3}
, {-0x1.05892e0000000p-5}
, {0x1.df2d0e0000000p-3}
}
, {{0x1.cb8bb40000000p-2}
, {0x1.ea00aa0000000p-3}
, {-0x1.0f22040000000p-4}
}
}
, {{{0x1.cbee3e0000000p-5}
, {0x1.cc7de40000000p-3}
, {0x1.1c64f60000000p-2}
}
, {{-0x1.05ce3c0000000p-3}
, {0x1.8ab3120000000p-4}
, {0x1.d8eaa40000000p-2}
}
, {{-0x1.849e7c0000000p-2}
, {-0x1.c0dad40000000p-2}
, {0x1.ed565a0000000p-4}
}
}
, {{{0x1.dbf4400000000p-3}
, {0x1.3749880000000p-2}
, {0x1.25683a0000000p-2}
}
, {{-0x1.0137e80000000p-2}
, {0x1.6a72d80000000p-6}
, {0x1.bf30f60000000p-3}
}
, {{-0x1.c9ff940000000p-2}
, {0x1.af71920000000p-5}
, {0x1.f5b7b40000000p-3}
}
}
, {{{0x1.b44d1e0000000p-3}
, {0x1.7617ae0000000p-3}
, {0x1.6c31920000000p-2}
}
, {{-0x1.10edb60000000p-3}
, {0x1.c5b0840000000p-3}
, {-0x1.2e535a0000000p-4}
}
, {{-0x1.50de1c0000000p-1}
, {-0x1.debe360000000p-2}
, {-0x1.005bb60000000p-1}
}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_GROUPS