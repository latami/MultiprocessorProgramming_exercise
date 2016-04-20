#ifndef DEPTH_OPENCL_AMD_H
#define DEPTH_OPENCL_AMD_H

#include "common_opencl.h"
//typedef enum {BRUTE_CL_A, HIERARCHIC_CL_A} searchMethod_ocl_amd;

//typedef enum {ACPU = 1, AGPU = 3} device_ocl_amd;

/* Generates post-processed depthmap. Takes: 2 32-bit stereo-images, their width
 * and height (mod 4). Wanted blocksize for a search, disparity-limit and search method.
 * Also selects either cpu or gpu depending on value in variable dev.
 * On success:
 *  Returns 1/4 by 1/4 image.
 * On failure:
 *  Returns NULL. */
unsigned char *generateDepthmap_opencl_amd(unsigned char *img0, unsigned char *img1,
                                             unsigned int width, unsigned height,
                                             unsigned int blockx, unsigned int blocky,
                                             unsigned int disp_limit, searchMethod_ocl select,
                                             device_ocl dev);
#endif
