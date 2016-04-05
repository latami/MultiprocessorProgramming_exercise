#ifndef DEPTH_OPENCL_H
#define DEPTH_OPENCL_H

typedef enum {BRUTE_CL, HIERARCHIC_CL} searchMethod_ocl;

typedef enum {CPU = 1, GPU = 2} device_ocl;

/* Generates post-processed depthmap. Takes: 2 32-bit stereo-images, their width
 * and height (mod 4). Wanted blocksize for a search, disparity-limit and search method.
 * Also selects either cpu or gpu depending on value in variable dev.
 * On success:
 *  Returns 1/4 by 1/4 image.
 * On failure:
 *  Returns NULL. */
unsigned char *generateDepthmap_opencl_basic(unsigned char *img0, unsigned char *img1,
                                             unsigned int width, unsigned height,
                                             unsigned int blockx, unsigned int blocky,
                                             unsigned int disp_limit, searchMethod_ocl select,
                                             device_ocl dev);
#endif
