#ifndef DEPTH_C_H
#define DEPTH_C_H

typedef enum {BRUTE, HIERARCHIC} searchMethod;

/* Generates post-processed depthmap. Takes: 2 32-bit stereo-images, their width
 * and height (mod 4). Wanted blocksize for a search, disparity-limit and search method.
 * On success:
 *  Returns 1/4 by 1/4 image.
 * On failure:
 *  Returns NULL. */
unsigned char *generateDepthmap(unsigned char *img0, unsigned char *img1,
								unsigned int width, unsigned height,
								unsigned int blockx, unsigned int blocky,
								unsigned int disp_limit, searchMethod select);

#endif
