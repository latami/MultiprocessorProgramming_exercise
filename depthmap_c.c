#include <stdio.h>
#include <stdlib.h>
#include <string.h> // memset
#include <math.h>
#include <float.h> // FLT_MAX definition
#include <assert.h>

#include "doubleTime.h"
#include "depthmap_c.h"

/* Blends 4x4 block of pixels (32bit, alpha is ignored) together to form 1 pixel,
 * which is converted to greyscale float image.
 * Returns:
 *  On success, memory-pointer to greyscale-image.
 *  On failure, returns NULL. */
float *blend4x4_cnvrtToGreyscale(unsigned char *image,
								 unsigned int w, unsigned int h) {
	int y, x, i, j, r, g, b;
	float *result;

	if ((w % 4 != 0) || (h % 4 != 0)) {
		fprintf(stderr, "blend4x4 does not currently handle resolutions not "
						"divisible by 4!\n");
		return NULL;
	}

	result = malloc(sizeof(float)*w*h/16);
	if (result == NULL) {
		fprintf(stderr, "Memory allocation failed!");
		return NULL;
	}

	/* x, y are indices for new resized image */
	for (y = 0; y < h/4; y++) {
		for (x = 0; x < w/4; x++) {
			r = 0;
			g = 0;
			b = 0;

			/* Blend 4x4 pixel-block to 1 pixel */
			for (j = 0; j < 4; j++) {
				for (i = 0; i < 4; i++) {
					r += image[((y*4+j)*w+(x*4+i))*4];
					g += image[((y*4+j)*w+(x*4+i))*4+1];
					b += image[((y*4+j)*w+(x*4+i))*4+2];
				}
			}
			/* Divide color-values by 16 */
			r = r >> 4;
			g = g >> 4;
			b = b >> 4;

			/* Convert to greyscale */
			result[y*(w/4)+x] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
		}
	}
	return result;
}

/* Blends 2x2 pixel together.
 * Returns: Pointer to resized image. */
float *blend_2x2(float *img, unsigned int w, unsigned int h) {
	int x, y, i, j;
	float pixel, *resized;

	resized = malloc(sizeof(float)*(w/2)*(h/2));

	for (y=0; y < h/2; y++) {
		for (x=0; x < w/2; x++) {
			pixel = 0.0f;
			for (j = 0; j < 2; j++) {
				for (i = 0; i < 2; i++) {
					pixel += img[((y*2+j)*w+(x*2+i))];
				}
			}
			pixel /= 4.0f;
			resized[y*(w/2)+x] = pixel;
		}
	}
	return resized;
}

/* Caches block elements subtracted by mean, and deviations.
 * Assumes cacheData memory is already correctly allocated. */
void scanline_cacheBlkData(float *img, unsigned int scanline, float *cacheData,
						   unsigned int width, unsigned int bx, unsigned int by) {
	float rcp_div_bxby, mean, *blkMean;
	unsigned int lineTop, lineBot;
	int x, y, i, bxSide;

	rcp_div_bxby = 1.0f/(float)(bx*by);
	lineTop = scanline-by/2;
	lineBot = scanline+by/2+1;
	bxSide = bx/2;

	/* Clear part of the memory. */
	memset(cacheData, 0, sizeof(float)*width);

	/* First stage for block mean calculation. Add columns (y-direction)
	 * together. */
	for (y = lineTop; y < lineBot; y++) {
		for (x = 0; x < width; x++) {
			cacheData[x] += img[y * width + x];
		}
	}

	/* Temp pointer for temporal mean's and deviation location on allocated
	 * memory. */
	blkMean = &cacheData[(width-(bx-1))*bx*by];

	/* Calculate mean-values for blocks. Overwritten by deviations at the end
	 * of the function. */
	for (x = bxSide; x < width - bxSide; x++) {
		mean = 0.0f;
		for (i = 0; i < bx; i++) {
			mean += cacheData[x + i - bxSide];
		}
		blkMean[x] = mean * rcp_div_bxby;
	}

	/* Cached elements are stored in serialized form, meaning that 1st 9x9-block
	 * occupies indices 0-80, 2nd indices 81-161 in 1D-array and so on. */
	float subtracted, squared_deviations;
	for (i=bxSide; i < width - bxSide; i++) {
		squared_deviations = 0.0f;
		mean = blkMean[i]; /* Avoids memory read in innermost loop */
		for (y=lineTop; y < lineBot; y++) {
			for (x=0; x < bx; x++) {
				subtracted = img[y*width+x+i-bxSide] - mean;
				cacheData[(i-bxSide)*bx*by + (y-lineTop)*bx + x] = subtracted;
				squared_deviations += subtracted*subtracted;
			}
		}
		/* Deviation is stored starting blockW*blockH*(windowW-blockW-1)
		 * for x=0. Actual stddev's are starting from x=blockW/2 because we
		 * calculate data only for full blocks. */
		blkMean[i] = sqrtf(squared_deviations);
	}
}

/* Searches best matches in stereo-images using zero-mean normalized cross correlation.
 * Returns:
 *  On success: returns 2 depthmap-pointers through a pointer.
 *  On failure: returns atleast 1 NULL depthmap-pointer. */
void zncc2way(float *left, float *right, unsigned int width,
						unsigned int height, unsigned int bx, unsigned int by,
						unsigned short *displacements,
						unsigned char **dMap1, unsigned char **dMap2) {

	unsigned int blkSidex, blkSidey, dlim, disp=0;
	int x, xx, i, d, scanline;
	float *cache_ccorrelations_dMap2;
	float *cache_blkData_r, *cache_blkData_l;


	(*dMap1) = malloc(width*height*sizeof(unsigned char));
	(*dMap2) = malloc(width*height*sizeof(unsigned char));

	if ((*dMap1) == NULL || (*dMap2) == NULL)
		return;

	/* Wipe memory */
	memset((*dMap1), 0, sizeof(unsigned char)*width*height);
	memset((*dMap2), 0, sizeof(unsigned char)*width*height);

	/* Block distance from block-center to block-edge. */
	blkSidex = bx/2;
	blkSidey = by/2;

	/* Allocate memory for all block values in one scanline + deviation
	 * values at the end of allocated memory. */
	cache_blkData_l = malloc(sizeof(float)*((width-(bx-1))*bx*by+width));
	cache_blkData_r = malloc(sizeof(float)*((width-(bx-1))*bx*by+width));

	/* For comparing correlations for 2nd depthmap. */
	cache_ccorrelations_dMap2 = malloc(sizeof(float)*width);


	float temp1, temp2, deviations_left, deviations_right, val, maxVal;
#ifndef NDEBUG
	float maxValGlo=-FLT_MAX, minValGlo=FLT_MAX;
#endif

	/* Scanline to analyze */
	for (scanline = blkSidey; scanline < height - blkSidey; scanline++) {

		/* Calculate and cache block data. */
		scanline_cacheBlkData(left, scanline, cache_blkData_l, width, bx, by);
		scanline_cacheBlkData(right, scanline, cache_blkData_r, width, bx, by);

		/* Set maximum negative single precision floating point value. */
		for (i = 0; i < width; i++) {
			cache_ccorrelations_dMap2[i] = -FLT_MAX;
		}

		/* Go through one scanline */
		for (x = blkSidex; x < width - blkSidex; x++) {

			xx = x - blkSidex;

			deviations_left = cache_blkData_l[(width-(bx-1))*bx*by + x];

			maxVal = -FLT_MAX;
			/* Set disparity-range for a loop. */
			d = displacements[scanline*width*2+x*2];
			dlim = displacements[scanline*width*2+x*2+1];

			for (d=d; d <= dlim; d++) {
				deviations_right = cache_blkData_r[(width-(bx-1))*bx*by + x-d];

				/* Left element multiplied with right element. */
				/* Unrolled by 4. This makes function ~70% faster overall. */
				/* Sum values in parallel to avoid chain dependancy
				 * (3 adds is 9 cycles of latency on Intel processors) and
				 * to take advantage of instruction level parallelism. Most of
				 * the performance increase of unrolling is because of this. */
				float summed[4];
				summed[0]=0.0f;
				summed[1]=0.0f;
				summed[2]=0.0f;
				summed[3]=0.0f;
				/* Amount of elements is not divisible by 4, do not go over. */
				for (i = xx*bx*by; i < xx*bx*by+bx*by-3; i += 4) {
					temp1 = cache_blkData_r[i - d*bx*by];
					temp2 = cache_blkData_l[i];
					summed[0] += temp1*temp2;
					temp1 = cache_blkData_r[i+1 - d*bx*by];
					temp2 = cache_blkData_l[i+1];
					summed[1] += temp1*temp2;
					temp1 = cache_blkData_r[i+2 - d*bx*by];
					temp2 = cache_blkData_l[i+2];
					summed[2] += temp1*temp2;
					temp1 = cache_blkData_r[i+3 - d*bx*by];
					temp2 = cache_blkData_l[i+3];
					summed[3] += temp1*temp2;
				}
				summed[0] += summed[1];
				summed[2] += summed[3];
				summed[0] += summed[2];
				/* Calculate trailing loops. */
				for (i = i; i < xx*bx*by+bx*by; i++) {
					temp1 = cache_blkData_r[i - d*bx*by];
					temp2 = cache_blkData_l[i];
					summed[0] += temp1*temp2;
				}

				val = summed[0] / (deviations_left * deviations_right);

				/* Comparison for a first depthmap. */
				if (val > maxVal) {
					maxVal = val;
					disp = d;
				}
				/* Second depthmap is constructed using exact same calculations */
				if (val > cache_ccorrelations_dMap2[x-d]) {
					cache_ccorrelations_dMap2[x-d] = val;
					(*dMap2)[scanline*width + x-d] = d;
				}
#ifndef NDEBUG
				if (val < minValGlo) {
					minValGlo = val;
				}
				if (val > maxValGlo) {
					maxValGlo = val;
				}
#endif
			}
			(*dMap1)[scanline * width + x] = disp;
		}
	}

#ifndef NDEBUG
	printf("Min and max of zncc %lf, %lf\n", minValGlo, maxValGlo);
#endif

	free(cache_blkData_l);
	free(cache_blkData_r);
	free(cache_ccorrelations_dMap2);
}

/* Postprocess depthmaps.
 * Returns: One processed image, or NULL in case of allocation failures. */
unsigned char *postProcess(unsigned char *dMap1, unsigned char *dMap2,
						   unsigned int width, unsigned int height,
						   unsigned int dispLimit) {

	int x, y,pixel_l, pixel_r, diff;
	unsigned char *result;

	result = malloc(sizeof(unsigned char)*width*height);
	if (result == NULL) {
		fprintf(stderr, "Allocating memory failed in postProcess!");
		return NULL;
	}

	for (y=0; y < height; y++) {
		for (x=0; x < width; x++) {
			pixel_l = dMap1[y*width+x];
			pixel_r = dMap2[y*width+x-pixel_l];

			diff = abs(pixel_l - pixel_r);

			if (diff > 1) {
				result[y*width+x] = 0;
			}
			else {
				/* Rescale saved value */
				result[y*width+x] = dMap1[y*width+x] * 255.5f/dispLimit;
			}
		}
	}

	unsigned char *fill, *temp;

	fill = malloc(sizeof(unsigned char)*width*height);
	if (fill == NULL) {
		fprintf(stderr, "Allocating memory failed in postProcess!");
		return NULL;
	}
	memset(fill, 0, sizeof(unsigned char)*width*height);

	int val, val1, val2, val3, val4, i, pass, passes=2;

	/* Fill black pixel with average of 4 or less neighbouring pixels.
	 * Can do multiple passes. */
	for (pass=0; pass < passes; pass++) {
		for (y=1; y < height-1; y++) {
			for (x=1; x < width-1; x++) {
				if (result[y*width+x] == 0) {
					val1 = result[(y-1)*width+x];
					val2 = result[y*width+x-1];
					val3 = result[y*width+x+1];
					val4 = result[(y+1)*width+x];
					val = 0;
					i = 0;
					if (val1 != 0) {
						val += val1;
						i++;
					}
					if (val2 != 0) {
						val += val2;
						i++;
					}
					if (val3 != 0) {
						val += val3;
						i++;
					}
					if (val4 != 0) {
						val += val4;
						i++;
					}
					fill[y*width+x] = val/(float)i;
				}
				else {
					fill[y*width+x] = result[y*width+x];
				}
			}
		}
		/* Swap source and destination image for next pass. */
		temp = result;
		result = fill;
		fill = temp;
	}
	free(fill);

	return result;
}

/* Create array of disparity-range of 0-disp_limit for every pixel. */
unsigned short *initializeDisparity(unsigned int width, unsigned int height,
									unsigned int bx, unsigned int by,
									unsigned short disp_limit) {
	int x, y;
	unsigned short *disparitys, disp;

	disparitys = malloc(sizeof(unsigned short)*width*height*2);
	memset(disparitys, 0, sizeof(unsigned short)*width*height*2);

	for (y=by/2; y < height-by/2; y++) {
		for (x=bx/2; x < width-bx/2; x++) {
			disparitys[y*width*2+x*2] = 0;
			disp = disp_limit;
			/* Scale disparity near left edge. */
			if (disp_limit > x - bx/2) {
				disp = x - bx/2;
			}
			disparitys[y*width*2+x*2+1] = disp;
		}
	}
	return disparitys;
}

/* Figure out decent disparity-ranges for a 2x2 times bigger image.
 * Depthmap1 and 2 are meant to be halfsized.
 * Width, height and disp_limit should be full-size. */
unsigned short *disparityLimits_2x2(unsigned char *depthMap1, unsigned char *depthMap2,
									unsigned int width, unsigned int height,
									unsigned int bx, unsigned int by,
									unsigned short disp_limit) {
	int x, y, i, max, min, halfx, halfy, halfWidth, val, val2, val3, bxSide, bySide;
	unsigned short *newLimits;

	newLimits = malloc(sizeof(unsigned short)*width*height*4*2);

	/* Distance from block center to edges. Also casts value to signed,
	 * avoiding hard to see effects when mixing unsigned values with possibly
	 * negative values. */
	bxSide = bx/2;
	bySide = by/2;

	halfWidth = width/2;
	for (y=by/2; y < height-bySide; y++) {
		halfy = y/2;
		for (x=bx/2; x < width-bxSide; x++) {
			/* This codeblock trys to figure out decent range for displacement */

			halfx = x/2;
			val = depthMap1[halfy*halfWidth+halfx];

			/* If halfsize depthmap has value 0, set range to maximum.
			 * Probably happens only in edges. */
			if (val==0) val=disp_limit-1;

			/* Add +-1 to value. Improves image.
			 * FIXME: Is this redundant now that val2 and val3 are in the mix also? */
			max = val+1;
			/* Reminder: Comparison was added just because range clipping did not
			 * work properly while code underneath compared value to value computed
			 * with unsigned variable. */
			if (val > 0) min = val-1;
			else min = val;

			/* Selecting two auxiliary values 2-pixel away help with sharp
			 * edges somewhat */
			val2 = depthMap1[halfy*halfWidth+halfx-2];
			val3 = depthMap1[halfy*halfWidth+halfx+2];
			if (val2 > max) max = val2;
			if (val3 > max) max = val3;
			if (val2 < min) min = val2;
			if (val3 < min) min = val3;

			/* Compare against second depthmaps values and update min and max. */
			for (i=0; i <= val; i++) {

				val2 = depthMap2[halfy*halfWidth+halfx-i];
				if (max < val2) {
					max = val2;
				}
				if (min > val2) {
					min = val2;
				}
			}
			/* Clip range to honor image borders */
			if (min*2 > x-bxSide) {
				min = (x-bxSide)/2;
			}
			if (max*2 > x-bxSide) {
				max = (x-bxSide)/2;
			}
			newLimits[y*width*2+x*2] = min*2;
			newLimits[y*width*2+x*2+1] = max*2;
		}
	}

	return newLimits;
}

/* Generates post-processed depthmap. Takes: 2 32-bit stereo-images, their width
 * and height (mod 4). Wanted blocksize for a search, disparity-limit and search method.
 * On success:
 *  Returns 1/4 by 1/4 image.
 * On failure:
 *  Returns NULL. */
unsigned char *generateDepthmap(unsigned char *img0, unsigned char *img1,
								unsigned int width, unsigned int height,
								unsigned int blockx, unsigned int blocky,
								unsigned int dispLimit, searchMethod select) {

	float *converted0, *converted1;
	double time1, time2, total1, total2;

	if ( (blockx % 2 != 1) || (blocky % 2 != 1) || blockx == 1 || blocky == 1 ) {
		fprintf(stderr, "Blocksize must be odd in both dimensions and more than 1!\n");
		return NULL;
	}

	total1 = doubleTime();

	/* Convert images to 1/4 greyscale images. */
	time1 = doubleTime();
	converted0 = blend4x4_cnvrtToGreyscale(img0, width, height);
	converted1 = blend4x4_cnvrtToGreyscale(img1, width, height);

	if (converted0 == NULL || converted1 == NULL)
		return NULL;
	time2 = doubleTime();
	printf("\nBlend 4x4 and greyscaling: %6.1lf ms.\n", (time2-time1)*1000);


	unsigned char *depthMap0, *depthMap1;
	unsigned short *dLimits;

	if (select == HIERARCHIC) {
		float *convHalf0, *convHalf1;

		/* Halve dimensions */
		time1 = doubleTime();
		convHalf0 = blend_2x2(converted0, width/4, height/4);
		convHalf1 = blend_2x2(converted1, width/4, height/4);

		if (convHalf0 == NULL || convHalf1 == NULL)
			return NULL;
		time2 = doubleTime();
		printf("Blend 2x2:                 %6.1lf ms.\n", (time2-time1)*1000);


		/* Disparity-range for every pixel. In this case 0-dispLimit/2. */
		dLimits = initializeDisparity(width/8, height/8, blockx, blocky, dispLimit/2);

		time1 = doubleTime();
		zncc2way(convHalf0, convHalf1, width/8, height/8, blockx, blocky,
				 dLimits, &depthMap0, &depthMap1);
		time2 = doubleTime();
		printf("Half-sized zncc:           %6.1lf ms.\n", (time2-time1)*1000);


		free(dLimits);
		free(convHalf0);
		free(convHalf1);


		time1 = doubleTime();
		/* Figure out decent disparity-range for 2x2 times bigger image. */
		dLimits = disparityLimits_2x2(depthMap0, depthMap1, width/4, height/4,
									  blockx, blocky, dispLimit);
		time2 = doubleTime();
		printf("Disparity-limits:          %6.1lf ms.\n", (time2-time1)*1000);


		/* Free half-resolution depthmaps */
		free(depthMap0);
		free(depthMap1);
	}
	else {
		/* Full disparity-range. */
		dLimits = initializeDisparity(width/4, height/4, blockx, blocky, dispLimit);
	}

	time1 = doubleTime();
	zncc2way(converted0, converted1, width/4, height/4, blockx, blocky,
			 dLimits, &depthMap0, &depthMap1);
	time2 = doubleTime();
	printf("zncc:                      %6.1lf ms.\n", (time2-time1)*1000);


	free(dLimits);
	free(converted0);
	free(converted1);


	time1 = doubleTime();
	unsigned char *ppo;
	ppo = postProcess(depthMap0, depthMap1, width/4, height/4, dispLimit);
	time2 = doubleTime();
	printf("Post-processing:           %6.1lf ms.\n", (time2-time1)*1000);


	free(depthMap0);
	free(depthMap1);

	total2 = doubleTime();
	printf("Total time:                %6.1lf ms.\n\n", (total2-total1)*1000);

	return ppo;
}
