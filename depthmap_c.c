#include <stdio.h>
#include <stdlib.h>
#include <string.h> // memset
#include <math.h>
#include <float.h> // FLT_MAX definition
#include <pthread.h>
#include <unistd.h> // sysconf

#include "doubleTime.h"
#include "depthmap_c.h"

#define MAXTHREADS 8

struct blend4x4Data {
    int threadsN;
    pthread_mutex_t *lock_firstAvailable;
    int *firstAvailable;

    unsigned char *image32Bit;
    unsigned int width;
    unsigned int height;

    float *resized;
};

struct disparityData {
    int threadsN;
    pthread_mutex_t *lock_firstAvailable;
    int *firstAvailable;

    unsigned int width;
    unsigned int height;
    unsigned int bx;
    unsigned int by;
    unsigned int disp_limit;
    unsigned char *dmap1;
    unsigned char *dmap2;
    unsigned short *newLimits;
};

struct znccData {
    int threadsN;
    pthread_mutex_t *lock_firstAvailable;
    int *firstAvailable;

    unsigned int width;
    unsigned int height;
    float *greyImage0;
    float *greyImage1;
    unsigned int bx;
    unsigned int by;
    float *cache_blk_l;
    float *cache_blk_r;
    float *cache_ccorrelations_dMap2;
    unsigned short *displacements;
    unsigned char *dmap1;
    unsigned char *dmap2;
};

void *blendWorker(void *threadData) {

    int x, y, i, j, w, r, g, b, lasty;
    int sclines_increment = 32;

    struct blend4x4Data *thData = (struct blend4x4Data *)threadData;

    w = thData->width;

    while (1) {
        /* Set next work item for the thread. */
        pthread_mutex_lock(thData->lock_firstAvailable);
        y = (*thData->firstAvailable);
        lasty = y + sclines_increment;
        /* With one thread, process all the scanlines in one go. */
        if (thData->threadsN == 1)
            lasty = thData->height/4;
        (*thData->firstAvailable) = lasty;
        pthread_mutex_unlock(thData->lock_firstAvailable);

        /* If no scanlines to process, break from loop. */
        if (y > thData->height/4-1)
            break;
        /* Last work item might be smaller than sclines_increment */
        if (lasty > thData->height/4)
            lasty = thData->height/4;

        /* x, y are indices for new resized image */
        for (y = y; y < lasty; y++) {
            for (x = 0; x < w/4; x++) {
                r = 0;
                g = 0;
                b = 0;

                /* Blend 4x4 pixel-block to 1 pixel */
                for (j = 0; j < 4; j++) {
                    for (i = 0; i < 4; i++) {
                        r += thData->image32Bit[((y*4+j)*w+(x*4+i))*4];
                        g += thData->image32Bit[((y*4+j)*w+(x*4+i))*4+1];
                        b += thData->image32Bit[((y*4+j)*w+(x*4+i))*4+2];
                    }
                }
                /* Divide color-values by 16 */
                r = r >> 4;
                g = g >> 4;
                b = b >> 4;

                /* Convert to greyscale */
                thData->resized[y*(w/4)+x] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
            }
        }
    }
    return NULL;
}

/* Blends 4x4 block of pixels (32bit, alpha is ignored) together to form 1 pixel,
 * which is converted to greyscale float image.
 * Returns:
 *  On success, memory-pointer to greyscale-image.
 *  On failure, returns NULL. */
float *blend4x4_cnvrtToGreyscale(struct blend4x4Data *data) {
    int i;

    if ((data->width % 4 != 0) || (data->height % 4 != 0)) {
        fprintf(stderr, "blend4x4 does not currently handle resolutions not "
                        "divisible by 4!\n");
        return NULL;
    }

    data->resized = malloc(sizeof(float)*data->width*data->height/16);
    if (data->resized == NULL) {
        fprintf(stderr, "Memory allocation failed!");
        return NULL;
    }

    pthread_t threads[MAXTHREADS-1];

    (*data->firstAvailable) = 0;
    for (i=0; i < data->threadsN-1; i++) {
        pthread_create(&threads[i], NULL, &blendWorker, data);
    }
    /* Main thread */
    blendWorker(data);

    for (i = 0; i < data->threadsN-1; i++) {
        pthread_join(threads[i], NULL);
    }

    return data->resized;
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
        /* Reciprocal of deviation is stored starting blockW*blockH*(windowW-blockW-1)
         * for x=0. Actual rcp_dev's are starting from x=blockW/2 because we
         * calculate data only for full blocks. */
        blkMean[i] = 1.0f / sqrtf(squared_deviations);
    }
}

void *znccWorker(void *data) {

    struct znccData *thData;
    int scanline, lastscanline, width, bx, by, blkSidex, blkSidey, i, x, xx;
    int d, dlim, disp;
    float deviations_left, deviations_right, maxVal, temp1, temp2, val;

    thData = (struct znccData *)data;

    width = thData->width;
    bx = thData->bx;
    by = thData->by;
    blkSidex = bx/2;
    blkSidey = by/2;

    int sclines_increment = 10;
    while (1) {
        /* Set next work item for the thread. */
        pthread_mutex_lock(thData->lock_firstAvailable);
        scanline = (*thData->firstAvailable);
        lastscanline = scanline + sclines_increment;
        /* With one thread, process all the scanlines in one go. */
        if (thData->threadsN == 1)
            lastscanline = thData->height-blkSidey;
        (*thData->firstAvailable) = lastscanline;
        pthread_mutex_unlock(thData->lock_firstAvailable);

        /* If no scanlines to process, break from loop. */
        if (scanline > thData->height-1-blkSidey)
            break;
        /* Last work item might be smaller than sclines_increment */
        if (lastscanline > thData->height-blkSidey)
            lastscanline = thData->height-blkSidey;

        /* Scanline to analyze */
        for (scanline = scanline; scanline < lastscanline; scanline++) {

            /* Calculate and cache block data. */
            scanline_cacheBlkData(thData->greyImage0, scanline, thData->cache_blk_l, width, bx, by);
            scanline_cacheBlkData(thData->greyImage1, scanline, thData->cache_blk_r, width, bx, by);

            /* Set maximum negative single precision floating point value. */
            for (i = 0; i < width; i++) {
                thData->cache_ccorrelations_dMap2[i] = -FLT_MAX;
            }

            /* Go through one scanline */
            for (x = blkSidex; x < width - blkSidex; x++) {

                xx = x - blkSidex;

                deviations_left = thData->cache_blk_l[(width-(bx-1))*bx*by + x];

                maxVal = -FLT_MAX;
                /* Set disparity-range for a loop. */
                d = thData->displacements[scanline*width*2+x*2];
                dlim = thData->displacements[scanline*width*2+x*2+1];

                for (d=d; d <= dlim; d++) {
                    deviations_right = thData->cache_blk_r[(width-(bx-1))*bx*by + x-d];

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
                        temp1 = thData->cache_blk_r[i - d*bx*by];
                        temp2 = thData->cache_blk_l[i];
                        summed[0] += temp1*temp2;
                        temp1 = thData->cache_blk_r[i+1 - d*bx*by];
                        temp2 = thData->cache_blk_l[i+1];
                        summed[1] += temp1*temp2;
                        temp1 = thData->cache_blk_r[i+2 - d*bx*by];
                        temp2 = thData->cache_blk_l[i+2];
                        summed[2] += temp1*temp2;
                        temp1 = thData->cache_blk_r[i+3 - d*bx*by];
                        temp2 = thData->cache_blk_l[i+3];
                        summed[3] += temp1*temp2;
                    }
                    summed[0] += summed[1];
                    summed[2] += summed[3];
                    summed[0] += summed[2];
                    /* Calculate trailing loops. */
                    for (i = i; i < xx*bx*by+bx*by; i++) {
                        temp1 = thData->cache_blk_r[i - d*bx*by];
                        temp2 = thData->cache_blk_l[i];
                        summed[0] += temp1*temp2;
                    }

                    val = summed[0] * (deviations_left * deviations_right);

                    /* Comparison for a first depthmap. */
                    if (val > maxVal) {
                        maxVal = val;
                        disp = d;
                    }
                    /* Second depthmap is constructed using exact same calculations */
                    if (val > thData->cache_ccorrelations_dMap2[x-d]) {
                        thData->cache_ccorrelations_dMap2[x-d] = val;
                        thData->dmap2[scanline*width + x-d] = d;
                    }
                }
                thData->dmap1[scanline * width + x] = disp;
            }
        }
    }

    return NULL;
}

/* Searches best matches in stereo-images using zero-mean normalized cross correlation.
 * Returns:
 *  On success: returns 2 depthmap-pointers through a struct.
 *  On failure: returns atleast 1 NULL depthmap-pointer. */
void zncc2way(struct znccData *data) {

    unsigned int blkSidey;

    data->dmap1 = malloc(data->width*data->height*sizeof(unsigned char));
    data->dmap2 = malloc(data->width*data->height*sizeof(unsigned char));

    if (data->dmap1 == NULL || data->dmap2 == NULL)
        return;

    /* Wipe memory */
    memset(data->dmap1, 0, sizeof(unsigned char)*data->width*data->height);
    memset(data->dmap2, 0, sizeof(unsigned char)*data->width*data->height);

    /* Block distance from block-center to block-edge. */
    blkSidey = data->by/2;

    /* Allocate memory for all block values in one scanline + deviation
     * values at the end of allocated memory. */
    data->cache_blk_l = malloc(sizeof(float)*((data->width-(data->bx-1))*data->bx*data->by+data->width));
    data->cache_blk_r = malloc(sizeof(float)*((data->width-(data->bx-1))*data->bx*data->by+data->width));

    /* For comparing correlations for 2nd depthmap. */
    data->cache_ccorrelations_dMap2 = malloc(sizeof(float)*data->width);

    if (data->cache_blk_l == NULL || data->cache_blk_r == NULL
            || data->cache_ccorrelations_dMap2 == NULL) {
        free(data->cache_blk_l);
        free(data->cache_blk_r);
        free(data->cache_ccorrelations_dMap2);
        free(data->dmap1);
        free(data->dmap2);
        data->dmap1 = NULL;
        data->dmap2 = NULL;
        return;
    }

    int i;
    pthread_t threads[MAXTHREADS];
    struct znccData thData[MAXTHREADS];

    /* Reset first available line for processing */
    (*data->firstAvailable) = blkSidey;
    /* Children threads */
    for (i=0; i < data->threadsN-1; i++) {
        /* Copy struct and add thread-specific stuff. */
        thData[i] = (*data);
        thData[i].cache_blk_l = malloc(sizeof(float)*((data->width-(data->bx-1))*data->bx*data->by+data->width));
        thData[i].cache_blk_r = malloc(sizeof(float)*((data->width-(data->bx-1))*data->bx*data->by+data->width));
        thData[i].cache_ccorrelations_dMap2 = malloc(sizeof(float)*data->width);
        /* Check these allocations also for consistency. Actual failure untested. */
        if (thData[i].cache_blk_l == NULL || thData[i].cache_blk_r == NULL
                || thData[i].cache_ccorrelations_dMap2 == NULL) {
            free(thData[i].cache_blk_l);
            free(thData[i].cache_blk_r);
            free(thData[i].cache_ccorrelations_dMap2);
            /* Cancel previously launched threads */
            for(i=i-1; i >= 0; i--) {
                pthread_cancel(threads[i]);
                free(thData[i].cache_blk_l);
                free(thData[i].cache_blk_r);
                free(thData[i].cache_ccorrelations_dMap2);
            }
            free(data->cache_blk_l);
            free(data->cache_blk_r);
            free(data->cache_ccorrelations_dMap2);
            free(data->dmap1);
            free(data->dmap2);
            data->dmap1 = NULL;
            data->dmap2 = NULL;
            return;
        }

        pthread_create(&threads[i], NULL, znccWorker, &thData[i]);
    }
    /* Main thread */
    znccWorker(data);

    for (i = 0; i < data->threadsN-1; i++) {
        pthread_join(threads[i], NULL);
        free(thData[i].cache_blk_l);
        free(thData[i].cache_blk_r);
        free(thData[i].cache_ccorrelations_dMap2);
    }

    free(data->cache_blk_l);
    free(data->cache_blk_r);
    free(data->cache_ccorrelations_dMap2);
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

void *disparityWorker(void *data) {

    int x, y, lasty, i, bxSide, bySide, halfWidth, halfy, halfx, min, max;
    int val, val2, val3;
    int sclines_increment = 10;
    struct disparityData *thData;

    thData = (struct disparityData *)data;

    /* Distance from block center to edges. Also casts value to signed,
     * avoiding hard to see effects when mixing unsigned values with possibly
     * negative values. */
    bxSide = thData->bx/2;
    bySide = thData->by/2;
    halfWidth = thData->width/2;

    while (1) {
        /* Set next work item for the thread. */
        pthread_mutex_lock(thData->lock_firstAvailable);
        y = (*thData->firstAvailable);
        lasty = y + sclines_increment;
        /* With one thread, process all the scanlines in one go. */
        if (thData->threadsN == 1)
            lasty = thData->height-bySide;
        (*thData->firstAvailable) = lasty;
        pthread_mutex_unlock(thData->lock_firstAvailable);

        /* If no scanlines to process, break from loop. */
        if (y > thData->height-bySide-1)
            break;
        /* Last work item might be smaller than sclines_increment */
        if (lasty > thData->height-bySide)
            lasty = thData->height-bySide;

        for (y=y; y < lasty; y++) {
            halfy = y/2;
            for (x=bxSide; x < thData->width-bxSide; x++) {
                /* This codeblock trys to figure out decent range for displacement */

                halfx = x/2;
                val = thData->dmap1[halfy*halfWidth+halfx];

                /* If halfsize depthmap has value 0, set range to maximum.
                 * Probably happens only in edges. */
                if (val==0) val=thData->disp_limit-1;

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
                val2 = thData->dmap1[halfy*halfWidth+halfx-2];
                val3 = thData->dmap1[halfy*halfWidth+halfx+2];
                if (val2 > max) max = val2;
                if (val3 > max) max = val3;
                if (val2 < min) min = val2;
                if (val3 < min) min = val3;

                /* Compare against second depthmaps values and update min and max. */
                for (i=0; i <= val; i++) {

                    val2 = thData->dmap2[halfy*halfWidth+halfx-i];
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
                thData->newLimits[y*thData->width*2+x*2] = min*2;
                thData->newLimits[y*thData->width*2+x*2+1] = max*2;
            }
        }
    }
    return NULL;
}

/* Figure out decent disparity-ranges for a 2x2 times bigger image.
 * Depthmap1 and 2 are meant to be halfsized.
 * Width, height and disp_limit should be full-size. */
unsigned short *disparityLimits_2x2(struct disparityData *data) {

    int i;

    data->newLimits = malloc(sizeof(unsigned short)*data->width*data->height*4*2);

    (*data->firstAvailable) = data->by/2;

    pthread_t threads[MAXTHREADS-1];

    /* Launch threads */
    for (i=0; i < data->threadsN-1; i++) {
        pthread_create(&threads[i], NULL, &disparityWorker, data);
    }
    /* Main thread */
    disparityWorker(data);

    for (i=0; i < data->threadsN-1; i++) {
        pthread_join(threads[i], NULL);
    }

    return data->newLimits;
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
                                unsigned int dispLimit, searchMethod select, int threads) {

    double time1, time2, total1, total2;
    struct znccData Data;
    struct blend4x4Data blend;
    struct disparityData dispData;

    if ( (blockx % 2 != 1) || (blocky % 2 != 1) || blockx == 1 || blocky == 1 ) {
        fprintf(stderr, "Blocksize must be odd in both dimensions and more than 1!\n");
        return NULL;
    }
    if (threads == 0)
        threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (threads > MAXTHREADS) {
        fprintf(stderr, "Maximum threads allowed is %d.\n", MAXTHREADS);
        return NULL;
    }

    total1 = doubleTime();

    Data.threadsN = threads;
    blend.threadsN = threads;
    dispData.threadsN = threads;
    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    int lineAvailable = 0;
    Data.lock_firstAvailable = &mutex1;
    Data.firstAvailable = &lineAvailable;
    /* Can share mutex */
    blend.lock_firstAvailable = &mutex1;
    blend.firstAvailable = &lineAvailable;
    dispData.lock_firstAvailable = &mutex1;
    dispData.firstAvailable = &lineAvailable;

    /* Convert images to 1/4 greyscale images. */
    time1 = doubleTime();
    Data.width = width/4;
    Data.height = height/4;
    blend.width = width;
    blend.height = height;
    blend.image32Bit = img0;
    Data.greyImage0 = blend4x4_cnvrtToGreyscale(&blend);
    blend.image32Bit = img1;
    Data.greyImage1 = blend4x4_cnvrtToGreyscale(&blend);

    if (Data.greyImage0 == NULL || Data.greyImage1 == NULL)
        return NULL;
    time2 = doubleTime();
    printf("\nBlend 4x4 and greyscaling: %6.1lf ms.\n", (time2-time1)*1000);


    Data.bx = blockx;
    Data.by = blocky;

    if (select == HIERARCHIC) {
        /* Halve dimensions */
        struct znccData DataHalf;

        time1 = doubleTime();
        DataHalf = Data;
        DataHalf.width = Data.width/2;
        DataHalf.height = Data.height/2;

        DataHalf.greyImage0 = blend_2x2(Data.greyImage0, width/4, height/4);
        DataHalf.greyImage1 = blend_2x2(Data.greyImage1, width/4, height/4);

        if (DataHalf.greyImage0 == NULL || DataHalf.greyImage1 == NULL)
            return NULL;
        time2 = doubleTime();
        printf("Blend 2x2:                 %6.1lf ms.\n", (time2-time1)*1000);

        /* Disparity-range for every pixel. In this case 0-dispLimit/2. */
        DataHalf.displacements = initializeDisparity(width/8, height/8, blockx, blocky, dispLimit/2);

        time1 = doubleTime();
        zncc2way(&DataHalf);
        time2 = doubleTime();
        printf("Half-sized zncc:           %6.1lf ms.\n", (time2-time1)*1000);

        free(DataHalf.displacements);
        free(DataHalf.greyImage0);
        free(DataHalf.greyImage1);

        time1 = doubleTime();
        /* Figure out decent disparity-range for 2x2 times bigger image. */
        dispData.dmap1 = DataHalf.dmap1;
        dispData.dmap2 = DataHalf.dmap2;
        dispData.width = Data.width;
        dispData.height = Data.height;
        dispData.bx = blockx;
        dispData.by = blocky;
        dispData.disp_limit = dispLimit;
        Data.displacements = disparityLimits_2x2(&dispData);
        time2 = doubleTime();
        printf("Disparity-limits:          %6.1lf ms.\n", (time2-time1)*1000);


        /* Free half-resolution depthmaps */
        free(DataHalf.dmap1);
        free(DataHalf.dmap2);
    }
    else {
        /* Full disparity-range. */
        Data.displacements = initializeDisparity(width/4, height/4, blockx, blocky, dispLimit);
    }

    time1 = doubleTime();
    zncc2way(&Data);
    time2 = doubleTime();
    printf("zncc:                      %6.1lf ms.\n", (time2-time1)*1000);


    free(Data.displacements);
    free(Data.greyImage0);
    free(Data.greyImage1);


    time1 = doubleTime();
    unsigned char *ppo;
    ppo = postProcess(Data.dmap1, Data.dmap2, width/4, height/4, dispLimit);
    time2 = doubleTime();
    printf("Post-processing:           %6.1lf ms.\n", (time2-time1)*1000);


    free(Data.dmap1);
    free(Data.dmap2);

    pthread_mutex_destroy(&mutex1);

    total2 = doubleTime();
    printf("Total time:                %6.1lf ms.\n\n", (total2-total1)*1000);

    return ppo;
}
