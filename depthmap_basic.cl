/* Convert rgba-image to 1/4 dimensions greyscale float-image */
__kernel void blend4x4_cnvrtToGreyscale(__global uchar *data,
                                        uint width,
                                        __global float *converted) {
    size_t x, y;
    uint r, g, b, i, j;

    x = get_global_id(0);
    y = get_global_id(1);

    r = 0;
    g = 0;
    b = 0;
    for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++) {
            r += data[((y*4+j)*width+(x*4+i))*4+0];
            g += data[((y*4+j)*width+(x*4+i))*4+1];
            b += data[((y*4+j)*width+(x*4+i))*4+2];
        }
    }
    r = r >> 4;
    g = g >> 4;
    b = b >> 4;

    converted[y*(width/4)+x] = 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

/* Halve input resolution by blending 4 values together */
__kernel void blend2x2(__global float *input,
                       uint inputWidth,
                       __global float *output) {

    uint x, y, width;
    float val;

    x = get_global_id(0)*2;
    y = get_global_id(1)*2;

    val = input[y*inputWidth+x] + input[y*inputWidth+x+1]
        + input[(y+1)*inputWidth+x] + input[(y+1)*inputWidth+x];

    val *= 0.25f;

    output[get_global_id(1)*get_global_size(0)+get_global_id(0)] = val;
}

/* Cache mean-subtracted blocks */
__kernel void cacheBlkData(__global float *image,
                           __global float *cache,
                           uint width,
                           uint height,
                           uint bx,
                           uint by) {
    uint iterx, itery, iterxx, iteryy, index;
    int i, j, ibx, iby;
    float mean, subtracted, sq_dev;

    iterx = get_global_id(0);
    itery = get_global_id(1);

    iterxx = iterx + bx/2;
    iteryy = itery + by/2;
    ibx = bx;
    iby = by;

    /* Calculate block mean-value */
    mean = 0.0f;
    for (j=-iby/2; j <= iby/2; j++) {
        for (i=-ibx/2; i <= ibx/2; i++) {
            mean += image[(iteryy+j)*width+iterxx+i];
        }
    }
    mean /= bx*by;

    /* For the same block, now elements get subtracted by mean
     * and deviation is calculated. */
    index = (itery*(width-bx+1) + iterx)*bx*by;
    sq_dev = 0.0f;
    for (j=-iby/2; j <= iby/2; j++) {
        for (i=-ibx/2; i <= ibx/2; i++) {
            subtracted = image[(iteryy+j)*width+iterxx+i] - mean;
            cache[index] = subtracted;
            sq_dev += subtracted*subtracted;
            index++;
        }
    }
    /* reciprocal of deviations */
    cache[(width-bx+1)*(height-by+1)*bx*by + itery*(width-bx+1)+iterx] = 1.0f /sqrt(sq_dev);
}

__kernel void initDisparitys(__global ushort *displacements,
                             uint width,
                             uint bx,
                             uint dlimit) {
    uint x, y, dlim;
    int ibx;

    x = get_global_id(0);
    y = get_global_id(1);

    ibx = bx;

    displacements[y*width*2+x*2+0] = 0;

    dlim = dlimit;
    if (dlimit > x - ibx/2)
        dlim = x - ibx/2;

    displacements[y*width*2+x*2+1] = dlim;
}

/* Remove and use clEnqueueFillBuffer */
__kernel void initccor(__global float *ccor) {

    uint i;

    i = get_global_id(0);
    ccor[i] = -FLT_MAX;
}

/* Remove and use clEnqueueFillBuffer */
__kernel void zero_clMem(__global uchar *data) {

    uint x, y, w;

    x = get_global_id(0);
    y = get_global_id(1);
    w = get_global_size(0);

    data[y*w+x] = 0;
}

/* zero-mean normalized cross-correlation */
__kernel void zncc(__global float *cache_blk_l,
                   __global float *cache_blk_r,
                   __global ushort *displacements,
                   __global uchar *dmap1,
                   __global float *ccor,
                   uint width,
                   uint height,
                   uint bx,
                   uint by,
                   uint dlimit) {
    uint iterx, itery, i, blockStart, devBase;
    uchar d, dlim;
    float deviations_left, deviations_right, temp1, temp2, summed, val, max_val;

    iterx = get_global_id(0);
    itery = get_global_id(1);

    /* Condition check to enable testing different local item sizes that don't
     * divide evenly with image-width. */
    if (iterx < width-bx+1) {

        max_val = -FLT_MAX;

        d = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+0];
        dlim = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+1];

        /* 1st block index for current iteration */
        blockStart = (itery*(width-bx+1)+iterx)*bx*by;
        /* Base index for reciprocal deviations */
        devBase = (width-bx+1)*(height-by+1)*bx*by;

        deviations_left = cache_blk_l[devBase + itery*(width-bx+1)+iterx];
        for (d=d; d <= dlim; d++) {
            deviations_right = cache_blk_r[devBase + itery*(width-bx+1)+iterx-d];

            summed = 0.0f;
            for (i = blockStart; i < blockStart+bx*by; i++) {

                temp1 = cache_blk_r[i-d*bx*by];
                temp2 = cache_blk_l[i];
                summed += temp1*temp2;
            }
            val = summed * (deviations_left * deviations_right);

            if (val > max_val) {
                max_val = val;
                dmap1[(itery+by/2)*width+iterx+bx/2] = d;
            }

            ccor[((itery)*(width-bx+1)+iterx)*(dlimit+1)+d] = val;
        }
    }
}

/* zncc2way saves calculated cross correlation values certain way
 * that can be used to calculate depthmaps */
__kernel void constructDmap2(__global uchar *dmap2,
                             __global float *ccor,
                             uint dlim,
                             uint width,
                             uint bx,
                             uint by) {
    uint x, d, dlimit, scanline;
    int ibx;
    float val, val2;

    /* Scanline for dmap2 is scanline+by/2 */
    scanline = get_global_id(0);

    ibx = bx;

    /* Reconstruct right depthmap using saved correlation values */
    for (x = 0; x < width-bx+1; x++) {
        val = ccor[(scanline*(width-bx+1)+x)*(dlim+1)];
        dmap2[(scanline+by/2)*width+x+bx/2] = 0;

        dlimit = dlim;
        /* Do not go over left border */
        if ((dlim+x) > (width-bx))
            dlimit = width-bx-x;

        for(d=1; d <= dlimit; d++) {
            val2 = ccor[(scanline*(width-bx+1)+x)*(dlim+1)+d*(dlim+1)+d];
            if (val2 > val) {
                val = val2;
                dmap2[(scanline+by/2)*width+x+bx/2] = d;
            }
        }

        /* This commented code reconstructs left depthmap */
//        /* Do not go over right border */
//        if (dlim > x)
//            dlimit -= dlim-x;

//        for(d=1; d <= dlimit; d++) {
//            val2 = ccor[(scanline*(width-bx+1)+x)*(dlim+1)+d];
//            if (val2 > val) {
//                val = val2;
//                dmap2[(scanline+by/2)*width+x+bx/2] = d;
//            }
//        }
    }
}

/* Creates crosschecked dephmap from 2 dephtmaps */
__kernel void postCrossCorrelation(__global uchar *dMap1,
                                   __global uchar *dMap2,
                                   __global uchar *result,
                                   uint dispLimit) {
    uint width, x, y;
    int pixel_l, pixel_r, diff;

    x = get_global_id(0);
    y = get_global_id(1);
    width = get_global_size(0);

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

/* Fills black pixel with using up to 4 neighbouring values,
 * if they are nonzero. */
__kernel void postFill(__global uchar *input,
                       __global uchar *fill,
                       uint width) {
    ushort val, val1, val2, val3, val4;
    uint i, x, y;

    /* Edge pixels would result in overread */
    x = get_global_id(0)+1;
    y = get_global_id(1)+1;


    if (input[y*width+x] == 0) {
        val1 = input[(y-1)*width+x];
        val2 = input[y*width+x-1];
        val3 = input[y*width+x+1];
        val4 = input[(y+1)*width+x];
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
        fill[y*width+x] = input[y*width+x];
    }

}

/* Half-resolution depthmaps, fullsize disparity-buffer,
 * fullsize width, height and disp_limit */
__kernel void disparityLimits_2x2(__global uchar *dmap1,
                                  __global uchar *dmap2,
                                  __global ushort *newLimits,
                                  uint width,
                                  uint height,
                                  uint disp_limit,
                                  uint bx,
                                  uint by) {

    int x, y, lasty, i, bxSide, bySide, halfWidth, halfy, halfx, min, max;
    int val, val2, val3;

    bxSide = bx/2;
    bySide = by/2;
    halfWidth = width/2;

    x = (get_global_id(0)+bxSide);
    y = (get_global_id(1)+bySide);

    /* This codeblock trys to figure out decent range for displacement.
     * Copied from c-implementation. */
    halfx = x/2;
    halfy = y/2;
    val = dmap1[halfy*halfWidth+halfx];

    /* If halfsize depthmap has value 0, set range to maximum.
     * Probably happens only in edges. */
    /* Note: subtraction is undone by next line of code */
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
    val2 = dmap1[halfy*halfWidth+halfx-2];
    val3 = dmap1[halfy*halfWidth+halfx+2];
    if (val2 > max) max = val2;
    if (val3 > max) max = val3;
    if (val2 < min) min = val2;
    if (val3 < min) min = val3;

    /* Compare against second depthmaps values and update min and max. */
    for (i=0; i <= val; i++) {

        val2 = dmap2[halfy*halfWidth+halfx-i];
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
