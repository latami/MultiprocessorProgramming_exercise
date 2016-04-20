/* Convert rgba-image to 1/4 dimensions greyscale float-image */
__kernel void blend4x4_cnvrtToGreyscale(__global uchar *data,
                                        uint width,
                                        __global float *converted) {
    size_t x, y;
    uint r, g, b, i, j;

    x = get_global_id(0);
    y = get_global_id(1);

    if (x < width/4) {
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
}

/* Cache mean-subtracted blocks */
__kernel void cacheBlkData(__global float *image,
                           __global float *cache,
                           uint width,
                           uint height,
                           uint bx,
                           uint by) {
    uint iterx, itery, iterxx, iteryy, index, blkStride, interleave, lineStride;
    int i, j, ibx, iby;
    float mean, subtracted, sq_dev;
    uint iteryOffset;

    iterx = get_global_id(0);
    itery = get_global_id(1);
    iteryOffset = get_global_offset(1);

    blkStride = (64*bx*by+63)/64 * 64; // stride for 64 interleaved blocks
    /* One extra blockStride for block padding at the end of the scanline */
    lineStride = ((width-bx+1+63)/64) * blkStride + blkStride;


    if (iterx < width-bx+1) {

        /* pixel iterators */
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
        interleave = get_local_id(0); /* interleave blocks */
        index = ((itery-iteryOffset)*lineStride + (iterx/64)*blkStride) + interleave; // align to 256 bytes
        sq_dev = 0.0f;
        for (j=-iby/2; j <= iby/2; j++) {
            for (i=-ibx/2; i <= ibx/2; i++) {
                subtracted = image[(iteryy+j)*width+iterxx+i] - mean;
                cache[index] = subtracted;
                sq_dev += subtracted*subtracted;
                index += 64;
            }
        }
        /* reciprocal of deviations */
        cache[(get_global_size(1))*lineStride + (itery-iteryOffset)*(width-bx+1)+iterx] = 1.0f /sqrt(sq_dev);
    }
}

__kernel void initDisparitys(__global ushort *displacements,
                             uint width,
                             uint bx,
                             uint dlimit) {
    uint x, y, dlim;

    x = get_global_id(0);
    y = get_global_id(1);

    displacements[y*width*2+x*2+0] = 0;

    dlim = dlimit;

    if (dlimit + x >= width-bx/2)
        dlim = width-bx/2-1 - x;

    displacements[y*width*2+x*2+1] = dlim;
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
__kernel void zncc_vector(__global float *cache_blk_l,
                          __global float *cache_blk_r,
                          __global ushort *displacements,
                          __global uchar *dmap2,
                          __global float *ccor,
                          uint width,
                          uint height,
                          uint bx,
                          uint by,
                          uint dlimit) {
    uint iterx, itery, iteryOffset, i, blockStart, devBase, blkStride, lineStride;
    uchar d, dlim;
    int interleave, jump, jump2;

    iterx = get_global_id(0)*8;
    itery = get_global_id(1);
    iteryOffset = get_global_offset(1);

    /* Condition check to enable testing different local item sizes that don't
     * divide evenly with image-width. */
    if (iterx < width-bx+1) {

        /* Does not control d-iterations exactly because of the way kernel works */
        d = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+0];
        dlim = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+1];

        /* 1st block index for current iteration */
        blkStride = (64*bx*by+63)/64 * 64;
        interleave = get_local_id(0);
        /* last blkStride is for padding */
        lineStride = ((width-bx+1+63)/64) * blkStride + blkStride;
        blockStart = ((itery-iteryOffset)*lineStride + (iterx/64)*blkStride) + interleave*8;
        /* Base index for reciprocal deviations */
       devBase = (get_global_size(1))*lineStride;

        float8 blk8_r;
        float8 blk8_l1, blk8_l2, temp8;
        float8 summed8[8], deviations8_left[8], deviations8_right;

        deviations8_right = vload8(0, &cache_blk_r[devBase + (itery-iteryOffset)*(width-bx+1)+iterx]);
        for (d=d; d <= dlim; d+=8) {
            deviations8_left[0] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d]);
            deviations8_left[1] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+1]);
            deviations8_left[2] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+2]);
            deviations8_left[3] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+3]);
            deviations8_left[4] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+4]);
            deviations8_left[5] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+5]);
            deviations8_left[6] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+6]);
            deviations8_left[7] = vload8(0, &cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d+7]);

            /* Jump between block-groups */
            jump = (d+interleave*8)/64;
            /* Second load needs separate jump */
            jump2 = (d+interleave*8+8)/64;
            /* When accessing next blockgroup, start from last block in a group */
            jump = jump * (blkStride - 64);
            jump2 = jump2 * (blkStride - 64);

            summed8[0] = (float8)(0.0f);
            summed8[1] = (float8)(0.0f);
            summed8[2] = (float8)(0.0f);
            summed8[3] = (float8)(0.0f);
            summed8[4] = (float8)(0.0f);
            summed8[5] = (float8)(0.0f);
            summed8[6] = (float8)(0.0f);
            summed8[7] = (float8)(0.0f);
            for (i = blockStart; i < blockStart+bx*by*64; i += 64) {
                blk8_r = vload8(0, &cache_blk_r[i]);
                blk8_l1 = vload8(0, &cache_blk_l[i+d+jump]);
                blk8_l2 = vload8(0, &cache_blk_l[i+d+jump2+8]);

                summed8[0] += blk8_r*blk8_l1;

                temp8 = blk8_l1.s12345670;  // shift 1 index forward
                temp8.s7 = blk8_l2.s0;
                summed8[1] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s1;
                summed8[2] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s2;
                summed8[3] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s3;
                summed8[4] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s4;
                summed8[5] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s5;
                summed8[6] += blk8_r*temp8;

                temp8 = temp8.s12345670;
                temp8.s7 = blk8_l2.s6;
                summed8[7] += blk8_r*temp8;
            }

            for (i=0; i < 8; i++) {

                temp8 = summed8[i] * (deviations8_right * deviations8_left[i]);

                uint pos = ((itery-iteryOffset)*(width-bx+1)+iterx)*(((dlimit+1)+7)/8)*8+d+i;
                ccor[pos] = temp8.s0;
                ccor[pos+(((dlimit+1)+7)/8)*8*1] = temp8.s1;
                ccor[pos+(((dlimit+1)+7)/8)*8*2] = temp8.s2;
                ccor[pos+(((dlimit+1)+7)/8)*8*3] = temp8.s3;
                ccor[pos+(((dlimit+1)+7)/8)*8*4] = temp8.s4;
                ccor[pos+(((dlimit+1)+7)/8)*8*5] = temp8.s5;
                ccor[pos+(((dlimit+1)+7)/8)*8*6] = temp8.s6;
                ccor[pos+(((dlimit+1)+7)/8)*8*7] = temp8.s7;
            }
        }

    }
}

/* zero-mean normalized cross-correlation */
__kernel void zncc_scalar(__global float *cache_blk_l,
                          __global float *cache_blk_r,
                          __global ushort *displacements,
                          __global uchar *dmap2,
                          __global float *ccor,
                          uint width,
                          uint height,
                          uint bx,
                          uint by,
                          uint dlimit) {
    uint iterx, itery, iteryOffset, i, blockStart, devBase, blkStride, lineStride;
    uchar d, dlim;
    float deviations_left, deviations_right, temp1, temp2, summed, val;
    int interleave, jump;

    iterx = get_global_id(0);
    itery = get_global_id(1);
    iteryOffset = get_global_offset(1);

    /* Condition check to enable testing different local item sizes that don't
     * divide evenly with image-width. */
    if (iterx < width-bx+1) {

        d = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+0];
        dlim = displacements[(itery+by/2)*width*2+(iterx+bx/2)*2+1];

        /* 1st block index for current iteration */
        blkStride = (64*bx*by+63)/64 * 64;
        interleave = get_local_id(0);
        lineStride = ((width-bx+1+63)/64) * blkStride + blkStride;
        blockStart = ((itery-iteryOffset)*lineStride + (iterx/64)*blkStride) + interleave;
        /* Base index for reciprocal deviations */
        devBase = (get_global_size(1))*lineStride;

        deviations_right = cache_blk_r[devBase + (itery-iteryOffset)*(width-bx+1)+iterx];
        for (d=d; d <= dlim; d++) {
            deviations_left = cache_blk_l[devBase + (itery-iteryOffset)*(width-bx+1)+iterx+d];

            /* Jump between block-groups */
            jump = (d+interleave)/64;
            /* When accessing next blockgroup, start from first block in a group */
            if (jump > 0)
                jump = jump * blkStride - jump * 64;

            summed = 0.0f;
            for (i = blockStart; i < blockStart+bx*by*64; i += 64) {
                temp1 = cache_blk_r[i];
                temp2 = cache_blk_l[i+d+jump];
                summed += temp1*temp2;
            }
            val = summed * (deviations_left * deviations_right);

            ccor[((itery-iteryOffset)*(width-bx+1)+iterx)*((dlimit+1+7)/8)*8+d] = val;
        }
    }
}

/* zncc2way saves calculated cross correlation values certain way
 * that can be used to calculate depthmaps */
__kernel void constructDmaps(__global uchar *dmap1,
                             __global uchar *dmap2,
                             __global float *ccor,
                             uint dlim,
                             uint width,
                             uint bx,
                             uint by) {
    uint x, d, dlimit, scanline, scLineOffset, dlimBlock;
    float val, val2;

    /* Scanline for dmap2 is scanline+by/2 */
    x = get_global_id(0);
    scanline = get_global_id(1);
    scLineOffset = get_global_offset(1);

    dlimBlock = (((dlim+1)+7)/8)*8;

    if (x < width) {
        dmap1[(scanline+by/2)*width+x] = 0;
        dmap2[(scanline+by/2)*width+x] = 0;

        if ((x >= bx/2) && (x < width-bx/2)) {
            val = ccor[((scanline-scLineOffset)*(width-bx+1)+x-bx/2)*(dlimBlock)];
            dlimit = dlim;

            /* Constructs dmap2 */
            if (dlim+x >= width-bx/2)
                dlimit = width-x-bx/2-1;

            for(d=1; d <= dlimit; d++) {
                val2 = ccor[((scanline-scLineOffset)*(width-bx+1)+x-bx/2)*(dlimBlock)+d];
                if (val2 > val) {
                    val = val2;
                    dmap2[(scanline+by/2)*width+x] = d;
                }
            }

            val = ccor[((scanline-scLineOffset)*(width-bx+1)+x-bx/2)*(dlimBlock)];
            dlimit = dlim;

            /* Constructs dmap1 */
            if (dlim+bx/2 > x)
                dlimit = x-bx/2;

            for(d=1; d <= dlimit; d++) {
                val2 = ccor[((scanline-scLineOffset)*(width-bx+1)+x-bx/2)*(dlimBlock)-d*(dlimBlock)+d];
                if (val2 > val) {
                    val = val2;
                    dmap1[(scanline+by/2)*width+x] = d;
                }
            }
        }
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
