#include <stdio.h>
#include <float.h>
//#include <CL/cl.h>

#include "depthmap_opencl.h"
#include "common_opencl.h"
#include "doubleTime.h"

/* OpenCL 1.1 doesn't have clEnqueueFillBuffer */
/* Fills buffer with width*height zero bytes */
int zeroMem_kernel(cl_program program, cl_command_queue queue,
                   cl_mem data, cl_uint width, cl_uint height) {
    cl_kernel kernZero;
    size_t global[2];
    cl_int err;

    kernZero = clCreateKernel(program, "zero_clMem", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a kernel! Code: %d"
                        "(file %s line %d)\n", err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    global[0] = width;
    global[1] = height;
    clSetKernelArg(kernZero, 0, sizeof(cl_mem), &data);
    err = clEnqueueNDRangeKernel(queue, kernZero, 2, NULL, global, NULL,
                                 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't enqueue the kernel. Code %d. %s line %d\n",
                err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    clReleaseKernel(kernZero);

    return EXIT_SUCCESS;
}

/* Converts rgba-image to greyscale float-image  with 1/4 resolution on both axis */
int blend4x4CnvrtToGrey(cl_program program, cl_command_queue queue,
                        cl_mem *input_img0, cl_mem *input_img1,
                        cl_mem *greyImage0, cl_mem *greyImage1,
                        cl_uint width, cl_uint height) {
    cl_event event[2];
    cl_int err, err2;
    size_t global[2];
    cl_kernel blendAndGreyscale;

    if ((width % 4 != 0) || (height % 4 != 0)) {
        fprintf(stderr, "blend4x4 does not currently handle resolutions not "
                        "divisible by 4!\n");
        return EXIT_FAILURE;
    }

    blendAndGreyscale = clCreateKernel(program, "blend4x4_cnvrtToGreyscale", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a kernel! Code: %d"
                        "(file %s line %d)\n", err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    global[0] = width/4;
    global[1] = height/4;
    clSetKernelArg(blendAndGreyscale, 0, sizeof(cl_mem), input_img0);
    clSetKernelArg(blendAndGreyscale, 1, sizeof(cl_uint), &width);
    clSetKernelArg(blendAndGreyscale, 2, sizeof(cl_mem), greyImage0);
    err = clEnqueueNDRangeKernel(queue, blendAndGreyscale, 2, NULL, global, NULL,
                                 0, NULL, &event[0]);

    clSetKernelArg(blendAndGreyscale, 0, sizeof(cl_mem), input_img1);
    clSetKernelArg(blendAndGreyscale, 1, sizeof(cl_uint), &width);
    clSetKernelArg(blendAndGreyscale, 2, sizeof(cl_mem), greyImage1);
    err2 = clEnqueueNDRangeKernel(queue, blendAndGreyscale, 2, NULL, global, NULL,
                                 0, NULL, &event[1]);
    if (err != CL_SUCCESS || err2 != CL_SUCCESS) {
        fprintf(stderr, "Couldn't enqueue the kernel. Code %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(2, event);

    float time;
    time = eventRuntime(event[0]);
    time += eventRuntime(event[1]);
    printf("Blend4x4 and greyscaling:       %6.1f ms.\n", time);

    /* Make sure queue is empty before attempting to release resources. */
    clFinish(queue); // clWaitForEvents makes this redundant

    clReleaseMemObject((*input_img0));
    clReleaseMemObject((*input_img1));
    clReleaseKernel(blendAndGreyscale);

    return EXIT_SUCCESS;
}

/* reduce float-image dimensions by half */
int blend2x2(cl_context context, cl_program program, cl_command_queue queue,
             cl_mem *greyImage0, cl_mem *greyImage1,
             cl_mem *halfImage0, cl_mem *halfImage1,
             cl_uint width, cl_uint height) {
    cl_kernel blend;
    size_t global[2], imgSize;
    cl_event event[2];
    cl_int err, err2;

    blend = clCreateKernel(program, "blend2x2", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a kernel! Code: %d"
                        "(file %s line %d)\n", err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    /* Allocate space for half-images */
    imgSize = (width/2)*(height/2)*sizeof(cl_float);
    (*halfImage0) = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize, NULL, &err);
    (*halfImage1) = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize, NULL, &err2);
    if (err != CL_SUCCESS || err2 != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a buffer. %s line %d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    global[0] = width/2;
    global[1] = height/2;
    clSetKernelArg(blend, 0, sizeof(cl_mem), greyImage0);
    clSetKernelArg(blend, 1, sizeof(cl_uint), &width);
    clSetKernelArg(blend, 2, sizeof(cl_mem), halfImage0);
    err = clEnqueueNDRangeKernel(queue, blend, 2, NULL, global, NULL,
                                 0, NULL, &event[0]);

    clSetKernelArg(blend, 0, sizeof(cl_mem), greyImage1);
    clSetKernelArg(blend, 1, sizeof(cl_uint), &width);
    clSetKernelArg(blend, 2, sizeof(cl_mem), halfImage1);
    err2 = clEnqueueNDRangeKernel(queue, blend, 2, NULL, global, NULL,
                                 0, NULL, &event[1]);
    if (err != CL_SUCCESS || err2 != CL_SUCCESS) {
        fprintf(stderr, "Couldn't enqueue the kernel. Code %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(2, event);

    float time;
    time = eventRuntime(event[0]);
    time += eventRuntime(event[1]);
    printf("Blend2x2:                       %6.1f ms.\n", time);

    clReleaseKernel(blend);

    return EXIT_SUCCESS;
}

/* Input is required to be fullsize width and height, halfsize dmaps,
 * function allocates newDisparitys */
int estimateDisparitys_2x2(cl_context context, cl_program program, cl_command_queue queue,
                           cl_mem dmap1, cl_mem dmap2, cl_mem *newDisparitys,
                           cl_uint width, cl_uint height, cl_uint disp_limit,
                           cl_uint bx, cl_uint by) {
    cl_kernel limits;
    size_t global[2], size;
    cl_event event;
    cl_int err;

    limits = clCreateKernel(program, "disparityLimits_2x2", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a kernel! Code: %d"
                        "(file %s line %d)\n", err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    /* Allocate space for disparitys */
    size = width*height*2*sizeof(cl_ushort);
    (*newDisparitys) = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a buffer. %s line %d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    /* Abuse kernel to clear whole ushort area with uchar kernel */
    zeroMem_kernel(program, queue, (*newDisparitys), width*2*sizeof(ushort), height);

    global[0] = width-bx+1;
    global[1] = height-by+1;
    clSetKernelArg(limits, 0, sizeof(cl_mem), &dmap1);
    clSetKernelArg(limits, 1, sizeof(cl_mem), &dmap2);
    clSetKernelArg(limits, 2, sizeof(cl_mem), newDisparitys);
    clSetKernelArg(limits, 3, sizeof(cl_uint), &width);
    clSetKernelArg(limits, 4, sizeof(cl_uint), &height);
    clSetKernelArg(limits, 5, sizeof(cl_uint), &disp_limit);
    clSetKernelArg(limits, 6, sizeof(cl_uint), &bx);
    clSetKernelArg(limits, 7, sizeof(cl_uint), &by);
    err = clEnqueueNDRangeKernel(queue, limits, 2, NULL, global, NULL,
                                 0, NULL, &event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't enqueue the kernel. Code %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event);

    float time;
    time = eventRuntime(event);
    printf("Disparity limits:               %6.1f ms.\n", time);

    clReleaseKernel(limits);

    return EXIT_SUCCESS;
}

/* Fill disparitys with range 0-disp_limit, except for left edge,
 * in where disp_limit is scaled in order to avoid overread. */
int initDisparitys(cl_context context, cl_program program, cl_command_queue queue,
                   cl_uint width, cl_uint height, cl_uint bx, cl_uint disp_limit,
                   cl_mem *disparitys) {
    cl_kernel initDisparitys;
    cl_int err;
    size_t global[2], bufSize;
    cl_event event;

    initDisparitys = clCreateKernel(program, "initDisparitys", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a kernel! Code: %d"
                        " %s line %d\n", err, __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    bufSize = width*height*2*sizeof(cl_ushort);
    (*disparitys) = clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a buffer %s line %d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    /* Kernel clears uchar elements, so abuse it a bit */
    zeroMem_kernel(program, queue, (*disparitys), width*2*sizeof(cl_ushort), height);

    global[0] = width;
    global[1] = height;
    clSetKernelArg(initDisparitys, 0, sizeof(cl_mem), disparitys);
    clSetKernelArg(initDisparitys, 1, sizeof(cl_uint), &width);
    clSetKernelArg(initDisparitys, 2, sizeof(cl_uint), &bx);
    clSetKernelArg(initDisparitys, 3, sizeof(cl_uint), &disp_limit);
    err = clEnqueueNDRangeKernel(queue, initDisparitys, 2, NULL, global, NULL,
                                 0, NULL, &event);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue initDisparitys. Code %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event);
    float time = eventRuntime(event);

    printf("Init disparitys                 %6.1f ms.\n", time);

    clReleaseKernel(initDisparitys);

    return EXIT_SUCCESS;
}

/* Calculate zero-mean cross-correlations and constructs 2 depthmaps.
 * Allocates buffers for dmaps. */
int znccFunc(cl_context context, cl_program program, cl_command_queue queue,
             cl_mem img0, cl_mem img1, cl_mem disparitys, cl_uint disp_limit,
             cl_uint width, cl_uint height,
             cl_uint bx, cl_uint by,
             cl_mem *dmap1, cl_mem *dmap2) {

    cl_event event[2];
    cl_mem cacheBlks_l, cacheBlks_r, ccor;
    size_t cacheSize, size, ccSize, global[2], local[2];
    cl_int err, errs[5], i;
    cl_kernel cacheBlkData, initCcors, zncc, constructDmap2;
    float time1, time2, time3, time4;

    cacheBlkData =   clCreateKernel(program, "cacheBlkData", &errs[0]);
    initCcors =	     clCreateKernel(program, "initccor", &errs[1]);
    zncc =	         clCreateKernel(program, "zncc", &errs[2]);
    constructDmap2 = clCreateKernel(program, "constructDmap2", &errs[3]);

    for(i=0; i < 4; i++) {
        if (errs[i] != CL_SUCCESS) {
            fprintf(stderr, "Couldn't create a kernel number %d! Code: %d"
                            " %s line %d\n", i, errs[0], __FILE__, __LINE__);
            return EXIT_FAILURE;
        }
    }

    cacheSize = (bx*by*(width-bx+1)*(height-by+1)
                 +(width-bx+1)*(height-by+1))*sizeof(float);
    size = width*height*sizeof(cl_uchar);
    ccSize = (width)*(height)*(disp_limit+1)*sizeof(cl_float);
    /* Allocate block caches */
    cacheBlks_l = clCreateBuffer(context, CL_MEM_READ_WRITE, cacheSize, NULL, &errs[0]);
    cacheBlks_r = clCreateBuffer(context, CL_MEM_READ_WRITE, cacheSize, NULL, &errs[1]);
    (*dmap1) = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &errs[2]);
    (*dmap2) = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &errs[3]);
    ccor = clCreateBuffer(context, CL_MEM_READ_WRITE, ccSize, NULL, &errs[4]);

    /* Check success of creating buffers */
    for (i=0; i < 5; i++) {
        if (errs[i] != CL_SUCCESS) {
            fprintf(stderr, "Couldn't create a buffer %d %s line %d\n"
                    , i, __FILE__, __LINE__);
            return EXIT_FAILURE;
        }
    }
    zeroMem_kernel(program, queue, (*dmap1), width, height);
    zeroMem_kernel(program, queue, (*dmap2), width, height);
    clFinish(queue);

    global[0] = width-bx+1;
    global[1] = height-by+1;
    clSetKernelArg(cacheBlkData, 0, sizeof(cl_mem), &img0);
    clSetKernelArg(cacheBlkData, 1, sizeof(cl_mem), &cacheBlks_l);
    clSetKernelArg(cacheBlkData, 2, sizeof(cl_uint), &width);
    clSetKernelArg(cacheBlkData, 3, sizeof(cl_uint), &height);
    clSetKernelArg(cacheBlkData, 4, sizeof(cl_uint), &bx);
    clSetKernelArg(cacheBlkData, 5, sizeof(cl_uint), &by);
    errs[0] = clEnqueueNDRangeKernel(queue, cacheBlkData, 2, NULL, global, NULL,
                                     0, NULL, &event[0]);

    clSetKernelArg(cacheBlkData, 0, sizeof(cl_mem), &img1);
    clSetKernelArg(cacheBlkData, 1, sizeof(cl_mem), &cacheBlks_r);
    clSetKernelArg(cacheBlkData, 2, sizeof(cl_uint), &width);
    clSetKernelArg(cacheBlkData, 3, sizeof(cl_uint), &height);
    clSetKernelArg(cacheBlkData, 4, sizeof(cl_uint), &bx);
    clSetKernelArg(cacheBlkData, 5, sizeof(cl_uint), &by);
    errs[1] = clEnqueueNDRangeKernel(queue, cacheBlkData, 2, NULL, global, NULL,
                                     0, NULL, &event[1]);
    if (errs[0] < 0 || errs[1] < 0) {
        fprintf(stderr, "Couldn't enqueue cacheBlkData! Codes %d %d",
                errs[0], errs[1]);
        return EXIT_FAILURE;
    }
    clWaitForEvents(2, event);
    time1 = eventRuntime(event[0]);
    time1 += eventRuntime(event[1]);


    /* Fill buffer with -FLT_MAX */
    global[0] = (width)*(height)*(disp_limit+1);
    clSetKernelArg(initCcors, 0, sizeof(cl_mem), &ccor);
    err = clEnqueueNDRangeKernel(queue, initCcors, 1, NULL, global, NULL,
                                 0, NULL, &event[0]);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue initCcors: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event[0]);
    time2 = eventRuntime(event[0]);

    /* clEnqueueFillBuffer requires OpenCL 1.2 */
//	float pattern = -FLT_MAX;
//	size_t patternSize = (width/4)*(height/4)*(disp_limit+1);
//	clEnqueueFillBuffer(queue, ccor, &pattern, sizeof(float), 0, patternSize, 0,
//						NULL, NULL);


    global[0] = width-bx+1;
    global[1] = height-by+1;
    local[0] = 1;
    local[1] = 1;

    clSetKernelArg(zncc, 0, sizeof(cl_mem), &cacheBlks_l);
    clSetKernelArg(zncc, 1, sizeof(cl_mem), &cacheBlks_r);
    clSetKernelArg(zncc, 2, sizeof(cl_mem), &disparitys);
    clSetKernelArg(zncc, 3, sizeof(cl_mem), dmap1);
    clSetKernelArg(zncc, 4, sizeof(cl_mem), &ccor);
    clSetKernelArg(zncc, 5, sizeof(cl_uint), &width);
    clSetKernelArg(zncc, 6, sizeof(cl_uint), &height);
    clSetKernelArg(zncc, 7, sizeof(cl_uint), &bx);
    clSetKernelArg(zncc, 8, sizeof(cl_uint), &by);
    clSetKernelArg(zncc, 9, sizeof(cl_uint), &disp_limit);
    err = clEnqueueNDRangeKernel(queue, zncc, 2, NULL, global, local,
                                 0, NULL, &event[0]);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue zncc: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, event);
    time3 = eventRuntime(event[0]);

    /* Caches not needed anymore */
    clReleaseMemObject(cacheBlks_l);
    clReleaseMemObject(cacheBlks_r);


    global[0] = height-by+1;
    clSetKernelArg(constructDmap2, 0, sizeof(cl_mem), dmap2);
    clSetKernelArg(constructDmap2, 1, sizeof(cl_mem), &ccor);
    clSetKernelArg(constructDmap2, 2, sizeof(cl_uint), &disp_limit);
    clSetKernelArg(constructDmap2, 3, sizeof(cl_uint), &width);
    clSetKernelArg(constructDmap2, 4, sizeof(cl_uint), &bx);
    clSetKernelArg(constructDmap2, 5, sizeof(cl_uint), &by);
    err = clEnqueueNDRangeKernel(queue, constructDmap2, 1, NULL, global, NULL,
                                 0, NULL, &event[0]);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue constructDmap2: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, event);
    time4 = eventRuntime(event[0]);

    clReleaseMemObject(ccor);

    printf("zncc:                           %6.1f ms.\n", time1+time2+time3+time4);
    printf(" cacheData:                     %6.1f ms.\n", time1);
    printf(" init cross-correlation buffer: %6.1f ms.\n", time2);
    printf(" zncc:                          %6.1f ms.\n", time3);
    printf(" construct dmap2:               %6.1f ms.\n", time4);

    clReleaseKernel(cacheBlkData);
    clReleaseKernel(initCcors);
    clReleaseKernel(zncc);
    clReleaseKernel(constructDmap2);

    return EXIT_SUCCESS;
}

/* Calls kernels to postprocess depthmaps */
int postProcessDmaps(cl_context context, cl_command_queue queue, cl_program program,
                     cl_mem dmap1, cl_mem dmap2,
                     cl_uint width, cl_uint height, cl_uint disp_limit,
                     cl_mem *processed) {

    cl_kernel postCross, postFill;
    cl_mem postpMem1, postpMem2;
    size_t global[2], size;
    cl_int errs[2], err, i;
    cl_event event;
    float time;

    size = width*height*sizeof(cl_uchar);
    postpMem1 = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &errs[0]);
    postpMem2 = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &errs[1]);
    for (i=0; i < 2; i++) {
        if (errs[i] != CL_SUCCESS) {
            fprintf(stderr, "Couldn't create a buffer %d in file %s line %d\n"
                    , i, __FILE__, __LINE__);
            return EXIT_FAILURE;
        }
    }
    zeroMem_kernel(program, queue, postpMem1, width, height);
    zeroMem_kernel(program, queue, postpMem2, width, height);
    clFinish(queue);

    postCross = clCreateKernel(program, "postCrossCorrelation", &errs[0]);
    postFill = clCreateKernel(program, "postFill", &errs[1]);
    for (i=0; i < 2; i++) {
        if (errs[i] != CL_SUCCESS) {
            fprintf(stderr, "Couldn't create a kernel %d: %s line %d\n"
                    , i, __FILE__, __LINE__);
            return EXIT_FAILURE;
        }
    }

    global[0] = width;
    global[1] = height;
    clSetKernelArg(postCross, 0, sizeof(cl_mem), &dmap1);
    clSetKernelArg(postCross, 1, sizeof(cl_mem), &dmap2);
    clSetKernelArg(postCross, 2, sizeof(cl_mem), &postpMem1);
    clSetKernelArg(postCross, 3, sizeof(cl_uint), &disp_limit);
    err = clEnqueueNDRangeKernel(queue, postCross, 2, NULL, global, NULL,
                                 0, NULL, &event);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue postCrossCorr: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event);
    time = eventRuntime(event);

    /* Left 1-pixel on all sides to avoid overread.
     * NOTE: Atleast with opensource Clover OpenCL 1.1-driver
     * global offsets seems not to work. */
    global[0] = width-2;
    global[1] = height-2;
    clSetKernelArg(postFill, 0, sizeof(cl_mem), &postpMem1);
    clSetKernelArg(postFill, 1, sizeof(cl_mem), &postpMem2);
    clSetKernelArg(postFill, 2, sizeof(cl_uint), &width);
    err = clEnqueueNDRangeKernel(queue, postFill, 2, NULL, global, NULL,
                                 0, NULL, &event);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue postFill: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event);
    time += eventRuntime(event);

    clSetKernelArg(postFill, 0, sizeof(cl_mem), &postpMem2);
    clSetKernelArg(postFill, 1, sizeof(cl_mem), &postpMem1);
    err = clEnqueueNDRangeKernel(queue, postFill, 2, NULL, global, NULL,
                                 0, NULL, &event);
    if (err < 0) {
        fprintf(stderr, "Couldn't enqueue postFill: code: %d\n", err);
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event);
    time += eventRuntime(event);

    clReleaseMemObject(postpMem2);

    (*processed) = postpMem1;

    printf("Post processing:                %6.1f ms.\n", time);

    clReleaseKernel(postCross);
    clReleaseKernel(postFill);

    return EXIT_SUCCESS;
}

unsigned char *generateDepthmap_opencl_basic(unsigned char *img0, unsigned char *img1,
                                             unsigned int width, unsigned height,
                                             unsigned int blockx, unsigned int blocky,
                                             unsigned int disp_limit, searchMethod_ocl select,
                                             device_ocl dev) {

    cl_platform_id platform;
    cl_device_type device_type;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue queue;
    cl_int err, errs[16], i;
    double hostTime1, hostTime2;

    if ( (blockx % 2 != 1) || (blocky % 2 != 1) || blockx == 1 || blocky == 1 ) {
        fprintf(stderr, "Blocksize must be odd in both dimensions and more than 1!\n");
        return NULL;
    }

    if (dev == CPU) device_type = CL_DEVICE_TYPE_CPU;
    else if (dev == GPU) device_type = CL_DEVICE_TYPE_GPU;
    else {
        fprintf(stderr, "Wrong opencl device\n");
        return NULL;
    }

    if (initOpenCL(&platform, &device, &context, &program, device_type)
            == EXIT_FAILURE) {
        return NULL;
    }

    hostTime1 = doubleTime();

    cl_mem input_img0, input_img1, greyImage0, greyImage1, disparitys;

    input_img0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                width*height*4*sizeof(unsigned char), img0, &errs[0]);
    greyImage0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                (width/4)*(height/4)*sizeof(float), NULL, &errs[1]);
    input_img1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                width*height*4*sizeof(unsigned char), img1, &errs[2]);
    greyImage1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                (width/4)*(height/4)*sizeof(float), NULL, &errs[3]);

    /* Check success of creating buffers */
    for (i=0; i < 4; i++) {
        if (errs[i] != CL_SUCCESS) {
            fprintf(stderr, "Couldn't create a buffer %d in file %s line %d\n"
                    , i, __FILE__, __LINE__);
            return NULL;
        }
    }

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't create a command queue\n");
        return NULL;
    }

    if (blend4x4CnvrtToGrey(program, queue, &input_img0, &input_img1,
                            &greyImage0, &greyImage1, width, height) == EXIT_FAILURE) {
        fprintf(stderr, "Image converting kernel failed!\n");
        return NULL;
    }

    cl_uint greyImgWidth = width/4;
    cl_uint greyImgHeight = height/4;
    cl_mem dmap1, dmap2;

    if (select == HIERARCHIC_CL) {
        /* Estimate search ranges from half-resolution depthmaps */
        cl_uint widthH, heightH;
        cl_mem hImg0, hImg1;

        widthH = greyImgWidth/2;
        heightH = greyImgHeight/2;
        if (initDisparitys(context, program, queue,
                           widthH, heightH, blockx, disp_limit/2,
                           &disparitys) == EXIT_FAILURE) {
            return NULL;
        }
        if (blend2x2(context, program, queue, &greyImage0, &greyImage1,
                     &hImg0, &hImg1, greyImgWidth, greyImgHeight) == EXIT_FAILURE) {
            return NULL;
        }
        if (znccFunc(context, program, queue,
                     hImg0, hImg1, disparitys, disp_limit,
                     widthH, heightH, blockx, blocky,
                     &dmap1, &dmap2) == EXIT_FAILURE) {
            return NULL;
        }
        /* Release memory because next function allocates new memory for the variable */
        clReleaseMemObject(disparitys);
        if (estimateDisparitys_2x2(context, program, queue, dmap1, dmap2,
                                   &disparitys, greyImgWidth, greyImgHeight,
                                   disp_limit, blockx, blocky) == EXIT_FAILURE) {
            return NULL;
        }
        /* Not needed anymore */
        clReleaseMemObject(dmap1);
        clReleaseMemObject(dmap2);
    }
    else {
        /* If brute is selected, don't do fancy things to limit search ranges. */
        if (initDisparitys(context, program, queue,
                           greyImgWidth, greyImgHeight, blockx, disp_limit,
                           &disparitys) == EXIT_FAILURE) {
            return NULL;
        }
    }
    cl_mem postResult;

    if (znccFunc(context, program, queue,
                 greyImage0, greyImage1, disparitys, disp_limit,
                 greyImgWidth, greyImgHeight, blockx, blocky,
                 &dmap1, &dmap2) == EXIT_FAILURE) {
        return NULL;
    }
    if (postProcessDmaps(context, queue, program, dmap1, dmap2, greyImgWidth,
                         greyImgHeight, disp_limit, &postResult) == EXIT_FAILURE) {
        return NULL;
    }

    /* Allocate host memory and read the resulting image from OpenCL device */
    unsigned char *res;
    res = malloc(sizeof(unsigned char)*(width/4)*(height/4));

    err = clEnqueueReadBuffer(queue, postResult, CL_TRUE, 0,
                              sizeof(cl_uchar)*(width/4)*(height/4),
                              res, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Couldn't read postprocessed image from buffer.\n");
        return NULL;
    }

    clReleaseMemObject(greyImage0);
    clReleaseMemObject(greyImage1);
    clReleaseMemObject(dmap1);
    clReleaseMemObject(dmap2);
    clReleaseMemObject(postResult);

    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device);

    hostTime2 = doubleTime();
    printf("Total time (host):              %6.1lf ms.\n\n",
           (hostTime2-hostTime1)*1e3);

    return res;
}
