#ifndef COMMON_OPENCL_H
#define COMMON_OPENCL_H

#include <CL/cl.h>

int initOpenCL(cl_platform_id *platform, cl_device_id *device,
               cl_context *context, cl_program *program, cl_device_type devType);

cl_int buildOCLProgram(cl_device_id device, cl_context context,
                       const char *filename, cl_program *program);

void printfInfo(cl_device_id device);

cl_float eventRuntime(cl_event event);

#endif
