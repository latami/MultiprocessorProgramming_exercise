#include <stdio.h>
#include <CL/cl.h>

/* Print info required by task2-phase. */
void printfInfo(cl_device_id device) {

    printf("\n");
    cl_uint value;
    cl_ulong value2;

    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
                    sizeof(cl_uint), &value, NULL);
    if (value == CL_LOCAL) {
        printf("Device has local memory: ");
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(cl_ulong), &value2, NULL);
        printf("         %6ld KiB\n", value2/1024);
    }
    else
        printf("Device has no local memory.\n");

    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(cl_uint), &value, NULL);
    printf("Device compute units:             %6d\n", value);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(cl_uint), &value, NULL);
    printf("Device maximum clock frequency:   %6d MHz\n", value);

    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                    sizeof(cl_ulong), &value2, NULL);
    printf("Maximum constant buffer size:     %6ld KiB\n", value2/1024);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(cl_uint), &value, NULL);
    printf("Maximum workgroup size:           %6d\n", value);

    size_t *itemSizes;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(cl_uint), &value, NULL);
    itemSizes = malloc(sizeof(size_t)*value);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(size_t)*value, itemSizes, NULL);
    printf("Max item sizes:             ");
    for (int i=0; i < value; i++)
        printf(" %ld", itemSizes[i]);
    printf("\n");
    free(itemSizes);

    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &value2, NULL);
    printf("Maximum allocateable object size: %6ld MiB\n\n", value2/1048576);
}

cl_float eventRuntime(cl_event event) {
    cl_ulong time_start, time_end, time_executing;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(time_end), &time_end, NULL);
    time_executing = time_end-time_start;

    return (cl_float)time_executing/1e6f;
}

cl_int buildOCLProgram(cl_device_id device, cl_context context,
                       const char *filename, cl_program *program) {
        /* Build program */
        FILE *handle;
        char *buffer, *log;
        size_t program_size, log_size;
        cl_int err;

        handle = fopen(filename, "r");
        if (handle == NULL) {
                perror("Couldn't open the program file");
                return EXIT_FAILURE;
        }
        fseek(handle, 0, SEEK_END);
        program_size = ftell(handle);
        rewind(handle);
        buffer = malloc(program_size+1);
        buffer[program_size] = '\0';
        fread(buffer, sizeof(char), program_size, handle);
        fclose(handle);

        (*program) = clCreateProgramWithSource(context, 1, (const char **)&buffer, &program_size, &err);
        if (err < 0) {
                perror("Couldn't create OpenCL-program");
                return EXIT_FAILURE;
        }
        free(buffer);

        err = clBuildProgram((*program), 0, NULL, NULL, NULL, NULL);
        if (err < 0) {
                clGetProgramBuildInfo((*program), device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                log = malloc(log_size+1);
                clGetProgramBuildInfo((*program), device, CL_PROGRAM_BUILD_LOG, log_size+1, log, NULL);
                fprintf(stderr, "%s\n", log);
                free(log);
                return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
}

int initOpenCL(cl_platform_id *platform, cl_device_id *device,
               cl_context *context, cl_program *program, cl_device_type devType,
               const char *sourceFile) {
        int err;

        err = clGetPlatformIDs(1, platform, NULL);
        if (err < 0) {
                perror("Couldn't identify a platform");
                return EXIT_FAILURE;
        }

        err = clGetDeviceIDs((*platform), devType, 1, device, NULL);
        if (err < 0) {
                fprintf(stderr, "Couldn't access ");
                if (devType == CL_DEVICE_TYPE_CPU)
                        fprintf(stderr, "CPU!\n");
                else if (devType == CL_DEVICE_TYPE_GPU)
                        fprintf(stderr, "GPU!\n");
                else
                        fprintf(stderr, "device!\n");

                return EXIT_FAILURE;
        }

        printfInfo((*device));

        (*context) = clCreateContext(NULL, 1, device, NULL, NULL, &err);
        if (err < 0) {
                perror("Couldn't create a context");
                return EXIT_FAILURE;
        }

        /* Build OpenCL source-file */
        if (buildOCLProgram((*device), (*context), sourceFile, program) == EXIT_FAILURE) {
                fprintf(stderr, "cl-file did not compile.\n");
                return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
}
