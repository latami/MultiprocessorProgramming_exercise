#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <pthread.h>

#include "lodepng.h"
#include "depthmap_c.h"
#include "depthmap_opencl.h"
#include "doubleTime.h"

#define DEF_THREADS 0
#define DEF_DISABLE_ASM 0

struct threadData {
    unsigned int error;
    unsigned char *image;
    unsigned int w;
    unsigned int h;
    char *name;
};

/* integer conversion with error checking */
int parse_int( const char *str, int *d_error ) {

    char *pend;
    int i;

    errno = 0;
    i = strtol(str, &pend, 10);
    if (pend == str || (*pend) != '\0' || errno != 0)
        (*d_error) = EXIT_FAILURE;
    else
        (*d_error) = EXIT_SUCCESS;

    return i;
}

void *imageLoader(void *data) {
    struct threadData *thData;

    thData = (struct threadData *)data;
    thData->error = lodepng_decode32_file(&thData->image, &thData->w, &thData->h, thData->name);
    if (thData->error) {
        fprintf(stderr, "error %u: %s\n", thData->error, lodepng_error_text(thData->error));
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    int error;
    double time1, time2, timeTotal1, timeTotal2;
    char c;
    int blockx, blocky, disp_limit, threads, disableAsm;
    unsigned int setOpencl;
    searchMethod select;

    /* defaults */
    threads = DEF_THREADS;
    disableAsm = DEF_DISABLE_ASM;
    blockx = 9;
    blocky = 9;
    disp_limit = 65;
    select = HIERARCHIC;
    setOpencl = 0;

    /* Parse command line */
    while (1) {
        c = getopt(argc, argv, "x:y:d:bt:sa:");
        if (c == -1)
            break;
        switch (c) {
        case 'x':
            blockx = parse_int(optarg, &error);
            if (error == EXIT_FAILURE) {
                fprintf(stderr, "Error parsing x!\n");
                return EXIT_FAILURE;
            }
            break;
        case 'y':
            blocky = parse_int(optarg, &error);
            if (error == EXIT_FAILURE) {
                fprintf(stderr, "Error parsing y!\n");
                return EXIT_FAILURE;
            }
            break;
        case 'd':
            disp_limit = parse_int(optarg, &error);
            if (error == EXIT_FAILURE) {
                fprintf(stderr, "Error parsing disparity limit!\n");
                return EXIT_FAILURE;
            }
            break;
        case 'b':
            select = BRUTE;
            break;
        case 't':
            threads = parse_int(optarg, &error);
            if (error == EXIT_FAILURE) {
                fprintf(stderr, "Error parsing number of threads!\n");
                return EXIT_FAILURE;
            }
            break;
        case 's':
            disableAsm = 1;
            break;
        case 'a':
            setOpencl = parse_int(optarg, &error);
            if (error == EXIT_FAILURE || setOpencl > 2) {
                fprintf(stderr, "Error parsing OpenCL argument!\n");
                return EXIT_FAILURE;
            }
            break;
        default:
            printf("Options:\n"
                   "-x <>   set blocksize in x-direction\n"
                   "-y <>   set blocksize in y-direction\n"
                   "-d <>   set maximum distance to search matches\n"
                   "-b      toggle bruteforcing depthmaps\n"
                   "-t <>   set number of threads\n"
                   "-s      toggle to disable assembly-code\n"
                   "-a <>   select opencl version\n"
                   "        1: basic cpu\n"
                   "        2: basic gpu\n");
            return EXIT_FAILURE;
            break;
        }
    }
    if ((DEF_DISABLE_ASM != disableAsm || DEF_THREADS != threads) && setOpencl > 0)
        printf("Arguments used, that have no effect with OpenCL.\n");

    timeTotal1 = doubleTime();

    time1 = doubleTime();

    /* Load images */
    struct threadData thread0, thread1;
    pthread_t helperThread;

    thread0.name = "im0.png";
    thread1.name = "im1.png";
    /* Launch thread to decode another image */
    pthread_create(&helperThread, NULL, imageLoader, (void *)&thread1);
    /* Decode image also in mainthread. */
    imageLoader((void *)&thread0);

    /* Wait thread to finish before proceeding. */
    pthread_join(helperThread, NULL);
    if (thread0.error || thread1.error)
        return EXIT_FAILURE;

    if ((thread0.w != thread1.w) || (thread0.h != thread1.h)) {
        fprintf(stderr, "Image dimensions did not match!\n");
        free(thread0.image);
        free(thread1.image);
        return EXIT_FAILURE;
    }
    time2 = doubleTime();
    printf("Image decoding time: %.3lf seconds.\n", time2-time1);

    unsigned char *finalDepthmap;
    if (setOpencl != 0) {
        finalDepthmap = generateDepthmap_opencl_basic(thread0.image, thread1.image,
                                                      thread0.w, thread0.h,
                                                      blockx, blocky,
                                                      disp_limit, select, setOpencl);
    }
    else
        finalDepthmap = generateDepthmap(thread0.image, thread1.image,
                                         thread0.w, thread0.h,
                                         blockx, blocky,
                                         disp_limit, select, threads, disableAsm);

    if (finalDepthmap == NULL) {
        fprintf(stderr, "GenerateDepthmap failed!\n");
        return EXIT_FAILURE;
    }
    error = lodepng_encode_file("depth01p.png", finalDepthmap,
                                thread0.w/4, thread0.h/4, LCT_GREY, 8);
    if (error) {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
    }

    timeTotal2 = doubleTime();

    printf("Program total time: %.3lf seconds.\n", timeTotal2-timeTotal1);

    return EXIT_SUCCESS;
}
