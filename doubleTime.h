#ifndef DOUBLETIME_H
#define DOUBLETIME_H

#include <time.h>

/* Returns time conveniently as a double-value. */
static double doubleTime(void) {
       struct timespec timeNow;

       clock_gettime(CLOCK_MONOTONIC, &timeNow);
       return (double)timeNow.tv_sec + (double)timeNow.tv_nsec/1e9;
}

#endif
