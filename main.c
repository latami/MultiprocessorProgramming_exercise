#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>

#include "lodepng.h"
#include "depthmap_c.h"
#include "doubleTime.h"

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

int main(int argc, char *argv[])
{
	int error;
	unsigned char *image0, *image1;
	unsigned int width0, width1, height0, height1;
	double time1, time2, timeTotal1, timeTotal2;
	char c;
	int blockx, blocky, disp_limit;
	searchMethod select;

	/* defaults */
	blockx = 9;
	blocky = 9;
	disp_limit = 65;
	select = HIERARCHIC;

	/* Parse command line */
	while (1) {
		c = getopt(argc, argv, "x:y:d:b");
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
		default:
			printf("Options:\n"
				   "-x <>   sets blocksize in x-direction\n"
				   "-y <>   sets blocksize in y-direction\n"
				   "-d <>   sets maximum distance to search matches\n"
				   "-b      toggle bruteforcing depthmaps\n");
			return EXIT_FAILURE;
			break;
		}
	}

	timeTotal1 = doubleTime();

	time1 = doubleTime();

	/* Load images */
	error = lodepng_decode32_file(&image0, &width0, &height0, "im0.png");
	if (error) {
		fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
		return EXIT_FAILURE;
	}
	error = lodepng_decode32_file(&image1, &width1, &height1, "im1.png");
	if (error) {
		fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
		return EXIT_FAILURE;
	}
	if ((width0 != width1) || (height0 != height1)) {
		fprintf(stderr, "Image dimensions did not match!\n");
		free(image0);
		free(image1);
		return EXIT_FAILURE;
	}
	time2 = doubleTime();
	printf("Image decoding time: %.3lf seconds.\n", time2-time1);

	unsigned char *finalDepthmap;
	finalDepthmap = generateDepthmap(image0, image1, width0, height0, blockx, blocky, disp_limit, select);
	//int error;
	error = lodepng_encode_file("depth01p.png", finalDepthmap,
								width0/4, height0/4, LCT_GREY, 8);
	if (error) {
		fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
	}

	timeTotal2 = doubleTime();

	printf("Program total time: %.3lf seconds.\n", timeTotal2-timeTotal1);

	return EXIT_SUCCESS;
}
