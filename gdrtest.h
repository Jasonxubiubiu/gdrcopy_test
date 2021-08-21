#ifndef __GDR_TEST
#define __GDR_TEST

#include <gdrapi.h>

#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))
#define MYCLOCK CLOCK_MONOTONIC


typedef struct gdr_info_tmp{
    gdr_t g;
    gdr_mh_t mh;
    void *map_d_ptr;
    size_t size;
}gdr_info_tmp;

// do the job of initalization of GPU computing
// no parameters
// return : A pointer of GPU memory
CUdeviceptr gdr_init();

// Map the GPU address to user-space
uint32_t* gdr_map_addr(CUdeviceptr d_A, size_t size, gdr_info_tmp* gi);

//Unmap the GPU address
void gdr_unmap_addr(gdr_info_tmp* gi);
#endif
