#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <gdrapi.h>

#define SIZE    (10*1024*1024)

void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
{
    uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
    unsigned w;
    for(w = 0; w<size/sizeof(uint32_t); ++w)
        h_buf[w] = base_value ^ (1<< (w%32));
}

void cuda_gdr_test(size_t size){
    uint32_t *init_buf = NULL;
    cuMemAllocHost((void **)&init_buf, size);
    init_hbuf_walking_bit(init_buf, size);
    // Create a gdr object
    gdr_t g = gdr_open();
    // Create a gdr handler
    gdr_mh_t mh;
    
    return;
}

int main( void ) {
    size_t size = 0;
    cuda_gdr_test(size);
    return 0;
}
