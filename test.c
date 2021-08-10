#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <gdrapi.h>

#define SIZE    (10*1024*1024)
#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

typedef struct gpuMemHandle 
{
    CUdeviceptr ptr; 
    union {
        CUdeviceptr unaligned_ptr;
        #if CUDA_VERSION >= 11000
        CUmemGenericAllocationHandle handle;
        #endif
    };
    size_t size;
    size_t allocated_size;
} gpu_mem_handle_t;

CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
{
    CUresult ret = CUDA_SUCCESS;
    CUdeviceptr ptr, out_ptr;
    size_t allocated_size;

    if (aligned_mapping)
        allocated_size = size + GPU_PAGE_SIZE - 1;
    else
        allocated_size = size;

    ret = cuMemAlloc(&ptr, allocated_size);
    if (ret != CUDA_SUCCESS)
        return ret;

    if (set_sync_memops) {
        unsigned int flag = 1;
        ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
        if (ret != CUDA_SUCCESS) {
            cuMemFree(ptr);
            return ret;
        }
    }

    if (aligned_mapping)
        out_ptr = ROUND_UP(ptr, GPU_PAGE_SIZE);
    else
        out_ptr = ptr;

    handle->ptr = out_ptr;
    handle->unaligned_ptr = ptr;
    handle->size = size;
    handle->allocated_size = allocated_size;

    return CUDA_SUCCESS;
} 

void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
{
    uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
    unsigned w;
    for(w = 0; w<size/sizeof(uint32_t); ++w)
        h_buf[w] = base_value ^ (1<< (w%32));
}

void cuda_gdr_test(CUdeviceptr d_A, size_t size){
    int ret = 0;
    uint32_t *init_buf = NULL;
    cuMemAllocHost((void **)&init_buf, size);
    init_hbuf_walking_bit(init_buf, size);
    // Create a gdr object
    gdr_t g = gdr_open();
    // Create a gdr handler
    gdr_mh_t mh;
    //ret = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
    if (ret != 0){
        printf("gdr_pin_buffer error!\n");
    }
    //gdr_unpin_buffer(g, mh);
    return;
}

int main( void ) {
    size_t size = 128*1024;
    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    
    cuda_gdr_test(d_A, size);
    return 0;
}
