#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <gdrapi.h>

#define SIZE    (10*1024*1024)
#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

int num_write_iters = 10000;
size_t copy_size = 0;
size_t copy_offset = 0;

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
    int iter = 0;
    uint32_t *init_buf = NULL;
    CUresult result = cuMemAllocHost((void **)&init_buf, size);
    if (result != CUDA_SUCCESS){
        const char *_err_name;
        cuGetErrorName(result, &_err_name);
        printf("CUDA error: %s\n", _err_name);
    }
    init_hbuf_walking_bit(init_buf, size);
    // Create a gdr object
    gdr_t g = gdr_open();
    // Create a gdr handler
    gdr_mh_t mh;
    
    //start map and data copy
    ret = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
    if (ret != 0){
        printf("gdr_pin_buffer error!\n");
    }
    
    void *map_d_ptr  = NULL;
    if (gdr_map(g, mh, &map_d_ptr, size) != 0){
        printf("gdr_map error!\n");
    }
    
    printf("map_d_ptr: %x\n", map_d_ptr);
    
    gdr_info_t info;
    if (gdr_get_info(g, mh, &info) != 0){
        printf("gdr_get_info error!\n");
    }
    
    printf("info.va: %x\n", info.va);
    printf("info.mapped_size: %x\n", info.mapped_size);
    printf("info.page_size: %x\n", info.page_size);
    printf("info.mapped: %x\n", info.mapped);
    printf("info.wc_mapping: %x\n", info.wc_mapping);
    
    int off = info.va - d_A;
    printf("page offset: %d\n", off);

    uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
    printf("user-space pointer: %x\n", buf_ptr);
    
    for (iter=0; iter<num_write_iters; ++iter)
        gdr_copy_to_mapping(mh, buf_ptr + copy_offset/4, init_buf, copy_size);    
    
    if(gdr_unmap(g, mh, map_d_ptr, size) != 0){
        printf("gdr_unmap error!\n");
    }
    
    if(gdr_unpin_buffer(g, mh) != 0){
       printf("gdr_unpin_buffer error!\n"); 
    }
    
    if(gdr_close(g) != 0){
       printf("gdr_close error!\n"); 
    }

    
    return;
}

int main( void ) {
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext dev_ctx;
    cuDevicePrimaryCtxRetain(&dev_ctx, dev);
    cuCtxSetCurrent(dev_ctx);
    size_t size = 128*1024;
    //size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    gpu_mem_alloc(&mhandle, size, true, true);
    d_A = mhandle.ptr;
    cuda_gdr_test(d_A, size);
    return 0;
}
