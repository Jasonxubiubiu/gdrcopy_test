#include <memory.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>
#include <stdbool.h>
#include "gdrtest.h"

size_t copy_size = 131072;

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


CUdeviceptr gdr_init(){
    CUdeviceptr d_A;
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext dev_ctx;
    cuDevicePrimaryCtxRetain(&dev_ctx, dev);
    cuCtxSetCurrent(dev_ctx);
    gpu_mem_handle_t mhandle;
    size_t size = copy_size;
    gpu_mem_alloc(&mhandle, size, true, true);
    d_A = mhandle.ptr;
    return d_A;
}

uint32_t* gdr_map_addr(CUdeviceptr d_A, size_t size, gdr_info_tmp* gi){
    int ret = 0;
    // Create a gdr object
    gdr_t g = gdr_open();
    gi->g = g;
    // Create a gdr handler
    gdr_mh_t mh;
    //start map and data copy
    ret = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
    if (ret != 0){
        printf("gdr_pin_buffer error!\n");
    }
    gi->mh = mh;
    void *map_d_ptr  = NULL;
    if (gdr_map(g, mh, &map_d_ptr, size) != 0){
        printf("gdr_map error!\n");
    }
    gi->map_d_ptr = map_d_ptr;
    gi->size = copy_size;
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
    return buf_ptr; 
}

void gdr_unmap_addr(gdr_info_tmp* gi){
    if(gdr_unmap(gi->g, gi->mh, gi->map_d_ptr, gi->size) != 0){
        printf("gdr_unmap error!\n");
    }

    if(gdr_unpin_buffer(gi->g, gi->mh) != 0){
       printf("gdr_unpin_buffer error!\n");
    }

    if(gdr_close(gi->g) != 0){
       printf("gdr_close error!\n");
    }
    return;
}
