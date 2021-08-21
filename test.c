#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <gdrapi.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include "gdrtest.h"

#define SIZE    (10*1024*1024)
//#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))
//#define MYCLOCK CLOCK_MONOTONIC

char pathname[] = "./data.txt";
int num_write_iters = 100;
extern size_t copy_size;
size_t copy_offset = 0;


void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
{
    uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
    unsigned w;
    for(w = 0; w<size/sizeof(uint32_t); ++w)
        h_buf[w] = base_value ^ (1<< (w%32));
}

ssize_t pread_buf_from_file(uint32_t *buf){
    ssize_t res = 0;
    int f_id; // file descriptor
    ssize_t nread;
    f_id = open(pathname, O_RDWR | O_CREAT);
    if (f_id == -1){
        printf("open file error for %s\n", pathname);
    }
    int nbytes = 131072;
    res = pread(f_id, buf, nbytes, 0); 
    if (res == -1){
        printf("pread error\n");
    }
    return res;
}

void write_file(){
    int fd;
    fd = open(pathname, O_WRONLY|O_CREAT);
    if (fd == -1){ 
        printf("open file error (write) for %s\n", pathname);
    }
    uint32_t test_buf = NULL;
    test_buf = (uint32_t *)malloc(copy_size); 
    init_hbuf_walking_bit(test_buf, copy_size);
    write(fd, test_buf, copy_size);
    return;
}

void cuda_gdr_test(CUdeviceptr d_A, size_t size){
    int ret = 0;
    int iter = 0;
    //CUresult result = cuMemAllocHost((void **)&init_buf, size); 
    //CUresult result = cudaHostAlloc((void **)&init_buf, size, cudaHostAllocDefault);
    //if (result != CUDA_SUCCESS){
    //    const char *_err_name;
    //    cuGetErrorName(result, &_err_name);
    //    printf("CUDA error: %s\n", _err_name);
    //}
    //init_hbuf_walking_bit(init_buf, size);
    // Create a gdr object
    //gdr_t g = gdr_open();
    // Create a gdr handler
    //gdr_mh_t mh;
    
    //start map and data copy
    //ret = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
    //if (ret != 0){
    //    printf("gdr_pin_buffer error!\n");
    //}
    //
    //void *map_d_ptr  = NULL;
    //if (gdr_map(g, mh, &map_d_ptr, size) != 0){
    //    printf("gdr_map error!\n");
    //}
    //
    //printf("map_d_ptr: %x\n", map_d_ptr);
    //
    //gdr_info_t info;
    //if (gdr_get_info(g, mh, &info) != 0){
    //    printf("gdr_get_info error!\n");
    //}
    //
    //printf("info.va: %x\n", info.va);
    //printf("info.mapped_size: %x\n", info.mapped_size);
    //printf("info.page_size: %x\n", info.page_size);
    //printf("info.mapped: %x\n", info.mapped);
    //printf("info.wc_mapping: %x\n", info.wc_mapping);
    //
    //int off = info.va - d_A;
    //printf("page offset: %d\n", off);

    //uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
    //printf("user-space pointer: %x\n", buf_ptr);
    gdr_info_tmp gi; 
    uint32_t *buf_ptr =  gdr_map_addr(d_A, size, &gi);
    struct timespec beg, end;
    clock_gettime(MYCLOCK, &beg);
    
    for (iter=0; iter<num_write_iters; ++iter){
        if(pread_buf_from_file(buf_ptr) == -1){ 
            printf("pread error!\n");
        }
        else{ 
            printf("pread success!\n");
        }
    }
    clock_gettime(MYCLOCK, &end);
    double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("pread+gdrcopy time: %f ms\n", dt_ms);

    //clock_gettime(MYCLOCK, &beg);
    //for (iter=0; iter<num_write_iters; ++iter)
        
       // gdr_copy_to_mapping(mh, buf_ptr + copy_offset/4, init_buf, copy_size);    
    
    //clock_gettime(MYCLOCK, &end);
    //dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    //printf("cpu-gpu copy time: %f ms\n", dt_ms);
    
    
    //if(gdr_unmap(g, mh, map_d_ptr, size) != 0){
    //    printf("gdr_unmap error!\n");
    //}
    //
    //if(gdr_unpin_buffer(g, mh) != 0){
    //   printf("gdr_unpin_buffer error!\n"); 
    //}
    //
    //if(gdr_close(g) != 0){
    //   printf("gdr_close error!\n"); 
    //}
    gdr_unmap_addr(&gi);
    
    return;
}

void traditional_data_transfer_between_ssd_and_gpu(){
    int f_id;
    uint32_t *dev_gpu;  
    uint32_t trad_buf = NULL;
    int res = 0;
    struct timespec beg, end;
    trad_buf = (uint32_t *)malloc(copy_size); 
    f_id = open(pathname, O_RDWR | O_CREAT);
    if (f_id == -1){
        printf("open file error for %s\n", pathname);
    }
    int nbytes = 131072;
    cudaMalloc((void **)&dev_gpu, copy_size);
    int iter;
    //Start time tick!
    clock_gettime(MYCLOCK, &beg);
    for (iter=0; iter<num_write_iters; ++iter){    
        res = read(f_id, trad_buf, nbytes); 
        if (res == -1){
            printf("read error\n");
        }
        cudaMemcpy(dev_gpu, trad_buf, copy_size, cudaMemcpyHostToDevice);
    }
    clock_gettime(MYCLOCK, &end);
    double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Traditional data copy time: %f ms\n", dt_ms);
    cudaFreeHost(trad_buf);
    return;
}


void traditional_pin_data_transfer_between_ssd_and_gpu(){
    int f_id;
    uint32_t *dev_gpu;  
    uint32_t trad_buf = NULL;
    int res = 0;
    struct timespec beg, end;
    trad_buf = (uint32_t *)malloc(copy_size);
     
    //CUresult result = cudaHostAlloc((void **)&trad_buf, copy_size, cudaHostAllocDefault);
    CUresult result = cudaHostRegister(trad_buf, copy_size, cudaHostRegisterDefault);
    if (result != CUDA_SUCCESS){
        const char *_err_name;
        cuGetErrorName(result, &_err_name);
        printf("CUDA error: %s\n", _err_name);
    }

    f_id = open(pathname, O_RDWR);
    if (f_id == -1){
        printf("open file error for %s\n", pathname);
    }
    int nbytes = 131072;
    cudaMalloc((void **)&dev_gpu, copy_size);
    int iter;
    //Start time tick!
    clock_gettime(MYCLOCK, &beg);
    for (iter=0; iter<num_write_iters; ++iter){    
        //res = read(f_id, trad_buf, nbytes); 
        res = pread(f_id, trad_buf, nbytes, 0);
        if (res == -1){
            printf("read error\n");
        }
       cudaMemcpy(dev_gpu, trad_buf, copy_size, cudaMemcpyHostToDevice);
    }
    clock_gettime(MYCLOCK, &end);
    double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
    printf("Traditional pinned buffer data copy time: %f ms\n", dt_ms);
    return;
}

int main( void ) {
    write_file();
    size_t size = 128*1024;
    ////size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    CUdeviceptr d_A = gdr_init();
    cuda_gdr_test(d_A, size);
    traditional_data_transfer_between_ssd_and_gpu();
    traditional_pin_data_transfer_between_ssd_and_gpu();
    return 0;
}
