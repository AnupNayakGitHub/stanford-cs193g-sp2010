/* This is machine problem 2, part 2: brute force k nearest neighbors
 * You are given a large number of particles, and are asked
 * to find the k particles that are nearest to each one.
 * Look at the example in /tutorials/thread_local_variables.cu
 * for how you can use per thread arrays for sorting.
 * Using that example, port the cpu reference code to the gpu in a first step.
 * In a second step, modify your code so that the per-thread arrays are in 
 * shared memory. You should submit this second version of your code. 
 */
 
/*
 * SUBMISSION INSTRUCTIONS
 * =========================
 * 
 * You can submit the assignment from any of the cluster machines by using
 * our submit script. Th submit script bundles the entire current directory into
 * a submission. Thus, you use it by CDing to a the directory for your assignment,
 * and running:
 * 
 *   > cd *some directory*
 *   > /usr/class/cs193g/bin/submit mp2
 * 
 * This will submit the current directory as your assignment. You can submit
 * as many times as you want, and we will use your last submission.
 */
 
#include <cassert>

#include "mp2-util.h"

// TODO enable this to print debugging information
//const bool print_debug = true;
const bool print_debug = false;

event_pair timer;

inline __device__ __host__ float3 operator -(float3 a, float3 b)
{
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__host__ __device__
float dist2(float3 a, float3 b)
{
  float3 d = a - b;
  float d2 = d.x*d.x + d.y*d.y + d.z*d.z;
  return d2;
}

template
<typename T>
__host__ __device__
void init_list(T *base_ptr, unsigned int size, T val)
{
  for(int i=0;i<size;i++)
  {
    base_ptr[i] = val;
  }
}

__host__ __device__
void insert_list(float *dist_list, int *id_list, int size, float dist, int id)
{
 int k;
 for (k=0; k < size; k++) {
   if (dist < dist_list[k]) {
     // we should insert it in here, so push back and make it happen
     for (int j = size - 1; j > k ; j--) {
       dist_list[j] = dist_list[j-1];
       id_list[j] = id_list[j-1];
     }
     dist_list[k] = dist;
     id_list[k] = id;
     break;
   }
 }
}

template
  <int num_neighbors>
void host_find_knn(float3 *particles, int *knn, int array_length)
{
  for(int i=0;i<array_length;i++)
  {
    float3 p = particles[i];
    float neigh_dist[num_neighbors];
    int neigh_ids[num_neighbors];
    
    init_list(&neigh_dist[0],num_neighbors,2.0f);
    init_list(&neigh_ids[0],num_neighbors,-1);
    for(int j=0;j<array_length;j++)
    {
      if(i != j)
      {
        float rsq = dist2(p,particles[j]);
        insert_list(&neigh_dist[0], &neigh_ids[0], num_neighbors, rsq, j);
      }
    }
    for(int j=0;j<num_neighbors;j++)
    {
      knn[num_neighbors*i + j] = neigh_ids[j];
    }
  }
}

template
  <int num_neighbors>
__global__ void device_find_knn_local_mem(float3 *particles, int *knn, int array_length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= array_length) return;

  {
    float3 p = particles[i];
    float neigh_dist[num_neighbors];
    int neigh_ids[num_neighbors];

    init_list(&neigh_dist[0],num_neighbors,2.0f);
    init_list(&neigh_ids[0],num_neighbors,-1);
    for(int j=0;j<array_length;j++)
    {
      if(i != j)
      {
        float rsq = dist2(p,particles[j]);
        insert_list(&neigh_dist[0], &neigh_ids[0], num_neighbors, rsq, j);
      }
    }
    for(int j=0;j<num_neighbors;j++)
    {
      knn[num_neighbors*i + j] = neigh_ids[j];
    }
  }
}

template
  <int num_neighbors>
__global__ void device_find_knn_shared_mem(float3 *particles, int *knn, int array_length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= array_length) return;

  {
    float3 p = particles[i];
    extern __shared__ float neigh_dist[]; 
                      int * neigh_ids = (int*)&neigh_dist[num_neighbors];

    init_list(&neigh_dist[0],num_neighbors,2.0f);
    init_list(&neigh_ids[0],num_neighbors,-1);
    for(int j=0;j<array_length;j++)
    {
      if(i != j)
      {
        float rsq = dist2(p,particles[j]);
        insert_list(&neigh_dist[0], &neigh_ids[0], num_neighbors, rsq, j);
      }
    }
    for(int j=0;j<num_neighbors;j++)
    {
      knn[num_neighbors*i + j] = neigh_ids[j];
    }
  }
}


void allocate_host_memory(int num_particles, int num_neighbors,
                          float3 *&h_particles, int *&h_knn, int *&h_knn_checker)
{
  // malloc host array
  h_particles = (float3*)malloc(num_particles * sizeof(float3));
  h_knn = (int*)malloc(num_particles * num_neighbors * sizeof(int));
  h_knn_checker = (int*)malloc(num_particles * num_neighbors * sizeof(int));

  // if either memory allocation failed, report an error message
  if(h_particles == 0 || h_knn == 0 || h_knn_checker == 0)
  {
    printf("couldn't allocate host memory\n");
    exit(1);
  }
}

#define CHECK(call) { \
  cudaError_t err = cudaSuccess; \
  if ( (err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  }\
}

void allocate_device_memory(int num_particles, int num_neighbors,
                            float3 *&d_particles, int *&d_knn)
{
  // device memory allocations here
  CHECK(cudaMalloc((void**)&d_particles, num_particles * sizeof(float3)));
  CHECK(cudaMalloc((void**)&d_knn, num_particles * num_neighbors *  sizeof(int)));
}


void deallocate_host_memory(float3 *h_particles, int *h_knn, int *h_knn_checker)
{
  free(h_particles);
  free(h_knn);
  free(h_knn_checker);
}


void deallocate_device_memory(float3 *d_particles, int *d_knn)
{
  // device memory deallocations here
  CHECK(cudaFree(d_particles));
  CHECK(cudaFree(d_knn));
}


bool cross_check_results(int * reference_knn, int * knn, int num_particles, int num_neighbors)
{
  int error = 0;


  for(int i=0;i<num_particles;i++)
  {
    for(int j=0;j<num_neighbors;j++)
    {
      if(reference_knn[i*num_neighbors + j] != knn[i*num_neighbors + j])
      {
        if(print_debug) printf("particle %d, neighbor %d is %d on cpu, %d on gpu\n",i,j,reference_knn[i*num_neighbors + j],knn[i*num_neighbors + j]);
        error = 1;
      }
    }

  }

  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else {
    printf("Worked! CUDA and reference output match. \n");
  }
  return error;
}

int main(void)
{  
  // create arrays of 8K elements
  int num_particles = 20*1024;
  const int num_neighbors = 5;

  // pointers to host arrays
  float3 *h_particles = 0;
  int    *h_knn = 0;
  int    *h_knn_checker = 0;

  // pointers to device arrays
  float3 *d_particles = 0;
  int    *d_knn = 0;

  allocate_host_memory(num_particles, num_neighbors, h_particles, h_knn, h_knn_checker);
  allocate_device_memory(num_particles, num_neighbors, d_particles, d_knn);

  // generate random input
  // initialize
  srand(13);

  for(int i=0;i< num_particles;i++)
  {
    h_particles[i] = make_float3((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX);
  }

  // copy input to GPU
  start_timer(&timer);
  //copy of input from host to device here
  CHECK(cudaMemcpy(d_particles,h_particles, num_particles * sizeof (float3), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_knn,h_knn, num_particles * num_neighbors * sizeof (int), cudaMemcpyHostToDevice));
  stop_timer(&timer,"copy to gpu");

  dim3 THRD_SZ(512);
  dim3 GRID_SZ((num_particles + THRD_SZ.x-1)/THRD_SZ.x);

  start_timer(&timer);  
  // kernel launch which uses local memory
  device_find_knn_local_mem<num_neighbors><<<GRID_SZ, THRD_SZ>>>(d_particles, d_knn, num_particles);
  check_cuda_error("brute force knn");
  stop_timer(&timer,"brute force knn");

  start_timer(&timer);  
  // kernel launch which uses __shared__ memory
  int shmem_size = THRD_SZ.x * (sizeof (float) +  sizeof(int)) * num_neighbors;
  device_find_knn_shared_mem<num_neighbors><<<GRID_SZ, shmem_size>>>(d_particles, d_knn, num_particles);
  check_cuda_error("shared meme knn");
  stop_timer(&timer,"shared mem knn");

  // download and inspect the result on the host
  start_timer(&timer);
  // copy results from device to host here
  cudaMemcpy(h_knn,d_knn, num_particles * num_neighbors * sizeof (int), cudaMemcpyDeviceToHost);
  check_cuda_error("copy from gpu");
  stop_timer(&timer,"copy back from gpu memory");

  // generate reference output
  start_timer(&timer);
  host_find_knn<num_neighbors>(h_particles, h_knn_checker, num_particles);
  stop_timer(&timer,"cpu brute force knn");

  // check CUDA output versus reference output
  cross_check_results(h_knn_checker, h_knn, num_particles, num_neighbors);

  deallocate_host_memory(h_particles, h_knn, h_knn_checker);
  deallocate_device_memory(d_particles, d_knn);

  return 0;
}

