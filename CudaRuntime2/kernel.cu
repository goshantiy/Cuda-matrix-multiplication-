
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <random>
#include <chrono>

__global__ void mulMtrx(double* a, double* b, double* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}
void initMatrix(std::vector<double>& matrix)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> rg(1, 100);
	for (int i=0;i<matrix.size();i++)
	{
		matrix[i] = rg(gen);
	}
}
void printMatrix(std::vector<double>& matrix)
{
	for (auto it : matrix)
		std::cout << it<<' ';
}
void cpu_matrix_mult(const std::vector<double> &h_a,const std::vector<double> &h_b, std::vector<double> &h_result, int m, int n, int k) {
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			double tmp = 0;
			for (int h = 0; h < n; ++h)
			{
				tmp += h_a[i * n + h] * h_b[h * k + j];
			}
			h_result[i * k + j] = tmp;
		}
	}
}

int main()
{
size_t m, n, k;
std::cin >> m >> n >> k;
   std::vector<double> h_a(m*n);
   std::vector<double> h_b(n*k);
   std::vector<double> h_c(m*k);
   
   initMatrix(h_a);
   initMatrix(h_b);
  // printMatrix(h_a);
   std::cout << "\n";
  // printMatrix(h_b);

   //MEMORY ALLOCATION
   double* d_a, * d_b, * d_c;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaMalloc(&d_a, m * n * sizeof(double));
   cudaMalloc(&d_b, n * k * sizeof(double));
   cudaMalloc(&d_c, m * k * sizeof(double));
   
   //MEMCPY
   cudaEventRecord(start);
   cudaMemcpy(d_a, h_a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, h_b.data(), n * k * sizeof(double), cudaMemcpyHostToDevice);
   int BLOCK_SIZE = 32;
   unsigned int grid_rows = ceil( double(m) / double(BLOCK_SIZE));
   unsigned int grid_cols = ceil( double(k) / double(BLOCK_SIZE));
   dim3 dimGrid(grid_cols, grid_rows);
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   //GPU RUN
  
   mulMtrx<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
   cudaThreadSynchronize();
   
   cudaMemcpy(h_c.data(), d_c, m * k * sizeof(double), cudaMemcpyDeviceToHost);
   
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   std::cout << "\ngpu milliseconds elapsed: " << milliseconds;
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
  
   //std::cout << "\nresult:\n";
   //printMatrix(h_c);

   //CPU RUN
   std::vector<double>h_cc(m * k);
   auto begin = std::chrono::steady_clock::now();
   cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
   auto end = std::chrono::steady_clock::now();
   auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
   std::cout << "\ncpu milliseconds elapsed: " << elapsed_ms.count();
  
   bool all_ok = 1;
   for (int i = 0; i < m; ++i)
   {
	   for (int j = 0; j < k; ++j)
		   if (h_cc[i * k + j] != h_c[i * k + j])
		   {
			   all_ok = 0;
		   }
   }

   if (all_ok)
   {
	   std::cout << "\nok\n";
	  // printMatrix(h_cc);
   }
   else
   {
	   std::cout << "\nne ok\n";
	  // printMatrix(h_cc);
   }
   std::cout << '\n';
  // printMatrix(h_c);
 }