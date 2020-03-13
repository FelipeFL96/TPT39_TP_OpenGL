__kernel void vector_add(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z)
{
    size_t id = get_global_id(0);    
    z[id] = x[id] + y[id];
}

__kernel void matrix_multiplication(__global const int *dim, 
                        __global const float *x, 
                        __global const float *y, 
                        __global float *restrict z)
{
    size_t id = get_global_id(0);
    int M = dim[0], N = dim[1], K = dim[2];
    int lin = (int) id/M;
    int col = (int) id%M;
    
    float acc = 0;

    for(int i = 0; i < N; i++) {
        acc += x[lin*M+i] * y[i*N+col]; //x[lin][i] * y[i][col]
    }
    z[lin*M+col] = acc;
}
