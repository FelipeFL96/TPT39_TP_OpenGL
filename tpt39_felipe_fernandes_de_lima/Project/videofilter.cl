__kernel void videofilter(global const float *gaussian_kernel,
                        __global const unsigned char *i_frame, 
                        __global char *restrict o_frame)
{
    size_t id_x = get_global_id(0);
    size_t id_y = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);

    int mask_size = 3;
    float sum = 0;
    unsigned char r = 0;
    
    for(int i = 0; i < mask_size; i++) {
        for(int j = 0; j <  mask_size; j++) {
            int a = (id_x - mask_size/2) + i;
            int b = (id_y - mask_size/2) + j;
            
            if (id_x >= mask_size/2 && id_y >= mask_size/2 && id_x < (width - mask_size/2) && id_y < (height - mask_size/2)) {
                r = (float) i_frame[b*width+a];
            } 
            else {
                r = 0;
            }
            sum += r * gaussian_kernel[i*mask_size+j];
            //if (id_x == 160 && id_y == 115) printf("%.2x\n", (unsigned char)sum);
        }
    }
    ///if (id_x == 0 && id_y == 0) printf("%f  %f\n", sum, i_frame[id_y*width+id_x]);
    
    o_frame[id_y*width+id_x] = (unsigned char) sum;
    /*o_frame[id_x*height+id_y] = i_frame[id_x*height+id_y];*/
    /*o_frame[id_y*width+id_x] = i_frame[id_y*width+id_x];*/
}