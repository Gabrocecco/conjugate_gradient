#include<stdio.h>

/*
Compile: 
riscv64-unknown-elf-gcc -o redcution_rv reduction.c
Run: 
spike pk reduction_rv

Produce assembly: 
riscv64-unknown-elf-gcc -S -o reduction_rv.s reduction.c
*/ 

int reduce_sum(int *vec, int dim){
    int sum = 0;
    for(int i = 0; i < dim; i++){
        sum += vec[i];
    }
    
    return sum;
}

int main(){
	printf("Reduction test started\n");
    
    int vec[5] = {1,2,3,4,5};
    int sum = 0; 

    sum = reduce_sum(vec, 5);

    printf("Reduction (sum) = %d\n", sum);

    printf("Reduction test ended\n");
    

return 0; 
}
