#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void write_mat(int dim, float fmax){
    int i,j ;
    float mat[dim][dim] ;
    float sum[dim] ;
    for (i=0; i<dim; i++){sum[i] = 0;}
    for (i=0; i<dim; i++){
        for (j=i+1; j<dim; j++){
            float x = fmax * ((float)rand()/(float)(RAND_MAX)) - fmax/2. ;
            mat[i][j] = x ;
            mat[j][i] = x ;
            sum[i] += fabs(x) ;
            sum[j] += fabs(x) ;
            //fprintf(f, "%f ", x) ;
        }
        mat[i][i] = sum[i] + fmax*((float)rand()/(float)(RAND_MAX))/2. ;
        //fputs("\n",f) ;
    }
    FILE *f = fopen("mat.txt", "w");
    fprintf(f, "%d\n", dim);
    for (i=0; i<dim; i++){
        for (j=0; j<dim; j++){
            fprintf(f, "%f ", mat[i][j]) ;
        }
        fputs("\n", f) ;
    }
    fclose(f);
}

void write_mat_2(int dim){
    int i,j ;
    float mat[dim][dim] ;
    for (i=0; i<dim; i++){
        for (j=i+1; j<dim; j++){
            float dist = (float)abs(i-j), x ;
            if (dist <= ((float)dim) / 4.){
                x = pow(0.8, (float)(abs(i-j))) ;
            }else{ x = 0. ;}
            mat[i][j] = x ;
            mat[j][i] = x ;
        }
        mat[i][i] = dim ;
    }
    FILE *f = fopen("mat.txt", "w");
    fprintf(f, "%d\n", dim);
    for (i=0; i<dim; i++){
        for (j=0; j<dim; j++){
            fprintf(f, "%f ", mat[i][j]) ;
        }
        fputs("\n", f) ;
    }
    fclose(f);
}

void write_param(int dim, float fmax){
    FILE *f = fopen("param.txt", "w");
    fprintf(f, "%d\n", dim);
    int i;
    float tab[dim] ;
    float a = rand();
    for (i=0; i<dim; i++){
        float x =  fmax * ((float)rand()/(float)(RAND_MAX)) ;
        tab[i] = x ;
        fprintf(f, "%f ", x) ;
    }
    fclose(f);
}

void write_init_zero(int dim){
    FILE *f = fopen("init.txt", "w");
    fprintf(f, "%d\n", dim);
    int i;
    for (i=0; i<dim; i++){
        fprintf(f, "%f ", 0.) ;
    }
    fclose(f);
}

void write_init(int dim, float fmax){
    FILE *f = fopen("init.txt", "w");
    fprintf(f, "%d\n", dim);
    int i;
    float tab[dim] ;
    for (i=0; i<dim; i++){
        float x =  fmax * ((float)rand()/(float)(RAND_MAX)) - fmax/2. ;
        tab[i] = x ;
        fprintf(f, "%f ", x) ;
    }
    fclose(f);
}

int main(int argc, char* argv[]){
    int x;
    int args;

    printf("Enter an integer: ");
    if (( args = scanf("%d", &x)) == 0) {
        printf("Error: not an integer\n");
    } else {
        printf("Read in %d\n", x);
    }
    write_mat_2(x) ;
    write_init_zero(x) ;
    write_param(x, (float)x) ;
    return 0;
}
