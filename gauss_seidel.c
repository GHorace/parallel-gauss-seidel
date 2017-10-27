#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "mpi.h"

int dim = 12; /* taille des vecteurs, matrices, etc. */

void read_param(int dim, float result[dim]){
    FILE *myFile;
    myFile = fopen("param.txt", "r");
    int size ;
    fscanf(myFile, "%d", &size);
    if (size != dim){
        printf("Size is not correct !");
        //return 0 ;
    } else {
        //read file into array
        int i;
        for (i = 0; i < dim; i++) { fscanf(myFile, "%f", &result[i]); }
        fclose(myFile) ;
        printf("param loaded\n") ;
    }

//    for (i = 0; i < dim; i++) { printf("Reading - Number |%i| : %f\n", i, vec_complet[i]); }
}

void read_init(int dim, float result[dim]){
    FILE *myFile;
    myFile = fopen("init.txt", "r");
    int size ;
    fscanf(myFile, "%d", &size);
    if (size != dim){
        printf("Size is not correct !");
    } else {
        //read file into array
        int i;
        for (i = 0; i < dim; i++) { fscanf(myFile, "%f", &result[i]); }
        fclose(myFile) ;
        printf("init vector x(0) loaded\n") ;
    }
}

void print_vec(int n, float vec[]){
    //printf("Vector :\n|") ;
    printf("|");
    int i ;
    for (i = 0; i < n; i++) {
        printf(" %f |", vec[i]) ;
    }
    printf("\n\n") ;
}

void read_mat(int dim, float result[dim][dim]){
    FILE *myFile;
    int i,j ;
    myFile = fopen("mat.txt", "r");

    int size ;
    fscanf(myFile, "%d", &size);
    if (size != dim){
        printf("Size is not correct !");
    //    return 0 ;
    } else {
        for (i = 0; i < dim; i++) {
            for (j=0; j < dim; j++){
                fscanf(myFile, "%f", &result[i][j]);
            }
        }
        fclose(myFile);
        printf("matrix loaded\n") ;
    }
}

void print_mat(int n, int m, float mat[n][m]){
    printf("[") ;
    int i,j;
    for (i = 0; i < n; i++) {
        printf("|") ;
        for (j = 0; j < m; j++) {
            printf(" %f |", mat[i][j]);
        }
        if (i < n-1){
            printf("\n");
        }
    }
    printf("]\n\n") ;
}

void print_tensor(int n, int m, int q, float tens[n][m][q]){
    int i,j,k ;
    for (i = 0; i < n; i++) {
        printf("|") ;
        for (j = 0; j < m; j++) {
            printf("|") ;
            for (k=0; k<q; k++){
                printf(" %f |", tens[i][j][k]);
            }
            printf("\n") ;
        }
        //printf("\n");
    }
    printf("\n") ;
}

void write_solution(int dim, float sol[dim]){
    FILE *f = fopen("final.txt", "w");
    fprintf(f, "%d\n", dim);
    int i;
    for (i=0; i<dim; i++){
        fprintf(f, "%f ", sol[i]) ;
    }
    fclose(f);
}

int main (int argc, char* argv[]) {
    int my_rank, p ; /* Mon rang, nb de proc. */

    MPI_Status status ;
    /* On utilise MPI */
    MPI_Init(&argc, &argv) ;
    /* Qui suis-je ? */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) ;
    /* Combien sommes-nous ? */
    MPI_Comm_size(MPI_COMM_WORLD, &p) ;

    float epsilon = 0.001 ;// seuil de convergence
    int max_iter = 10; // nb d'iterations maximale, atteint faute de convergence
    int g = 3 ; // hauteur d'un bloc de lignes. On suppose que g divise dim.
    int total_blocs_count = dim / g ;// nombre total de blocs dans la matrice, ou dans un vecteur.
    // Si on fait la div. eucl. : total = k*p + r, les r premiers processeurs prennent en charge k+1 blocs de taille g,
    // les (p-r) derniers prennent en charge k blocs de taille g.

    int own_blocs_count = total_blocs_count / p ; // ci-dessus appelé k. On l'incrémente plus loin si le proc. est dans les "reste" premiers
    int reste = total_blocs_count % p ; // les r premiers proc. font un tour de plus
    int last = (p + reste-1)%p ; // l'identifiant du dernier processeur, qui devra transmettre le resultat de l'iteration à 0.

    float param[dim] ; // Vecteur b parametre
    float init[dim] ; // solution initiale x0

    float x_pred[dim] ; // coord. x(k) calculées pendant les itérations préedentes
    float x_new[dim] ; // où ce proc. stocke ses coord. de x(k+1)
    float matrix[dim][dim] ; // pas besoin de charger la matrice partout, par simplicité on l'initialise partout.
    float T[own_blocs_count][g] ; // pour stocker les valerus de t_i^(k). On a T[q][r] = t_{p*q + r}^(k)
    // Ici on n'a pas besoin de garder les valeurs après chaque itération, donc k n'apparait pas et on efface à chaque iteration.
    float Z[own_blocs_count][g] ;
    // Meme reflexion pour les z_i, avec k+1 cette fois ; Z[q][r] = z_{p*q+r}^(k+1)
    int sign ;

    printf("p%d waking up.\n", my_rank) ;
    fflush(stdout) ;

    if (my_rank < reste){
        own_blocs_count += 1 ;
    } // les r premiers proc. ont un bloc de plus que les autres

    float own_blocs[own_blocs_count][g][dim] ;
    /*
    own_blocs[q][r][j] = matrix[my_rank*g + q*p*g + r][j]
    //                    = matrix[(my_rank + q*p)*g + r][j]
    // On définit ainsi une permutation [q][r] qui va nous suivre par la suite.
    */

    float blocs_feeder[own_blocs_count][p*g][dim] ;
    /*
    // blocs_feeder[q][r][j] = matrix[q*p*g + r][j] ;
    // On partitionne la matrice en blocs de blocs pour pouvoir disperser chaque bloc
    // dans les own_blocs, avec la fonction MPI_Scatter.
    */

    if (my_rank == 0){
        printf("p0 : Is dimension = %d dividable by g = %d? ", dim, g) ;
        if (dim%g == 0){
            printf("Yes. Algorithm will begin.\n") ;
        }else{
            printf("No. Aborting.\n") ;
        }
        assert(dim%g == 0) ;

        printf("p0 : Is total blocs count (dim/g) = %d dividable by number of processors %d? ", total_blocs_count, p) ;
        if (reste == 0){
            printf("Yes. Last processor is %d.\n", last) ;
        }else{
            printf("No. Last processor is %d.\n", last) ;
        }

        printf("p0 : problem loading. dim : %d | g : %d | total_blocs_count : %d | own_blocs_count : %d | reste : %d | last : %d\n",
               dim, g, total_blocs_count, own_blocs_count, reste, last) ;

        read_param(dim, param) ;
        print_vec(dim, param) ;

        read_init(dim, init) ;
        print_vec(dim, init) ;

        read_mat(dim, matrix) ;
        print_mat(dim, dim, matrix) ;

        fflush(stdout) ;

        int i,j,k ;
        for (i=0 ; i < own_blocs_count; i++){
            for(j=0 ; j < p*g ; j++){
                for (k=0 ; k < dim ; k++){
                    blocs_feeder[i][j][k] = matrix[i*p*g + j][k] ;
                }
            }
        }
        printf("Matrix splitted. \n") ;
        //print_tensor(own_blocs_count, p*g, dim, blocs_feeder) ;
    }

    fflush(stdout) ;

    MPI_Bcast(param, dim, MPI_FLOAT, 0, MPI_COMM_WORLD) ;
    MPI_Bcast(init, dim, MPI_FLOAT, 0, MPI_COMM_WORLD) ;

    void scatter_all(){
        int i;
        for (i=0 ; i < own_blocs_count ; i++){
            MPI_Scatter(blocs_feeder[i], dim*g, MPI_FLOAT, own_blocs[i], dim*g, MPI_FLOAT, 0, MPI_COMM_WORLD) ;
        }
    }
    scatter_all() ;
    printf("p%d received matrix blocs.\n", my_rank) ;
    fflush(stdout) ;

    /*
    printf("Parameters vector (b):\n") ;
    print_vec(dim, param) ;
    printf("Initial vector (x0):\n") ;
    print_vec(dim, init) ;
    printf("Values used by me (p%d):\n", my_rank) ;
    print_tensor(own_blocs_count, g, dim, own_blocs) ;
    */

    void recopie_x0(){
        int i;
        for (i=0; i<dim; i++){
            x_pred[i] = init[i] ;
        }
    }
    recopie_x0() ;

    void init_x_new(){
        int i;
        for (i=0; i<dim; i++){
            x_new[i] = 0;
        }
    }
    init_x_new();

    void compute_ts(){
        //rappel : T[q][r] = t_{my_rank*g + q*p*g + r}^(k)

        int q,r ;
        for (q=0; q < own_blocs_count; q++){
            //printf("q = %d |", q);
            for (r=0; r < g; r++){
                //printf("r = %d |", r);
                float val = param[my_rank*g + q*p*g + r] ;
                //printf("init val = %f |", val) ;
                int j;
                for (j = my_rank*g + q*p*g + r + 1; j < dim; j++){
                    //printf("j = %d \n", j) ;
                    val -= own_blocs[q][r][j] * x_pred[j] ;
                    // matrix[my_rank*q + q*p*g + r][j]
                    //printf("ob[%d][%d][%d] = %f | x_pred[%d] = %f \n", q, r, j, own_blocs[q][r][j], j, x_pred[j]) ;
                    // i.e. a_{(m_r +pq)g+r,j} * x_j^(k)
                }
                T[q][r] = val ;
                // T[q][r] = t_{my_rank*g + q*p*g + r}^(k)
                //t_{pgq + r}^(k)
            }
        }
    }

    void init_zs(){
        int q,r ;
        for (q=0; q < own_blocs_count; q++){
            for(r=0; r < g; r++){
                Z[q][r] = 0;
            }
        }
    }

    void update_zs(int own_bloc_id, float new_x[dim]){
        /*
        Rappel : Z[q][r] = z_{my_rank*g + q*p*g + r}^(k+1)
        En dessinant la partie triangulaire inférieure de la matrice, soit les cofficients qui rentrent en compte
        // pour le calcul des z_i, et en entourant et indiquant le numéro de l'étape où tel coeffs sont utilisés pour
        // calculer tels z_i, on arrive à la formule (peu intuitive) suivante.

        // Noter qu'ici on doit avoir 0 <= own_bloc_id <= own_blocs_count - 2 ;
        // car lors du dernier calcul, il ne reste plus rien à anticiper.
        */
        int q, r;
        for (q = own_bloc_id+1; q < own_blocs_count; q++){
            // On appelle la fonction apres le calcul des x_i^(k+1) du bloc own_bloc_id
            // donc on ne regarde que les blocs suivants
            for (r=0; r < g; r++){
                if (own_bloc_id == 0){
                    // Les premieres sommes partielles sont plus courtes, ensuite leur longueur est stationnaire à p*g
                    int j;
                    printf("    p%d computing partial z_%d sum for 0th loop. %d values added, from %d to %d\n",
                           my_rank, my_rank*g + q*p*g +r, (my_rank+1)*g, 0, (my_rank+1)*g) ;

                    for (j=0; j < (my_rank+1)*g; j++){
                        Z[q][r] -= own_blocs[q][r][j] * new_x[j] ;
                        /*
                        printf("z_%d = Z[%d][%d] for p%d -= %f * %f = %f | ",
                               my_rank*g + p*g*q + r, q, r, my_rank, own_blocs[q][r][j], new_x[j], Z[q][r]);
                        */
                    }
                }else{
                    // Z[q][r] = z_{(my_rank+q*p)*g + r}^(k+1)
                    int i0 = (my_rank+q*p)*g ;
                    // Alors i := (my_rank+q*p)*g + r = i0 + r
                    int j;

                    printf("    p%d computing partial z_%d sum for loop %d. %d values added, from column %d to %d\n",
                           my_rank, i0+r, own_bloc_id, p*g, i0 - (p+1)*g, i0 - g) ;

                    for (j = i0 - (p+1)*g ; j < i0 - g ; j++){
                        /*
                        // (j = my_rank*g + (own_bloc_id-1)*p*g + g ; j < my_rank*g + own_bloc_id*p*g + g, j++)
                        // en notant my_rank = m, on a :
                        // j = mg + [1+(q-1)p]g, ..., mg + (1+qp)g - 1
                        // j = mg + (q-1)pg + g, ..., mg + qpg + g-1
                        // cela concerne donc les colonnes à partir de celle suivant l'indice [q-1][g-1],
                        // la derniere ligne du bloc q-1, jusqu'à la ligne [q][g-1] incluse.
                        // C'est bien la portion manquante à ce processeur, de longueur pg.
                        */
                        Z[q][r] -= own_blocs[q][r][j] * new_x[j] ;
                        /*
                        printf("z_%d = Z[%d][%d] for p%d -= %f*%f = %f ; column %d\n",
                               i0+r, q, r, my_rank, own_blocs[q][r][j], new_x[j], Z[q][r], j);
                        */
                    }
                }
                printf("        p%d's Z[%d][%d] = z_%d is now %f.\n",
                       my_rank, q, r, my_rank*g + p*g*q + r, Z[q][r]);
            }
        }
        fflush(stdout) ;
    }

    void compute_missing_z(int own_bloc_id, int z_num_r){
        // met à jour Z[own_bloc_id][z_num_r] à sa valeur finale z_i^(k+1) à l'iteration k
        // où i = my_rank*g + own_bloc_id*p*g + z_num_r
        if (own_bloc_id == 0){
            int target_i = my_rank*g + own_bloc_id*p*g + z_num_r;
            printf("    p%d computing entirely z_%d sum for 0th loop. %d values added, from column %d to %d\n",
                   my_rank, target_i, target_i, 0, target_i) ;

            int j;
            for (j=0; j < target_i; j++){
                Z[own_bloc_id][z_num_r] -= own_blocs[own_bloc_id][z_num_r][j] * x_new[j] ;
            }
        }else{
            int i0 = my_rank*g + own_bloc_id*p*g ;
            printf("    p%d completing z_%d sum for loop %d. %d values added, from column %d to %d\n",
                   my_rank, i0 + z_num_r, own_bloc_id, (p-1)*g + z_num_r, i0 - (p-1)*g, i0 + z_num_r) ;
            int j;
            for (j = i0 - (p-1)*g ; j < i0 + z_num_r; j++){
                // Le nombre de valeurs calculées va bien de (p-1)g pour r=0, jusqu'à pg pour r=g-1
                // voir schéma pour s'en convaincre.
                Z[own_bloc_id][z_num_r] -= own_blocs[own_bloc_id][z_num_r][j] * x_new[j] ;
            }
        }
    }

    sign = 0;
    /*
    A priori x(0) n'est pas solution
    Noter bien qu'ici la convention est chagée par rapport a l'article de ref
    Afin de calculer sign = new_sign * previous_sign.
    Une erreur le bloquerait donc à 0 tout le long du programme
    */
    int k = 0;
    while ((k < max_iter)&&(sign == 0)){
        sign = 1 ;
        init_zs() ;
        compute_ts() ;
        printf("p%d computed t_i^(%d) for his own values of i.\n", my_rank, k) ;
        /*
        print_mat(own_blocs_count, g, T) ;
        fflush(stdout) ;
        */
        int qb ; // analogue à q, renomme pour eviter les collapse avec l'autre update_zs()
        for (qb=0 ; qb < own_blocs_count; qb++){
            if ((qb > 0)||(my_rank > 0)){
                /*
                printf("p%d waiting to receive data from p%d\n", my_rank, (my_rank+ p-1)%p ) ;
                fflush(stdout);
                */
                MPI_Recv(x_new, qb*p*g + my_rank*g, MPI_FLOAT, (my_rank + p-1)%p, 50, MPI_COMM_WORLD, &status) ;
                MPI_Recv(&sign, 1, MPI_INT, (my_rank + p-1)%p, 100, MPI_COMM_WORLD, &status) ;
            }

            // compute x_{q*p + j*g}, ..., x_{q*p + (j+1)*g-1}
            // Then check if precision is reached for these coordinates and set sign accordingly
            int r;
            printf("p%d completing z's sums while computing x's values.\n", my_rank) ;
            for(r=0; r < g; r++){
                compute_missing_z(qb, r) ;
                int i = my_rank*g + qb*p*g + r;
                x_new[i] = (Z[qb][r] + T[qb][r])/own_blocs[qb][r][i] ;
                if (fabs(x_new[i] - x_pred[i]) > epsilon){ sign = 0; }
                /*
                printf("p%d computed x^(%d)_%d : %f. \n",
                        my_rank, k+1, i, x_new[i]) ;
                fflush(stdout) ;
                */
            }

            printf("p%d computed x^(%d)_i for %d <= i < %d. ",
                    my_rank, k+1, my_rank*g + qb*p*g, (my_rank+1)*g + qb*p*g);

            int dest ;
            if ((qb < own_blocs_count-1)||(my_rank != last)){
                dest = (my_rank + 1)%p ;
                printf("p%d waitind to send data to p%d.\n", my_rank, (my_rank+1)%p);
                fflush(stdout);

                MPI_Send(x_new, qb*p*g + (my_rank+1)*g, MPI_FLOAT, (my_rank + 1)%p, 50, MPI_COMM_WORLD) ;
                MPI_Send(&sign, 1, MPI_INT, (my_rank+1)%p, 100, MPI_COMM_WORLD) ;
            }else{
                if (my_rank != 0){
                    dest = 0;
                    printf("\np%d : End of iteration %d, sending result to p0.\n", my_rank, k);
                    fflush(stdout) ;

                    MPI_Send(x_new, dim, MPI_FLOAT, 0, 50, MPI_COMM_WORLD) ;
                    MPI_Send(&sign, 1, MPI_INT, 0, 100, MPI_COMM_WORLD) ;
                }else{
                    printf("\np%d : End of iteration %d. p0 (me) already has result since I am the last one.\n", my_rank, k);
                    fflush(stdout) ;
                }
            }

            printf("p%d completed loop %d and sent data to p%d. \n\n", my_rank, qb, dest);
            fflush(stdout);
            if (qb < own_blocs_count-1){
                printf("p%d updating partial sums of z's.\n", my_rank);
                update_zs(qb, x_new) ; // On ne perd pas de temps
                /*
                print_mat(own_blocs_count, g, Z);
                */
                fflush(stdout);
            }
        }
        if (my_rank == 0){ // receive result and broadcast it
            /*
            printf("p0 waiting to receive final iteration %d vector from p%d\n", k, last);
            fflush(stdout) ;
            */
            if (last != 0){
                MPI_Recv(x_new, dim, MPI_FLOAT, last , 50, MPI_COMM_WORLD, &status) ;
                MPI_Recv(&sign, 1, MPI_INT, last, 100, MPI_COMM_WORLD, &status) ;
            }

            printf("Sign is %d. Vector x(%d) computed. \n", sign, k+1);
            print_vec(dim, x_new) ;
            /*
            printf("p0 waiting to send broadcast\n");
            */
            fflush(stdout);

            if ((sign == 0)&&(k+1 < max_iter)){
                printf("p0 : -------------------------------------- Iteration %d will begin -----------------------------------\n\n", k+1) ;
            }

            void recopie_x(){
                int i;
                for (i=0; i<dim; i++){x_pred[i] = x_new[i] ;} // on recopie x(k+1) a la place de x(k)
            }
            recopie_x() ;

            MPI_Bcast(x_pred, dim, MPI_FLOAT, 0, MPI_COMM_WORLD) ;
            MPI_Bcast(&sign, 1, MPI_INT, 0, MPI_COMM_WORLD) ;

        }else{
            /*
            printf("p%d waiting to receive broadcast from p0\n", my_rank);
            fflush(stdout);
            */
            MPI_Bcast(x_pred, dim, MPI_FLOAT, 0, MPI_COMM_WORLD) ; // tous les processeurs recopient x(k+1) a la place de x(k)
            MPI_Bcast(&sign, 1, MPI_INT, 0, MPI_COMM_WORLD) ;
        }
        /*
        printf("Final values of p%d's Z's :\n", my_rank);
        print_mat(own_blocs_count, g, Z) ;
        */
        fflush(stdout);

        k++ ;
    }
    if (my_rank == 0){
        if (sign == 1){
            printf("p0 : Algorithm converged with precision %f, in %d iterations\nResult (written in final.txt) :\n", epsilon, k) ;
            print_vec(dim, x_new);
            write_solution(dim, x_new);
        }else{
            printf("p0 : Algorithm diverged with precision %f. %d iterations completed.\nFollowing result will not be written on drive:\n", epsilon, k) ;
            print_vec(dim, x_new);
        }
    }

    printf("p%d ended.\n", my_rank) ;
    fflush(stdout) ;

    MPI_Finalize() ;
    return 0 ;
}
