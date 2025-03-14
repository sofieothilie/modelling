#include <stdio.h>


const int N = 10, M = 10;
FILE* parf;

void printState() {
    for (int y = 0; y < N; y++) {
        //print parf
        double next = 0;

        //print seqf
        for (int x = 0; x < M; x++) {
            fread(&next, sizeof(double), 1, parf);
            printf("%f ", next);
        }
        printf("\n");

    }
}


int main(int argc, char **argv){
    char parstr[100] = {0};
    sprintf(parstr, "data/%s.dat", argv[1]);


    parf = fopen(parstr, "r");

    printState();

}
