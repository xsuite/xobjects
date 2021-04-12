typedef long long int i64;


#define NN 50000

typedef struct {
    double x;
    double y;
} B;

typedef struct {
    i64 a;
    B b[NN];
} S;


typedef struct{} Bs;
typedef struct{} Ss;

i64 Ss_get_a(Ss* s){ return   ((i64 *) s)[0]; };
i64 S_get_a(S* s){ return   s->a; };

void Ss_set_a(Ss* s, i64 val ){ ((i64 *) s)[0]=val; };
void S_set_a(S* s, i64 val){  s->a=val; };


Bs* Ss_get_b(Ss* s, i64 i){ return   (Bs*) ( ((i64*)s) [1+2*i]); };
B* S_get_b(S* s, i64 i){ return  &(s->b[i]); };

double Ss_get_b_x(Ss* s, i64 i){ return  ((double *) s)[1+2*i]; };
double Ss_get2_b_x(Ss* s, i64 i){ return ((double*) Ss_get_b(s,i))[0]; };
double S_get_b_x(S* s, i64 i){ return s->b[i].x; };

void Ss_set_b_x(Ss* s, i64 i, double val ){ ((double *) s)[1+2*i]=val; };
void S_set_b_x(S* s, i64 i, double val){  s->b[i].x=val; };

double Ss_get_b_y(Ss* s, i64 i){ return  ((double *) s)[1+2*i+1]; };
double Ss_get2_b_y(Ss* s, i64 i){ return ((double*) Ss_get_b(s,i))[1]; };
double S_get_b_y(S* s, i64 i){ return s->b[i].y; };

void Ss_set_b_y(Ss* s, i64 i, double val ){ ((double *) s)[1+2*i+1]=val; };
void S_set_b_y(S* s, i64 i, double val){  s->b[i].y=val; };

double S_f(S* restrict s){
    int l = s->a;
    double res=0;
    for (int i=0; i<l; i++){
        res+=s->b[i].x+s->b[i].y;
    }
    return res;
};


double Ss_f(Ss* restrict s){
    int l = Ss_get_a(s);
    double res=0;
    for (int i=0; i<l; i++){
        res+=Ss_get_b_x(s,i)+Ss_get_b_y(s,i);
    }
    return res;
};

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main (void){
    int begin;
    double r1,r2;
    S* s1 = (S*) malloc(sizeof(S));
    Ss* s2 = (Ss*) malloc(sizeof(S));

    S_set_a(s1,NN);
    Ss_set_a(s2,NN);

    for(int i; i<NN; i++){
       S_set_b_x(s1,i,i);
       S_set_b_y(s1,i,2*i);
//       printf("%g %g\n",S_get_b_x(s1,i), S_get_b_y(s1,i));
    };

    for(int i; i<NN; i++){
       Ss_set_b_x(s2,i,1*i);
       Ss_set_b_y(s2,i,2*i);
//       printf("%g %g\n",Ss_get_b_x(s2,i), Ss_get_b_y(s2,i));
    };


    printf("%d\n",S_get_a(s1));
    printf("%d\n",Ss_get_a(s2));

    begin=clock();
    r1=S_f(s1);
    printf("%30.25e %g\n",(double)(clock() - begin) / CLOCKS_PER_SEC, r1);

    begin=clock();
    r2=Ss_f(s2);
    printf("%30.25e %g\n",(double)(clock() - begin) / CLOCKS_PER_SEC, r2);

    return 1;
};
