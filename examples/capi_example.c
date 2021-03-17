typedef long long int i64;


#define NN 50000

typedef struct {
    double x;
    double y;
} Field;

typedef struct {
    i64 order;
    Field field[NN];
} Multipole;


i64 Multipole_get_order(Multipole* mult){ return   mult->order; };
void Multipole_set_order(Multipole* mult, i64 val){  mult->order=val; };
Field* Multipole_get_field(Multipole* mult, i64 i){ return  &(mult->field[i]); };

double Multipole_get_field_normal(Multipole* mult, i64 i){ return mult->field[i].x; };
void Multipole_set_field_normal(Multipole* mult, i64 i, double val){  mult->field[i].x=val; };

double Multipole_get_field_skew(Multipole* mult, i64 i){ return mult->field[i].y; };
void Multipole_set_field_skew(Multipole* mult, i64 i, double val){  mult->field[i].y=val; };


//xobjects
typedef struct{} XField;
typedef struct{} XMultipole;

i64 XMultipole_get_order(XMultipole* mult){ return ((i64 *) mult)[0]; };
void XMultipole_set_order(XMultipole* mult, i64 val ){ ((i64 *) mult)[0]=val; };

XField* XMultipole_get_field(XMultipole* mult, i64 i){ return (XField*) ( ((i64*)mult) [1+2*i]); };

double XMultipole_get_field_normal(XMultipole* mult, i64 i){ return  ((double *) mult)[1+2*i]; };
void XMultipole_set_field_normal(XMultipole* mult, i64 i, double val ){ ((double *) mult)[1+2*i]=val; };

double XMultipole_get_field_skew(XMultipole* mult, i64 i){ return  ((double *) mult)[1+2*i+1]; };
void XMultipole_set_field_skew(XMultipole* mult, i64 i, double val ){ ((double *) mult)[1+2*i+1]=val; };

double Multipole_f(Multipole* restrict mult){
    int l = mult->order;
    double res=0;
    for (int i=0; i<l; i++){
        res+=mult->field[i].x+mult->field[i].y;
    }
    return res;
};

double XMultipole_f(XMultipole* restrict mult){
    int l = XMultipole_get_order(mult);
    double res=0;
    for (int i=0; i<l; i++){
        res+=XMultipole_get_field_normal(mult,i) +
             XMultipole_get_field_skew(mult,i);
    }
    return res;
};

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main (void){
    int begin;
    double r1,r2;
    Multipole* mult1 = (Multipole*) malloc(sizeof(Multipole));
    XMultipole* mult2 = (XMultipole*) malloc(sizeof(Multipole));

    Multipole_set_order(mult1,NN);
    XMultipole_set_order(mult2,NN);

    for(int i; i<NN; i++){
       Multipole_set_field_normal(mult1,i,i);
       Multipole_set_field_skew(mult1,i,2*i);
//       printf("%g %g\n",Multipole_get_field_normal(mult1,i), Multipole_get_field_skew(mult1,i));
    };

    for(int i; i<NN; i++){
       XMultipole_set_field_normal(mult2,i,1*i);
       XMultipole_set_field_skew(mult2,i,2*i);
//       printf("%g %g\n",XMultipole_get_field_normal(mult2,i), XMultipole_get_field_skew(mult2,i));
    };


    printf("%d\n",Multipole_get_order(mult1));
    printf("%d\n",XMultipole_get_order(mult2));

    begin=clock();
    r1=Multipole_f(mult1);
    printf("%30.25e %g\n",(double)(clock() - begin) / CLOCKS_PER_SEC, r1);

    begin=clock();
    r2=XMultipole_f(mult2);
    printf("%30.25e %g\n",(double)(clock() - begin) / CLOCKS_PER_SEC, r2);

    return 1;
};
