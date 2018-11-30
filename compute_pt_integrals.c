#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>

// Code to compute the 1-loop perturbation theory integrals required for the FullShape model. This is performed in C for speed
// and simply compiled and called from python using the command line and STDIN/STDOUT. It requires the gsl libraries to be
// installed (or loaded on whatever machine you are using)

// Global parameters and arrays
int NK;
char *Pk_file, *Output_file;
double pkkmin;         // The minimum kmin to integrate over, based on the input power spectrum file
double pkkmax;         // The maximum k in the input power spectrum. The maximum k to integrate over is the smallest of this or kmax
gsl_spline * pk_spline;
gsl_interp_accel * pk_acc;

// Prototypes
void read_power();
double Pk_interp(double x);
double G_func(int n, int m, double r);
double pk_integrand(double q, void * p);
double Inm_inner_integrand(double x, void * p);
double Inm_outer_integrand(double r, void * p);
double Jnm_integrand(double q, void * p);
double Knm_inner_integrand(double x, void * p);
double Knm_outer_integrand(double r, void * p);
double sigma3_inner_integrand(double x, void * p);
double sigma3_outer_integrand(double r, void * p);
double sigma_v_integrand(double q, void * p);
double F_func(int n, int m, double r, double x, double y);
double K_func(int n, int m, int s, double r, double x, double y);
void printProgress (double percentage);

static struct option long_options[] =
{
    {"infile",  required_argument, NULL, 'a'},
    {"outfile", required_argument, NULL, 'b'},
    {"koutmin", optional_argument, NULL, 'c'},
    {"koutmax", optional_argument, NULL, 'd'},
    {"nkout",   optional_argument, NULL, 'e'},
    {NULL, 0, NULL, 0}
};

// The main driver routine
int main(int argc, char **argv) {

    // This code is configured to take k and pk as the inputs from an input file and output the evaluations of the perturbation
    // theory integrals to another file. Additional, optional inputs from are the minimum and maximum k-values and the number of k-bins for the output. 

    // Parse the command line arguments for all the necessary inputs
    FILE * fout;
    int i, ch, nkout = 40;
    unsigned long nevals;
    double result, error, koutmin = 0.01, koutmax = 1.0;
    while ((ch = getopt_long_only(argc, argv, "a:b:c::d::e::", long_options, NULL)) != -1) {
        switch(ch) {
            case 'a':
                Pk_file = optarg;
                break;
            case 'b':
                Output_file = optarg;
                break;
            case 'c':
                koutmin = atof(optarg);
                break;
            case 'd':
                koutmax = atof(optarg);
                break;
            case 'e':
                nkout = atoi(optarg);
                break;
            case '?':
                break;
            default:
                abort();
        }
    }
    
    if (koutmin < 1.0e-10) {
        printf("Error: koutmin close to or equal to zero, and so can't be logged, please choose a more suitable value\n");\
        exit(1);
    }

    // Read in the input power spectrum
    read_power();

    // Open the output file
    if(!(fout = fopen(Output_file, "w"))) {
        printf("\nERROR: Can't open output file '%s'.\n\n", Output_file);
        exit(0);
    }
    fprintf(fout, "#   kval   P_lin   I_00   I_01   I_02   I_03   I_10   I_11   I_12   I_13   I_20   I_21   I_22   I_23   I_30   I_31   I_32   I_33   J_00   J_01   J_02   J_10   J_11   J_20   K_00   K_00_s   K_01   K_01_s   K_02_s   K_10   K_10_s   K_11   K_11_s   K_20   K_20_s   K_30   K_30_s   sigma_3\n");

    // Perform all the integrations
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;

    F.function = &pk_integrand;
    gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
    double norm = result/(2.0*M_PI*M_PI);

    double kwidth = (log10(koutmax) - log10(koutmin))/nkout;
    for (i=0; i<=nkout; i++) {
        double kval = pow(10.0,log10(koutmin) + i*kwidth);

        double P_lin = Pk_interp(kval);

        double parameters[3] = {0, 0, kval};
        F.function = &Inm_outer_integrand;
        F.params = &parameters;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_00 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 0;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_01 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 0;
        parameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_02 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 0;
        parameters[1] = 3;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_03 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_10 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_11 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_12 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 3;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_13 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 2;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_20 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 2;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_21 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 2;
        parameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_22 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 2;
        parameters[1] = 3;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_23 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 3;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_30 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 3;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_31 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 3;
        parameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_32 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        parameters[0] = 3;
        parameters[1] = 3;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double I_33 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        F.function = &Jnm_integrand;

        parameters[0] = 0;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_00 = result/(2.0*M_PI*M_PI);

        parameters[0] = 0;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_01 = result/(2.0*M_PI*M_PI);

        parameters[0] = 0;
        parameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_02 = result/(2.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_10 = result/(2.0*M_PI*M_PI);

        parameters[0] = 1;
        parameters[1] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_11 = result/(2.0*M_PI*M_PI);

        parameters[0] = 2;
        parameters[1] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double J_20 = result/(2.0*M_PI*M_PI);

        double Kparameters[4] = {0, 0, 0, kval};
        F.function = &Knm_outer_integrand;
        F.params = &Kparameters;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_00 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_00_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[1] = 1;
        Kparameters[2] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_01 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);
        K_01 -= norm;

        Kparameters[1] = 1;
        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_01_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);
        K_01_s -= 4.0/9.0*norm;

        Kparameters[1] = 2;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_02_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);
        K_02_s -= 2.0/3.0*norm;

        Kparameters[0] = 1;
        Kparameters[1] = 0;
        Kparameters[2] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_10 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_10_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[1] = 1;
        Kparameters[2] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_11 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_11_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[0] = 2;
        Kparameters[1] = 0;
        Kparameters[2] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_20 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_20_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[0] = 3;
        Kparameters[1] = 0;
        Kparameters[2] = 0;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_30 = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        Kparameters[2] = 1;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double K_30_s = (kval*kval*kval*result)/(4.0*M_PI*M_PI);

        double sigma3_parameters[1] = {kval};
        F.function = &sigma3_outer_integrand;
        F.params = &sigma3_parameters;
        gsl_integration_cquad(&F, pkkmin, pkkmax, 0, 1.0e-6, w, &result, &error, &nevals);
        double sigma_3 = (105.0/16.0)*(kval*kval*kval*result)/(4.0*M_PI*M_PI);
        
        double perc = (double)i/(double)nkout;
        printProgress(perc);
        if (i == nkout) {
            printf("\n");
        }

        fprintf(fout, "%g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g  %g\n", kval, P_lin, I_00, I_01, I_02, I_03, I_10, I_11, I_12, I_13, I_20, I_21, I_22, I_23, I_30, I_31, I_32, I_33, J_00, J_01, J_02, J_10, J_11, J_20, K_00, K_00_s, K_01, K_01_s, K_02_s, K_10, K_10_s, K_11, K_11_s, K_20, K_20_s, K_30, K_30_s, sigma_3);

    }

    gsl_integration_cquad_workspace_free(w);

    gsl_interp_accel_free(pk_acc);
    gsl_spline_free(pk_spline);
    fclose(fout);

    return 0;
}

double pk_integrand(double q, void * p) {

    double * params = (double * )p;
    unsigned long nevals;
    double result, error;
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;

    double P_lin = Pk_interp(q);

    gsl_integration_cquad_workspace_free(w);

    return q*q*P_lin*P_lin;
}

double Jnm_integrand(double q, void * p) {

    double * params = (double * )p;
    int n = (int)params[0];
    int m = (int)params[1];
    double kval = params[2];
    return G_func(n, m, q/kval)*Pk_interp(q);

}

double Inm_outer_integrand(double r, void * p) {

    double * params = (double * )p;
    unsigned long nevals;
    int n = (int)params[0];
    int m = (int)params[1];
    double kval = params[2];
    double result, error;

    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &Inm_inner_integrand;
    double parameters[4] = {n, m, kval, r};
    F.params = &parameters;
    gsl_integration_cquad(&F, -1.0, 1.0, 0, 1.0e-6, w, &result, &error, &nevals);
    gsl_integration_cquad_workspace_free(w);
    return r*r*Pk_interp(kval*r)*result;

}

double Inm_inner_integrand(double x, void * p) {

    double * params = (double * )p;
    int n = (int)params[0];
    int m = (int)params[1];
    double kval = params[2];
    double rval = params[3];
    double result, error;

    double y = sqrt(1.0 + rval*rval - 2.0*rval*x);
    return F_func(n, m, rval, x, y)*Pk_interp(kval*y);

}

double Knm_outer_integrand(double r, void * p) {

    double * params = (double * )p;
    unsigned long nevals;
    int n = (int)params[0];
    int m = (int)params[1];
    int s = (int)params[2];
    double kval = params[3];
    double result, error;

    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &Knm_inner_integrand;
    double parameters[5] = {n, m, s, kval, r};
    F.params = &parameters;
    gsl_integration_cquad(&F, -1.0, 1.0, 0, 1.0e-6, w, &result, &error, &nevals);
    gsl_integration_cquad_workspace_free(w);
    return r*r*Pk_interp(kval*r)*result;

}

double Knm_inner_integrand(double x, void * p) {

    double * params = (double * )p;
    int n = (int)params[0];
    int m = (int)params[1];
    int s = (int)params[2];
    double kval = params[3];
    double rval = params[4];
    double result, error;

    double y = sqrt(1.0 + rval*rval - 2.0*rval*x);
    return K_func(n, m, s, rval, x, y)*Pk_interp(kval*y);

}

double sigma3_outer_integrand(double r, void * p) {

    double * params = (double * )p;
    unsigned long nevals;
    double kval = params[0];
    double result, error;

    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &sigma3_inner_integrand;
    double parameters[2] = {kval, r};
    F.params = &parameters;
    gsl_integration_cquad(&F, -1.0, 1.0, 0, 1.0e-6, w, &result, &error, &nevals);
    gsl_integration_cquad_workspace_free(w);
    return r*r*Pk_interp(kval*r)*result;

}

double sigma3_inner_integrand(double x, void * p) {

    double * params = (double * )p;
    double kval = params[0];
    double rval = params[1];
    double result, error;

    double y = sqrt(1.0 + rval*rval - 2.0*rval*x);
    double numer = (2.0/7.0)*(x*x - 1.0)*(3.0*x*x - 4.0*rval*x + 2.0*rval*rval - 1.0);
    return numer/(3.0*y*y) + 8.0/63.0;

}

double Pk_interp(double x) {

    /*int low, middle, high;

    if ((x < pkkmin) || (x > pkkmax)) {
        return 0;
    } 
    
    low = 0;
    high = NK-1;
    middle = (high+low)/2;
    while (low < high-1) {
        if (x > karray[middle]) {
            low = middle;
        } else {
            high = middle;
        }
        middle = (high+low)/2;
    }

    double fdiff = (x - karray[low])/(karray[high] - karray[low]);
    double pkinterp = log10(pkarray[low]) + fdiff*(log10(pkarray[high])-log10(pkarray[low]));
    return pow(10.0,pkinterp);*/

    if ((x < pkkmin) || (x > pkkmax)) {
        return 0;
    } else { 
        return pow(10.0,gsl_spline_eval(pk_spline, log10(x), pk_acc));
    }

}

double F_func(int n, int m, double r, double x, double y) {

    if (n == 0) {
        if (m == 0) {
            double numer = 7.0*x + 3.0*r - 10.0*r*x*x;
            return (numer*numer)/(14.0*14.0*r*r*y*y*y*y);
        } else if (m == 1) {
            double numer = (7.0*x + 3.0*r - 10.0*r*x*x)*(7.0*x - r - 6.0*r*x*x);
            return numer/(14.0*14.0*r*r*y*y*y*y);
        } else if (m == 2) {
            double numer = (x*x - 1.0)*(7.0*x + 3.0*r - 10.0*r*x*x);
            return numer/(14.0*r*y*y*y*y);
        } else if (m == 3) {
            double numer = (1.0 - x*x)*(3.0*r*x - 1.0);
            return numer/(r*r*y*y);
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 1) {
        if (m == 0) {
            double numer = x*(7.0*x + 3.0*r - 10.0*r*x*x);
            return numer/(14.0*r*r*y*y);
        } else if (m == 1) {
            double numer = 7.0*x - r - 6.0*r*x*x;
            return (numer*numer)/(14.0*14.0*r*r*y*y*y*y);
        } else if (m == 2) {
            double numer = (x*x - 1.0)*(7.0*x - r - 6.0*r*x*x);
            return numer/(14.0*r*y*y*y*y);
        } else if (m == 3) {
            double numer = 4.0*r*x + 3.0*x*x - 6.0*r*x*x*x - 1.0;
            return numer/(2.0*r*r*y*y); 
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 2) {
        if (m == 0) {
            double numer = (2.0*x + r - 3.0*r*x*x)*(7.0*x + 3.0*r - 10.0*r*x*x);
            return numer/(14.0*r*r*y*y*y*y);
        } else if (m == 1) {
            double numer = (2.0*x + r - 3.0*r*x*x)*(7.0*x - r - 6.0*r*x*x);
            return numer/(14.0*r*r*y*y*y*y);
        } else if (m == 2) {
            double numer = x*(7.0*x - r - 6.0*r*x*x);
            return numer/(14.0*r*r*y*y);
        } else if (m == 3) {
            double numer = 3.0*(1.0-x*x)*(1.0-x*x);
            return numer/(y*y*y*y);
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 3) {
        if (m == 0) {
            double numer = 1.0 - 3.0*x*x - 3.0*r*x + 5.0*r*x*x*x;
            return numer/(r*r*y*y);
        } else if (m == 1) {
            double numer = (1.0 - 2*r*x)*(1.0 - x*x);
            return numer/(2.0*r*r*y*y);
        } else if (m == 2) {
            double numer = (1.0 - x*x)*(2.0 - 12.0*r*x - 3.0*r*r + 15.0*r*r*x*x);
            return numer/(r*r*y*y*y*y);
        } else if (m == 3) {
            double numer = -4.0 + 12.0*x*x + 24.0*r*x - 40.0*r*x*x*x + 3.0*r*r - 30.0*r*r*x*x + 35.0*r*r*x*x*x*x;
            return numer/(r*r*y*y*y*y);
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else {
        printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
        exit(0);
    }
}

double G_func(int n, int m, double r) {

    if (n == 0) {
        if (m == 0) {
            return (12.0/(r*r) - 158.0 + 100.0*r*r - 42.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(7.0*r*r + 2.0)*log((r + 1.0)/fabs(r - 1.0)))/3024.0;
        } else if (m == 1) {
            return (24.0/(r*r) - 202.0 + 56.0*r*r - 30.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(5.0*r*r + 4.0)*log((r + 1.0)/fabs(r - 1.0)))/3024.0;
        } else if (m == 2) {
            return (2.0*(r*r + 1.0)*(3.0*r*r*r*r - 14.0*r*r + 3.0)/(r*r) - (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*log((r + 1.0)/fabs(r - 1.0)))/224.0;
        } else {
            printf("Invalid choice of n=%d and m=%d for G_func\n", n, m);
            exit(0);
        }
    } else if (n == 1) {
        if (m == 0) {
            return (-38.0 +48.0*r*r - 18.0*r*r*r*r + (9.0/r)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*log((r + 1.0)/fabs(r - 1.0)))/1008.0;
        } else if (m == 1) {
            return (12.0/(r*r) - 82.0 + 4.0*r*r - 6.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r + 2.0)*log((r + 1.0)/fabs(r - 1.0)))/1008.0;
        } else {
            printf("Invalid choice of n=%d and m=%d for G_func\n", n, m);
            exit(0);
        }
    } else if (n == 2) {
        if (m == 0) {
            return (2.0*(9.0 - 109.0*r*r + 63.0*r*r*r*r - 27.0*r*r*r*r*r*r)/(r*r) + (9.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(3.0*r*r + 1.0)*log((r + 1.0)/fabs(r - 1.0)))/672.0;
        } else {
            printf("Invalid choice of n=%d and m=%d for G_func\n", n, m);
            exit(0);
        }
    } else {
        printf("Invalid choice of n=%d and m=%d for G_func\n", n, m);
        exit(0);
    }
}

double K_func(int n, int m, int s, double r, double x, double y) {

    if (n == 0) {
        if (m == 0) {
            if (s == 0) {
                double numer = 7.0*x + 3.0*r - 10.0*r*x*x;
                return numer/(14.0*r*y*y);
            } else if (s == 1) {
                double numer = (7.0*x + 3.0*r - 10.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return numer/(14.0*3.0*r*y*y*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else if (m == 1) {
            if (s == 0) {
                return 1.0;
            } else if (s == 1) {
                double numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return (numer*numer)/(3.0*3.0*y*y*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else if (m == 2) {
            if (s == 1) {
                double numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return numer/(3.0*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 1) {
        if (m == 0) {
            if (s == 0) {
                double numer = 7.0*x - r - 6.0*r*x*x;
                return numer/(14.0*r*y*y);
            } else if (s == 1) {
                double numer = (7.0*x - r - 6.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return numer/(14.0*3.0*r*y*y*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else if (m == 1) {
            if (s == 0) {
                return x/r;
            } else if (s == 1) {
                double numer = (3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0)*x/r;
                return numer/(3.0*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 2) {
        if (m == 0) {
            if (s == 0) {
                double numer = x*x - 1.0;
                return numer;
            } else if (s == 1) {
                double numer = (x*x - 1.0)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return numer/(3.0*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else if (n == 3) {
        if (m == 0) {
            if (s == 0) {
                double numer = 2.0*x + r - 3.0*r*x*x;
                return numer;
            } else if (s == 1) {
                double numer = (2.0*x + r - 3.0*r*x*x)*(3.0*x*x - 4.0*r*x + 2.0*r*r - 1.0);
                return numer/(3.0*y*y);
            } else {
                printf("Invalid choice of n=%d, m=%d and s=%d for K_func\n", n, m, s);
                exit(0);
            }
        } else {
            printf("Invalid choice of n=%d and m=%d for F_func\n", n, m);
            exit(0);
        }
    } else {
        printf("Invalid choice of n=%d and m=%d for G_func\n", n, m);
        exit(0);
    }
}

// Routine to read in the linear power spectrum.
void read_power() {
    
    FILE * fp;
    char buf[500];
    int i;

    if(!(fp = fopen(Pk_file, "r"))) {
        printf("\nERROR: Can't open power file '%s'.\n\n", Pk_file);
        exit(0);
    }

    NK = 0;
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, tpk;
            if(sscanf(buf, "%lf %lf\n", &tk, &tpk) != 2) {printf("Pk read error\n"); exit(0);};
            NK++;
        }
    }
    fclose(fp);

    double *karray = (double *)calloc(NK, sizeof(double));
    double *pkarray = (double *)calloc(NK, sizeof(double));

    NK = 0;
    fp = fopen(Pk_file, "r");
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, tpk;
            if(sscanf(buf, "%lf %lf\n", &tk, &tpk) != 2) {printf("Pk read error\n"); exit(0);};
            karray[NK] = log10(tk);
            pkarray[NK] = log10(tpk);
            NK++;
        }
    }
    fclose(fp);

    pkkmin = pow(10.0,karray[0]);
    pkkmax = pow(10.0,karray[NK-1]);

    pk_acc = gsl_interp_accel_alloc();
    pk_spline = gsl_spline_alloc(gsl_interp_cspline, NK);
    gsl_spline_init(pk_spline, karray, pkarray, NK);

    return;
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 50

void printProgress (double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}