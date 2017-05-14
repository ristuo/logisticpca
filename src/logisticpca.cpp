#include "logisticpca.h"
#include <math.h>
#include <stdlib.h>
using namespace arma;

double sigmoid(double x)
{
    return 1.0 / (1.0 + (1.0 / exp(x)));
}

double logit(double x)
{
    return log(x / (1.0 - x));
}

vec initialize_mu(const mat &x)
{
    vec res(x.n_cols);
    for (long int j = 0; j < x.n_cols; j++) 
    {
        double colmean = 0;
        for (long int i = 0; i < x.n_rows; i++)
        {
            colmean += x.at(i, j);
        }
        colmean = colmean / (double)x.n_rows;
        res(j) = logit(colmean);
    }
    return res;
}

mat find_q(const mat &x)
{
    mat res(x.n_rows, x.n_cols); 
    for (long int i = 0; i < res.n_rows; i++)
    {
        for (long int j = 0; j < res.n_cols; j++)
        {
            res(i, j) = 2.0 * x(i, j) - 1.0;
        }
    }
    return res;
}

mat initialize_u(const mat &x, const mat &q, long int k)
{
    mat u;
    vec s;
    mat v;
    if (!svd(u, s, v, q))
    {
        std::cout << "Error in computing svd when intilizing U!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return v.head_cols(k); 
}

mat initialize_u(const mat &x, long int k)
{
    return initialize_u(x, find_q(x), k);
}

double find_theta_ij(long int i, long int j, const vec &mu_t0, 
                     const mat &u_t0, const mat &theta_tilde)
{
    mat rhs = (u_t0 * u_t0.t() * (theta_tilde.rows(i, i).t() - mu_t0));
    return mu_t0(j) + rhs(j, 0);
}

mat find_z(const mat &x, const vec &mu_t0, const mat &u_t0, 
           const mat &theta_tilde)
{
    mat res = mat(x.n_rows, x.n_cols);
    for (long int i = 0; i < res.n_rows; i ++)
    {
        for (long int j = 0; j < res.n_cols; j++)
        {
            double theta_ij = find_theta_ij(i, j, mu_t0, u_t0, theta_tilde);
            res(i, j) = theta_ij + 4.0 * (x(i,j) - sigmoid(theta_ij));
        }
    }
    return res;
}

vec find_mu(const mat &x, const mat &z, const mat &theta_tilde, 
            const mat &u_t0)
{
    vec ones = vec(x.n_rows); 
    ones.ones();
    mat tmp = z - theta_tilde * u_t0 * u_t0.t();
    vec result = (1.0 / x.n_rows) * tmp.t() * ones;
    return result;
}

mat find_u(const mat &x, const mat &z, const mat &theta_tilde, 
           const vec &mu_t, long int k)
{
    vec ones_n = vec(x.n_rows);
    ones_n.ones();
    mat t_tilde_minus_mu = theta_tilde - ones_n * mu_t.t();
    mat z_minus_mu = z - ones_n * mu_t.t();
    mat fst_term = t_tilde_minus_mu.t() * z_minus_mu;
    mat snd_term = z_minus_mu.t() * t_tilde_minus_mu;
    mat trd_term = t_tilde_minus_mu.t() * t_tilde_minus_mu;
    mat tmp = fst_term + snd_term - trd_term;
    mat e;  
    vec d;
    eig_sym(d, e, tmp);
    return e.head_cols(k);
}

double trace_inner_prod(const mat &a, const mat &b)
{
    mat tmp = a.t() * b;
    return trace(tmp);
}

double deviance(const mat &x, const vec &mu, const mat &theta_tilde, 
                const mat &u)
{
    vec ones_n = vec(x.n_rows);
    ones_n.ones();
    mat rmat = ones_n * mu.t() + (theta_tilde - ones_n * mu.t()) * u * u.t();
    double fst_term = (-2.0) * trace_inner_prod(x, rmat);
    
    double snd_term = 0.0;
    for (long int i = 0; i < x.n_rows; i++)
    {
        for (long int j = 0; j < x.n_cols; j++) 
        {
            mat uut = u * u.t() * (theta_tilde.rows(i,i).t() - mu);
            double expterm = mu(j) + uut(j);
            snd_term += log(1.0 + exp(expterm));
        }
    }
    snd_term *= 2.0;
    return fst_term + snd_term;
}

mat logisticpca(const mat &x, long int k, double epsilon, double m,
                long int max_iter, bool verbose,
                long int dev_check_frequency) 
{
    vec mu = initialize_mu(x);
    mat q = find_q(x);
    mat u = initialize_u(x, q, k);
    mat theta_tilde = m * q;
    double dev_t0 = deviance(x, mu, theta_tilde, u);
    double dev_t = dev_t0;
    for (int i = 0; i < max_iter; i++) 
    {
        mat z = find_z(x, mu, u, theta_tilde);
        mu = find_mu(x, z, theta_tilde, u);
        u = find_u(x, z, theta_tilde, mu, k);
        if (i % dev_check_frequency == 0)
        {
            dev_t = deviance(x, mu, theta_tilde, u);
            if (abs(dev_t0 - dev_t) < epsilon)
            {
                if (verbose)
                {
                    std::cout << "Reached convergence after " << i;
                    std::cout << " iterations" << std::endl;
                }
                return(u);
            }
            dev_t0 = dev_t;
            if (verbose)
            {
                std::cout << "After " << i << " iterations ";
                std::cout << "deviance was " << dev_t << std::endl;
            }
        }
    }
    return u;
}
