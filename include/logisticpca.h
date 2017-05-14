#ifndef LOGISTICPCA
#define LOGISTICPCA
#include "armadillo"
arma::mat logisticpca(const arma::mat &x, long int k = 2, 
                      double epsilon = 0.000000001, double m = 4, 
                      long int max_iter = 1000,
                      bool verbose = false, 
                      long int dev_check_frequency = 100);
#endif
