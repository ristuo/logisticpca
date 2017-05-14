#include<iostream>
#include "armadillo"
#include <stdlib.h>
#include <unistd.h>
#include "logisticpca.h"
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "USAGE: " << argv[0] << " <data csv path>";
        std::cout << " <output file path>" << std::endl;
        exit(EXIT_FAILURE);
    }
    arma::mat data;
    if (!data.load(argv[1], arma::csv_ascii))
    {
        if (access(argv[1], F_OK) == -1)
        {
            std::cout << "The file " << argv[1] << " does not exists!";
            std::cout << std::endl;
        }
        else
        {
            std::cout << "Failed to data as csv from file " << argv[1];
            std::cout << " probably due to formating issues";
            std::cout << std::endl;
        }
        exit(EXIT_FAILURE);
    }
    data = data;
    arma::mat u = logisticpca(data);
    arma::mat res =  data * u;
    res.save(argv[2], arma::csv_ascii);
    return 0;
}
