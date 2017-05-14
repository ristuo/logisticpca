#!/bin/bash
Rscript R/write.R
./build/main "datasets/votes.csv" "results/omatulos.csv"
Rscript R/plotty.R
