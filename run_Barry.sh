#!/bin/bash

for i in {19080..19099}
do
    python BAO_fitter.py 1 2 ${i}
done