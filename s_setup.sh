#!/bin/bash
pushd ..
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
mkdir results
mkdir plots
popd