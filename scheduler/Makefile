# Copyright 2019, Gurobi Optimization, LLC
PLATFORM = linux64
TWOUP    = ../..
KEEPTU   = ../..
INC      = $(TWOUP)/include/
CC       = gcc
CPP      = g++
CARGS    = -std=c++14 -m64 -g
CPPLIB   = -L$(TWOUP)/lib -lgurobi_c++ -lgurobi100

exec: main.cpp input.cpp intermed.cpp abs.cpp rel.cpp prior_rel_opt.cpp  
	$(CPP) $(CARGS) -o $@ $^ -I$(INC) $(CPPLIB) -lm 
	
run_delay: exec
	./exec

clear_output:
	cd ../../files/output/ && rm *


                



