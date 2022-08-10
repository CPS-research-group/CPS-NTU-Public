# Deadline-constrained Multi-resource Task Mapping and Allocation for Edge-Cloud Systems

### 1. Environment

OS: windows/linux

language: python 3.9 or above

ILP solver: [Gurobi - The Fastest Solver - Gurobi](https://www.gurobi.com/) 



### 2. Dataset generation

The code for generating data set is in file `Data_creation.ipynb`.

- function `set_server_variables()` define edge-cloud system related parameters.
- function `generate_tasksets()` generate required taskset
- function `create_tasksets_and_analysis()` generate the taskset and do a basic analysis for the generated dataset.
- function `main()` run the code and label the data.

A summary of the data is in file `experiment data.txt`



### 3. Zero Slack Greedy Algorithm

The code for ZSG is in file `Heuristic_results_new_gamma_2.ipynb`.

The result is split into several data files for parallel processing of ILP solver.



### 4. Linear Discretization Method

The code for ILP solver of LDM is in files `Gurobi_Test_Size_*0_*.ipynb`.

Multiple files are used for parallel processing to shorten the code running time.



### 5. Result Analysis

The code for result analysis is in file `Analyze_result.ipynb`.

You can get the experiment data from this [link](https://doi.org/10.21979/N9/5D1FBL).

