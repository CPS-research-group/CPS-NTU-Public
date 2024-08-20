# RTSS2024 Paper Source Code

Paper: Energy-Efficient Real-Time Job Mapping and Resource Management in Mobile-Edge Computing

## RTSS2024 Experiment Code Introduction (#250)

[Download Source Code](https://doi.org/10.21979/N9/VJTMBM)

### Python Environment Setup

#### Hardware Requirement

- CPU with up to 16 cores (32 threads) (more cores can let the ILP solver run faster)

- 32GB RAM

#### Required Python Package (python version >= 3.8)

```tex
json, pickle, random, sys, time, gurobipy, math, multiprocessing, matplotlib, numpy, pandas, seaborn, scienceplots
```

among those packages, there are some python built-in package, which do not need to manually install

##### Built-in package

```tex
json, pickle, random, sys, time, math, multiprocessing, 
```

##### Packages needs to manually install

```tex
gurobipy, matplotlib, numpy, pandas, seaborn, scienceplots
```

#### Install Packages

1. install anaconda [Installation — Anaconda documentation](https://docs.anaconda.com/anaconda/install/) 

2. activate conda environment

3. install packages

   ```bash
   # install gurobipy
   (base) $ conda install -c gurobi gurobi
   # install matplotlib, numpy, pandas, seaborn, scienceplots
   (base) $ conda install matplotlib
   (base) $ conda install numpy
   (base) $ conda install pandas
   (base) $ conda install seaborn
   (base) $ pip install SciencePlots
   ```

4. The gurobi license `gurobi.lic` is inside the folder `gurobi license`. 

   - Copy the gurobi license `gurobi.lic`  (not the folder) to the **home directory** (for windows `C:\Users\username\` , for linux `/home/usename/`)
   - make sure your PC is connected to the internet when executing the python scripts (for license verification)



### Draw results directly based on our experiment data

```bash
$ cd ~/full		# go to the folder 'full'
$ python draw_results.py 3a
$ python draw_results.py 3b-up
$ python draw_results.py 3b-low
$ python draw_results.py 3c
$ python draw_results.py 4
$ python draw_results.py 5
$ python draw_results.py 6-left
$ python draw_results.py 6-right
```



### MEC Data

- folder `~/full/data/profiling` contains the profiled job processing time and energy consumption during job processing
- folder `~/full/data/sumo_small` contains the MEC information for the small-scale MEC (base MEC), which contains 13 APs, 6 GPU server, 6 CPU server
- folder `~/full/data/sumo_medium` contains the MEC information for the small-scale MEC (base MEC), which contains 16 APs, 9 GPU server, 9 CPU server
- folder `~/full/data/sumo_large` contains the MEC information for the small-scale MEC (base MEC), which contains 21 APs, 12 GPU server, 12 CPU server

### Jobset Data

#### 1. Small-scale MEC

- folder `~/full/data/datasetS` contains the jobset and MEC synthesized for small-scale MEC

  - `edge.pickle` : small-scale MEC

  - `jobsets.pickle` : the synthesized 3000 jobsets for performance evaluation

  - `jobsets_scale.pickle` : the synthesized 100 jobsets for scalability evaluation, each jobset constains 80 jobs.

  - folder `~/full/data/datasetS/solution` contains the algorithm solution with respect to small-scale MEC

    - `lhjs_solution.pickle` : offline, solution of LHJS for  `jobsets.pickle` 

    - `round_solution.pickle` : offline, solution of RandRound in LHJS for the first 900 jobsets in  `jobsets.pickle` 

    - `scale_solution.pickle` : offline, solution of LHJS for  `jobsets_scale.pickle` 

    - `search_solution.pickle` : offline, solution of offline baseline algorithm SEARCH for  `jobsets.pickle`  

      

    - `sol_base_LBS.pickle` : online, solution of LBS for  `jobsets.pickle` 

    - `sol_base_LBSLate.pickle` : online, solution of LBSLate for  `jobsets.pickle` 

    - `sol_base_LCEarly.pickle` : online, solution of LCEarly for  `jobsets.pickle` 

    - `sol_base_LCLate.pickle` : online, solution of LCLate for  `jobsets.pickle` 

    - `sol_scale_LBS.pickle` : online, solution of LBS for  `jobsets_scale.pickle` 

#### 2. Medium-scale MEC

- folder `~/full/data/datasetM` contains the jobset and MEC synthesized for medium-scale MEC

  - `edge.pickle` : medium-scale MEC

  - `jobsets_scale.pickle` : the synthesized 100 jobsets for scalability evaluation, each jobset constains 140 jobs

  - folder `~/full/data/datasetM/solution` contains the algorithm solution with respect to medium-scale MEC
    - `scale_solution.pickle` : offline, solution of LHJS for  `jobsets_scale.pickle` 
    - `sol_scale_LBS.pickle` : online, solution of LBS for  `jobsets_scale.pickle` 

#### 3. Large-scale MEC

- folder `~/full/data/datasetL` contains the jobset and MEC synthesized for large-scale MEC

  - `edge.pickle` : large-scale MEC

  - `jobsets_scale.pickle` : the synthesized 100 jobsets for scalability evaluation, each jobset constains 200 jobs

  - folder `~/full/data/datasetL/solution` contains the algorithm solution with respect to large-scale MEC
    - `scale_solution.pickle` : offline, solution of LHJS for  `jobsets_scale.pickle` 
    - `sol_scale_LBS.pickle` : online, solution of LBS for  `jobsets_scale.pickle` 



### Python Scripts Instruction

#### 1. Go to the jobset folder

```bash
$ cd ~/full
```

#### 2. Generate Jobsets

```bash
$ python job_init.py
$ python job_init_scale.py small
$ python job_init_scale.py medium
$ python job_init_scale.py large
```

two python scripts can be used to generate jobsets.

1. `job_init.py` : generate 3000 jobsets for performance evaluation
   - generated jobsets are saved to `./data/datasetS/jobsets.pickle` 
2. `job_init_scale.py` : generate 100 jobsets for scalability evaluation
   - argument `small` : small-scale MEC
     - generated jobsets are saved to `./data/datasetS/jobsets_scale.pickle` 
   - argument `medium` : medium-scale MEC
     - generated jobsets are saved to `./data/datasetM/jobsets_scale.pickle` ,
   - argument `large` : large-scale MEC
     - generated jobsets are saved to `./data/datasetL/jobsets_scale.pickle` 

#### 3. Run Algorithms

##### 3.1 Offline

```bash
$ python alg_lhjs.py
$ python alg_search_ILP.py
$ python alg_lhjs_round.py
$ python alg_sortAll.py
$ python alg_lhjs_scale.py small
$ python alg_lhjs_scale.py medium
$ python alg_lhjs_scale.py large
```

- `alg_lhjs.py` solve the offline EMJS with algorithm LHJS, run RandRound for 50 times. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/lhjs_solution.pickle`
- `alg_search_ILP.py` solve the offline EMJS with the baseline algorithm SEARCH. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/search_solution.pickle` 

- `alg_sortAll.py` solve the offline EMJS with algorithm baseline algorithm SortAll. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` and `./data/datasetS/solution/lhjs_solution.pickle` 

  - saved solution: `./data/datasetS/solution/sortAll_solution.pickle`
- `alg_lhjs_round.py` solve the offline EMJS with algorithm LHJS, run RandRound for 50 times. 
  - Input jobsets: the 900 jobsets of different jobset sizes in  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/round_solution.pickle` 
- `alg_lhjs_scale.py` solve the offline EMJS with algorithm LHJS for different MEC scales. 
  - argument `small`

    - Input jobsets:  `./data/datasetS/jobsets_scale.pickle` 

    - saved solution: `./data/datasetS/solution/scale_solution.pickle` 

  - argument `medium`

    - Input jobsets:  `./data/datasetM/jobsets_scale.pickle` 
    - saved solution: `./data/datasetM/solution/scale_solution.pickle` 

  - argument `large`

    - Input jobsets:  `./data/datasetL/jobsets_scale.pickle` 
    - saved solution: `./data/datasetL/solution/scale_solution.pickle` 

##### 3.2 Online

```bash
$ python alg_LBS.py
$ python alg_base_LBSLate.py
$ python alg_base_LCEarly.py
$ python alg_base_LCLate.py
$ python alg_LBS_scale.py small
$ python alg_LBS_scale.py medium
$ python alg_LBS_scale.py large
```

- `alg_LBS.py` solve the online EMJS with algorithm LBS for small-scale MEC. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/sol_base_LBS.pickle` 

- `alg_base_LBSLate.py` solve the online EMJS with algorithm LBSLate for small-scale MEC. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/sol_base_LBSLate.pickle` 

- `alg_base_LCEarly.py` solve the online EMJS with algorithm LCEarly for small-scale MEC. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/sol_base_LCEarly.pickle` 

- `alg_base_LCLate.py` solve the online EMJS with algorithm LCLate for small-scale MEC. 
  - Input jobsets:  `./data/datasetS/jobsets.pickle` 

  - saved solution: `./data/datasetS/solution/sol_base_LCLate.pickle` 

- `alg_LBS_scale.py` solve the online EMJS with algorithm LBS for different MEC scales. 
  - argument `small`

    - Input jobsets:  `./data/datasetS/jobsets_scale.pickle` 

    - saved solution: `./data/datasetS/solution/sol_scale_LBS.pickle` 

  - argument `medium`

    - Input jobsets:  `./data/datasetM/jobsets_scale.pickle` 
    - saved solution: `./data/datasetM/solution/sol_scale_LBS.pickle` 

  - argument `large`

    - Input jobsets:  `./data/datasetL/jobsets_scale.pickle` 
    - saved solution: `./data/datasetL/solution/sol_scale_LBS.pickle` 

#### 4. Draw Results

```bash
$ python draw_results.py 3a
$ python draw_results.py 3b-up
$ python draw_results.py 3b-low
$ python draw_results.py 3c
$ python draw_results.py 4
$ python draw_results.py 5
$ python draw_results.py 6-left
$ python draw_results.py 6-right
```

`draw_results.py` draws the experiment results.

- argument `3a` : draw Fig. 3(a)
  - Performance ratios of LHJS, SEARCH, and SortAll under different AP-server utilization range combinations in offline job scheduling
- argument `3b-up` : Fig. 3(b) consists of two subfigures, `3b-up` draws the upper half 
  - Intermediate results of LHJS compared to the optimal saved energy under different AP-server utilization range combinations
- argument `3b-low` : Fig. 3(b) consists of two subfigures, `3b-low` draws the lower half 
  - Intermediate results of LHJS compared to the optimal saved energy under different AP-server utilization range combinations
- argument `3c` : draw Fig. 3(c)
  - A comparison between the integral solution obtained by RandRound and the approximation bound of RandRound (as in Theorem 1)
  - the x axis label needs to manually adjusted as it is draw based on jobset index (so that you can see the scatter figure). **The mapping between jobset index and jobset size is as follows:**
    - 0~30 (jobset index) : 60 (jobset size)
    - 30~120 : 70
    - 120~210 : 80
    - 210~300 : 90
    - 300~390 : 100
    - 390~480 : 110
    - 480~570 : 120
    - 570~660 : 130
    - 660~750 : 140
    - 750~840 : 150
    - 840~900 : 160
- argument `4` : draw Fig. 4
  - Runtime analysis of LHJS for different scales of MEC
- argument `5` : draw Fig. 5
  - Performance ratios of LBS and its variants under different AP-server utilization range combinations for online job scheduling
- argument `6-left` : Fig. 6 consists of two subfigures, `6-left` draws the left half 
  - Runtime of online algorithms under different jobset sizes
- argument `6-right` : Fig. 6 consists of two subfigures, `6-right` draws the right half 
  - Runtime of LBS under different MEC scales

