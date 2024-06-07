#include "gurobi_c++.h"
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include "task_struct.h"
#include <sstream>

using namespace std;

string path_to_output = "../output/";

extern bool time_limit_expired_this_test;
extern int no_input, no_tasks, no_ap_upload, no_server, no_ap_download;
extern vector <int> prior_of_task, task_at_prior, _1dint;
extern vector < vector < int > > num_max_terms, max_terms_rev_sorted, var_index;

extern vector <int> proc_time_s1, proc_time_s2, proc_time_s3;

int ubound;
int tlimit = 600;
