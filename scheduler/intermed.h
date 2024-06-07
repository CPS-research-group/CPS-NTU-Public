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

vector <int> prior_of_task, task_at_prior, _1dint;
vector < vector < int > > num_max_terms, max_terms_rev_sorted, var_index;

vector <int> proc_time_s1, proc_time_s2, proc_time_s3;

extern int no_input, no_tasks, no_ap_upload, no_server, no_ap_download;
extern vector <int> uplink_rate, compute_rate, downlink_rate; 