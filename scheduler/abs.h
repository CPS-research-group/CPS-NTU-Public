#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include "task_struct.h"

using namespace std;

extern string path_to_output;
extern int no_input, no_tasks, no_ap_upload, no_server, no_ap_download;
extern int no_tasks_scheduled_this_task_set;
extern vector <int> prior_of_task, task_at_prior, _1dint;
extern vector < vector < int > > num_max_terms, max_terms_rev_sorted;
extern vector <int> proc_time_s1, proc_time_s2, proc_time_s3;

extern void clear_2d_vc_initialize_zero(vector < vector < int > > &, int, int);

vector <int> delay_as_per_cur_prior;//store delay as per assigned priority
vector <int> discarded;

extern double heav_accepted_opa_partial, heav_rejected_opa_partial;

extern double get_current_heaviness(int);
extern bool is_present(int, vector <int>);

vector <vector <double> > load_at_each_stage;

// decomposed task
//define task type
struct dTask{
    int tid;//task number...not index 
    int atime;//arrival time
    int ptime;//processing time of this stage
    int deadline;//deadline
    int rs;//resource to which it is mapped...counting starts with 1 at Stage 1
    int pr;//whether this task is preemptive or npr...1 - pr, 0 - npr

    dTask(int _tid, int _atime, int _ptime, int _deadline, int _rs, int _pr) : tid(_tid),  atime(_atime), ptime(_ptime), deadline(_deadline), rs(_rs), pr(_pr) {}
};

vector <dTask> dtasks;
vector <double> _1ddouble;
