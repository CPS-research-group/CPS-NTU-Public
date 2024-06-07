#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include "task_struct.h"
#include <sstream>
#include <limits>

using namespace std;

extern string path_to_output;
extern int no_input, no_tasks, no_ap_upload, no_server, no_ap_download;
extern vector <int> _1dint;
extern vector < vector < int > > num_max_terms;
extern vector < int > delay_as_per_cur_prior;
extern vector <int> _1dint;
extern int max_viol_allowed;
extern vector <int> discarded;
extern vector < vector < int > > num_max_terms, max_terms_rev_sorted;
extern vector <int> proc_time_s1, proc_time_s2, proc_time_s3;
extern int no_tasks_scheduled_this_task_set;

extern int get_delay_tindex(int, vector <int>);
extern void clear_2d_vc_initialize_neg1(vector < vector <int> > &, int, int);
extern void print_deadline();
extern void print_task_to_resource_mapping();
extern void print_1d_vc(vector <int>);
extern void print_prior();
extern bool is_deadline_satisfied(int , int);
extern void get_higher_priority_tasks(int, vector <int> &, vector < vector <int> >);
extern bool do_two_tasks_meet(int, int);
extern bool compute_priority_ordering_if_feasible_rel_heur();
extern void assign_prior_dm_heur();
extern bool is_present(int, vector <int>);

vector <int> viol_count;
vector < vector <int> > rel_prior_of_task;

extern double heav_accepted_rel_heur_partial, heav_rejected_rel_heur_partial, heav_accepted_rel_baseline_partial , heav_rejected_rel_baseline_partial;

