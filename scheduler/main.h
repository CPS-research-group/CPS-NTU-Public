#include <iostream>
#include <assert.h>
#include <fstream>
#include <string>

using namespace std;

extern string path_to_input;

extern int no_input, no_tasks;
extern void get_static_input();
extern void get_input(int);

extern bool assign_rel_priority_opt();
extern bool assign_abs_priority_opa();
extern bool assign_abs_priority_opa_partial();
extern bool assign_abs_baseline_partial();
extern bool assign_abs_priority_dm();
extern bool assign_rel_priority_heur();
extern bool assign_rel_heur_partial();
extern bool assign_rel_priority_heur_extended();
extern bool assign_rel_priority_baseline();
extern bool assign_rel_baseline_partial();
extern void compute_common_intermed_info();
extern bool decompostion_baseline();

int no_tasks_scheduled_this_task_set;
extern int no_tasks;
bool time_limit_expired_this_test;

double heav_accepted_opa_partial, heav_rejected_opa_partial, heav_accepted_rel_heur_partial, heav_rejected_rel_heur_partial, heav_accepted_rel_baseline_partial , heav_rejected_rel_baseline_partial;

extern string path_to_output;
        
