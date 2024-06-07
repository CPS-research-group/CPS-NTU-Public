
#include "abs.h"

//clear and initialize an 1d vector with -1
void clear_1d_vc_initialize_neg1(vector <int> &vc, int sz){
    vc.clear();

    for(int i = 0; i < sz; i++)
        vc.push_back(-1);
} 


//print deadline of each tasks
void print_deadline(){
    cout << "Deadline is \n";
    for(int tindex = 0; tindex < no_tasks; tindex++)
        cout << tasks[tindex].deadline << "\t";
    cout << "\n";
}


//print 1d vec
void print_1d_vec(vector <int> vc){
    for(int i = 0; i < vc.size(); i++){
        cout << vc[i] << "\t";
    }
    cout << "\n";
}


//return delay of task index with given hp set
int get_delay_tindex(int tindex, vector <int> hp){
    vector <int> lp;
    for(int tsk = 1; tsk <= no_tasks; tsk++){
        if(tsk == tindex + 1 || is_present(tsk, hp)){
            continue;
        }
        lp.push_back(tsk);
    }
    
    int delay = 0;

    delay += max_terms_rev_sorted[tindex*no_tasks + tindex][0];//max term of task itself
    for(int i = 0; i < hp.size(); i++){
        int hp_tindex = hp[i] - 1;
        for(int j = 0; j < num_max_terms[tindex][hp_tindex]; j++){
            delay += max_terms_rev_sorted[(tindex)*no_tasks + hp_tindex][j];
        }
    }

    int val = proc_time_s1[tindex];//initialize with self
    for(int i = 0; i < hp.size(); i++){
        if(tasks[tindex].s1 == tasks[hp[i] - 1].s1)//if they meet at first stage
            val = max(val, proc_time_s1[hp[i] - 1]);
    }
    delay += val;

    val = proc_time_s2[tindex];//initialize with self
    for(int i = 0; i < hp.size(); i++){
        if(tasks[tindex].s2 == tasks[hp[i] - 1].s2)//if they meet at second stage
            val = max(val, proc_time_s2[hp[i] - 1]);
    }
    delay += val;

    val = 0;//initialize
    for(int i = 0; i < lp.size(); i++){
        int l_tindex = lp[i] - 1;
        if(tasks[tindex].s3 == tasks[l_tindex].s3)//if they meet at third stage
            val = max(val, proc_time_s3[l_tindex]);
    }
    delay += val;

    return delay;

}


bool inc_deadline(const Task& a, const Task& b) {
    return a.deadline > b.deadline;
}


bool is_feasible(int tindex, vector <int> hp){
    
    int delay = get_delay_tindex(tindex, hp);
    
    if(delay > tasks[tindex].deadline){
        return false;
    }
    return true;
}

//compute a priority ordering if feasible 
bool compute_priority_ordering_if_feasible_dm(){
    vector <pair <int, int> > tsk_pair;

    for(int i = 0; i < no_tasks; i++){
        tsk_pair.push_back(make_pair(tasks[i].tid, tasks[i].deadline));
    }

    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for(int i = 0, plevel = no_tasks; i < no_tasks; i++, plevel--){
        task_at_prior[plevel - 1] = tsk_pair[i].first;
        prior_of_task[tsk_pair[i].first - 1] = plevel;
    }

    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(-1);

    for(int i = 0; i < no_tasks; i++){
        vector <int> hp;
        int _p = prior_of_task[i];

        for(int p = 0; p < _p - 1; p++) {
            hp.push_back(task_at_prior[p]);
        }
        int delay = get_delay_tindex(i, hp);
        delay_as_per_cur_prior[i] = delay;
        if(delay > tasks[i].deadline){
            return false;
        }
    }

    //all well
    return true;
}


//assign absolute priority to each task
bool assign_abs_priority_dm(){
    
    clear_1d_vc_initialize_neg1(prior_of_task, no_tasks);//clear and initialize an 1d vector with -1
    clear_1d_vc_initialize_neg1(task_at_prior, no_tasks);//clear and initialize an 1d vector with -1

    bool status = compute_priority_ordering_if_feasible_dm();

    //all well
    return status;
}
        
//compute a priority ordering if feasible 
bool compute_priority_ordering_if_feasible_abs_opa(){
    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(-1);
    
    vector < int > infeasible;//yet to assigned
    for(int i = 1; i <= no_tasks; i++){
        infeasible.push_back(i);
    }

    vector <int> hp; //high prior task...auxiliary
    //for each priority level
    for(int plevel = no_tasks; plevel > 0; plevel--){
        bool assigned_at_this_prior = false;
        for(int ind = 0; ind < infeasible.size(); ind++){
            int tindex = infeasible[ind]  - 1;
            hp.clear();
            for(int i = 0; i < ind ; i++){
                hp.push_back(infeasible[i]);
            }
            for(int i = ind  + 1; i < infeasible.size(); i ++){
                hp.push_back(infeasible[i]);
            }
            if(is_feasible(tindex, hp)){//if this task index feasible at this priority
                prior_of_task[tindex] = plevel;
                task_at_prior[plevel - 1] = tindex + 1;
                delay_as_per_cur_prior[tindex] = get_delay_tindex(tindex, hp);
                infeasible.erase(infeasible.begin() + ind);
                assigned_at_this_prior = true;
                break;
            }
        }
        if(!assigned_at_this_prior){
            return false;
        }
    }

    //all well
    return true;
}
   

//assign absolute priority to each task
bool assign_abs_priority_opa(){
    
    clear_1d_vc_initialize_neg1(prior_of_task, no_tasks);//clear and initialize an 1d vector with -1
    clear_1d_vc_initialize_neg1(task_at_prior, no_tasks);//clear and initialize an 1d vector with -1
    
    bool status = compute_priority_ordering_if_feasible_abs_opa();

    //all well
    return status;
}
        
void compute_priority_ordering_abs_opa_partial(){

    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(-1);
    
    vector < int > feasible, infeasible;
    for(int i = 0; i < no_tasks; i++){
        infeasible.push_back(i + 1);
    }
    vector <pair <int, int> > tsk_pair;
    vector <int> hp; //high prior task...auxiliary
    for(int plevel = no_tasks; plevel > 0; plevel--){
        if(feasible.size() == 0 && infeasible.size() == 0){
            break;
        }

        tsk_pair.clear();
        for(int ind = 0; ind < infeasible.size(); ind++){
            int tsk = infeasible[ind];
            hp.clear();
            for(int i = 0; i < ind ; i++){
                hp.push_back(infeasible[i]);
            }
            for(int i = ind  + 1; i < infeasible.size(); i ++){
                hp.push_back(infeasible[i]);
            }
            for(int i = 0; i < feasible.size(); i++){
                hp.push_back(feasible[i]);
            }
            if(is_feasible(tsk - 1, hp)){//if this task index feasible at this priority
                feasible.push_back(tsk);
                infeasible.erase(infeasible.begin() + ind);
                ind--;
            }
            else{
                int delay = get_delay_tindex(tsk - 1, hp);
                tsk_pair.push_back(make_pair(tsk, (tasks[tsk - 1].deadline - delay)));
            }
        }

        while(feasible.size() == 0 && infeasible.size() != 0){
            sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

            int _i = -1;
            for(int ind = 0; ind < infeasible.size(); ind++){
                if(infeasible[ind] == tsk_pair[0].first){  
                    _i = ind;
                    break;
                }
            }
            assert(_i != -1);
            
            infeasible.erase(infeasible.begin() + _i);
            tsk_pair.erase(tsk_pair.begin());
            
            for(int ind = 0; ind < infeasible.size(); ind++){
                int tsk = infeasible[ind];
                hp.clear();
                for(int i = 0; i < ind ; i++){
                    hp.push_back(infeasible[i]);
                }
                for(int i = ind  + 1; i < infeasible.size(); i ++){
                    hp.push_back(infeasible[i]);
                }
                for(int i = 0; i < feasible.size(); i++){
                    hp.push_back(feasible[i]);
                }
                if(is_feasible(tsk - 1, hp)){
                    feasible.push_back(tsk);
                    infeasible.erase(infeasible.begin() + ind);
                    ind--;
                }
            }
        }   

        if(feasible.size() == 0){
            break;
        }
        int tindex = feasible[0] - 1;
        prior_of_task[tindex] = plevel;
        task_at_prior[plevel - 1] = tindex + 1;
        feasible.erase(feasible.begin());
        no_tasks_scheduled_this_task_set++;

        hp.clear();
        for(int i = 0; i < feasible.size(); i++)
            hp.push_back(feasible[i]);
        for(int i = 0; i < infeasible.size(); i++)
            hp.push_back(infeasible[i]);
        delay_as_per_cur_prior[tindex] = get_delay_tindex(tindex, hp);
    }

}
   

bool assign_abs_priority_opa_partial(){
    
    clear_1d_vc_initialize_neg1(prior_of_task, no_tasks);//clear and initialize an 1d vector with -1
    clear_1d_vc_initialize_neg1(task_at_prior, no_tasks);//clear and initialize an 1d vector with -1
    
    compute_priority_ordering_abs_opa_partial();

    if(no_tasks_scheduled_this_task_set == no_tasks){
        for(int tindex = 0; tindex < no_tasks; tindex++){
            heav_accepted_opa_partial += get_current_heaviness(tindex);
        }
        return true;
    }
    
    //add discarded and accepted heavines
    for(int tindex = 0; tindex < no_tasks; tindex++){
        double _h = get_current_heaviness(tindex);
        if(is_present(tindex + 1, discarded)){
            heav_rejected_opa_partial += _h;
        }
        else{
            heav_accepted_opa_partial += _h;
        }
    }
    return false;
}

//return true if given val is present in vector
bool is_present(int val, vector <int> vc){
    for(int i = 0; i < vc.size(); i++){
        if(vc[i] == val)
            return true;
    }
    return false;
}

//compute a priority ordering 
void compute_priority_ordering_baseline_partial(){

    vector <int> infeasible;
    vector <pair <int, int> > tsk_pair;
    discarded.clear();

    while(true){
        tsk_pair.clear();
        infeasible.clear();

        clear_1d_vc_initialize_neg1(prior_of_task, no_tasks);//clear and initialize an 1d vector with -1
        clear_1d_vc_initialize_neg1(task_at_prior, no_tasks);//clear and initialize an 1d vector with -1

        for(int i = 0; i < no_tasks; i++){
            if(!is_present(i + 1, discarded)){
                tsk_pair.push_back(make_pair(tasks[i].tid, tasks[i].deadline));
            }
        }

        sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        for(int i = 0, j = 0, plevel = no_tasks; i < no_tasks; i++){
            if(is_present(i + 1, discarded)){
                continue;
            }
            task_at_prior[plevel - 1] = tsk_pair[j].first;
            prior_of_task[tsk_pair[j].first - 1] = plevel;
            j++;
            plevel--;
        }

        delay_as_per_cur_prior.clear();
        for(int i = 0; i < no_tasks; i++)
            delay_as_per_cur_prior.push_back(-1);

        for(int i = 0; i < no_tasks; i++){
            if(is_present(i + 1, discarded)){//if a discarded task
                continue;
            }

            vector <int> hp;
            int _p = prior_of_task[i];

            for(int p = 0; p < _p - 1; p++) {
                if(task_at_prior[p] != -1)
                    hp.push_back(task_at_prior[p]);
            }

            int delay = get_delay_tindex(i, hp);
            delay_as_per_cur_prior[i] = delay;
            if(delay > tasks[i].deadline){
                infeasible.push_back(i + 1);
            }
        }

        if(infeasible.size() == 0){
            break;
        }

        int _i = 0;
        int max_lag = tasks[0].deadline - delay_as_per_cur_prior[0];
        for(int i = 1; i < infeasible.size(); i++){
            int tindex = infeasible[i] - 1;
            int lag = tasks[tindex].deadline - delay_as_per_cur_prior[tindex];
            if(max_lag > lag){
                max_lag = lag;
                _i = i;
            }
        }   
        discarded.push_back(infeasible[_i]);
    }

    no_tasks_scheduled_this_task_set = no_tasks - discarded.size();
}


//assign absolute priority to each task
bool assign_abs_baseline_partial(){
    
    clear_1d_vc_initialize_neg1(prior_of_task, no_tasks);//clear and initialize an 1d vector with -1
    clear_1d_vc_initialize_neg1(task_at_prior, no_tasks);//clear and initialize an 1d vector with -1

    compute_priority_ordering_baseline_partial();

    if(no_tasks_scheduled_this_task_set == no_tasks)
        return true;
    return false;
}

struct tempTask {
    int id;
    int arrival_time;
    int execution_time;
    int deadline;

    tempTask(int _id, int _arrival_time, int _execution_time, int _deadline) : id(_id), arrival_time(_arrival_time), execution_time(_execution_time), deadline(_deadline) {}
};

vector <tempTask> tmp_tasks;

bool compareAtimeDdline(const tempTask& task1, const tempTask& task2) {
    if (task1.arrival_time == task2.arrival_time) {
        return task1.deadline < task2.deadline;
    }
    return task1.arrival_time < task2.arrival_time;
}


//get index of this task index in tmp_tasks vector 
int get_index_in_tmp_tasks(int tindex){
    for(int i = 0; i < tmp_tasks.size(); i++){
        if(tmp_tasks[i].id == tindex + 1){
            return i;
        }
    }
    assert(0 == 1);
}

//check if all tasks finished
bool all_tasks_finished(vector <bool> finished){
    for(int i = 0; i < finished.size(); i++){
        if(!finished[i]){//not finished
            return false;
        }
    }
    return true;
}


//return false, if tmp_tasks are infeasible with pr dm
bool is_pr_feasbile(){
    
    if(tmp_tasks.size() == 1)
        return true;

    int ctime = 0;

    vector <bool> finished;
    for(int i = 0; i < tmp_tasks.size(); i++){
        finished.push_back(false);
    }

    vector <int> rem_exec;
    for(int i = 0; i < tmp_tasks.size(); i++){
        rem_exec.push_back(tmp_tasks[i].execution_time);
    }

    vector <int> avail_unifinished_tasks;

    while(!all_tasks_finished(finished)){
        avail_unifinished_tasks.clear();
        
        for(int i = 0; i < tmp_tasks.size(); i++){
            if(!finished[i] && tmp_tasks[i].arrival_time <= ctime){
                avail_unifinished_tasks.push_back(tmp_tasks[i].id);
            }
        }

        if(avail_unifinished_tasks.size() != 0){
            int _ind = 0;
            for(int i = 1; i < avail_unifinished_tasks.size(); i++){
                if(dtasks[avail_unifinished_tasks[i] - 1].deadline < dtasks[avail_unifinished_tasks[_ind] - 1].deadline){
                    _ind = i;
                }
            }
            int tindex = avail_unifinished_tasks[_ind] - 1; 
            int j = get_index_in_tmp_tasks(tindex);
            if(tmp_tasks[j].deadline < ctime){
                return false;
            }
            rem_exec[j]--;
            if(rem_exec[j] == 0){
                finished[j] = true;
            }
        }
        ctime++;
    }
    //all well 
    return true;
}


//return false, if tmp_tasks are infeasible with npr dm
bool is_npr_feasbile(){
    if(tmp_tasks.size() == 1)
        return true;

    int ctime = 0;

    for (const tempTask& task : tmp_tasks) {
        if (ctime < task.arrival_time) {
            ctime = task.arrival_time;
        }

        if (ctime + task.execution_time > task.deadline) {
            return false; // miss
        }

        ctime += task.execution_time;
    }

    return true; 
}

//run dm for all these tasks mapped to the same resource
bool run_dm_for_tasks_at_a_resource(vector <int> tasks_mapped_to_this_res){
    tmp_tasks.clear();
    for(int i = 0; i < tasks_mapped_to_this_res.size(); i++){
        tmp_tasks.emplace_back(0, 0, 0, 0);
        
        int tindex = tasks_mapped_to_this_res[i] - 1;
        
        tmp_tasks[i].id = dtasks[tindex].tid;
        tmp_tasks[i].arrival_time = dtasks[tindex].atime;
        tmp_tasks[i].execution_time = dtasks[tindex].ptime;
        tmp_tasks[i].deadline = dtasks[tindex].deadline;
    }
    sort(tmp_tasks.begin(), tmp_tasks.end(), compareAtimeDdline);

    if(dtasks[tasks_mapped_to_this_res[0] - 1].pr == 0){//npr tasks
        if(!is_npr_feasbile()){
            return false;
        }
    }
    else{//pr tasks
        if(!is_pr_feasbile()){
            return false;
        }

    }
    return true;
}

//list of all new tasks mapped to a resource
void get_tasks_mapped_to_this_res_index(vector <int> & tasks_mapped_to_this_res, int rindex){
    tasks_mapped_to_this_res.clear();
    for(int tindex = 0; tindex < 3*no_tasks; tindex++){
        if(dtasks[tindex].rs == rindex + 1){
            tasks_mapped_to_this_res.push_back(dtasks[tindex].tid);
        }
    }
}

bool run_dm_decomposed_tasks(){
    int _t_rs = no_ap_upload + no_server + no_ap_download; 

    vector <int> tasks_mapped_to_this_res;

    //at each resource run dm and check if any task fails at this resource
    for(int rindex = 0; rindex < _t_rs; rindex++){
        get_tasks_mapped_to_this_res_index(tasks_mapped_to_this_res, rindex);

        if(tasks_mapped_to_this_res.size() > 0 && !run_dm_for_tasks_at_a_resource(tasks_mapped_to_this_res)){
            return false;
        }
    }

    return true;
}

//get deomposed tasks
void get_decomposed_tasks(){
    int new_tid_ind = 0;

    //now add decomposed tasks
    dtasks.clear();

    for (int i = 0; i < 3*no_tasks; ++i) {
        //3 new tasks corresponding to this task
        dtasks.emplace_back(0, 0, 0, 0, 0, 0);
    }

    //for each task...replace with 3 new tasks based on decomposition
    for(int tindex = 0; tindex < no_tasks; tindex++){
        int _d = tasks[tindex].deadline;//orig deadline of this task

        int _new_d1 = max(1, (int)floor(_d * ((load_at_each_stage[0][tasks[tindex].s1 - 1])/(load_at_each_stage[0][tasks[tindex].s1 - 1] + load_at_each_stage[1][tasks[tindex].s2 - 1] + load_at_each_stage[2][tasks[tindex].s3 - 1]))));

        int _new_d2 = max(1, (int)floor(_d * ((load_at_each_stage[1][tasks[tindex].s2 - 1])/(load_at_each_stage[0][tasks[tindex].s1 - 1] + load_at_each_stage[1][tasks[tindex].s2 - 1] + load_at_each_stage[2][tasks[tindex].s3 - 1]))));
        
        int _new_d3 = _d - _new_d1 - _new_d2;

        //first new task
        dtasks[new_tid_ind].tid = new_tid_ind + 1;
        dtasks[new_tid_ind].atime = 0;
        dtasks[new_tid_ind].ptime = proc_time_s1[tindex];
        dtasks[new_tid_ind].deadline = dtasks[new_tid_ind].atime + _new_d1;//convert to absolute deadline
        dtasks[new_tid_ind].rs = tasks[tindex].s1;

        new_tid_ind++;

        //second new task
        dtasks[new_tid_ind].tid = new_tid_ind + 1;
        dtasks[new_tid_ind].atime = _new_d1;
        dtasks[new_tid_ind].ptime = proc_time_s2[tindex];
        dtasks[new_tid_ind].deadline = dtasks[new_tid_ind].atime + _new_d2;
        dtasks[new_tid_ind].rs = no_ap_upload + tasks[tindex].s2;//resource counting starts with 1 at stage 1
        dtasks[new_tid_ind].pr = 1;//other two stages npr

        new_tid_ind++;

        //third new task
        dtasks[new_tid_ind].tid = new_tid_ind + 1;
        dtasks[new_tid_ind].atime = _new_d1 + _new_d2;
        dtasks[new_tid_ind].ptime = proc_time_s3[tindex];
        dtasks[new_tid_ind].deadline = dtasks[new_tid_ind].atime + _new_d3;
        dtasks[new_tid_ind].rs = no_ap_upload + no_server + tasks[tindex].s3;

        new_tid_ind++;
    }

}


//get_heaviness_of_a_resource_at_a_stage
double get_heaviness_of_a_resource_at_a_stage(int stg, int res){
  double _hv = 0;
  if(stg == 1){//s1
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s1 == res){
        _hv += (double)(proc_time_s1[tindex])/tasks[tindex].deadline;
      }
    }
  }
  else if(stg == 2){//s2
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s2 == res){
        _hv += (double)(proc_time_s2[tindex])/tasks[tindex].deadline;
      }
    }
  }
  else{//s3
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s3 == res){
        _hv += (double)(proc_time_s3[tindex])/tasks[tindex].deadline;
      }
    }
  }
  return _hv;
}


//compute load at each resource at each stage
void compute_load_each_res_each_stage(){
    load_at_each_stage.clear();
    
    //stage1
    _1ddouble.clear();
    int stg = 1;
    for(int res = 1; res <= no_ap_upload; res++){
        _1ddouble.push_back(get_heaviness_of_a_resource_at_a_stage(stg, res));
    }
    load_at_each_stage.push_back(_1ddouble);

    //stage2
    _1ddouble.clear();
    stg = 2;
    for(int res = 1; res <= no_server; res++){
        _1ddouble.push_back(get_heaviness_of_a_resource_at_a_stage(stg, res));
    }
    load_at_each_stage.push_back(_1ddouble);

    //stage3
    _1ddouble.clear();
    stg = 3;
    for(int res = 1; res <= no_ap_download; res++){
        _1ddouble.push_back(get_heaviness_of_a_resource_at_a_stage(stg, res));
    }
    load_at_each_stage.push_back(_1ddouble);
}


//decompositon based baseline
bool decompostion_baseline(){
    //compute load at each resource at each stage
    compute_load_each_res_each_stage();

    //get deomposed tasks
    get_decomposed_tasks();

    //run dm for decomposed tasks...return false if any task fails
    bool status = run_dm_decomposed_tasks();

    return status;
}
