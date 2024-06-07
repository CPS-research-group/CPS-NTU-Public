#include "rel.h"

//get heaviness of task
double get_current_heaviness(int i){
  return (double)(proc_time_s1[i] + proc_time_s2[i] + proc_time_s3[i])/tasks[i].deadline;
}

//print_task_to_resource_mapping();
void print_task_to_resource_mapping(){
    cout << "S1\t";
    for(int i = 0; i < no_ap_upload; i++){
        cout << "R" << i + 1 << "-";
        for(int k = 0; k < no_tasks; k++){
            if(tasks[k].s1 == i + 1){
                cout << k + 1 << " ";
            }
        }
        cout << "\t";
    }
    cout << "\n";

    cout << "S2\t";
    for(int i = 0; i < no_server; i++){
        cout << "R" << i + 1 << "-";
        for(int k = 0; k < no_tasks; k++){
            if(tasks[k].s2 == i + 1){
                cout << k + 1 << " ";
            }
        }
        cout << "\t";
    }
    cout << "\n";

    cout << "S3\t";
    for(int i = 0; i < no_ap_download; i++){
        cout << "R" << i + 1 << "-";
        for(int k = 0; k < no_tasks; k++){
            if(tasks[k].s3 == i + 1){
                cout << k + 1 << " ";
            }
        }
        cout << "\t";
    }
    cout << "\n\n";


    for(int i = 0; i < no_tasks; i++){
        cout << "\tT" << i + 1 << "\t" << tasks[i].s1 << "\t" << tasks[i].s2 << "\t" << tasks[i].s3 << "\n";
    }
    cout << "\n";
}

//print an 1d int vector
void print_1d_vc(vector <int> vc){
    for(int i = 0; i < vc.size(); i++){
        cout << vc[i]  << "\t";
    }
    cout << "\n";
}

//print task prior
void print_prior(){
    for(int  i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(rel_prior_of_task[i][j] == 0){
                cout << j + 1 << ">" << i + 1 << "\t"; 
            }
            else if(rel_prior_of_task[i][j] == 1){
                cout << i + 1 << ">" << j + 1 << "\t"; 
            }
        }
    }
    cout << "\n";
}


//return true if deadline is satisfied for given task index
bool is_deadline_satisfied(int tindex, int delay){
    if(delay > tasks[tindex].deadline){
        return false;
    }
    return true;
}

//rearrance tsk_pair...first entry may be out of order
void rearrange_tsk_pair(vector <pair <int, int> > &tsk_pair){
    vector <pair <int, int> > tmp_tsk_pair;
    //find appropriate index for the first entry
    int ind = -1;
    for(int i = 1; i < no_tasks; i++){
        if(tsk_pair[0].second < tsk_pair[i].second){
            ind = i;
            break;
        }
    }

    //if not found
    if(ind == -1){
        for(int i = 1; i < no_tasks; i++)
            tmp_tsk_pair.push_back(tsk_pair[i]);
        tmp_tsk_pair.push_back(tsk_pair[0]);
    }
    else{//found
        for(int i = 1; i < ind - 1; i++)
            tmp_tsk_pair.push_back(tsk_pair[i]);
        tmp_tsk_pair.push_back(tsk_pair[0]);
        for(int i = ind; i < no_tasks; i++)
            tmp_tsk_pair.push_back(tsk_pair[ind]);
    }
    
    tsk_pair = tmp_tsk_pair;

}

//get higher priority tasks as per given priority assignment for tsk index
void get_higher_priority_tasks(int tindex, vector <int> &hp, vector < vector <int> > ptask){
    for(int j = 0; j < no_tasks; j++){
        if(ptask[tindex][j] == 0){//j is higher priority task
            hp.push_back(j + 1);
        }
    }
}


//swapping priorities of tsk1 and tsk2, earlier tsk1 violates deadline, tsk2 satisfies. After swapping if tsk2 satisfies, tsk1 delay either reduces or deadline satisfied
bool swap_priorities_if_helpful(int tindex1, int tindex2){
    vector < vector <int> > tmp_prior_of_task;
    tmp_prior_of_task = rel_prior_of_task;

    tmp_prior_of_task[tindex1][tindex2] = 1;
    tmp_prior_of_task[tindex2][tindex1] = 0;

    vector <int> tmp_delay_as_per_cur_prior;//store temp delay as per assigned priority 
    tmp_delay_as_per_cur_prior = delay_as_per_cur_prior;

    vector <int> hp;

    get_higher_priority_tasks(tindex2, hp, tmp_prior_of_task);

    int delay = get_delay_tindex(tindex2, hp);
    
    if(!is_deadline_satisfied(tindex2, delay)){//not satisfied
        return false;
    }
    
    tmp_delay_as_per_cur_prior[tindex2] = delay;

    hp.clear();
    get_higher_priority_tasks(tindex1, hp, rel_prior_of_task);
    delay = get_delay_tindex(tindex1, hp);
    
    hp.clear();

    get_higher_priority_tasks(tindex1, hp, tmp_prior_of_task);

    int new_delay = get_delay_tindex(tindex1, hp);

    if(new_delay < delay){
        tmp_delay_as_per_cur_prior[tindex1] = delay;
        
        rel_prior_of_task[tindex1][tindex2] = 1;
        rel_prior_of_task[tindex2][tindex1] = 0;

        delay_as_per_cur_prior[tindex1] = tmp_delay_as_per_cur_prior[tindex1]; 
        delay_as_per_cur_prior[tindex2] = tmp_delay_as_per_cur_prior[tindex2]; 

        return true;
    }
    else{
        return false;
    }
    
}

//check if two tasks meet
bool do_two_tasks_meet(int i, int j){
    if(tasks[i].s1 == tasks[j].s1 || tasks[i].s2 == tasks[j].s2 || tasks[i].s3 == tasks[j].s3)
        return true;
    return false;
}

//dm failed..repair the soln
bool repair_soln(){
    vector <pair <int, int> > tsk_pair;

    for(int i = 0; i < no_tasks; i++){
        tsk_pair.push_back(make_pair(tasks[i].tid, (tasks[i].deadline - delay_as_per_cur_prior[i])));
    }

    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    while(true){
        if(tsk_pair[0].second >= 0){//no deadline violation
            break;
        }

        int tindex1 = tsk_pair[0].first - 1;
        
        bool improved = false;

        for(int i = no_tasks - 1; i > 0; i--){
            int tindex2 = tsk_pair[i].first - 1;
            if(rel_prior_of_task[tindex1][tindex2] == 0 && tsk_pair[i].second > 0){
                if(swap_priorities_if_helpful(tindex1, tindex2)){//helpful and swapped
                    //update
                    tsk_pair[0].second = tasks[tindex1].deadline - delay_as_per_cur_prior[tindex1];
                    tsk_pair[i].second = tasks[tindex2].deadline - delay_as_per_cur_prior[tindex2];
                    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });
                    improved = true;
                    break;
                }
            }
        }
        if(!improved)
            return false;
    }
    return true;
}


//assign priority as per DM
void assign_prior_dm_heur(){
    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(num_max_terms[i][j] > 0){
                if(tasks[i].deadline < tasks[j].deadline){//shorter deadline
                    rel_prior_of_task[i][j] = 1;
                    rel_prior_of_task[j][i] = 0;
                }
                else{
                    rel_prior_of_task[i][j] = 0;
                    rel_prior_of_task[j][i] = 1;
                }
            }
        }
    }
}


//compute a priority ordering if feasible
bool compute_priority_ordering_if_feasible_rel_heur(){
    //assign relative priority as per 
    clear_2d_vc_initialize_neg1(rel_prior_of_task, no_tasks, no_tasks);

    //assign priority as per DM
    assign_prior_dm_heur();

    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(numeric_limits<int>::max());

    for(int i = 0; i < no_tasks; i++){
        vector <int> hp;

        get_higher_priority_tasks(i, hp, rel_prior_of_task);
        int delay = get_delay_tindex(i, hp);
        delay_as_per_cur_prior[i] = delay;
    }
    for(int i = 0; i < no_tasks; i++){
        if(delay_as_per_cur_prior[i] > tasks[i].deadline){
            return false;
        }
    }
    return true;

}

bool assign_rel_priority_heur(){

    bool status = compute_priority_ordering_if_feasible_rel_heur();

    if(!status){
        status = repair_soln();
    }

    //final status after repair
    return status;
}

//swapping priorities of tsk1 and tsk2, earlier tsk1 violates deadline, tsk2 satisfies. 
bool swap_priorities_if_helpful_extended(int tindex1, int tindex2){
    vector < vector <int> > tmp_prior_of_task;
    tmp_prior_of_task = rel_prior_of_task;

    tmp_prior_of_task[tindex1][tindex2] = 1;
    tmp_prior_of_task[tindex2][tindex1] = 0;

    vector <int> tmp_delay_as_per_cur_prior;
    tmp_delay_as_per_cur_prior = delay_as_per_cur_prior;

    vector <int> hp;
    get_higher_priority_tasks(tindex2, hp, tmp_prior_of_task);

    int delay = get_delay_tindex(tindex2, hp);
    
    if(!is_deadline_satisfied(tindex2, delay)){
        if(viol_count[tindex2] < max_viol_allowed){
            viol_count[tindex2]++;
        }
        else{
            return false;
        }
    }
    
    tmp_delay_as_per_cur_prior[tindex2] = delay;
    hp.clear();
    get_higher_priority_tasks(tindex1, hp, rel_prior_of_task);
    delay = get_delay_tindex(tindex1, hp);
    
    hp.clear();
    get_higher_priority_tasks(tindex1, hp, tmp_prior_of_task);

    int new_delay = get_delay_tindex(tindex1, hp);

    if(new_delay < delay){
        tmp_delay_as_per_cur_prior[tindex1] = delay;
        rel_prior_of_task[tindex1][tindex2] = 1;
        rel_prior_of_task[tindex2][tindex1] = 0;
        delay_as_per_cur_prior[tindex1] = tmp_delay_as_per_cur_prior[tindex1]; 
        delay_as_per_cur_prior[tindex2] = tmp_delay_as_per_cur_prior[tindex2]; 
        return true;
    }
    else{
        viol_count[tindex2]--;
        return false;
    }
}

//dm failed..repair the soln
bool repair_soln_extended(){

    vector <pair <int, int> > tsk_pair;

    for(int i = 0; i < no_tasks; i++){
        tsk_pair.push_back(make_pair(tasks[i].tid, (tasks[i].deadline - delay_as_per_cur_prior[i])));
    }
    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    while(true){
        if(tsk_pair[0].second >= 0){//no deadline violation
            break;
        }
        //first task is violating deadline
        int tindex1 = tsk_pair[0].first - 1;
        bool improved = false;
        for(int i = no_tasks - 1; i > 0; i--){
            int tindex2 = tsk_pair[i].first - 1;
            if(rel_prior_of_task[tindex1][tindex2] == 0){
                if(swap_priorities_if_helpful_extended(tindex1, tindex2)){//helpful and swapped
                    //update
                    tsk_pair[0].second = tasks[tindex1].deadline - delay_as_per_cur_prior[tindex1];
                    tsk_pair[i].second = tasks[tindex2].deadline - delay_as_per_cur_prior[tindex2];
                    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });
                    improved = true;
                    break;
                }
            }
        }
        if(!improved)
            return false;
    }
    return true;
}


bool assign_rel_priority_heur_extended(){
    viol_count.clear();
    for(int i = 0; i < no_tasks; i++){
        viol_count.push_back(0);
    }

    bool status = compute_priority_ordering_if_feasible_rel_heur();

    if(!status){
        status = repair_soln_extended();
    }
    return status;
}
   
//assign priority as per DM
void assign_prior_dm_baseline(){
    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(num_max_terms[i][j] > 0){
                if(tasks[i].deadline < tasks[j].deadline){//shorter deadline
                    rel_prior_of_task[i][j] = 1;
                    rel_prior_of_task[j][i] = 0;
                }
                else{
                    rel_prior_of_task[i][j] = 0;
                    rel_prior_of_task[j][i] = 1;
                }
            }
        }
    }
}


//compute a priority ordering if feasible
bool compute_priority_ordering_if_feasible_rel_dm(){
    //assign relative priority as per 
    clear_2d_vc_initialize_neg1(rel_prior_of_task, no_tasks, no_tasks);

    //assign priority as per DM
    assign_prior_dm_baseline();

    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(numeric_limits<int>::max());

    //check each task must satisfy the priority assigned by DM
    for(int i = 0; i < no_tasks; i++){
        vector <int> hp;

        for(int j = 0; j < no_tasks; j++){
            if(rel_prior_of_task[j][i] == 1){//j is higher priority task
                hp.push_back(j + 1);
            }
        }

        //check whether task is feasible or not with this priority assignment
        int delay = get_delay_tindex(i, hp);
        delay_as_per_cur_prior[i] = delay;
        //check if timing constraint satisfied
        if(delay > tasks[i].deadline){
            return false;
        }
    }
    //all well
    return true;

}

bool assign_rel_priority_baseline(){

    bool status = compute_priority_ordering_if_feasible_rel_dm();

    //all well
    return status;
}

//return delay of task index with given hp set..while ignoring the discarded tasks
int get_delay_tindex_partial(int tindex, vector <int> hp){
    
    int delay = 0;

    delay += max_terms_rev_sorted[tindex*no_tasks + tindex][0];//max term of task itself
    for(int i = 0; i < hp.size(); i++){
        int hp_tindex = hp[i] - 1;
        for(int j = 0; j < num_max_terms[tindex][hp_tindex]; j++){
            delay += max_terms_rev_sorted[(tindex)*no_tasks + hp_tindex][j];
        }
    }

    //....second term....//
    int val = proc_time_s1[tindex];
    //compute max
    for(int i = 0; i < hp.size(); i++){
        if(tasks[tindex].s1 == tasks[hp[i] - 1].s1)
            val = max(val, proc_time_s1[hp[i] - 1]);
    }
    delay += val;

    //....third term....//
    val = proc_time_s2[tindex];//initialize with self
    //compute max
    for(int i = 0; i < hp.size(); i++){
        if(tasks[tindex].s2 == tasks[hp[i] - 1].s2)//if they meet at second stage
            val = max(val, proc_time_s2[hp[i] - 1]);
    }
    delay += val;


    //....fourth term....//
    val = 0;//initialize
    vector <int> lp;
    for(int tsk = 1; tsk <= no_tasks; tsk++){
        if(tsk == tindex + 1 || is_present(tsk, hp)){
            continue;
        }
        lp.push_back(tsk);
    }

    for(int i = 0; i < lp.size(); i++){
        int l_tindex = lp[i] - 1;
        if(is_present(l_tindex + 1, discarded)){
                continue;
        }
        if(tasks[tindex].s3 == tasks[l_tindex].s3)//if they meet at third stage
            val = max(val, proc_time_s3[l_tindex]);
    }
    delay += val;

    return delay;

}


//get higher priority tasks as per given priority assignment for tsk index..while ignoring discarded tasks
void get_higher_priority_tasks_partial(int tindex, vector <int> &hp, vector < vector <int> > ptask){
    for(int j = 0; j < no_tasks; j++){
        if(is_present(j + 1, discarded)){
                continue;
        }
        if(ptask[tindex][j] == 0){//j is higher priority task
            hp.push_back(j + 1);
        }
    }
}

//compute_updated_delay a task has been discarded
void compute_updated_delay_partial(){
    delay_as_per_cur_prior.clear();
    for(int i = 0; i < no_tasks; i++)
        delay_as_per_cur_prior.push_back(numeric_limits<int>::max());

    //check each task may not satisfy the priority assigned by DM
    for(int i = 0; i < no_tasks; i++){
        if(is_present(i + 1, discarded)){
                continue;
        }
        vector <int> hp;

        //get higher priority tasks
        get_higher_priority_tasks_partial(i, hp, rel_prior_of_task);

        //check whether task is feasible or not with this priority assignment
        int delay = get_delay_tindex_partial(i, hp);
        delay_as_per_cur_prior[i] = delay;
    }
}

//swapping priorities of two tasks...if helpful
bool swap_priorities_if_helpful_partial(int tindex1, int tindex2){
    vector < vector <int> > tmp_prior_of_task;
    tmp_prior_of_task = rel_prior_of_task;

    tmp_prior_of_task[tindex1][tindex2] = 1;
    tmp_prior_of_task[tindex2][tindex1] = 0;
    vector <int> tmp_delay_as_per_cur_prior;
    tmp_delay_as_per_cur_prior = delay_as_per_cur_prior;

    vector <int> hp;
    get_higher_priority_tasks_partial(tindex2, hp, tmp_prior_of_task);
    int delay = get_delay_tindex_partial(tindex2, hp);
    
    if(!is_deadline_satisfied(tindex2, delay)){//not satisfied
        return false;
    }
    tmp_delay_as_per_cur_prior[tindex2] = delay;

    hp.clear();
    get_higher_priority_tasks_partial(tindex1, hp, rel_prior_of_task);
    delay = get_delay_tindex_partial(tindex1, hp);
    
    hp.clear();
    get_higher_priority_tasks_partial(tindex1, hp, tmp_prior_of_task);
    int new_delay = get_delay_tindex_partial(tindex1, hp);

    if(new_delay < delay){
        tmp_delay_as_per_cur_prior[tindex1] = delay;
        rel_prior_of_task[tindex1][tindex2] = 1;
        rel_prior_of_task[tindex2][tindex1] = 0;
        delay_as_per_cur_prior[tindex1] = tmp_delay_as_per_cur_prior[tindex1]; 
        delay_as_per_cur_prior[tindex2] = tmp_delay_as_per_cur_prior[tindex2]; 
        //all well
        return true;
    }
    else{
        return false;
    }
    
}

//repair the first task by swapping with all tasks
bool first_task_satisfied_by_swapping_partial(vector <pair <int, int> > tsk_pair){

    int tindex1 = tsk_pair[0].first - 1;
    bool satis = false;
    for(int i = tsk_pair.size() - 1; i > 0; i--){
        int tindex2 = tsk_pair[i].first - 1;
        if(rel_prior_of_task[tindex1][tindex2] == 0 && tsk_pair[i].second > 0){
            if(swap_priorities_if_helpful_partial(tindex1, tindex2)){
                tsk_pair[0].second = tasks[tindex1].deadline - delay_as_per_cur_prior[tindex1];
                tsk_pair[i].second = tasks[tindex2].deadline - delay_as_per_cur_prior[tindex2];
                if(tsk_pair[0].second >= 0){
                    satis = true;
                    break;
                }
            }
        }
    }
    return satis;
}

//assign priority as per DM...ignore discarded tasks
void assign_prior_dm_heur_partial(){
    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(num_max_terms[i][j] > 0){
                if(tasks[i].deadline < tasks[j].deadline){//shorter deadline
                    rel_prior_of_task[i][j] = 1;
                    rel_prior_of_task[j][i] = 0;
                }
                else{
                    rel_prior_of_task[i][j] = 0;
                    rel_prior_of_task[j][i] = 1;
                }
            }
        }
    }
}

//previous iteration of repair failed. discard a task and repair
bool compute_partial_soln_repair_failed(){
    vector <pair <int, int> > tsk_pair;
    clear_2d_vc_initialize_neg1(rel_prior_of_task, no_tasks, no_tasks);
    assign_prior_dm_heur_partial();
    compute_updated_delay_partial();

    bool all_satis = true;
    for(int i = 0; i < no_tasks; i++){
        if(is_present(i + 1, discarded)){
            continue;
        }
        if(delay_as_per_cur_prior[i] > tasks[i].deadline){
            all_satis = false;
        }
        tsk_pair.push_back(make_pair(tasks[i].tid, (tasks[i].deadline - delay_as_per_cur_prior[i])));
    }
    if(all_satis){
        return true;
    }
    //sort in ascending order
    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    if(first_task_satisfied_by_swapping_partial(tsk_pair)){
        return true;
    }
    else{//add first to discard...return false
        discarded.push_back(tsk_pair[0].first);
        return false;
    }
}

//relative heuristic partial - first assign with dm, if not ok repair, if not ok dicard one try to repair, repeat
bool assign_rel_heur_partial(){
    bool status = compute_priority_ordering_if_feasible_rel_heur();

    //if it failed...repair
    if(!status){
        status = repair_soln();
    }

    if(status){//regular heurstic works
        no_tasks_scheduled_this_task_set = no_tasks;
        for(int tindex = 0; tindex < no_tasks; tindex++){
            heav_accepted_rel_heur_partial += get_current_heaviness(tindex);
        }
        
        return true;
    }
    else{
        discarded.clear();
        while(!compute_partial_soln_repair_failed()){
            ;//do nothing call again
        }

        no_tasks_scheduled_this_task_set = no_tasks - discarded.size();
        for(int tindex = 0; tindex < no_tasks; tindex++){
            double _h = get_current_heaviness(tindex);
            if(is_present(tindex + 1, discarded)){
                heav_rejected_rel_heur_partial += _h;
            }
            else{
                heav_accepted_rel_heur_partial += _h;
            }
        }
        return false; //not all scheduled
    }

}
   
bool compute_partial_soln_baseline(){
    vector <pair <int, int> > tsk_pair;
    clear_2d_vc_initialize_neg1(rel_prior_of_task, no_tasks, no_tasks);
    assign_prior_dm_heur_partial();//ignore discarded tasks
    compute_updated_delay_partial();

    bool all_satis = true;
    //check if timing constraint is not satisfied for any task
    for(int i = 0; i < no_tasks; i++){
        if(is_present(i + 1, discarded)){
            continue;
        }
        if(delay_as_per_cur_prior[i] > tasks[i].deadline){
            all_satis = false;
        }
        tsk_pair.push_back(make_pair(tasks[i].tid, (tasks[i].deadline - delay_as_per_cur_prior[i])));
    }
    //if all satis after deletion
    if(all_satis){
        return true;
    }
    //one or more tasks are not satisfied
    sort(tsk_pair.begin(), tsk_pair.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    discarded.push_back(tsk_pair[0].first);
    return false;
}

//relative baseline partial
bool assign_rel_baseline_partial(){
    bool status = compute_priority_ordering_if_feasible_rel_dm();
    
    if(!status){//failed try by discarding
        discarded.clear();
    
        while(!compute_partial_soln_baseline()){
            ; //do nothing call again
        }
        no_tasks_scheduled_this_task_set = no_tasks - discarded.size();
        for(int tindex = 0; tindex < no_tasks; tindex++){
            double _h = get_current_heaviness(tindex);
            if(is_present(tindex + 1, discarded)){
                heav_rejected_rel_baseline_partial += _h;
            }
            else{
                heav_accepted_rel_baseline_partial += _h;
            }
        }
        return false;
    }
    else{
        no_tasks_scheduled_this_task_set = no_tasks;
        for(int tindex = 0; tindex < no_tasks; tindex++){
            heav_accepted_rel_baseline_partial += get_current_heaviness(tindex);
        }
        return true;
    }

}

