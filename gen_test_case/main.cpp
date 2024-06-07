
#include "main.h"


void wait_for_char(){
  char ch; cin >> ch;
}

//clear and initialize a 2d vector with 0
void clear_2d_vc_initialize_zero(vector < vector < int > > &vc, int sz1, int sz2){
    vc.clear();
    _1dint.clear();

    for(int i = 0 ; i < sz2; i++){
        _1dint.push_back(0);
    }

    for(int i = 0; i < sz1; i++){
        vc.push_back(_1dint);
    }
}

//clear and initialize a 2d vector with -1
void clear_2d_vc_initialize_neg1(vector < vector <int> > &vc, int sz1, int sz2){
    vc.clear();
    
    for(int i = 0; i < sz1; i++){
        _1dint.clear();
        for(int j = 0; j < sz2; j++){
            _1dint.push_back(-1);
        }
        vc.push_back(_1dint);
    }
}

//get_heaviness_of_a_resource_at_a_stage
double get_heaviness_of_a_resource_at_a_stage(int stg, int res){
  double _hv = 0;
  if(stg == 1){//s1
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s1 == res){//include its heaviness
        _hv += (double)(tasks[tindex].stage1_time)/tasks[tindex].deadline;
      }
    }
  }
  else if(stg == 2){//s2
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s2 == res){//include its heaviness
        _hv += (double)(tasks[tindex].stage2_time)/tasks[tindex].deadline;
      }
    }
  }
  else{//s3
    for(int tindex = 0; tindex < no_tasks; tindex++){
      if(tasks[tindex].s3 == res){//include its heaviness
        _hv += (double)(tasks[tindex].stage3_time)/tasks[tindex].deadline;
      }
    }
  }
  return _hv;
}


//new function
void print_heaviness_of_resource(){
  for(int i = 0; i < no_ap_up; i++){
    cout << get_heaviness_of_a_resource_at_a_stage(1, i + 1) << "\t";
  }
  cout << "\n\n";
  for(int i = 0; i < no_server; i++){
    cout << get_heaviness_of_a_resource_at_a_stage(2, i + 1) << "\t";
  }
  cout << "\n\n";
  for(int i = 0; i < no_ap_down; i++){
    cout << get_heaviness_of_a_resource_at_a_stage(3, i + 1) << "\t";
  }
  cout << "\n";
  
}


//return the heaviest resrouce at stage
int get_heaviest_resource_at_stage(int stg){
  int heav_res;
  double cur_highest_heav, _hv;

  heav_res = 1;
  cur_highest_heav = get_heaviness_of_a_resource_at_a_stage(stg, 1);//get heaviness of first resource...initialize

  if(stg == 1){
    for(int i = 1; i < no_ap_up; i++){
      _hv = get_heaviness_of_a_resource_at_a_stage(stg, i + 1);
      if(_hv > cur_highest_heav){
        heav_res = i + 1;
        cur_highest_heav = _hv;
      }
    }
  }
  else if(stg == 2){
    for(int i = 1; i < no_server; i++){
      _hv = get_heaviness_of_a_resource_at_a_stage(stg, i + 1);
      if(_hv > cur_highest_heav){
        heav_res = i + 1;
        cur_highest_heav = _hv;
      }
    }    
  }
  else{
    for(int i = 1; i < no_ap_down; i++){
      _hv = get_heaviness_of_a_resource_at_a_stage(stg, i + 1);
      if(_hv > cur_highest_heav){
        heav_res = i + 1;
        cur_highest_heav = _hv;
      }
    }
  }

  return heav_res;
}

//get current heaviness of taskset...as per max at each stage calculation
double get_heaviness_taskset(){

    int _h = get_heaviest_resource_at_stage(1);//get heaviest resource
    int heaviest_resource = _h;//get heaviest resource
    double _heav = get_heaviness_of_a_resource_at_a_stage(1, _h);
    double cur_heav = _heav;

    _h = get_heaviest_resource_at_stage(2);//get heaviest resource
    _heav = get_heaviness_of_a_resource_at_a_stage(2, _h);

    if(_heav > cur_heav){
        heaviest_resource = _h;
        cur_heav = _heav;
    }

    _h = get_heaviest_resource_at_stage(3);//get heaviest resource
    _heav = get_heaviness_of_a_resource_at_a_stage(3, _h);

    if(_heav > cur_heav){
        heaviest_resource = _h;
        cur_heav = _heav;
    }

    return cur_heav;


}

//get and print heavy tasks and requirements
void get_and_print_heavy_tasks_and_requirement(){
  int c1 = 0, c2 = 0, c3 = 0;

  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage1_time)/tasks[i].deadline >= heavy){
      c1++;
    }
  }
  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage2_time)/tasks[i].deadline >= heavy){
      c2++;
    }
  }

  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage3_time)/tasks[i].deadline >= heavy){
      c3++;
    }
  }
  cout << "\tstage heaviness count " << c1 << "\t" << c2 << "\t" << c3 << "\n";
  cout << "\tstage heaviness req count " << no_heavy_tasks_req_s1 << "\t" << no_heavy_tasks_req_s2 << "\t" << no_heavy_tasks_req_s3 << "\n";

  //total heaviness
  cout << "\t" << get_heaviness_taskset() << "\n";
}

//get current average avg_heaviness
double get_current_avg_heaviness(){
  double hv = 0;
  for(int i = 0; i < no_tasks; i++){
    hv += (double)(tasks[i].stage1_time + tasks[i].stage2_time + tasks[i].stage3_time)/tasks[i].deadline;
  }
  return hv/no_tasks;
}

//get heaviness of task
double get_current_heaviness(int i){
  return (double)(tasks[i].stage1_time + tasks[i].stage2_time + tasks[i].stage3_time)/tasks[i].deadline;
}


//get number of heavy tasks at stage1 
int get_number_heavy_tasks_at_stage1(){
  int cnt = 0;

  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage1_time)/tasks[i].deadline >= heavy)
      cnt++;
  }
  return cnt;
}

//get number of heavy tasks at stage2 
int get_number_heavy_tasks_at_stage2(){
  int cnt = 0;

  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage2_time)/tasks[i].deadline >= heavy)
      cnt++;
  }
  return cnt;
}

//get number of heavy tasks at stage3 
int get_number_heavy_tasks_at_stage3(){
  int cnt = 0;

  for(int i = 0; i < no_tasks; i++){
    if((double)(tasks[i].stage3_time)/tasks[i].deadline >= heavy)
      cnt++;
  }
  return cnt;
}

//get stage 3 heaviness of this task index
double get_stage3_heaviness(int i) {
  return (double)(tasks[i].stage3_time)/tasks[i].deadline;
}

//get stage 2 heaviness of this task index
double get_stage2_heaviness(int i) {
  return (double)(tasks[i].stage2_time)/tasks[i].deadline;
}

//get stage 1 heaviness of this task index
double get_stage1_heaviness(int i) {
  return (double)(tasks[i].stage1_time)/tasks[i].deadline;
}

//print task indic stage wise heaviness
void print_task_heaviness(){
  for(int i = 0; i < no_tasks; i++){
    cout << i + 1 << "\t" << get_stage1_heaviness(i)  << "\t" << get_stage2_heaviness(i)  << "\t" << get_stage3_heaviness(i) << "\n";
  }
}

//return true if a heavy task at s3
bool is_currently_a_heavy_task_at_s3(int i){
  if(get_stage3_heaviness(i) < heavy)
    return false;
  return true;
}

//return true if a heavy task at s2
bool is_currently_a_heavy_task_at_s2(int i){
  if(get_stage2_heaviness(i) < heavy)
    return false;
  return true;
}

//return true if a heavy task at s1
bool is_currently_a_heavy_task_at_s1(int i){
  if(get_stage1_heaviness(i) < heavy)
    return false;
  return true;
}


//return true if a task is mapped to a resource at a stage...all arguments index
bool is_task_mapped_to_resource_at_stage(int tindex, int rindex, int sindex){
  if((sindex == 0 && tasks[tindex].s1 == rindex + 1)||(sindex == 1 && tasks[tindex].s2 == rindex + 1)||(sindex == 2 && tasks[tindex].s3 == rindex + 1)){
    return true;
  }
  return false;
}


//return true if this task index must be a heavy task at stage 3...it has been designated to be
bool is_a_designated_heavy_task_at_s3(int tindex){
  for(int i = 0; i < list_of_heavy_tasks_s3.size(); i++){
    if(list_of_heavy_tasks_s3[i] == tindex + 1){
      return true;
    }
  }
  return false;
}

//return true if this task index must be a heavy task at stage 2...it has been designated to be
bool is_a_designated_heavy_task_at_s2(int tindex){
  for(int i = 0; i < list_of_heavy_tasks_s2.size(); i++){
    if(list_of_heavy_tasks_s2[i] == tindex + 1){
      return true;
    }
  }
  return false;
}

//return true if this task index must be a heavy task at stage 1...it has been designated to be
bool is_a_designated_heavy_task_at_s1(int tindex){
  for(int i = 0; i < list_of_heavy_tasks_s1.size(); i++){
    if(list_of_heavy_tasks_s1[i] == tindex + 1){
      return true;
    }
  }
  return false;
}

// -- try reducing heaviness at s3 subject to per stage heaviness req at the heaviest resource
bool reduce_heaviness_of_heaviest_resource_at_s3(){
  int heaviest_resource = get_heaviest_resource_at_stage(3);
  
  bool reduced = false;
  random_shuffle(rand_order_of_task_index.begin(), rand_order_of_task_index.end());
  for(int i = 0; i < no_tasks; i++){//consider tasks in random order for reduction
    int tindex = rand_order_of_task_index[i];
    if(is_task_mapped_to_resource_at_stage(tindex, heaviest_resource - 1, 2)){
      if(is_a_designated_heavy_task_at_s3(tindex)){//cannot reduce heaviness less than threshold
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        if((double)(tasks[tindex].stage3_time)/_t_d >= heavy){//still a heavy task after reduction at s3
          bool stat = true;//by default true 
          if(is_a_designated_heavy_task_at_s1(tindex) && ((double)(tasks[tindex].stage1_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s2(tindex) && ((double)(tasks[tindex].stage2_time)/_t_d < heavy)){
            stat = false;
          }
          if(stat && _t_d <= max_deadline){
            tasks[tindex].deadline = _t_d;
            reduced = true;
            break;
          }
        }
      }
      else{//not a heavy task..can be reduced subject to other stages of this task...it may be heavy at other stages
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        bool stat = true;//by default true 
        if(is_a_designated_heavy_task_at_s1(tindex) && ((double)(tasks[tindex].stage1_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s2(tindex) && ((double)(tasks[tindex].stage2_time)/_t_d < heavy)){
          stat = false;
        }
        if(stat && _t_d <= max_deadline){
          tasks[tindex].deadline = _t_d;
          reduced = true;
          break;
        }
      }
    }
  }

  return reduced;
}

// -- try reducing heaviness at s2 subject to per stage heaviness req at the heaviest resource
bool reduce_heaviness_of_heaviest_resource_at_s2(){
  int heaviest_resource = get_heaviest_resource_at_stage(2);

  bool reduced = false;
  random_shuffle(rand_order_of_task_index.begin(), rand_order_of_task_index.end());
  for(int i = 0; i < no_tasks; i++){//consider tasks in random order for reduction
    int tindex = rand_order_of_task_index[i];
    if(is_task_mapped_to_resource_at_stage(tindex, heaviest_resource - 1, 1)){
      if(is_a_designated_heavy_task_at_s2(tindex)){//cannot reduce heaviness less than threshold
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        if((double)(tasks[tindex].stage2_time)/_t_d >= heavy){//still a heavy task after reduction
          bool stat = true;//by default true 
          if(is_a_designated_heavy_task_at_s1(tindex) && ((double)(tasks[tindex].stage1_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s3(tindex) && ((double)(tasks[tindex].stage3_time)/_t_d < heavy)){
            stat = false;
          }
          if(stat && _t_d <= max_deadline){
            tasks[tindex].deadline = _t_d;
            reduced = true;
            break;
          }
        }
      }
      else{//not a heavy task..can be reduced
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        bool stat = true;//by default true 
        if(is_a_designated_heavy_task_at_s1(tindex) && ((double)(tasks[tindex].stage1_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s3(tindex) && ((double)(tasks[tindex].stage3_time)/_t_d < heavy)){
          stat = false;
        }
        if(stat && _t_d <= max_deadline){
          tasks[tindex].deadline = _t_d;
          reduced = true;
          break;
        }
      }
    }
  }

  return reduced;
}

// -- try reducing heaviness at s1 subject to per stage heaviness req at the heaviest resource
bool reduce_heaviness_of_heaviest_resource_at_s1(){
  int heaviest_resource = get_heaviest_resource_at_stage(1);

  bool reduced = false;
  random_shuffle(rand_order_of_task_index.begin(), rand_order_of_task_index.end());
  for(int i = 0; i < no_tasks; i++){//consider tasks in random order for reduction
    int tindex = rand_order_of_task_index[i];
    if(is_task_mapped_to_resource_at_stage(tindex, heaviest_resource - 1, 0)){
      if(is_a_designated_heavy_task_at_s1(tindex)){//cannot reduce heaviness less than threshold
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        if((double)(tasks[tindex].stage1_time)/_t_d >= heavy){//still a heavy task after reduction
          bool stat = true;//by default true 
          if(is_a_designated_heavy_task_at_s2(tindex) && ((double)(tasks[tindex].stage2_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s3(tindex) && ((double)(tasks[tindex].stage3_time)/_t_d < heavy)){
            stat = false;
          }
          if(stat && _t_d <= max_deadline){
            tasks[tindex].deadline = _t_d;
            reduced = true;
            break;
          }
        }
      }
      else{//not a heavy task..can be reduced
        int _t_d = ceil(tasks[tindex].deadline *  distribution(generator));//potential updated deadline
        bool stat = true;//by default true 
        if(is_a_designated_heavy_task_at_s2(tindex) && ((double)(tasks[tindex].stage2_time)/_t_d < heavy)){
            stat = false;
          }
          if(is_a_designated_heavy_task_at_s3(tindex) && ((double)(tasks[tindex].stage3_time)/_t_d < heavy)){
          stat = false;
        }
        if(stat && _t_d <= max_deadline){
          tasks[tindex].deadline = _t_d;
          reduced = true;
          break;
        }
      }
    }
  }

  return reduced;
}

//repair task set to satisfy heaviness requirement
bool repair_taskset_for_heaviness_req(){
  int trial_count = 0;

  if(get_number_heavy_tasks_at_stage3() < no_heavy_tasks_req_s3){
    cout << "No. of heavy tasks is less than threshold\n";
    return false;
  }
  while(get_number_heavy_tasks_at_stage3() != no_heavy_tasks_req_s3){

      int tindex = rand()%no_tasks; //select a task randomly
      int _trail = 0;
      while(is_currently_a_heavy_task_at_s3(tindex)){//if this is a heavy task then decrease its s3 processing time till it is heavy at that stage
        
        tasks[tindex].stage3_time = max(1, (int)floor(tasks[tindex].stage3_time * distribution2(generator)));
        if(_trail++ > 10000 * no_tasks){//cannot be reduced further
          break;
        }
      }

      if(trial_count++ > no_tasks * 100000){
        cout << "Stage 3 stuck \t Going to wait for char\n";
        cout<<get_number_heavy_tasks_at_stage3() << " " <<  no_heavy_tasks_req_s3<<"\n";
        wait_for_char();
       }
  }


  trial_count = 0;

  if(get_number_heavy_tasks_at_stage1() < no_heavy_tasks_req_s1){
    cout << "No. of heavy tasks is less than threshold\n";
    return false;
  }
  while(get_number_heavy_tasks_at_stage1() != no_heavy_tasks_req_s1){
      int tindex = rand()%no_tasks; //select a task randomly
      int _trail = 0;
      while(is_currently_a_heavy_task_at_s1(tindex)){//if this is a heavy task then decrease its s1 processing time till it is heavy at that stage
        tasks[tindex].stage1_time = max(1, (int)floor(tasks[tindex].stage1_time * distribution2(generator)));

        if(_trail++ > 10000* no_tasks){//cannot be reduced further
          break;
        }
      }

      if(trial_count++ > no_tasks * 100000){
        cout << "Stage 1 stuck \t Going to wait for char\n";
        wait_for_char();
       }
  }


  trial_count = 0;

  if(get_number_heavy_tasks_at_stage2() < no_heavy_tasks_req_s2){
    cout << "No. of heavy tasks is less than threshold\n";
    return false;
  }
  while(get_number_heavy_tasks_at_stage2() != no_heavy_tasks_req_s2){

      int tindex = rand()%no_tasks; //select a task randomly
      int _trail = 0;
      while(is_currently_a_heavy_task_at_s2(tindex)){//if this is a heavy task then decrease its s2 processing time till it is heavy at that stage
        tasks[tindex].stage2_time = max(1, (int)floor(tasks[tindex].stage2_time * distribution2(generator)));
        if(_trail++ > 10000 * no_tasks){//cannot be reduced further
          break;
        }
      }

      if(trial_count++ > no_tasks * 200000){
        cout << "Stage 2 stuck \t Going to wait for char\n";
        wait_for_char();
       }
  }

  list_of_heavy_tasks_s1.clear();
  list_of_heavy_tasks_s2.clear();
  list_of_heavy_tasks_s3.clear();
  //get_list_of_heavy_tasks
  for(int tindex = 0; tindex < no_tasks; tindex++){
    if(is_currently_a_heavy_task_at_s1(tindex)){
      list_of_heavy_tasks_s1.push_back(tindex + 1);
    }

    if(is_currently_a_heavy_task_at_s2(tindex)){
      list_of_heavy_tasks_s2.push_back(tindex + 1);
    }

    if(is_currently_a_heavy_task_at_s3(tindex)){
      list_of_heavy_tasks_s3.push_back(tindex + 1);
    }
  }


  double allowed_heaviness_factor = 1;
  for(int i = 0; i < list_of_heavy_tasks_s1.size(); i++){
    int tindex = list_of_heavy_tasks_s1[i] - 1;
    while(get_stage1_heaviness(tindex) > heavy + allowed_heaviness_factor * heavy){
      tasks[tindex].stage1_time -=  ceil(tasks[tindex].stage1_time * 0.01);//decrease by 1%
    }
  }
  for(int i = 0; i < list_of_heavy_tasks_s2.size(); i++){
    int tindex = list_of_heavy_tasks_s2[i] - 1;
    while(get_stage2_heaviness(tindex) > heavy + allowed_heaviness_factor * heavy){
      tasks[tindex].stage2_time -=  ceil(tasks[tindex].stage2_time * 0.01);//decrease by 1%
    }
  }
  for(int i = 0; i < list_of_heavy_tasks_s3.size(); i++){
    int tindex = list_of_heavy_tasks_s3[i] - 1;
    while(get_stage3_heaviness(tindex) > heavy + allowed_heaviness_factor * heavy){
      tasks[tindex].stage3_time -=  ceil(tasks[tindex].stage3_time * 0.01);//decrease by 1%
    }
  }

  //while current heaviness is more than the bound..reduce heaviness for heaviest resource at a stage...update heaviest resource if required...check heaviness again...if req repeat...if heaviness does not decreases...then return false
  double cur_heav = get_heaviness_taskset();//get current heaviness of taskset...as per max at each stage calculation
  int trial = 0;
  bool reduced_this_iter = true;
  while(cur_heav > max_heaviness_taskset && reduced_this_iter){
    if(trial++ > 10000 * no_tasks){
      cout << "\t\t*Returning...Trials exhausted...Max heaviness req Not satisfied\n";
      return false;//unsuccesful
    }
    reduced_this_iter = false;

    int heaviest_resource1 = get_heaviest_resource_at_stage(1);//heaviest resource at S1
    int heaviest_resource2 = get_heaviest_resource_at_stage(2);//heaviest resource at S2
    int heaviest_resource3 = get_heaviest_resource_at_stage(3);//heaviest resource at S3
    bool status;

    int _rnd = rand()%3;
    if(_rnd == 0){//s1 s2 s3
      status = false;
      if(get_heaviness_of_a_resource_at_a_stage(1, heaviest_resource1) > max_heaviness_taskset)
        status = reduce_heaviness_of_heaviest_resource_at_s1();

      if(status){//reduction successful
        cur_heav = get_heaviness_taskset();
        reduced_this_iter = true;
      }
      else{//consider s2
        if(get_heaviness_of_a_resource_at_a_stage(2, heaviest_resource2) > max_heaviness_taskset)
          status = reduce_heaviness_of_heaviest_resource_at_s2();

        if(status){//reduction successful
          cur_heav = get_heaviness_taskset();
          reduced_this_iter = true;
        }
        else{//consider s3
          if(get_heaviness_of_a_resource_at_a_stage(3, heaviest_resource3) > max_heaviness_taskset)
            status = reduce_heaviness_of_heaviest_resource_at_s3();

          if(status){//reduction successful
            cur_heav = get_heaviness_taskset();
            reduced_this_iter = true;
          }
          else{
            cout << "\t\t\tReturning...could not satis total heaviness req with per stage heav\n";
            return false;//unsuccesful
          }
        }
      }
    }
    else if(_rnd == 1){//s2 s3 s1
      status = false;
      
      if(get_heaviness_of_a_resource_at_a_stage(2, heaviest_resource2) > max_heaviness_taskset)
          status = reduce_heaviness_of_heaviest_resource_at_s2();

      if(status){//reduction successful
        cur_heav = get_heaviness_taskset();
        reduced_this_iter = true;
      }
      else{//consider s3
        if(get_heaviness_of_a_resource_at_a_stage(3, heaviest_resource3) > max_heaviness_taskset)
            status = reduce_heaviness_of_heaviest_resource_at_s3();

        if(status){//reduction successful
          cur_heav = get_heaviness_taskset();
          reduced_this_iter = true;
        }
        else{//consider s1
          if(get_heaviness_of_a_resource_at_a_stage(1, heaviest_resource1) > max_heaviness_taskset)
            status = reduce_heaviness_of_heaviest_resource_at_s1();

          if(status){//reduction successful
            cur_heav = get_heaviness_taskset();
            reduced_this_iter = true;
          }
          else{
            cout << "\t\t\tReturning...could not satis total heaviness req with per stage heav\n";
            return false;//unsuccesful
          }
        }
      }
    }
    else{//s3 s1 s2
      status = false;
      
      if(get_heaviness_of_a_resource_at_a_stage(3, heaviest_resource3) > max_heaviness_taskset)
        status = reduce_heaviness_of_heaviest_resource_at_s3();

      if(status){//reduction successful
        cur_heav = get_heaviness_taskset();
        reduced_this_iter = true;
      }
      else{//consider s1
        if(get_heaviness_of_a_resource_at_a_stage(1, heaviest_resource1) > max_heaviness_taskset)
          status = reduce_heaviness_of_heaviest_resource_at_s1();

        if(status){//reduction successful
          cur_heav = get_heaviness_taskset();
          reduced_this_iter = true;
        }
        else{//consider s2
          if(get_heaviness_of_a_resource_at_a_stage(2, heaviest_resource2) > max_heaviness_taskset)
            status = reduce_heaviness_of_heaviest_resource_at_s2();

          if(status){//reduction successful
            cur_heav = get_heaviness_taskset();
            reduced_this_iter = true;
          }
          else{
            cout << "\t\t\tReturning...could not satis total heaviness req with per stage heav\n";
            return false;//unsuccesful
          }
        }
      }
    }
  }

  if(get_heaviness_taskset() > max_heaviness_taskset){
    return false;
  }

  //all well
  return true;
}

//save heaviness of each task to a file
void export_heaviness(int test_count){
  ofstream fout(path_to_test + "heaviness" + to_string((long)test_count));
  assert(fout);
  fout << get_current_avg_heaviness() << "\n";
  
  int c1 = 0, c2 = 0, c3 = 0;
  for(int i = 0; i < no_tasks; i++){
    fout << (double)(tasks[i].stage1_time + tasks[i].stage2_time + tasks[i].stage3_time)/tasks[i].deadline << "\n";
    // fout << hv << "\n"; //avegrage is being added for each task...so as to get a straight line in plot
    cout << (double)(tasks[i].stage1_time)/tasks[i].deadline << "\t" << (double)(tasks[i].stage2_time)/tasks[i].deadline << "\t" << (double)(tasks[i].stage3_time)/tasks[i].deadline  << "\t"<< (double)(tasks[i].stage1_time + tasks[i].stage2_time + tasks[i].stage3_time)/tasks[i].deadline << "\n";
    if((double)(tasks[i].stage1_time)/tasks[i].deadline >= heavy)
      c1++;
    if((double)(tasks[i].stage2_time)/tasks[i].deadline >= heavy)
      c2++;
    if((double)(tasks[i].stage3_time)/tasks[i].deadline >= heavy)
      c3++;
    
  }
  cout << "stage heaviness count " << c1 << "\t" << c2 << "\t" << c3 << "\n";
  cout << "stage heaviness req count " << no_heavy_tasks_req_s1 << "\t" << no_heavy_tasks_req_s2 << "\t" << no_heavy_tasks_req_s3 << "\n";

  fout.close();
}

//write to file
void write_to_file(int test_count){
    //open file
    ofstream fout(path_to_test + "test" + to_string((long)test_count));
    assert(fout);

    //no of ap to upload
    fout << no_ap_up << "\n";

    //number of servers
    fout << no_server << "\n";

   //get number of tasks
    fout << no_tasks << "\n";

    //task parameters
    for(int i = 0; i < no_tasks; i++){

      //stage 1 proc time
      fout << tasks[i].stage1_time << "\n";

      //stage 2 proc time
      fout << tasks[i].stage2_time << "\n";

      //stage 3 proc time
      fout << tasks[i].stage3_time << "\n";

      //deadline
      fout << tasks[i].deadline << "\n";
      if(tasks[i].deadline > 100000000 || tasks[i].deadline < 0){
        cout << " Issue1 \n";
        wait_for_char();
      }

      // cout << tasks[i].stage1_time << "\t"  << tasks[i].stage2_time << "\t" << tasks[i].stage3_time << "\t" << tasks[i].deadline  << "\t" << (double)(tasks[i].stage1_time + tasks[i].stage2_time + tasks[i].stage3_time)/tasks[i].deadline  << "\n";

      //mapping 
      fout << tasks[i].s1 << "\n";
      fout << tasks[i].s2 << "\n";
      fout << tasks[i].s3 << "\n";

    } 

    fout.close();
}




//continue/start processing this task
void process_this_task(int tindex, int cur_time){
  rem_s2_time[tindex]--;
  if(rem_s2_time[tindex] == 0){//finished
    tasks[tindex].status_stage = 4;
  }
}

//get tasks waiting for or are being processed
void get_tasks_waiting_or_being_processed_at_this_server(int sindex){
  task_list.clear();
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].s2 == sindex + 1 && (tasks[i].status_stage == 2 || tasks[i].status_stage == 3)){
      task_list.push_back(i + 1);
    }
  }

}

//continue/start offloading this task
void offload_this_task(int tindex, int cur_time){
  rem_s1_time[tindex]--;
  if(rem_s1_time[tindex] == 0){//finished
    tasks[tindex].status_stage = 2;
  }
}


//continue/start downloading this task
void download_this_task(int tindex, int cur_time){
  if(cur_time > 100000000 || cur_time < 0){
    cout << cur_time << " Issue0 \n";
    wait_for_char();
  }
  rem_s3_time[tindex]--;
  if(rem_s3_time[tindex] == 0){//finished
    tasks[tindex].status_stage = 6;
    tasks[tindex].deadline = ceil(cur_time);//finish time is deadline
    assert(tasks[tindex].deadline <= max_deadline);

  }
}

//get the highest priority task
int get_highest_priority_task_of_task_list_vector(){
  if(task_list.size() == 1){
    return task_list[0] - 1;
  }

  int pos1, pos2, winner_tindex = task_list[0] - 1, sz = task_list.size();

  for(pos1 = 0; pos1 < sz - 1; pos1++){
    int tindex1 = task_list[pos1] - 1;
    for(pos2 = pos1 + 1; pos2 < sz; pos2++){
      int tindex2 = task_list[pos2] - 1;
      if(rel_prior_of_task[tindex1][tindex2] == -1)
        assert(0);

      if(rel_prior_of_task[tindex1][tindex2] == 1){//first is of higher prior
        if(rel_prior_of_task[tindex1][winner_tindex] == 1){//current task is of higher prior
          winner_tindex = tindex1;
        }
      }
      else{//second is of higher priority
        if(rel_prior_of_task[tindex2][winner_tindex] == 1){//current task is of higher prior
          winner_tindex = tindex2;
        }
      }
    }
  }

  return winner_tindex;
}


//get list of tasks waiting for offloading from this AP in task_list
void get_tasks_waiting_to_be_offloaded_this_ap(int ap_index){
  task_list.clear();
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].s1 == ap_index + 1 && tasks[i].status_stage == 0){
      task_list.push_back(i + 1);
    }
  }
}


//get list of tasks waiting for downloading from this AP in task_list
void get_tasks_waiting_to_be_downloaded_this_ap(int ap_index){
  task_list.clear();
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].s3 == ap_index + 1 && tasks[i].status_stage == 4){
      task_list.push_back(i + 1);
    }
  }
}


//get the task for which offloading is going on at this AP
int get_tasks_being_offloaded_this_ap(int ap_index){
  int ind = -1;
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].s1 == ap_index + 1 && tasks[i].status_stage == 1){
      return i;
    }
  }
  return -1;
}


//get the task for which downloading is going on at this AP
int get_tasks_being_downloaded_this_ap(int ap_index){
  int ind = -1;
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].s3 == ap_index + 1 && tasks[i].status_stage == 5){
      return i;
    }
  }
  return -1;
}

//return false if any task is yet to finish
bool all_tasks_finished(){
  for(int i = 0; i < no_tasks; i++){
    if(tasks[i].status_stage < 6)
      return false;
  }
  return true;
}


//compute meeting information...if two tasks are meeting each other at least once
void compute_meeting_info(){
  clear_2d_vc_initialize_zero(meeting, no_tasks, no_tasks);//clear and initialize a 2d vector with 0
  for(int i = 0; i < no_tasks - 1; i++){
    for(int j = i + 1; j < no_tasks; j++){
      if((tasks[i].s1 == tasks[j].s1 || tasks[i].s2 == tasks[j].s2) || tasks[i].s3 == tasks[j].s3){
        meeting[i][j] = 1;
        meeting[j][i] = 1;
      }
    }
  }

}



//assign relative priority to tasks
void assign_relative_priority(){
  clear_2d_vc_initialize_neg1(rel_prior_of_task, no_tasks, no_tasks);

  //compute meeting information...if two tasks are meeting each other at least once
  compute_meeting_info();//may be removed..for debugging

  vector <int> cand_tasks;
  for(int i = 0; i < no_tasks; i++){
    cand_tasks.push_back(i + 1);
  }
  random_shuffle(cand_tasks.begin(), cand_tasks.end());

  vector < vector <int> > partial_sched;
  //initialize with empty vector
  _1dint.clear();
  for(int i = 0; i < total_no_res; i++){
    partial_sched.push_back(_1dint);
  }

  while(cand_tasks.size() > 0){
    int tindex = cand_tasks[0] - 1;
    partial_sched[tasks[tindex].s1 - 1].emplace(partial_sched[tasks[tindex].s1 - 1].begin(), tindex  + 1);
    partial_sched[no_ap_up + tasks[tindex].s2 - 1].emplace(partial_sched[no_ap_up + tasks[tindex].s2 - 1].begin(), tindex  + 1);
    partial_sched[no_ap_up + no_server + tasks[tindex].s3 - 1].emplace(partial_sched[no_ap_up + no_server + tasks[tindex].s3 - 1].begin(), tindex  + 1);
    cand_tasks.erase(cand_tasks.begin());
  }
  for(int rindex = 0; rindex < total_no_res; rindex++){
    if(partial_sched[rindex].size() == 0)
      continue;

    for(int ind1 = 0; ind1 < partial_sched[rindex].size() - 1; ind1++){
      for(int ind2 = ind1 + 1; ind2 < partial_sched[rindex].size(); ind2++){
        int tindex1 = partial_sched[rindex][ind1] - 1;
        int tindex2 = partial_sched[rindex][ind2] - 1;
        if(rel_prior_of_task[tindex1][tindex2] != -1){//if modified
          assert(rel_prior_of_task[tindex1][tindex2] == 1);
        }

        rel_prior_of_task[tindex1][tindex2] = 1;
        rel_prior_of_task[tindex2][tindex1] = 0;

        assert((meeting[tindex1][tindex2] == 1));//may be removed...for debugging
      }
    }
  }
}

void print_task_to_resource_mapping(){
  cout << "Stage 1\n";
  for(int i = 0; i < no_ap_up; i++){
    cout << "AP" << i + 1 << "\t";
    for(int j = 0; j < no_tasks; j++){
      if(tasks[j].s1 == i + 1){
        cout << j + 1 << "\t";
      }
    }
    cout << "\n";
  }
  cout << "\n";
  
  cout << "Stage 2\n";
  for(int i = 0; i < no_server; i++){
    cout << "S" << i + 1 << "\t";
    for(int j = 0; j < no_tasks; j++){
      if(tasks[j].s2 == i + 1){
        cout << j + 1 << "\t";
      }
    }
    cout << "\n";
  }
  cout << "\n";
  
  cout << "Stage 3\n";
  for(int i = 0; i < no_ap_down; i++){
    cout << "AP" << i + 1 << "\t";
    for(int j = 0; j < no_tasks; j++){
      if(tasks[j].s3 == i + 1){
        cout << j + 1 << "\t";
      }
    }
    cout << "\n";
  }
  cout << "\n";
  

}

void determine_num_tasks_in_each_cluster(vector <int> & vc, int n, int m, double var){
  vc.clear();

  double lowerLimit = 1 - var;  // Lower limit (70% of the average)
  double upperLimit = 1 + var;  // Upper limit (130% of the average)

  double average = static_cast<double>(n) / m;

  random_device rd;
  default_random_engine generator(rd());

  vector<int> boxes(m, 0);

  for (int i = 0; i < n; ++i) {
      int boxIndex;
      do {
          uniform_int_distribution<int> distribution(0, m - 1);
          boxIndex = distribution(generator);
      } while (boxes[boxIndex] >= upperLimit * average);

      boxes[boxIndex]++;
  }

  for (int i = 0; i < m; ++i) {
      vc.push_back(boxes[i]);
  }
}



//assign assign_uplink_compute_downlink_rate
void assign_uplink_compute_downlink_rate(){
  //generate AP and server parameters
  uplink_rate.clear();
  compute_rate.clear();
  downlink_rate.clear();

  int sz = cand_uplink.size();
  for(int i = 0; i < no_ap_up; i++){
    uplink_rate.push_back(cand_uplink[rand()%sz]);
  }

  sz = cand_compute.size();
  for(int i = 0; i < no_server; i++){
      compute_rate.push_back(cand_compute[rand()%sz]);
  }

  sz = cand_downlink.size();
  for(int i = 0; i < no_ap_down; i++){
      downlink_rate.push_back(cand_downlink[rand()%sz]);
  }
}


//generate test case
void gen_test_case(){
  //generate required no of test cases
  for(int test_count = 1; test_count <= no_test; test_count++){
    cout << "\t****generating test case " << test_count << "\n";

    //assign //generate AP and server parameters
    assign_uplink_compute_downlink_rate();

    //generate task parameters

    tasks.clear();
    //initialize
    for (int i = 0; i < no_tasks; ++i) {
      tasks.emplace_back(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    //compute max deadline
    max_deadline = 1000*(ceil((double)(input_max * 1000)/uplink_rate[0]) + ceil((double)(cycle_max * 1000)/compute_rate[0]) + ceil((double)(output_max * 1000)/downlink_rate[0]));//max possible execution time x 1000


    if(no_tasks%no_ap_up !=0 || no_tasks%no_server != 0 || no_ap_up != no_ap_down){
      cout << "Issue... expecting number of tasks to be a mulitple of both no_ap and no_server\n";
      wait_for_char();
    }
    int num_clusters_ap = no_tasks/no_ap_up, num_clusters_ser = no_tasks/no_server;//number of clusters of tasks for AP and server allocations
    vector < vector <int> > tasks_in_cluster_ap, tasks_in_cluster_server;//tasks in each cluster
    _1dint.clear();
    for(int i = 0; i < no_ap_up; i++){
      tasks_in_cluster_ap.push_back(_1dint);
    }
    _1dint.clear();
    for(int i = 0; i < no_server; i++){
      tasks_in_cluster_server.push_back(_1dint);
    }
    vector <int> num_tasks_in_cluster_ap,  num_tasks_in_cluster_ser;//number of tasks in each cluster
    double variation = 0.3;//this denotes variation on both sides of average number of task per cluster - determines actual number of tasks in each cluster
    determine_num_tasks_in_each_cluster(num_tasks_in_cluster_ap, no_tasks, no_ap_up, variation);//determine number of tasks in each cluster of AP
    determine_num_tasks_in_each_cluster(num_tasks_in_cluster_ser, no_tasks, no_server, variation);//determine number of tasks in each cluster of server
    //add the corresponding number of tasks to each cluster of tasks for AP mapping and server mapping
    int tsk = 1;
    for(int i = 0; i < no_ap_up; i++){
      for(int j = 0; j < num_tasks_in_cluster_ap[i]; j++){
        tasks_in_cluster_ap[i].push_back(tsk);
        tsk++;
      }
      random_shuffle(tasks_in_cluster_ap[i].begin(), tasks_in_cluster_ap[i].end());
    }
    tsk = 1;
    for(int i = 0; i < no_server; i++){
      for(int j = 0; j < num_tasks_in_cluster_ser[i]; j++){
        tasks_in_cluster_server[i].push_back(tsk);
        tsk++;
      }
      random_shuffle(tasks_in_cluster_server[i].begin(), tasks_in_cluster_server[i].end());
    }

    //compute and assign task parameters except mapping and deadline
    for(int tindex = 0; tindex < no_tasks; tindex++){
      tasks[tindex].tid = tindex + 1;

      //get input size
      tasks[tindex].in_size = rand()%(input_max - input_min) + input_min;

      //get compute size
      tasks[tindex].cycles = rand()%(cycle_max - cycle_min) + cycle_min;

      //get output size
      tasks[tindex].out_size = rand()%(output_max - output_min) + output_min;
    }

    //assign mapping 
    double prob = 0.7;//assign tasks of a cluster with this prob to corresponding AP 
    // for each cluster...assign AP offload and download
    for(int cindex = 0; cindex < no_ap_up; cindex++){
      int res = cindex + 1;
      //for each cluster - assign the task of this cluster to corrsponding AP with prob  and to prev or next AP with 1-prob
      for(int ind = 0; ind < num_tasks_in_cluster_ap[cindex]; ind++){
        int tindex = tasks_in_cluster_ap[cindex][ind] - 1;
        //assign this task .. for offloading
        if(rand()%100 < (int)(prob * 100)){//assign to corresponding AP
          tasks[tindex].s1 = res;
        }
        else{
          if(cindex == 0){//if this is the first cluster
            if(rand()%2 == 0)//assign remaining to the next cluster
              tasks[tindex].s1 = res + 1;
            else //to randomly any other
              tasks[tindex].s1 = rand()%no_ap_up + 1;
          }
          else if(cindex == no_ap_up - 1){//if last
            if(rand()%2 == 0)//assign remaining to the prev cluster
              tasks[tindex].s1 = res - 1;
            else //to randomly any other
              tasks[tindex].s1 = rand()%no_ap_up + 1;
          }
          else{//assign randomly to prev or next
            if(rand()%2){
              if(rand()%2){
                tasks[tindex].s1 = res - 1;
              }
              else{
                tasks[tindex].s1 = res + 1;
              }
            }
            else{//to randomly any other
              tasks[tindex].s1 = rand()%no_ap_up + 1;
            }
          }
        }

        //assign this task .. for downloading
        if(rand()%100 < (int)(prob * 100)){//assign to corresponding AP
          tasks[tindex].s3 = res;
        }
        else{
          if(cindex == 0){//if this is the first cluster
            if(rand()%2 == 0)//assign remaining to the next cluster
              tasks[tindex].s3 = res + 1;
            else //to randomly any other
              tasks[tindex].s3 = rand()%no_ap_up + 1;
          }
          else if(cindex == no_ap_up - 1){//if last
            if(rand()%2 == 0)//assign remaining to the prev cluster
              tasks[tindex].s3 = res - 1;
            else //to randomly any other
              tasks[tindex].s3 = rand()%no_ap_up + 1;
          }
          else{//assign randomly to prev or next
            if(rand()%2){
              if(rand()%2){
                tasks[tindex].s3 = res - 1;
              }
              else{
                tasks[tindex].s3 = res + 1;
              }
            }
            else{//to randomly any other
              tasks[tindex].s3 = rand()%no_ap_up + 1;
            }
          }
        }

      } 
    }
    prob = 0;
    // for each cluster...assign server
    for(int cindex = 0; cindex < no_server; cindex++){
      int res = cindex + 1;
      //for each cluster - assign the task of this cluster to corrsponding Server with prob  and to any other Server with 1-prob
      for(int ind = 0; ind < num_tasks_in_cluster_ser[cindex]; ind++){
        int tindex = tasks_in_cluster_server[cindex][ind] - 1;
        //assign this task .. for processing
        if(rand()%100 < (int)(prob * 100)){//assign to corresponding Server
          tasks[tindex].s2 = res;
        }
        else{
          // to any other server
            tasks[tindex].s2 = rand()%no_server + 1;
        }
      }
    }

    //assign stage processing times
    for(int tindex = 0; tindex < no_tasks; tindex++){
      //get processing times at stages
      tasks[tindex].stage1_time = ceil((double)(tasks[tindex].in_size * 1000)/uplink_rate[tasks[tindex].s1 - 1]);//offloading time...ms
      tasks[tindex].stage2_time = ceil((double)(tasks[tindex].cycles * 1000)/compute_rate[tasks[tindex].s2 - 1]);//computation time...ms
      tasks[tindex].stage3_time = ceil((double)(tasks[tindex].out_size * 1000)/downlink_rate[tasks[tindex].s3 - 1]);//donwload time...ms

    }

    //assign relative priority to tasks
    assign_relative_priority();

    //initialize
    int cur_time = 0;

    rem_s1_time.clear();
    rem_s2_time.clear();
    rem_s3_time.clear();

    for(int i = 0; i < no_tasks; i++){
      rem_s1_time.push_back(tasks[i].stage1_time);
      rem_s2_time.push_back(tasks[i].stage2_time);
      rem_s3_time.push_back(tasks[i].stage3_time);
    }

    while(!all_tasks_finished()){//while not all tasks are finished
      cur_time++;//time in future...actual current time is cur_time

      //consider every AP one by one
      for(int ap_index = 0; ap_index <no_ap_down; ap_index++){
        //get the task for which downloading is going on at this AP
        int tindex = get_tasks_being_downloaded_this_ap(ap_index);

        //if any task with downloadin going on
        if(tindex != -1){
          download_this_task(tindex, cur_time);//continue downloading this task
        }
        else{
          get_tasks_waiting_to_be_downloaded_this_ap(ap_index);
          //if at least one task in task_list
          if(task_list.size() > 0){
            //get the highest priority task
            tindex = get_highest_priority_task_of_task_list_vector();
            tasks[tindex].status_stage = 5;
            download_this_task(tindex, cur_time);//start downloading this task
          }
        }
      }
      
      for(int sindex = 0; sindex < no_server; sindex++){
        //get tasks waiting for or are being processed
        get_tasks_waiting_or_being_processed_at_this_server(sindex);
        //if at least one task in task_list
        if(task_list.size() > 0){
          int tindex = get_highest_priority_task_of_task_list_vector();
          tasks[tindex].status_stage = 3;//actually required if 2 then 3, it might be already 3
          process_this_task(tindex, cur_time);//continue/start processing this task
        }
      }
      for(int ap_index = 0; ap_index <no_ap_up; ap_index++){
        //get the task for which offloading is going on at this AP
        int tindex = get_tasks_being_offloaded_this_ap(ap_index);
        //if any task with offloading going on
        if(tindex != -1){
          offload_this_task(tindex, cur_time);//continue offloading this task
        }
        else{
          get_tasks_waiting_to_be_offloaded_this_ap(ap_index);
          if(task_list.size() > 0){
            tindex = get_highest_priority_task_of_task_list_vector();
            tasks[tindex].status_stage = 1;
            offload_this_task(tindex, cur_time);//start downloading this task
          }
        }
      }
    }

    if(repair_taskset_for_heaviness_req()){//heaviness req satisfied
      write_to_file(test_count);

    }
    else{//heaviness req failed..discard this task set...begin again
      test_count--;
    }
  }
}


//read input for test case generation
void get_input(){
  //open input file
  fstream finput(path_to_input + "input_test");
  assert(finput);
  
  //variables to read file
  int var;
  string name;
  double dvar;
  
  //read no_test
  finput >> var >> name;
  no_test = var;

  //read no_ap_up
  finput >> var >> name;
  no_ap_up = var;

  no_ap_down = no_ap_up;

  //read no_server
  finput >> var >> name;
  no_server = var;

  //read no_tasks
  finput >> var >> name;
  no_tasks = var;

  //read max_heaviness_taskset
  finput >> dvar >> name;
  max_heaviness_taskset = dvar;

  //read heaviness req stage 1
  finput >> dvar >> name;
  heaviness_req_s1 = dvar;
  
  no_heavy_tasks_req_s1 = ceil(heaviness_req_s1 * no_tasks);

  //read heaviness req stage 2
  finput >> dvar >> name;
  heaviness_req_s2 = dvar;

  no_heavy_tasks_req_s2 = ceil(heaviness_req_s2 * no_tasks);

  //read heaviness req stage 3
  finput >> dvar >> name;
  heaviness_req_s3 = dvar;

  no_heavy_tasks_req_s3 = ceil(heaviness_req_s3 * no_tasks);

  //read heaviness definition...tasks heavier than this value are termed to be heavy
  finput >> dvar >> name;
  heavy = dvar;

  total_no_res = no_ap_up + no_server + no_ap_down;


}

int main(){
    //few initial things
    srand(time(NULL));

    //read input
    get_input();

    //initailize random order of tasks vector
    rand_order_of_task_index.clear();
    for(int tindex = 0; tindex < no_tasks; tindex++)
      rand_order_of_task_index.push_back(tindex);
    random_shuffle(rand_order_of_task_index.begin(), rand_order_of_task_index.end());

    //generate test case
    gen_test_case();

    return 0;
}

