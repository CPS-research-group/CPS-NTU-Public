#include "input.h"
#include "task_struct.h"

vector <Task> tasks;

//function to receive static input
void get_static_input(){
    //open input file
    fstream finput(path_to_input + "input_test");
    assert(finput);
  
    //variables to read file
    int var;
    string name;
  
    //read no_input
    finput >> var >> name;
    no_input = var;
}

//get test case specific input
void get_input(int cur_inp){
    ifstream fin(path_to_test_case + "test" + to_string((long long)cur_inp));
    assert(fin);

    int var;

    proc_time_s1.clear();
    proc_time_s2.clear();
    proc_time_s3.clear();

    //get number of APs to upload
    fin >> var;
    no_ap_upload = var;

    //get number of servers
    fin >> var;
    no_server = var;

    no_ap_download = no_ap_upload;

   //get number of tasks
    fin >> var;
    no_tasks = var;

    tasks.clear(); 

    for (int i = 0; i < no_tasks; ++i) {
        tasks.emplace_back(0, 0, 0, 0, 0);
    }

    //get task parameters
    for(int i = 0; i < no_tasks; i++){
        tasks[i].tid = i + 1;

        //get intput size
        fin >> var;
        proc_time_s1.push_back(var);
        
        //get compute size
        fin >> var;
        proc_time_s2.push_back(var);
        
        //get output size
        fin >> var;
        proc_time_s3.push_back(var);
        
        //get dedaline
        fin >> var;
        tasks[i].deadline = var;
        
        //get ap mapping offload
        fin >> var;
        tasks[i].s1 = var;
        
        //get server offload
        fin >> var;
        tasks[i].s2 = var;
        
        //get ap mapping download
        fin >> var;
        tasks[i].s3 = var;
        
    } 
 
}

