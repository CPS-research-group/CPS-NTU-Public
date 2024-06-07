#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <assert.h>
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <random>
#include <unordered_set>
#include <random>

using namespace std;

//....parameters
vector <double> cand_uplink = {500, 1000, 1500, 2000, 2500};//Mbps
//from https://www.intel.com/content/www/us/en/gaming/resources/wifi-6.html ... link rate Wi-Fi 6 (802.11ax) 	2.4/5 GHz 	600–9608 Mbit/s
//from paper Online Optimal Service Selection, Resource Allocation and Task Offloading for Multi-Access Edge Computing: A Utility-Based Approach - 1 - 3 Mbps with step size 0.5
vector <double> cand_downlink = {500, 750, 1000};//Mbps    downlink is half of uplink such as in wifi uplink 2.4 and 5, but typically donwlink at 2.4
vector <double> cand_compute = {5000, 6000, 7000, 8000, 9000, 10000};//Mega cycles/s     [5Ghz - 10 GHz for each proc]

//task
int input_min = 5;//Mb
int input_max = 100;//Mb
int cycle_min = 500;//M cycles
int cycle_max = 2500;//M cycles
int output_min = 2;//Mb
int output_max = 50;//Mb

double heavy;//a task with heaviness this value or more is termed as heavy
int max_deadline;//max possible deadline
//....regular code variables and data structures

// Create a random number generator
random_device rd;
mt19937 generator(rd());
uniform_real_distribution<double> distribution(1.01, 1.1); //for deadline adjustment
uniform_real_distribution<double> distribution2(0.9, 1.0);//for exec time adjustment

vector <int> _1dint;
int total_no_res;

string path_to_input="../input/";
string path_to_test="../test_case/";

// uplink_min , uplink_max ,downlink_min , downlink_max ,, deadline_min , deadline_max compute_min , compute_max ,
int no_test, no_ap_up,  no_server ,  no_ap_down ,  no_tasks;
double heaviness_req_s1,  heaviness_req_s2,  heaviness_req_s3; 
int no_heavy_tasks_req_s1,  no_heavy_tasks_req_s2,  no_heavy_tasks_req_s3;
//define task type
struct Task{
    int tid;//task number...not index 
    int in_size;//input size
    int cycles;//computation cycles
    int out_size;//output size
    int stage1_time;//processing time at stage 1
    int stage2_time;//processing time at stage 2
    int stage3_time;//processing time at stage 3
    int deadline;//deadline
    int s1;//stage 1 mapping
    int s2;//stage 2 mapping
    int s3;//stage 3 mapping
    //int prior;//task priority
    int status_stage;//0 - initialize/waiting to be offloaded, 1 - offloading going on, 2 - offloading finished/waiting to be processed, 3 - procecssing going on, 4 - processing finished/waiting to be downloaded,  5 - downloading going on, 6- task finished
    

    // Task(int _tid, int _deadline, int _s1, int _s2, int _s3) : tid(_tid), deadline(_deadline), s1(_s1), s2(_s2), s3(_s3) {}
    Task(int _tid, int _in_size, int _cycles, int _out_size, int _stage1_time, int _stage2_time, int _stage3_time,  int _deadline, int _s1, int _s2, int _s3, int _status_stage) : tid(_tid), in_size(_in_size), cycles(_cycles), out_size(_out_size), stage1_time(_stage1_time), stage2_time(_stage2_time), stage3_time(_stage3_time), deadline(_deadline), s1(_s1), s2(_s2), s3(_s3), status_stage(_status_stage) {}
    // Task(int _tid, int _in_size, int _cycles, int _out_size, int _stage1_time, int _stage2_time, int _stage3_time,  int _deadline, int _s1, int _s2, int _s3, int _prior, int _status_stage) : tid(_tid), in_size(_in_size), cycles(_cycles), out_size(_out_size), stage1_time(_stage1_time), stage2_time(_stage2_time), stage3_time(_stage3_time), deadline(_deadline), s1(_s1), s2(_s2), s3(_s3), prior(_prior), status_stage(_status_stage) {}

};

vector <Task> tasks;
vector <int> uplink_rate, compute_rate, downlink_rate;
double avg_heaviness, max_heaviness_taskset;
vector <int> task_list;
// vector <bool> offloading_going_on, processing_going_on, downloading_going_on;
vector <int> rem_s1_time, rem_s2_time, rem_s3_time;
vector <int> list_of_heavy_tasks_s1, list_of_heavy_tasks_s2, list_of_heavy_tasks_s3;
vector <int> rand_order_of_task_index;

vector < vector <int> > rel_prior_of_task, meeting;