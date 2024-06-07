#include "intermed.h"


bool reverse_order(int a, int b) {
    return a > b;
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

//store max terms in the decreasing order for each task in a separate vector
void compute_store_max_terms_sorted_order(){

    max_terms_rev_sorted.clear();

    for(int tindex = 0; tindex < no_tasks; tindex++){
        for(int tindex2 = 0; tindex2 < no_tasks; tindex2++){

            _1dint.clear();

            if(tindex == tindex2){//include processing time as it is
                _1dint.push_back(proc_time_s1[tindex2]);//offloading time    
                _1dint.push_back(proc_time_s2[tindex2]);//compute time    
                _1dint.push_back(proc_time_s3[tindex2]);//downloading time
            }
            else{//include if meet at a stage, otherwise 0
                if(tasks[tindex].s1 == tasks[tindex2].s1)//mapped to same resource
                    _1dint.push_back(proc_time_s1[tindex2]);//offloading time    
                else//mapped to diff resource
                    _1dint.push_back(0);//0

                if(tasks[tindex].s2 == tasks[tindex2].s2)//mapped to same resource
                    _1dint.push_back(proc_time_s2[tindex2]);//compute time    
                else//mapped to diff resource
                    _1dint.push_back(0);//0
                
                if(tasks[tindex].s3 == tasks[tindex2].s3)//mapped to same resource
                    _1dint.push_back(proc_time_s3[tindex2]);//downloading time    
                else//mapped to diff resource
                    _1dint.push_back(0);//0
            }
            
            //sort
            sort(_1dint.begin(), _1dint.end(), reverse_order);

            //push
            max_terms_rev_sorted.push_back(_1dint);
        }
    }
}

//compute how many max term to be included, if task becomes higher priority task
void compute_no_max_terms(){
    clear_2d_vc_initialize_zero(num_max_terms, no_tasks, no_tasks);

    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){

            if((tasks[i].s1 == tasks[j].s1 && tasks[i].s2 == tasks[j].s2) || (tasks[i].s2 == tasks[j].s2 && tasks[i].s3 == tasks[j].s3)){
                num_max_terms[i][j] += 2;
                num_max_terms[j][i] += 2;

            }
            else{
                if(tasks[i].s1 == tasks[j].s1){
                    num_max_terms[i][j] ++;
                    num_max_terms[j][i] ++;
                }
                if(tasks[i].s2 == tasks[j].s2){
                    num_max_terms[i][j] ++;
                    num_max_terms[j][i] ++;
                }
                if(tasks[i].s3 == tasks[j].s3){
                    num_max_terms[i][j] ++;
                    num_max_terms[j][i] ++;
                }
            }
        }
    }
}

//compute some intermediate common information for all approaches
void compute_common_intermed_info(){

    //compute how many max term to be included, if task becomes higher priority task
    compute_no_max_terms();

    //store max terms in the decreasing order for each task
    compute_store_max_terms_sorted_order();
}