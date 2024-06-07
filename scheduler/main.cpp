#include "main.h"

//wait for a character...used for debugging
void wait_for_char(){
    char ch; 
    cin >> ch;
}

int main(){
    srand(time(NULL));

    //get_static_input
    get_static_input();

    //number of test cases which are successfully scheduled
    int succ_count_opa = 0;
    int succ_count_rel_opt = 0;
    int succ_count_rel_heur = 0;
    int succ_count_rel_baseline = 0;
    int succ_count_decomp_baseline = 0;

    //% of tasks scheduled by partial approaches, of the test case for which corresponding approach cannot schedule that task set
    double accumulated_perc_tasks_scheduled_opa_partial = 0;
    double accumulated_perc_tasks_scheduled_heur_partial = 0;
    double accumulated_perc_tasks_scheduled_baseline_partial = 0;
    
    //ratio of heaviness rejected accepted during partial execution
    double acccumulated_ratio_rej_acc_opa_partial = 0;
    double acccumulated_ratio_rej_acc_heur_partial = 0;
    double acccumulated_ratio_rej_acc_baseline_partial = 0;

    //number of test cases for which partial approaches executed
    int partial_exec_count = 0;
    int time_expired_count = 0;
    //for each input file
    for(int cur_inp = 1; cur_inp <= no_input ; cur_inp++){
        //get input
        cout << "\n\t\tTest Case \t" << cur_inp << "************\n";
        get_input(cur_inp);

        //compute some intermediate common information for all approaches
        compute_common_intermed_info();

        //assign priorities
        bool status_opt = false, status_opa = false, status_rel_heur = false, status_rel_baseline = false, status_opa_partial = false, status_rel_heur_partial = false, status_rel_baseline_partial = false, status_decomp = false;;

        time_limit_expired_this_test = false;

        //relative optimal
        status_opt = assign_rel_priority_opt();
        if(time_limit_expired_this_test)
        time_expired_count++;

        if(status_opt){//successful
            succ_count_rel_opt++;
        }

            //absolute OPA
            status_opa = assign_abs_priority_opa();
            if(status_opa){//successful
                succ_count_opa++;
            }
            
            //relative heur
            status_rel_heur = assign_rel_priority_heur();
            if(status_rel_heur){//successful
                succ_count_rel_heur++;
            }
            
            //relative baseline
            status_rel_baseline = assign_rel_priority_baseline();
            if(status_rel_baseline){//successful
                succ_count_rel_baseline++;
            }
        
        
        //decomposition based approach
        status_decomp = decompostion_baseline();
        if(status_decomp){
            succ_count_decomp_baseline++;
        }

        //run partial approaches
        if(!status_opa &&  !status_rel_heur && !status_rel_baseline){//failed
            // initialize
            heav_accepted_opa_partial = 0;
            heav_rejected_opa_partial = 0;
            heav_accepted_rel_heur_partial = 0;
            heav_rejected_rel_heur_partial = 0;
            heav_accepted_rel_baseline_partial = 0;
            heav_rejected_rel_baseline_partial = 0;

            //absolute opa partial...opa does not stop but discard the task with highest lag and continues assigning priorities
            partial_exec_count++;
            
            no_tasks_scheduled_this_task_set = 0;
            status_opa_partial = assign_abs_priority_opa_partial();
            if(status_opa_partial){//successful
                assert(0 == 1);
            }
            accumulated_perc_tasks_scheduled_opa_partial += (double)no_tasks_scheduled_this_task_set*100/(no_tasks);
            //compute and add to ratio
            acccumulated_ratio_rej_acc_opa_partial += heav_rejected_opa_partial/(heav_rejected_opa_partial + heav_accepted_opa_partial);

            //relative heuristic partial...discard tasks that are not feasible
            no_tasks_scheduled_this_task_set = 0;
            status_rel_heur_partial = assign_rel_heur_partial();
            if(status_rel_heur_partial){
                assert(0 == 1);//though it may be possible as heur is not deterministic
            }
            accumulated_perc_tasks_scheduled_heur_partial += (double)no_tasks_scheduled_this_task_set*100/(no_tasks);
            acccumulated_ratio_rej_acc_heur_partial += heav_rejected_rel_heur_partial/(heav_rejected_rel_heur_partial + heav_accepted_rel_heur_partial);
            
            //relative heuristic...discard tasks that are not feasible
            no_tasks_scheduled_this_task_set = 0;
            status_rel_baseline_partial = assign_rel_baseline_partial();
            if(status_rel_baseline_partial){
                assert(0 == 1);//though it may be possible as heur is not deterministic
            }
            accumulated_perc_tasks_scheduled_baseline_partial += (double)no_tasks_scheduled_this_task_set*100/(no_tasks);
            acccumulated_ratio_rej_acc_baseline_partial += heav_rejected_rel_baseline_partial/(heav_rejected_rel_baseline_partial + heav_accepted_rel_baseline_partial);
        }

        if(!status_opt && !time_limit_expired_this_test && (status_opa||status_rel_heur||status_rel_baseline||status_opa_partial||status_rel_heur_partial||status_rel_baseline_partial)/*rel opt failed but any other succeeds*/){
            cout << "Issue\n";
            cout << status_opt << "\t" << status_opa  << "\t" << status_rel_heur << "\t" << status_rel_baseline << "\t" << status_opa_partial << "\t" << status_rel_heur_partial << "\t" << status_rel_baseline_partial << "\n"; 
            wait_for_char();
        }
    }


    //following may be removed..reading test case parameters to add it to output file
    fstream finput(path_to_input + "input_test");
    assert(finput);
    //variables to read file
    int v1;
    double v2, v3, v4, v5, v6;
    string name;
    finput >> v1 >> name;//read no_input...do not save
    finput >> v1 >> name;//read no_ap_up...do not save
    finput >> v1 >> name;//read no_server...do not save
    finput >> v1 >> name;//read no_tasks
    finput >> v2 >> name;//read max_heaviness_per_task
    finput >> v3 >> name;//read heaviness_req_s1
    finput >> v4 >> name;//read heaviness_req_s2
    finput >> v5 >> name;//read heaviness_req_s3
    finput >> v6 >> name;//read heavy_definition
    finput.close();

    //output files
    ofstream ofs_succ_ratio(path_to_output + "succ_ratio", ofstream::app);
    assert(ofs_succ_ratio);
    ofstream ofs_succ_ratio_val_only(path_to_output + "succ_ratio_val_only", ofstream::app);
    assert(ofs_succ_ratio_val_only);
    ofstream ofs_succ_ratio_partial(path_to_output + "succ_ratio_partial", ofstream::app);
    assert(ofs_succ_ratio_partial);
    ofstream ofs_succ_ratio_val_only_partial(path_to_output + "succ_ratio_val_only_partial", ofstream::app);
    assert(ofs_succ_ratio_val_only_partial);
    //store failure of rel_opt and decomposition baseline
    ofstream ofs_succ_ratio_decomp_baseline_rel_opt(path_to_output + "succ_ratio_decomp_baseline_rel_opt", ofstream::app);
    assert(ofs_succ_ratio_decomp_baseline_rel_opt);
    ofstream ofs_val_plot(path_to_output + "val_plot", ofstream::app);
    assert(ofs_val_plot);

    ofs_succ_ratio << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";
    ofs_succ_ratio_val_only << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";
    ofs_succ_ratio_partial << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";
    ofs_succ_ratio_val_only_partial << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";
    ofs_succ_ratio_decomp_baseline_rel_opt << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";
    ofs_val_plot << "\n\n**********\t" << "#Tasks " << v1 << " max heaviness " << v2 << " heaviness req " << v3 << " " << v4 << " " << v5 << " heavy def " << v6 << "\n\n";

    
    ofs_succ_ratio << "time expired for " << time_limit_expired_this_test << "\n";
        
    if(succ_count_rel_opt > 0){
        ofs_succ_ratio << "\nrel_opt\t" << succ_count_rel_opt << " Ratio " << (double)succ_count_rel_opt*100/no_input << "\n";
        ofs_succ_ratio_val_only << (double)succ_count_rel_opt*100/no_input << "\n";
        
    
        ofs_succ_ratio << "abs_opa\t"  << (double)succ_count_opa*100/no_input << "\n";
        ofs_succ_ratio_val_only << (double)succ_count_opa*100/no_input << "\n";
        
        ofs_succ_ratio << "rel_heur\t"  << (double)succ_count_rel_heur*100/no_input << "\n";
        ofs_succ_ratio_val_only << (double)succ_count_rel_heur*100/no_input << "\n";

        ofs_succ_ratio << "rel_baseline\t" << (double)succ_count_rel_baseline*100/no_input << "\n";
        ofs_succ_ratio_val_only << (double)succ_count_rel_baseline*100/no_input << "\n";

        ofs_val_plot << (double)succ_count_rel_baseline*100/no_input << "\n";
        ofs_val_plot << (double)succ_count_rel_heur*100/no_input - (double)succ_count_rel_baseline*100/no_input << "\n";
        ofs_val_plot << (double)succ_count_opa*100/no_input - (double)succ_count_rel_heur*100/no_input << "\n";
        ofs_val_plot << (double)succ_count_rel_opt*100/no_input - (double)succ_count_opa*100/no_input << "\n";


    }
    

    if(partial_exec_count > 0){
        double p1 = accumulated_perc_tasks_scheduled_opa_partial/(partial_exec_count);
        double p2 = accumulated_perc_tasks_scheduled_heur_partial/(partial_exec_count);
        double p3 = accumulated_perc_tasks_scheduled_baseline_partial/(partial_exec_count);
    
        ofs_succ_ratio_partial << "\nopa_partial\t" << p2 << "\n";
        ofs_succ_ratio_val_only_partial << p2 << "\n";

        ofs_succ_ratio_partial << "rel_heur_partial\t" << p3  << "\n";
        ofs_succ_ratio_val_only_partial << p3 << "\n";

        ofs_succ_ratio_partial << "rel_baseline_partial\t" << p1 << "\n";
        ofs_succ_ratio_val_only_partial << p1 << "\n";

        double r1 = acccumulated_ratio_rej_acc_baseline_partial/partial_exec_count;
        double r2 = acccumulated_ratio_rej_acc_opa_partial/partial_exec_count;
        double r3 = acccumulated_ratio_rej_acc_heur_partial/partial_exec_count;

        ofs_succ_ratio_partial << "\n\n heaviness rejected ratio \t" << r2 << "\n";
        ofs_succ_ratio_val_only_partial << "\n" << r2*100 << "\n";

        ofs_succ_ratio_partial << " heaviness rejected ratio \t" << r3  << "\n";
        ofs_succ_ratio_val_only_partial << r3*100 << "\n";

        ofs_succ_ratio_partial << "heaviness rejected ratio\t" << r1 << "\n";
        ofs_succ_ratio_val_only_partial << r1*100 << "\n";

    }
    
    ofs_succ_ratio_decomp_baseline_rel_opt << (double)(succ_count_rel_opt * 100)/no_input << "\n";
    ofs_succ_ratio_decomp_baseline_rel_opt << (double)(succ_count_decomp_baseline * 100)/no_input << "\n";

    ofs_succ_ratio << "\nDecomp\t" << (double)(succ_count_decomp_baseline * 100)/no_input << "\n";
    ofs_succ_ratio_val_only << (double)(succ_count_decomp_baseline * 100)/no_input << "\n";
    ofs_val_plot << (double)(succ_count_decomp_baseline * 100)/no_input << "\n";

    ofs_succ_ratio.close();
    ofs_succ_ratio_val_only.close();
    ofs_succ_ratio_decomp_baseline_rel_opt.close();
    ofs_succ_ratio_partial.close();
    ofs_succ_ratio_val_only_partial.close();
    ofs_val_plot.close();
    
    return 0;
}       

