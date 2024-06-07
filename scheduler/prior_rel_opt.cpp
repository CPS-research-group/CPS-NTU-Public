#include "prior_rel_opt.h"

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

//store index of variable in the array var
void store_var_index(){
    clear_2d_vc_initialize_neg1(var_index, no_tasks, no_tasks);

    int ind = 0;
    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(num_max_terms[i][j] > 0){
                var_index[i][j] = ind;
                var_index[j][i] = ind;
                ind++;
            }
        }
    }
}

//number of binary variables 
int get_number_of_binary_variables(){
    int num = 0;
    for(int i = 0; i < no_tasks - 1; i++){
        for(int j = i + 1; j < no_tasks; j++){
            if(num_max_terms[i][j] > 0)
                num++;
        }
    }
    return num;
}

//compute a priority ordering if feasible
bool compute_priority_ordering_if_feasible_rel_opt(){
	bool sol_found = false;
	
	//initialize
    GRBEnv* env = 0;
    GRBVar* var = 0; //variable to indicate binary variable which is priority of Ti and Tk, i < k

    try{
        //create environment
        env = new GRBEnv();
        //obtain a mddel
        GRBModel model = GRBModel(*env);
        model.set(GRB_StringAttr_ModelName, "optimal");
        
		//set other attributes
        model.set(GRB_DoubleParam_TimeLimit, tlimit);
        // model.set(GRB_IntParam_Threads, 1);
	    model.set(GRB_IntParam_OutputFlag, 0);
		
		ostringstream cname, vname;

        //............add variables..........//

         int no_var = get_number_of_binary_variables();

        var = model.addVars(no_var, GRB_BINARY);
        for(int i = 0; i < no_var; i++){
            vname.str("");
            vname.clear();
            vname << "V" << i + 1;
            var[i].set(GRB_StringAttr_VarName, vname.str());
        }

        store_var_index();
 
        //.............add constraints..........//
        
        for(int tindex = 0; tindex < no_tasks; tindex++){
            GRBLinExpr delay = 0; //delay of current task

            //add job additive components
            for(int tindex2 = 0; tindex2 < no_tasks; tindex2++){
                if(tindex == tindex2){
                    delay += max_terms_rev_sorted[tindex*no_tasks + tindex][0];
                }
                else if(num_max_terms[tindex][tindex2] > 0){
                    int term1 = 0;

                    for(int j = 0; j < num_max_terms[tindex][tindex2]; j++){
                        term1 +=  max_terms_rev_sorted[(tindex)*no_tasks + tindex2][j];
                    }
                    if(tindex < tindex2){
                        delay += (1 - var[var_index[tindex][tindex2]]) * term1;
                    }
                    else if(tindex > tindex2){
                        delay += var[var_index[tindex][tindex2]] * term1;
                    }
                }
            }
            //maximum of offloading times of all higher priority tasks, including itself

            GRBVar term2 = model.addVar(0, ubound, 0, GRB_INTEGER);
            GRBVar* aux_var_max1 = 0;
            aux_var_max1 = model.addVars(no_tasks, GRB_BINARY);

            for(int tindex2 = 0; tindex2 < no_tasks; tindex2++){
                GRBLinExpr expr1;

                if(tindex == tindex2){
                    expr1 = proc_time_s1[tindex2];
                }
                else{
                    if(tasks[tindex].s1 == tasks[tindex2].s1){
                        if(tindex < tindex2){
                            expr1 = (1 - var[var_index[tindex][tindex2]])* proc_time_s1[tindex2];
                        }
                        else if(tindex > tindex2){
                            expr1 = var[var_index[tindex][tindex2]]* proc_time_s1[tindex2];
                        }
                    }
                    else{
                        expr1 = 0;
                    }   
                }
                cname.str(" ");
                cname.clear();
                cname << "T2MAX1." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term2 >= expr1, cname.str());
                cname.str(" ");
                cname.clear();
                cname << "T2MAX2." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term2 <= expr1 + (1 - aux_var_max1[tindex2])*ubound, cname.str());
            }
            GRBLinExpr expr2 = 0;
            for(int i = 0; i < no_tasks; i++){
                expr2 += aux_var_max1[i];
            }
            cname.str(" ");
            cname.clear();
            cname << "T2.AV." << tindex + 1;
            model.addConstr(expr2 == 1, cname.str());

            delay += term2;

            //maximum of compute times of all higher priority tasks, including itself

            GRBVar term3 = model.addVar(0, ubound, 0, GRB_INTEGER);//variable for third term of the delay of this task 
            GRBVar* aux_var_max2 = 0;
            aux_var_max2 = model.addVars(no_tasks, GRB_BINARY);

            for(int tindex2 = 0; tindex2 < no_tasks; tindex2++){
                GRBLinExpr expr3;

                if(tindex == tindex2){
                    expr3 = proc_time_s2[tindex2];
                }
                else{
                    if(tasks[tindex].s2 == tasks[tindex2].s2){//mapped to same resouce
                        if(tindex < tindex2){
                            expr3 = (1 - var[var_index[tindex][tindex2]])* proc_time_s2[tindex2];
                        }
                        else if(tindex > tindex2){
                            expr3 = var[var_index[tindex][tindex2]]* proc_time_s2[tindex2];
                        }
                    }
                    else{
                        expr3 = 0;
                    }   
                }

                cname.str(" ");
                cname.clear();
                cname << "T3M1." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term3 >= expr3, cname.str());

                cname.str(" ");
                cname.clear();
                cname << "T3M2." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term3 <= expr3 + (1 - aux_var_max2[tindex2])*ubound, cname.str());
            }
            GRBLinExpr expr4 = 0;
            for(int i = 0; i < no_tasks; i++){
                expr4 += aux_var_max2[i];
            }
            cname.str(" ");
            cname.clear();
            cname << "T3.AV." << tindex + 1;
            model.addConstr(expr4 == 1, cname.str());

            delay += term3;

            //maximum of download times of all tasks, excluding itself

            GRBVar term4 = model.addVar(0, ubound, 0, GRB_INTEGER);
            GRBVar* aux_var_max3 = 0;
            aux_var_max3 = model.addVars(no_tasks, GRB_BINARY);

            for(int tindex2 = 0; tindex2 < no_tasks; tindex2++){
                GRBLinExpr expr5;

                if(tindex == tindex2){
                    expr5 = 0;
                }
                else{
                    if(tasks[tindex].s3 == tasks[tindex2].s3){
                        if(tindex < tindex2){
                            expr5 = var[var_index[tindex][tindex2]]* proc_time_s3[tindex2];
                        }
                        else if(tindex > tindex2){
                            expr5 = (1 - var[var_index[tindex][tindex2]])* proc_time_s3[tindex2];
                        }
                    }
                    else{
                        expr5 = 0;
                    }   
                }

                cname.str(" ");
                cname.clear();
                cname << "T4M1." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term4 >= expr5, cname.str());
                cname.str(" ");
                cname.clear();
                cname << "T4M2." << tindex + 1 << "." << tindex2 + 1;
                model.addConstr(term4 <= expr5 + (1 - aux_var_max3[tindex2])*ubound, cname.str());
            }
            GRBLinExpr expr6 = 0;
            for(int i = 0; i < no_tasks; i++){
                expr6 += aux_var_max3[i];
            }
            cname.str(" ");
            cname.clear();
            cname << "T4.AV." << tindex + 1;
            model.addConstr(expr6 == 1, cname.str());
            delay += term4;
            cname.str(" ");
            cname.clear();
            cname << "D." << tindex + 1;
            model.addConstr(delay <= tasks[tindex].deadline, cname.str());
        
        }
        //solve model
        model.optimize();

        //check if computed successfully
        if(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){//optimization status found
			sol_found = true;//all good
        }
        else if(model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT){//terminating due to time bound
            cout << "terminating due to time limit\n";
            time_limit_expired_this_test = true;
        }
        else if(model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE){
            cout << "\tModel was proven to be infeasible" << endl;
        }
        else if(model.get(GRB_IntAttr_Status) == GRB_UNBOUNDED){
            cout << "Model was proven to be unbounded" << endl;
        }
        else{
            cout << "No solution" << endl;
        }
    }
	//end of the try block
	catch (GRBException e){
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...){
		cout << "Exception during optimization" << endl;
	}

	delete[] var;
	delete env;
	
	return sol_found;
}

//compute max processing time...to be used as upper bound in ilp
void compute_ub(){
    ubound = 0;
    for(int tindex = 0; tindex < no_tasks; tindex++){
        ubound = max(ubound, proc_time_s1[tindex]);
        ubound = max(ubound, proc_time_s2[tindex]);
        ubound = max(ubound, proc_time_s3[tindex]);
    }
}

bool assign_rel_priority_opt(){
    //compute max processing time...to be used as upper bound in ilp
    compute_ub();

    bool status = compute_priority_ordering_if_feasible_rel_opt();

    //all well
    return status;
}
   

