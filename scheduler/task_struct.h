#include <vector>
using namespace std;

//define task type
struct Task{
    int tid;//task number...not index 
    int deadline;//deadline
    int s1;//stage 1 mapping
    int s2;//stage 2 mapping
    int s3;//stage 3 mapping

    Task(int _tid, int _deadline, int _s1, int _s2, int _s3) : tid(_tid),  deadline(_deadline), s1(_s1), s2(_s2), s3(_s3) {}

};

extern vector <Task> tasks;

