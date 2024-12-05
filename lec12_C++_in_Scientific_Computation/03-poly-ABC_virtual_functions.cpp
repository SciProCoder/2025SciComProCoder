#include "include/polymorphism_example.h"

using namespace Training2024::poly_example04;

class Pi_Calculator : public OpenMPAlgoBase
{
public:
    Pi_Calculator(int _end_indx) : OpenMPAlgoBase()
    {
        begin_indx = 1;
        end_indx   = _end_indx;
        local_res.resize(num_of_threads, 0.0);
    }

    virtual void initialize() noexcept { std::cout << "Pi_Calculator::initialize()" << std::endl; }
    virtual void finish() noexcept
    {
        auto res = std::accumulate(local_res.begin(), local_res.end(), 0.0);
        res      = sqrt(res * 6.0);
        printf("Pi = %.10f\n", res);
    }
    virtual void initialize_local() noexcept
    {
        auto thread_id       = OMP_THREAD_LABEL;
        local_res[thread_id] = 0.0;
    }
    virtual void finish_local() noexcept
    {
#pragma omp critical
        {
            std::cout << "thread " << OMP_THREAD_LABEL << " finished" << std::endl;
        }
    }
    virtual void do_task(size_t _task_id)
    {
        auto thread_id = OMP_THREAD_LABEL;
        local_res[thread_id] += 1.0 / (double(_task_id) * double(_task_id));
    }
    virtual bool should_upload_data() noexcept { return false; }
    virtual bool should_upload_data_force() noexcept { return false; }
    virtual void upload_data() {}
    virtual void upload_data_force() {}

protected:
    std::vector<double> local_res;
};

int main()
{
    Pi_Calculator pi_calculator(100000000);
    pi_calculator.run();
    return 0;
}
