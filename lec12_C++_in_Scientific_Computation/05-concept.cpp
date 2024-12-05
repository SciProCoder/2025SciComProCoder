#include "./include/concept.example.h"

using namespace Training2024::concept_example;

int main()
{

    // MatrixView_1<int> mat_view_0(nullptr, 10, 10); // template constraint failure

    MatrixView_1<double> mat_view_1(nullptr, 10, 10);
    MatrixView_1<double> mat_view_2(nullptr, 10, 10);

    // Check_MatrixView(mat_view_1, mat_view_2); // note: the required type ‘typename T::iter_type’ is invalid
    //                                           // note: ‘t.data()’ does not satisfy return-type-requirement

    MatrixView_2<double> mat_view_3(nullptr, 10, 10);

    // Check_MatrixView(mat_view_1, mat_view_3);

    MatrixView_3<double> mat_view_4(nullptr, 10, 10);
    MatrixView_3<double> mat_view_5(nullptr, 10, 10);
    Check_MatrixView(mat_view_4, mat_view_5);

    MatrixView_3<float> mat_view_6(nullptr, 10, 10);

    // Check_MatrixView(mat_view_4, mat_view_6);  // scalar not the same

    // Check_CIVecView(mat_view_4, mat_view_5); // mat_view_4 and _5 are not CIVecView

    CIVecView_1<double> ci_vec_view_1(nullptr, 10, 10);
    CIVecView_1<double> ci_vec_view_2(nullptr, 10, 10);

    // Check_CIVecView(ci_vec_view_1, ci_vec_view_2);

}