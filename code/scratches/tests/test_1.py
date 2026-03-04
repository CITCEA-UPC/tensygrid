import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import PolynomialMatrixBuilder_class as PMB

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import os
    save_output = False

    eqs = [
        "3*dx1*y1 + 6*(1/2+1/2*z1)*(2/3-1/3*u1)*x1",
        "y1 - dx1"
    ]



    builder = PMB.PolynomialMatrixBuilder(eqs, ineqs=[], verbose=False)

    v_dict = {'dx1': 1.0, 'x1': 2.0, 'u1': 3.0, 'y1': 4.0, 'z1': 5.0}


    builder.linearize(v_dict)

    evals, e_left, e_right, p_matrix, stable, margin = builder.compute_stability()

    out_path = os.path.join(os.path.dirname(__file__), "matrices_output.txt") if save_output else None
    builder.report(
        eigenvalues    = evals,
        is_stable      = stable,
        max_real       = margin,
        print_matrices = True,
        save_path      = out_path,
    )

    assert stable == False, "Expected the system to be unstable based on the eigenvalues."
    assert round(margin, 3) == 0.400, f"Expected the maximum real part of eigenvalues to be 0.4 (check). Got {margin:.3f} instead."