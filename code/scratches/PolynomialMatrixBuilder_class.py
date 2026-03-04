import sympy as sp
import numpy as np
from scipy import sparse, linalg
from sympy.parsing.sympy_parser import parse_expr
from enum import Enum
import builtins
from typing import Callable

# ------------------------------------------------------------------
# Profiler Control
# ------------------------------------------------------------------
# If the script is not run via `mprof run` or `python -m memory_profiler`,
# the 'profile' function won't be in builtins. We create a dummy to ensure 
# zero-overhead execution as per our performance rules.

if 'profile' not in builtins.__dict__:
    def profile(func: Callable) -> Callable:
        """
        Dummy decorator to prevent profiling overhead when not explicitly invoked.

        :param func: The function to decorate.
        :type func: Callable
        :return: The unmodified function.
        :rtype: Callable
        """
        return func
else:
    pass # In Python 3, memory_profiler injects directly into builtins during a run.

# ------------------------------------------------------------------
# Enums for Options
# ------------------------------------------------------------------

class ConstraintType(Enum):
    """
    Enum representing the type of constraint to process.
    Replaces the usage of string options or ambiguous booleans.
    """
    EQUALITY = 1
    INEQUALITY = 2


# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

class PolynomialMatrixBuilder:
    """
    Builds the S_H / Phi_H (and S_W / Phi_W) matrices from a list of
    polynomial equations, and performs analytic linearization using the
    iMTI / CPN representation described in the TenSyGrid papers.
    """

    # Finalizing the class to prevent dynamic attribute assignment and save memory
    __slots__ = [
        'verbose', 'eqs', 'ineqs', 'all_symbols', 'sym_to_idx',
        'S_H', 'Phi_H', 'S_W', 'Phi_W', '_E', '_A', '_B'
    ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, eqs: list, ineqs: list, verbose: bool = False) -> None:
        """
        Initializes the PolynomialMatrixBuilder.

        :param eqs: Polynomial equality constraints as strings.
        :type eqs: list
        :param ineqs: Polynomial inequality constraints as strings.
        :type ineqs: list
        :param verbose: Print debugging information.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.verbose: bool = verbose

        # Initialize lists explicitly
        self.eqs: list = list()
        self.ineqs: list = list()

        # Parse equations with explicit loops
        for eq_str in eqs:
            parsed_eq: sp.Expr = parse_expr(eq_str, evaluate=False)
            self.eqs.append(parsed_eq)

        for ineq_str in ineqs:
            parsed_ineq: sp.Expr = parse_expr(ineq_str, evaluate=False)
            self.ineqs.append(parsed_ineq)

        self.all_symbols: list = list()
        self.sym_to_idx: dict = dict()
        
        # We merge lists for symbol extraction
        combined_exprs: list = list()
        for eq in self.eqs:
            combined_exprs.append(eq)
        for ineq in self.ineqs:
            combined_exprs.append(ineq)

        self.extract_symbols(combined_exprs)

        # Build the symbol dictionary index explicitly
        idx: int = 0
        for sym in self.all_symbols:
            self.sym_to_idx[sym.name] = idx
            idx = idx + 1

        S_H_sp: sparse.csc_matrix
        Phi_H_sp: sparse.csr_matrix
        S_H_sp, Phi_H_sp = self.matrix_creation(self.eqs)

        S_W_sp: sparse.csc_matrix
        Phi_W_sp: sparse.csr_matrix
        S_W_sp, Phi_W_sp = self.matrix_creation(self.ineqs)

        # Store dense arrays
        self.S_H: np.ndarray = S_H_sp.toarray()
        self.Phi_H: np.ndarray = Phi_H_sp.toarray()
        self.S_W: np.ndarray = S_W_sp.toarray()
        self.Phi_W: np.ndarray = Phi_W_sp.toarray()

        # Protected interactive states
        self._E: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Getters and Setters
    # ------------------------------------------------------------------

    def get_E(self) -> np.ndarray | None:
        """
        Gets the E matrix.
        
        :return: The E matrix.
        :rtype: np.ndarray | None
        """
        return self._E

    def set_E(self, val: np.ndarray) -> None:
        """
        Sets the E matrix.
        
        :param val: The new matrix array.
        :type val: np.ndarray
        :return: None
        :rtype: None
        """
        self._E = val

    def get_A(self) -> np.ndarray | None:
        """
        Gets the A matrix.
        
        :return: The A matrix.
        :rtype: np.ndarray | None
        """
        return self._A

    def set_A(self, val: np.ndarray) -> None:
        """
        Sets the A matrix.
        
        :param val: The new matrix array.
        :type val: np.ndarray
        :return: None
        :rtype: None
        """
        self._A = val

    def get_B(self) -> np.ndarray | None:
        """
        Gets the B matrix.
        
        :return: The B matrix.
        :rtype: np.ndarray | None
        """
        return self._B

    def set_B(self, val: np.ndarray) -> None:
        """
        Sets the B matrix.
        
        :param val: The new matrix array.
        :type val: np.ndarray
        :return: None
        :rtype: None
        """
        self._B = val

    # ------------------------------------------------------------------
    # Helper Functions (Replacing Lambdas and Nested Functions)
    # ------------------------------------------------------------------

    def _get_symbol_name(self, sym: sp.Symbol) -> str:
        """
        Returns the name of a sympy symbol to act as a sorting key.

        :param sym: SymPy Symbol object.
        :type sym: sp.Symbol
        :return: The string name of the symbol.
        :rtype: str
        """
        return sym.name

    def _sort_symbol(self, sym: sp.Symbol) -> tuple:
        """
        Provides the sorting key tuple for symbols based on TenSyGrid rules.

        :param sym: SymPy Symbol object.
        :type sym: sp.Symbol
        :return: A tuple containing the priority integer and the symbol name.
        :rtype: tuple
        """
        name: str = sym.name
        
        if name.startswith('dx'):
            return (0, name)
        elif name.startswith('xp'):
            return (0, name)
        elif name.startswith('x') and not name.startswith('xp'):
            return (1, name)
        elif name.startswith('u'):
            return (2, name)
        elif name.startswith('y'):
            return (3, name)
        elif name.startswith('z'):
            return (4, name)
        else:
            return (5, name)

    def _get_real_part(self, val: complex) -> float:
        """
        Returns the real part of a complex number to act as a sorting key.

        :param val: Complex number.
        :type val: complex
        :return: Real part of the complex number.
        :rtype: float
        """
        return float(val.real)

    def _arr2str(self, a: np.ndarray) -> str:
        """
        Formats a numpy array into a structured string.

        :param a: Input array.
        :type a: np.ndarray
        :return: Formatted string representation.
        :rtype: str
        """
        return np.array2string(
            np.asarray(a),
            max_line_width=99999,
            precision=6,
            suppress_small=True,
            threshold=99999,
        )

    def _section(self, title: str, content: str) -> str:
        """
        Formats text into a standardized section for reporting.

        :param title: Section title.
        :type title: str
        :param content: Section body content.
        :type content: str
        :return: Formatted section block.
        :rtype: str
        """
        bar: str = "=" * (len(title) + 4)
        return f"{bar}\n  {title}\n{bar}\n{content}\n"

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def extract_symbols(self, all_exprs: list) -> list:
        """
        Extracts and sorts symbols from a list of SymPy expressions.

        Ordering convention (TenSyGrid deliverable 1.1):
            dx* < x* < u* < y* < z* < everything else

        :param all_exprs: SymPy expressions to scan.
        :type all_exprs: list
        :return: Sorted list of unique symbols.
        :rtype: list
        """
        seen: set = set()
        extracted_symbols: list = list()

        for eq in all_exprs:
            # We explicitly gather and sort the free symbols to avoid lambdas inline
            eq_symbols: list = list(eq.free_symbols)
            eq_symbols.sort(key=self._get_symbol_name)
            
            for sym in eq_symbols:
                if sym not in seen:
                    seen.add(sym)
                    extracted_symbols.append(sym)
                else:
                    pass  # State must be explicit

        extracted_symbols.sort(key=self._sort_symbol)

        if self.verbose:
            print(f"Unified symbols order: {extracted_symbols}\n")
        else:
            pass  # State must be explicit

        self.all_symbols = extracted_symbols
        return extracted_symbols

    # ------------------------------------------------------------------
    # Matrix creation
    # ------------------------------------------------------------------

    @profile
    def matrix_creation(self, all_exprs: list) -> tuple:
        """
        Creates the S and Phi sparse matrices with automatic L1 normalisation.

        :param all_exprs: SymPy expressions to encode.
        :type all_exprs: list
        :return: Tuple containing S (csc_matrix) and Phi (csr_matrix).
        :rtype: tuple
        """
        S_list: list = list()
        Phi_data: list = list()
        monom_to_idx: dict = dict()

        if self.verbose:
            print("--- Analyzing Expressions for Matrices ---")
        else:
            pass

        eq_idx: int = 0
        for eq in all_exprs:
            if self.verbose:
                print(f"\nEquation {eq_idx + 1}: {eq}")
            else:
                pass

            terms: list
            if eq.is_Add:
                terms = list(eq.args)
            else:
                terms = [eq]

            current_eq_coeffs: dict = dict()

            for term in terms:
                current_eq_coeffs = self._get_monomial_weights(
                    term, S_list, monom_to_idx, current_eq_coeffs
                )

            Phi_data.append(current_eq_coeffs)
            eq_idx = eq_idx + 1

        # Assemble sparse matrices
        S: sparse.csc_matrix
        Phi: sparse.csr_matrix

        if len(S_list) > 0:
            S_raw: np.ndarray = np.vstack(S_list).T.astype(float)
            S = sparse.csc_matrix(S_raw)
            num_cols: int = S_raw.shape[1]
            phi_rows: list = list()
            
            for eq_dict in Phi_data:
                row: np.ndarray = np.zeros(num_cols)
                # Dictionary iteration is safe here as it represents matrix columns
                for col_idx, val in eq_dict.items():
                    row[col_idx] = val
                phi_rows.append(row)
                
            Phi = sparse.csr_matrix(np.vstack(phi_rows))
        else:
            # Matrices are empty, use correct shapes
            S = sparse.csc_matrix((len(self.all_symbols), 0))
            Phi = sparse.csr_matrix((len(all_exprs), 0))

        return S, Phi

    def _get_monomial_weights(
        self,
        term: sp.Expr,
        S_list: list,
        monom_to_idx: dict,
        current_eq_coeffs: dict
    ) -> dict:
        """
        Processes a term to extract its global coefficients and internal weights.
        If the term contains multi-variable factors, it expands it recursively.

        :param term: The mathematical term to be processed.
        :type term: sp.Expr
        :param S_list: The collection of unique monomial weight arrays.
        :type S_list: list
        :param monom_to_idx: Dictionary mapping a monomial weight tuple to its matrix index.
        :type monom_to_idx: dict
        :param current_eq_coeffs: Dictionary of global coefficients for the current equation.
        :type current_eq_coeffs: dict
        :return: The updated dictionary of global coefficients for the current equation.
        :rtype: dict
        """
        coeff_sym: sp.Expr
        symbolic_part: tuple
        coeff_sym, symbolic_part = term.as_coeff_mul()
        
        factors: tuple = sp.Mul.make_args(sp.Mul(*symbolic_part))
        requires_expansion: bool = False
        
        for f in factors:
            f_vars: list = list(f.free_symbols)
            if len(f_vars) > 1:
                requires_expansion = True
            else:
                pass 
                
        if requires_expansion:
            if self.verbose:
                print(f"   Expanding complex term: {term}")
            else:
                pass

            expanded_sub_terms: tuple = sp.Add.make_args(term.expand())
            
            for sub_t in expanded_sub_terms:
                current_eq_coeffs = self._get_monomial_weights(
                    sub_t, S_list, monom_to_idx, current_eq_coeffs
                )
            return current_eq_coeffs
            
        else:
            global_phi: float = float(coeff_sym)
            weights: list = list()
            
            for _ in range(len(self.all_symbols)):
                weights.append(0.0)

            for f in factors:
                f_vars_inner: list = list(f.free_symbols)

                if len(f_vars_inner) == 1:
                    s: sp.Symbol = f_vars_inner[0]
                    idx_val: int | None = self.sym_to_idx.get(s.name, None)
                    
                    if idx_val is not None:
                        b_val: float = float(sp.diff(f, s))
                        a_val: float = float(f.subs(s, 0))

                        scale: float = abs(a_val) + abs(b_val)
                        if scale == 0.0:
                            scale = 1.0
                        else:
                            pass

                        weights[idx_val] = b_val / scale
                        global_phi = global_phi * scale
                    else:
                        pass 

                elif len(f_vars_inner) == 0:
                    global_phi = global_phi * float(f)
                else:
                    pass

            monom_tuple: tuple = tuple(weights)
            existing_idx: int | None = monom_to_idx.get(monom_tuple, None)
            final_idx: int = 0
            
            if existing_idx is None:
                final_idx = len(S_list)
                monom_to_idx[monom_tuple] = final_idx
                S_list.append(np.array(monom_tuple))
            else:
                final_idx = existing_idx

            current_val: float | None = current_eq_coeffs.get(final_idx, None)
            
            if current_val is None:
                current_eq_coeffs[final_idx] = global_phi
            else:
                current_eq_coeffs[final_idx] = current_val + global_phi

            return current_eq_coeffs

    # ------------------------------------------------------------------
    # Linearisation – public entry point
    # ------------------------------------------------------------------

    @profile
    def linearize(
        self, v_dict: dict, c_type: ConstraintType = ConstraintType.EQUALITY
    ) -> np.ndarray:
        """
        Analytic linearization for iMTI models (CPN representation).

        :param v_dict: Variable name mapped to value at the operating point.
        :type v_dict: dict
        :param c_type: Enum indicating whether to use equality or inequality matrices.
        :type c_type: ConstraintType
        :return: Full EABC Jacobian matrix.
        :rtype: np.ndarray
        """
        S: np.ndarray
        Phi: np.ndarray
        S, Phi = self._get_matrices(c_type)
        
        v: np.ndarray = self._build_v_vector(v_dict)
        F: np.ndarray = self._compute_jacobian(S, v)
        EABC: np.ndarray = Phi @ F.T
        
        new_E: np.ndarray
        new_A: np.ndarray
        new_B: np.ndarray
        new_E, new_A, new_B = self._split_EABC(EABC)

        # Modifying internal state via setters
        self.set_E(new_E)
        self.set_A(new_A)
        self.set_B(new_B)
        
        return EABC

    # ------------------------------------------------------------------
    # Linearisation – private helpers
    # ------------------------------------------------------------------

    def _get_matrices(self, c_type: ConstraintType) -> tuple:
        """
        Return the (S, Phi) pair for equations or inequalities based on Enum.

        :param c_type: Type of constraint matrix to retrieve.
        :type c_type: ConstraintType
        :return: S and Phi arrays.
        :rtype: tuple
        """
        if c_type == ConstraintType.INEQUALITY:
            return self.S_W, self.Phi_W
        else:
            return self.S_H, self.Phi_H

    def _build_v_vector(self, v_dict: dict) -> np.ndarray:
        """
        Build the operating-point vector from a name to value dictionary.

        :param v_dict: Operating point variables.
        :type v_dict: dict
        :return: Vector array of operating points.
        :rtype: np.ndarray
        """
        v: np.ndarray = np.zeros(len(self.all_symbols))
        
        for name, val in v_dict.items():
            idx_opt: int | None = self.sym_to_idx.get(name, None)
            if idx_opt is not None:
                v[idx_opt] = val
            else:
                pass
                
        return v

    @profile
    def _compute_jacobian(self, S: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute the analytic Jacobian of the monomial basis at point *v*.

        :param S: Weight matrix (n_vars x n_monomials).
        :type S: np.ndarray
        :param v: Operating-point vector (n_vars,).
        :type v: np.ndarray
        :return: Jacobian F (n_vars x n_monomials).
        :rtype: np.ndarray
        """
        X: np.ndarray = (S.T * v) + (1.0 - np.abs(S.T))
        Y: np.ndarray = np.prod(X, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            invX: np.ndarray = np.where(np.abs(X) > 1e-12, 1.0 / X, 0.0)
            F: np.ndarray = S * (Y[:, np.newaxis] * invX).T

            zeros_per_col: np.ndarray = np.sum(np.abs(X) < 1e-12, axis=1)
            single_zero_cols: tuple = np.where(zeros_per_col == 1)
            
            for col in single_zero_cols[0]:
                zero_row_tuple: tuple = np.where(np.abs(X[col, :]) < 1e-12)
                zero_row: int = int(zero_row_tuple[0][0])
                other_factors: np.ndarray = np.delete(X[col, :], zero_row)
                F[zero_row, col] = S[zero_row, col] * np.prod(other_factors)

        return F

    def _split_EABC(self, EABC: np.ndarray) -> tuple:
        """
        Splits the EABC matrix into E, A, and B descriptor matrices.

        :param EABC: Full Jacobian matrix.
        :type EABC: np.ndarray
        :return: Tuple containing E, A, and B matrices.
        :rtype: tuple
        """
        idx_dx: list = list()
        idx_vars: list = list()
        idx_u: list = list()

        # Explicit loops to categorize indices
        idx_counter: int = 0
        for s in self.all_symbols:
            s_name: str = s.name
            if s_name.startswith('dx') or s_name.startswith('xp'):
                idx_dx.append(idx_counter)
            elif s_name.startswith('x') or s_name.startswith('y') or s_name.startswith('z'):
                idx_vars.append(idx_counter)
            elif s_name.startswith('u'):
                idx_u.append(idx_counter)
            else:
                pass
            idx_counter = idx_counter + 1

        n_eqs: int = int(EABC.shape[0])
        
        E: np.ndarray = np.zeros((n_eqs, n_eqs))
        
        for i in range(len(idx_dx)):
            if i < n_eqs:
                col_idx_e: int = idx_dx[i]
                E[:, i] = -EABC[:, col_idx_e]
            else:
                pass

        A: np.ndarray = np.zeros((n_eqs, n_eqs))
        
        cols_to_take: int = len(idx_vars)
        if cols_to_take > n_eqs:
            cols_to_take = n_eqs
        else:
            pass
            
        A[:, :cols_to_take] = EABC[:, idx_vars[:cols_to_take]]

        B: np.ndarray = EABC[:, idx_u]

        return E, A, B

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------

    @profile
    def compute_stability(self) -> tuple:
        """
        Compute generalised eigenvalues for the descriptor pair (A, E).

        :return: tuple of (eigenvalues, left_evecs, right_evecs, participation, is_stable, max_real)
        :rtype: tuple
        """
        current_a: np.ndarray | None = self.get_A()
        current_e: np.ndarray | None = self.get_E()

        if current_a is None or current_e is None:
            return None, None, None, None, False, None
        else:
            pass

        try:
            evals: np.ndarray
            evecs_left: np.ndarray
            evecs_right: np.ndarray
            evals, evecs_left, evecs_right = linalg.eig(current_a, current_e, left=True, right=True)
            
            finite_evs: np.ndarray = evals[np.isfinite(evals)]

            if len(finite_evs) == 0:
                return evals, evecs_left, evecs_right, None, True, -np.inf
            else:
                pass

            for i in range(len(evals)):
                scaling: complex = np.dot(evecs_left[:, i].conj(), evecs_right[:, i])
                evecs_left[:, i] = evecs_left[:, i] / scaling

            participation_matrix: np.ndarray = np.abs(evecs_right) * np.abs(evecs_left).T

            max_real: float = float(np.max(np.real(finite_evs)))
            is_stable: bool = False
            
            if max_real < 0.0:
                is_stable = True
            else:
                pass
                
            return evals, evecs_left, evecs_right, participation_matrix, is_stable, max_real

        except linalg.LinAlgError as e:
            # We catch strictly LinAlgError because Scipy internally throws this on convergence failure
            print(f"Error computing eigenvalues: {e}")
            if self.verbose:
                det_val: float = float(linalg.det(current_e - current_a))
                print(f"det(E - A) = {det_val}")
            else:
                pass
            return None, None, None, None, False, None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        eigenvalues: np.ndarray | None = None,
        is_stable: bool = False,
        max_real: float | None = None,
        print_matrices: bool = True,
        save_path: str | None = None,
    ) -> None:
        """
        Print and/or save all matrices and stability results.

        :param eigenvalues: Array of eigenvalues.
        :type eigenvalues: np.ndarray | None
        :param is_stable: Boolean flag indicating if the system is stable.
        :type is_stable: bool
        :param max_real: Maximum real part among the eigenvalues.
        :type max_real: float | None
        :param print_matrices: Flag to trigger stdout printing.
        :type print_matrices: bool
        :param save_path: Absolute or relative file path for writing outputs.
        :type save_path: str | None
        :return: None
        :rtype: None
        """
        if print_matrices:
            # Replaced comprehension with explicit loop of tuples
            matrix_refs: list = [
                ("S", self.S_H),
                ("Phi", self.Phi_H),
                ("E", self.get_E()),
                ("A", self.get_A()),
                ("B", self.get_B()),
            ]
            
            for label, mat in matrix_refs:
                if mat is not None:
                    print(f"\n--- {label} Matrix ---")
                    print(sparse.csc_matrix(mat))
                else:
                    pass

            print("\n--- Stability Analysis ---")
            print(f"Is stable:     {is_stable}")
            print(f"Max Real Part: {max_real}")
            
            if eigenvalues is not None:
                valid_evs: list = list(eigenvalues[np.isfinite(eigenvalues)])
                sorted_evs: list = sorted(valid_evs, key=self._get_real_part, reverse=True)
                print("Eigenvalues:")
                for ev in sorted_evs:
                    print(f"  {ev.real:.6f} + {ev.imag:.6f}j")
            else:
                pass
        else:
            pass

        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Using an explicit file writer structure
            with open(save_path, "w", encoding="utf-8") as f:
                
                matrix_write_refs: list = [
                    ("S Matrix", self.S_H),
                    ("Phi Matrix", self.Phi_H),
                    ("E Matrix", self.get_E()),
                    ("A Matrix", self.get_A()),
                    ("B Matrix", self.get_B()),
                ]
                
                for label_w, mat_w in matrix_write_refs:
                    if mat_w is not None:
                        formatted_mat: str = self._arr2str(mat_w)
                        section_str: str = self._section(label_w, formatted_mat)
                        f.write(section_str + "\n")
                    else:
                        pass

                stability_txt: str = (
                    f"Is stable:     {is_stable}\n"
                    f"Max Real Part: {max_real}\n"
                )
                
                if eigenvalues is not None:
                    evs_str_list: list = list()
                    for ev in eigenvalues:
                        evs_str_list.append(f"  {ev}")
                    
                    # Using a simple loop instead of join with comprehension
                    evs_txt: str = ""
                    for s_ev in evs_str_list:
                        evs_txt = evs_txt + s_ev + "\n"
                        
                    stability_txt = stability_txt + f"Eigenvalues:\n{evs_txt}\n"
                else:
                    pass

                final_section: str = self._section("Stability Analysis", stability_txt)
                f.write(final_section)

            print(f"\nMatrices saved to: {save_path}")
        else:
            pass