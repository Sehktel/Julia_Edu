module LinearSystems

using LinearAlgebra
using SparseArrays

# Включаем реализации всех методов
include("DirectMethods.jl")
include("GaussianElimination.jl")
include("ThomasAlgorithm.jl")
include("GaussSeidel.jl")
include("SimpleIteration.jl")
include("SeidelMethod.jl")
include("NewtonMethod.jl")

# Экспортируем публичные функции
export solve_direct, solve_lu, solve_cholesky
export gauss_elimination, gauss_elimination_pivot
export thomas_algorithm
export gauss_seidel, gauss_seidel_matrix_form, is_diagonally_dominant

# Экспортируем функции для нелинейных систем
export simple_iteration_solve, prepare_iteration_function
export seidel_solve
export newton_solve, modified_newton_solve, compute_jacobian

end # module 