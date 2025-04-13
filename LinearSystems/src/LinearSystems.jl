module LinearSystems

using LinearAlgebra
using SparseArrays

# Включаем реализации всех методов
include("DirectMethods.jl")
include("GaussianElimination.jl")
include("ThomasAlgorithm.jl")
include("GaussSeidel.jl")

# Экспортируем публичные функции
export solve_direct, solve_lu, solve_cholesky
export gauss_elimination, gauss_elimination_pivot
export thomas_algorithm
export gauss_seidel, gauss_seidel_matrix_form, is_diagonally_dominant

end # module 