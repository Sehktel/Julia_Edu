"""
    NonLinearSystems

Модуль, реализующий различные методы для решения систем нелинейных уравнений.
"""
module NonLinearSystems

using LinearAlgebra

# Включаем реализации всех методов
include("SimpleIteration.jl")
include("SeidelMethod.jl")
include("NewtonMethod.jl")

# Экспортируем публичные функции
export simple_iteration_solve, prepare_iteration_function, check_convergence_simple_iteration
export seidel_solve, check_convergence_seidel
export newton_solve, modified_newton_solve, compute_jacobian

end # module 