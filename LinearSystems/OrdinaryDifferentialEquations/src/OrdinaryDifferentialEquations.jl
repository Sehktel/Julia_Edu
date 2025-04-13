"""
    OrdinaryDifferentialEquations

Модуль, реализующий различные методы для решения обыкновенных дифференциальных уравнений.
"""
module OrdinaryDifferentialEquations

using LinearAlgebra

# Включаем реализации всех методов
include("EulerMethod.jl")
include("RungeKuttaMethod.jl")
include("AdamsMethod.jl")
include("BoundaryValueProblems.jl")
include("SystemsODE.jl")
include("HighOrderODE.jl")
include("JacobianMatrix.jl")

# Экспортируем публичные функции

# Методы Эйлера
export euler_solve, improved_euler_solve

# Методы Рунге-Кутта
export runge_kutta_solve, runge_kutta4_solve, dopri_solve

# Методы Адамса
export adams_bashforth_solve, adams_moulton_solve, adams_bashforth_moulton_solve

# Методы для краевых задач
export shooting_solve, finite_difference_solve

# Методы для систем ОДУ
export solve_system_ode, system_stability, system_eigenvalues

# Методы для ОДУ высокого порядка
export reduce_to_system, solve_high_order_ode

# Методы для работы с матрицей Якоби
export jacobian_matrix, jacobian_determinant, condition_number, 
       matrix_eigenvalues, matrix_eigenvectors, 
       analyze_critical_point, is_invertible

end # module 