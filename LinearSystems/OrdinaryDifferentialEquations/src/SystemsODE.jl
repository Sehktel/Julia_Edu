"""
    SystemsODE

Модуль, реализующий специализированные методы для решения систем обыкновенных дифференциальных уравнений.
"""
module SystemsODE

export solve_system_ode, system_stability, system_eigenvalues

using LinearAlgebra
include("RungeKuttaMethod.jl")

"""
    solve_system_ode(f, t_span, y0; 
                    method="RK4", step_size=0.01, kwargs...)

Решает систему обыкновенных дифференциальных уравнений первого порядка.

# Аргументы
- `f::Function`: Функция правой части системы ОДУ в форме Y' = f(t, Y), где Y - вектор
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Vector{Float64}`: Вектор начальных условий
- `method::String="RK4"`: Метод решения (возможные значения: "RK4", "DOPRI", "Adams")
- `step_size::Float64=0.01`: Размер шага интегрирования
- `kwargs...`: Дополнительные параметры для методов решения

# Возвращает
- `t::Vector{Float64}`: Сетка значений времени
- `y::Matrix{Float64}`: Приближенное решение на сетке t, где каждая строка соответствует одной временной точке,
  а столбцы - компонентам вектора решения
"""
function solve_system_ode(f, t_span, y0; 
                         method="RK4", step_size=0.01, kwargs...)
    # Решаем систему ОДУ выбранным методом
    if method == "RK4"
        t, y = RungeKuttaMethod.runge_kutta4_solve(f, t_span, y0, step_size=step_size)
    elseif method == "DOPRI"
        t, y = RungeKuttaMethod.dopri_solve(f, t_span, y0; step_size=step_size, kwargs...)
    elseif method == "Adams"
        # Предполагается, что модуль AdamsMethod импортирован и экспортирует функцию adams_bashforth_moulton_solve
        include("AdamsMethod.jl")
        t, y = AdamsMethod.adams_bashforth_moulton_solve(f, t_span, y0, step_size=step_size)
    else
        error("Неподдерживаемый метод: $method. Используйте 'RK4', 'DOPRI' или 'Adams'.")
    end
    
    return t, y
end

"""
    system_jacobian(f, t, y; h=1e-6)

Вычисляет матрицу Якоби для системы ОДУ в точке (t, y).

# Аргументы
- `f::Function`: Функция правой части системы ОДУ в форме Y' = f(t, Y)
- `t::Float64`: Текущий момент времени
- `y::Vector{Float64}`: Текущее значение вектора состояния
- `h::Float64=1e-6`: Шаг для конечно-разностной аппроксимации производных

# Возвращает
- `J::Matrix{Float64}`: Матрица Якоби размера n×n, где n - размерность системы
"""
function system_jacobian(f, t, y; h=1e-6)
    n = length(y)
    J = zeros(n, n)
    
    # Вычисляем базовое значение f(t, y)
    f0 = f(t, y)
    
    # Вычисляем частные производные по каждой компоненте yi
    for i in 1:n
        y_plus = copy(y)
        y_plus[i] += h
        
        f_plus = f(t, y_plus)
        
        # Аппроксимация производной по формуле (f(y+h) - f(y)) / h
        J[:, i] = (f_plus - f0) / h
    end
    
    return J
end

"""
    system_eigenvalues(f, t, y)

Вычисляет собственные значения матрицы Якоби системы ОДУ в точке (t, y).
Собственные значения характеризуют локальную устойчивость системы.

# Аргументы
- `f::Function`: Функция правой части системы ОДУ в форме Y' = f(t, Y)
- `t::Float64`: Текущий момент времени
- `y::Vector{Float64}`: Текущее значение вектора состояния

# Возвращает
- `eigvals::Vector{ComplexF64}`: Вектор собственных значений матрицы Якоби
"""
function system_eigenvalues(f, t, y)
    # Вычисляем матрицу Якоби
    J = system_jacobian(f, t, y)
    
    # Вычисляем собственные значения
    return eigvals(J)
end

"""
    system_stability(f, t, y)

Анализирует локальную устойчивость системы ОДУ в точке (t, y) на основе собственных значений матрицы Якоби.

# Аргументы
- `f::Function`: Функция правой части системы ОДУ в форме Y' = f(t, Y)
- `t::Float64`: Текущий момент времени
- `y::Vector{Float64}`: Текущее значение вектора состояния

# Возвращает
- `stable::Bool`: Флаг устойчивости (true, если все собственные значения имеют отрицательную действительную часть)
- `eigvals::Vector{ComplexF64}`: Вектор собственных значений матрицы Якоби
- `stability_type::String`: Тип устойчивости ("устойчивый узел", "устойчивый фокус", "неустойчивый узел", 
   "неустойчивый фокус", "седло", "центр", "нейтральный")
"""
function system_stability(f, t, y)
    # Вычисляем собственные значения
    λ = system_eigenvalues(f, t, y)
    
    # Проверяем действительные части собственных значений
    real_parts = real.(λ)
    imag_parts = imag.(λ)
    
    # Система устойчива, если все действительные части отрицательны
    stable = all(real_parts .< 0)
    
    # Определяем тип устойчивости
    if all(real_parts .< 0)
        if all(imag_parts .≈ 0)
            stability_type = "устойчивый узел"
        else
            stability_type = "устойчивый фокус"
        end
    elseif all(real_parts .> 0)
        if all(imag_parts .≈ 0)
            stability_type = "неустойчивый узел"
        else
            stability_type = "неустойчивый фокус"
        end
    elseif any(real_parts .> 0) && any(real_parts .< 0)
        stability_type = "седло"
    elseif all(real_parts .≈ 0) && any(imag_parts .!= 0)
        stability_type = "центр"
    else
        stability_type = "нейтральный"
    end
    
    return stable, λ, stability_type
end

end # module 