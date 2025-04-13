"""
    HighOrderODE

Модуль, реализующий методы для решения обыкновенных дифференциальных уравнений высокого порядка.
"""
module HighOrderODE

export reduce_to_system, solve_high_order_ode

include("RungeKuttaMethod.jl")

"""
    reduce_to_system(f, n)

Преобразует ОДУ n-го порядка в систему n ОДУ первого порядка.

# Аргументы
- `f::Function`: Функция правой части ОДУ n-го порядка в форме y^(n) = f(t, y, y', y'', ..., y^(n-1))
- `n::Int`: Порядок ОДУ

# Возвращает
- `system_f::Function`: Функция правой части системы n ОДУ первого порядка
"""
function reduce_to_system(f, n)
    function system_f(t, y)
        # Преобразование системы к виду:
        # y₁' = y₂
        # y₂' = y₃
        # ...
        # yₙ' = f(t, y₁, y₂, ..., yₙ)
        
        result = zeros(n)
        
        # Первые n-1 уравнений: yᵢ' = yᵢ₊₁
        for i in 1:n-1
            result[i] = y[i+1]
        end
        
        # Последнее уравнение: yₙ' = f(t, y₁, y₂, ..., yₙ)
        result[n] = f(t, y...)
        
        return result
    end
    
    return system_f
end

"""
    solve_high_order_ode(f, t_span, initial_conditions; 
                         step_size=0.01, method="RK4", kwargs...)

Решает задачу Коши для ОДУ n-го порядка, преобразуя его в систему n ОДУ первого порядка.

# Аргументы
- `f::Function`: Функция правой части ОДУ n-го порядка в форме y^(n) = f(t, y, y', y'', ..., y^(n-1))
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `initial_conditions::Vector{Float64}`: Вектор начальных условий [y(t0), y'(t0), ..., y^(n-1)(t0)]
- `step_size::Float64=0.01`: Размер шага интегрирования
- `method::String="RK4"`: Метод решения (возможные значения: "RK4", "DOPRI")
- `kwargs...`: Дополнительные параметры для методов решения

# Возвращает
- `t::Vector{Float64}`: Сетка значений времени
- `y::Matrix{Float64}`: Приближенное решение на сетке t, где столбцы соответствуют y, y', ..., y^(n-1)
"""
function solve_high_order_ode(f, t_span, initial_conditions; 
                             step_size=0.01, method="RK4", kwargs...)
    # Определяем порядок ОДУ по количеству начальных условий
    n = length(initial_conditions)
    
    # Преобразуем ОДУ n-го порядка в систему n ОДУ первого порядка
    system_f = reduce_to_system(f, n)
    
    # Решаем полученную систему выбранным методом
    if method == "RK4"
        t, y = RungeKuttaMethod.runge_kutta4_solve(system_f, t_span, initial_conditions, step_size=step_size)
    elseif method == "DOPRI"
        t, y = RungeKuttaMethod.dopri_solve(system_f, t_span, initial_conditions; step_size=step_size, kwargs...)
    else
        error("Неподдерживаемый метод: $method. Используйте 'RK4' или 'DOPRI'.")
    end
    
    return t, y
end

end # module 