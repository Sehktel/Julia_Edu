"""
    EulerMethod

Модуль, реализующий метод Эйлера и его модификацию с пересчетом для решения 
обыкновенных дифференциальных уравнений первого порядка.
"""
module EulerMethod

export euler_solve, improved_euler_solve

"""
    euler_solve(f, t_span, y0; step_size=0.01)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка методом Эйлера.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования

# Возвращает
- `t::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}` или `Matrix{Float64}`: Приближенное решение ОДУ
"""
function euler_solve(f, t_span, y0; step_size=0.01)
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем сетку точек
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Проверяем размерность задачи (скалярное ОДУ или система)
    if isa(y0, Number)
        # Скалярное ОДУ
        y = zeros(length(t))
        y[1] = y0
        
        # Основной цикл метода Эйлера
        for i in 1:(length(t) - 1)
            h = t[i+1] - t[i]  # Учитываем возможный неравномерный шаг в конце
            y[i+1] = y[i] + h * f(t[i], y[i])
        end
    else
        # Система ОДУ
        n = length(y0)
        y = zeros(length(t), n)
        y[1, :] = y0
        
        # Основной цикл метода Эйлера
        for i in 1:(length(t) - 1)
            h = t[i+1] - t[i]
            y[i+1, :] = y[i, :] + h * f(t[i], y[i, :])
        end
    end
    
    return t, y
end

"""
    improved_euler_solve(f, t_span, y0; step_size=0.01)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
методом Эйлера с пересчетом (модифицированный метод Эйлера).

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования

# Возвращает
- `t::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}` или `Matrix{Float64}`: Приближенное решение ОДУ
"""
function improved_euler_solve(f, t_span, y0; step_size=0.01)
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем сетку точек
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Проверяем размерность задачи (скалярное ОДУ или система)
    if isa(y0, Number)
        # Скалярное ОДУ
        y = zeros(length(t))
        y[1] = y0
        
        # Основной цикл улучшенного метода Эйлера
        for i in 1:(length(t) - 1)
            h = t[i+1] - t[i]
            
            # Предиктор: используем явный метод Эйлера
            y_pred = y[i] + h * f(t[i], y[i])
            
            # Корректор: усредняем производные в начальной и конечной точках
            y[i+1] = y[i] + h/2 * (f(t[i], y[i]) + f(t[i+1], y_pred))
        end
    else
        # Система ОДУ
        n = length(y0)
        y = zeros(length(t), n)
        y[1, :] = y0
        
        # Основной цикл улучшенного метода Эйлера
        for i in 1:(length(t) - 1)
            h = t[i+1] - t[i]
            
            # Предиктор
            y_pred = y[i, :] + h * f(t[i], y[i, :])
            
            # Корректор
            y[i+1, :] = y[i, :] + h/2 * (f(t[i], y[i, :]) + f(t[i+1], y_pred))
        end
    end
    
    return t, y
end

end # module 