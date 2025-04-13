"""
    RungeKuttaMethod

Модуль, реализующий методы Рунге-Кутта для решения 
обыкновенных дифференциальных уравнений первого порядка.
"""
module RungeKuttaMethod

export runge_kutta_solve, runge_kutta4_solve, dopri_solve

"""
    runge_kutta_solve(f, t_span, y0; step_size=0.01, method="RK4")

Решает задачу Коши для обыкновенного дифференциального уравнения методом Рунге-Кутта
с заданной схемой.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `method::String="RK4"`: Метод Рунге-Кутта; возможные значения: "Euler", "Midpoint", "Heun", "RK4", "DOPRI"

# Возвращает
- `t::Vector{Float64}`: Сетка значений времени
- `y::Union{Vector{Float64}, Matrix{Float64}}`: Приближенное решение на сетке t
"""
function runge_kutta_solve(f, t_span, y0; step_size=0.01, method="RK4")
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем временную сетку
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Получаем таблицу Бутчера для выбранного метода
    A, b, c = butcher_tableau(method)
    
    # Проверяем размерность начального условия
    scalar_problem = isa(y0, Number)
    
    # Инициализируем массив решения
    if scalar_problem
        y = zeros(length(t))
        y[1] = y0
    else
        y = zeros(length(t), length(y0))
        y[1, :] = y0
    end
    
    # Основной цикл метода Рунге-Кутта
    for i in 1:length(t)-1
        h = t[i+1] - t[i]  # Учитываем возможный неравномерный шаг
        
        # Текущие время и значение
        t_i = t[i]
        y_i = scalar_problem ? y[i] : y[i, :]
        
        # Количество стадий метода
        s = length(b)
        
        # Промежуточные значения k_i
        k = Vector{typeof(y_i)}(undef, s)
        
        # Вычисляем промежуточные значения k_i
        for j in 1:s
            # Сумма для вычисления аргумента функции f
            y_arg = copy(y_i)
            for l in 1:j-1
                if A[j, l] != 0
                    y_arg = y_arg .+ h * A[j, l] * k[l]
                end
            end
            
            k[j] = f(t_i + c[j] * h, y_arg)
        end
        
        # Вычисляем следующее значение решения
        y_next = copy(y_i)
        for j in 1:s
            y_next = y_next .+ h * b[j] * k[j]
        end
        
        # Сохраняем результат
        if scalar_problem
            y[i+1] = y_next
        else
            y[i+1, :] = y_next
        end
    end
    
    return t, y
end

"""
    butcher_tableau(method)

Возвращает таблицу Бутчера для заданного метода Рунге-Кутта.

# Аргументы
- `method::String`: Название метода Рунге-Кутта

# Возвращает
- `A::Matrix{Float64}`: Матрица коэффициентов a_{ij}
- `b::Vector{Float64}`: Вектор весовых коэффициентов b_i
- `c::Vector{Float64}`: Вектор узлов c_i
"""
function butcher_tableau(method)
    if method == "Euler" # Метод Эйлера (RK1)
        A = zeros(1, 1)
        b = [1.0]
        c = [0.0]
    elseif method == "Midpoint" # Метод средней точки (RK2)
        A = [0.0 0.0; 0.5 0.0]
        b = [0.0, 1.0]
        c = [0.0, 0.5]
    elseif method == "Heun" # Метод Хойна (RK2)
        A = [0.0 0.0; 1.0 0.0]
        b = [0.5, 0.5]
        c = [0.0, 1.0]
    elseif method == "RK3" # Метод Рунге-Кутта 3-го порядка
        A = [0.0 0.0 0.0; 0.5 0.0 0.0; -1.0 2.0 0.0]
        b = [1/6, 2/3, 1/6]
        c = [0.0, 0.5, 1.0]
    elseif method == "RK4" # Классический метод Рунге-Кутта (RK4)
        A = [0.0 0.0 0.0 0.0;
             0.5 0.0 0.0 0.0;
             0.0 0.5 0.0 0.0;
             0.0 0.0 1.0 0.0]
        b = [1/6, 1/3, 1/3, 1/6]
        c = [0.0, 0.5, 0.5, 1.0]
    elseif method == "DOPRI" # Метод Дормана-Принса (RK5(4))
        A = zeros(7, 7)
        A[2, 1] = 1/5
        A[3, 1:2] = [3/40, 9/40]
        A[4, 1:3] = [44/45, -56/15, 32/9]
        A[5, 1:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
        A[6, 1:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        A[7, 1:6] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        
        # Для RK5
        b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    else
        error("Неизвестный метод Рунге-Кутта: $method")
    end
    
    return A, b, c
end

"""
    runge_kutta4_solve(f, t_span, y0; step_size=0.01)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
классическим методом Рунге-Кутта 4-го порядка (RK4).

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования

# Возвращает
- `t::Vector{Float64}`: Сетка значений времени
- `y::Union{Vector{Float64}, Matrix{Float64}}`: Приближенное решение на сетке t
"""
function runge_kutta4_solve(f, t_span, y0; step_size=0.01)
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем временную сетку
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Проверяем размерность начального условия
    scalar_problem = isa(y0, Number)
    
    # Инициализируем массив решения
    if scalar_problem
        y = zeros(length(t))
        y[1] = y0
    else
        y = zeros(length(t), length(y0))
        y[1, :] = y0
    end
    
    # Основной цикл метода Рунге-Кутта 4-го порядка
    for i in 1:length(t)-1
        h = t[i+1] - t[i]  # Учитываем возможный неравномерный шаг
        
        # Текущие время и значение
        t_i = t[i]
        y_i = scalar_problem ? y[i] : y[i, :]
        
        # Вычисляем коэффициенты k1, k2, k3, k4
        k1 = f(t_i, y_i)
        k2 = f(t_i + h/2, y_i + h/2 * k1)
        k3 = f(t_i + h/2, y_i + h/2 * k2)
        k4 = f(t_i + h, y_i + h * k3)
        
        # Вычисляем следующее значение решения
        y_next = y_i + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Сохраняем результат
        if scalar_problem
            y[i+1] = y_next
        else
            y[i+1, :] = y_next
        end
    end
    
    return t, y
end

"""
    dopri_solve(f, t_span, y0; atol=1e-6, rtol=1e-3, initial_step=0.01, max_step=1.0, min_step=1e-10, max_steps=10000)

Решает задачу Коши для обыкновенного дифференциального уравнения методом Дормана-Принса (RK5(4))
с адаптивным выбором шага.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `atol::Float64=1e-6`: Абсолютная допустимая погрешность
- `rtol::Float64=1e-3`: Относительная допустимая погрешность
- `initial_step::Float64=0.01`: Начальный размер шага
- `max_step::Float64=1.0`: Максимальный допустимый размер шага
- `min_step::Float64=1e-10`: Минимальный допустимый размер шага
- `max_steps::Int=10000`: Максимальное количество шагов

# Возвращает
- `t::Vector{Float64}`: Адаптивная сетка значений времени
- `y::Union{Vector{Float64}, Matrix{Float64}}`: Приближенное решение на сетке t
"""
function dopri_solve(f, t_span, y0; atol=1e-6, rtol=1e-3, initial_step=0.01, max_step=1.0, min_step=1e-10, max_steps=10000)
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Проверяем размерность начального условия
    scalar_problem = isa(y0, Number)
    
    # Инициализируем массивы для решения с адаптивным шагом
    t = [t0]
    if scalar_problem
        y = [y0]
    else
        y = [y0]
    end
    
    # Получаем коэффициенты метода Дормана-Принса
    A, b5, b4, c = dopri_coefficients()
    
    # Текущее время и значение
    t_current = t0
    y_current = y0
    h = initial_step
    
    # Основной цикл с адаптивным шагом
    step_count = 0
    while t_current < tf && step_count < max_steps
        # Ограничиваем шаг, чтобы не выйти за пределы интервала
        h = min(h, tf - t_current)
        
        # Промежуточные значения k_i
        k = Vector{typeof(y_current)}(undef, 7)
        
        # Вычисляем промежуточные значения k_i для метода DOPRI
        k[1] = f(t_current, y_current)
        
        y_temp = y_current + h * A[2, 1] * k[1]
        k[2] = f(t_current + c[2] * h, y_temp)
        
        y_temp = y_current + h * (A[3, 1] * k[1] + A[3, 2] * k[2])
        k[3] = f(t_current + c[3] * h, y_temp)
        
        y_temp = y_current + h * (A[4, 1] * k[1] + A[4, 2] * k[2] + A[4, 3] * k[3])
        k[4] = f(t_current + c[4] * h, y_temp)
        
        y_temp = y_current + h * (A[5, 1] * k[1] + A[5, 2] * k[2] + A[5, 3] * k[3] + A[5, 4] * k[4])
        k[5] = f(t_current + c[5] * h, y_temp)
        
        y_temp = y_current + h * (A[6, 1] * k[1] + A[6, 2] * k[2] + A[6, 3] * k[3] + A[6, 4] * k[4] + A[6, 5] * k[5])
        k[6] = f(t_current + c[6] * h, y_temp)
        
        y_temp = y_current + h * (A[7, 1] * k[1] + A[7, 2] * k[2] + A[7, 3] * k[3] + A[7, 4] * k[4] + A[7, 5] * k[5] + A[7, 6] * k[6])
        k[7] = f(t_current + c[7] * h, y_temp)
        
        # Вычисляем решения 5-го и 4-го порядков
        y5 = y_current
        y4 = y_current
        
        for j in 1:7
            y5 = y5 + h * b5[j] * k[j]
            if j < 7  # b4[7] = 0
                y4 = y4 + h * b4[j] * k[j]
            end
        end
        
        # Оцениваем локальную ошибку
        if scalar_problem
            err = abs(y5 - y4)
            sc = atol + rtol * max(abs(y_current), abs(y5))
        else
            err = norm(y5 - y4)
            sc = sqrt(sum((atol .+ rtol .* max.(abs.(y_current), abs.(y5))).^2))
        end
        
        # Вычисляем оптимальный размер шага
        err_ratio = err / sc
        if err_ratio == 0
            # Безопасность от деления на ноль
            h_new = 2 * h
        else
            # Стандартная формула для вычисления нового шага
            h_new = 0.9 * h * (1 / err_ratio)^(1/5)
        end
        
        # Ограничиваем новый шаг
        h_new = min(max(h_new, min_step), max_step)
        
        # Проверяем, принимаем ли мы текущий шаг
        if err_ratio <= 1.0
            # Шаг принят
            t_current = t_current + h
            y_current = y5
            
            # Сохраняем результат
            push!(t, t_current)
            if scalar_problem
                push!(y, y_current)
            else
                push!(y, copy(y_current))
            end
        end
        
        # Обновляем размер шага для следующей итерации
        h = h_new
        
        # Увеличиваем счетчик шагов
        step_count += 1
    end
    
    # Проверяем, достигли ли мы конца интервала
    if t[end] < tf && abs(t[end] - tf) > min_step
        # Добавляем последнюю точку, если не достигли
        h = tf - t[end]
        
        # Используем стандартный метод RK4 для последнего шага
        k1 = f(t[end], y[end])
        k2 = f(t[end] + h/2, y[end] + h/2 * k1)
        k3 = f(t[end] + h/2, y[end] + h/2 * k2)
        k4 = f(t[end] + h, y[end] + h * k3)
        
        y_last = y[end] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        push!(t, tf)
        push!(y, y_last)
    end
    
    # Преобразуем результаты в нужный формат
    if scalar_problem
        y_out = collect(y)
    else
        y_out = hcat(y...)
        if size(y_out, 1) > 1
            y_out = y_out'
        end
    end
    
    return t, y_out
end

"""
    dopri_coefficients()

Возвращает коэффициенты метода Дормана-Принса (RK5(4)).

# Возвращает
- `A::Matrix{Float64}`: Матрица коэффициентов a_{ij}
- `b5::Vector{Float64}`: Вектор весовых коэффициентов для 5-го порядка
- `b4::Vector{Float64}`: Вектор весовых коэффициентов для 4-го порядка
- `c::Vector{Float64}`: Вектор узлов
"""
function dopri_coefficients()
    # Коэффициенты для метода Дормана-Принса
    A = zeros(7, 7)
    A[2, 1] = 1/5
    A[3, 1:2] = [3/40, 9/40]
    A[4, 1:3] = [44/45, -56/15, 32/9]
    A[5, 1:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
    A[6, 1:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    A[7, 1:6] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    
    # Коэффициенты для 5-го порядка
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    
    # Коэффициенты для 4-го порядка (для оценки ошибки)
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    # Узлы
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    return A, b5, b4, c
end

end # module 