"""
    AdamsMethod

Модуль, реализующий методы Адамса для решения 
обыкновенных дифференциальных уравнений первого порядка.
"""
module AdamsMethod

export adams_bashforth_solve, adams_moulton_solve, adams_bashforth_moulton_solve

"""
    adams_bashforth_solve(f, t_span, y0; step_size=0.01, order=4)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
явным методом Адамса-Башфорта заданного порядка.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Башфорта (от 1 до 5)

# Возвращает
- `t::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}` или `Matrix{Float64}`: Приближенное решение ОДУ
"""
function adams_bashforth_solve(f, t_span, y0; step_size=0.01, order=4)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем сетку точек
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Получаем коэффициенты метода Адамса-Башфорта
    ab_coeffs = adams_bashforth_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    if isa(y0, Number)
        # Скалярное ОДУ
        y = zeros(length(t))
        y[1] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order, length(t) - 1)
            k1 = f(t[i], y[i])
            k2 = f(t[i] + step_size/2, y[i] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i] + step_size * k3)
            
            y[i+1] = y[i] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i] = f(t[i], y[i])
        end
        
        # Основной цикл метода Адамса-Башфорта
        for i in (order+1):length(t)
            # Сдвигаем массив производных
            for j in 1:(order-1)
                fs[j] = fs[j+1]
            end
            
            # Добавляем новую производную
            fs[order] = f(t[i-1], y[i-1])
            
            # Вычисляем следующее значение
            y[i] = y[i-1] + step_size * sum(ab_coeffs .* fs)
        end
    else
        # Система ОДУ
        n = length(y0)
        y = zeros(length(t), n)
        y[1, :] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order, n)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order, length(t) - 1)
            k1 = f(t[i], y[i, :])
            k2 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i, :] + step_size * k3)
            
            y[i+1, :] = y[i, :] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i, :] = f(t[i], y[i, :])
        end
        
        # Основной цикл метода Адамса-Башфорта
        for i in (order+1):length(t)
            # Сдвигаем массив производных
            for j in 1:(order-1)
                fs[j, :] = fs[j+1, :]
            end
            
            # Добавляем новую производную
            fs[order, :] = f(t[i-1], y[i-1, :])
            
            # Вычисляем следующее значение для каждой компоненты
            for j in 1:n
                y[i, j] = y[i-1, j] + step_size * sum(ab_coeffs .* fs[:, j])
            end
        end
    end
    
    return t, y
end

"""
    adams_moulton_solve(f, t_span, y0; step_size=0.01, order=4, tol=1e-6, max_iter=10)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
неявным методом Адамса-Мультона заданного порядка.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Мультона (от 1 до 5)
- `tol::Float64=1e-6`: Допустимая погрешность для итерационного процесса
- `max_iter::Int=10`: Максимальное число итераций для неявной схемы

# Возвращает
- `t::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}` или `Matrix{Float64}`: Приближенное решение ОДУ
"""
function adams_moulton_solve(f, t_span, y0; step_size=0.01, order=4, tol=1e-6, max_iter=10)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем сетку точек
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Получаем коэффициенты метода Адамса-Мультона
    am_coeffs = adams_moulton_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    if isa(y0, Number)
        # Скалярное ОДУ
        y = zeros(length(t))
        y[1] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order-1, length(t) - 1)
            k1 = f(t[i], y[i])
            k2 = f(t[i] + step_size/2, y[i] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i] + step_size * k3)
            
            y[i+1] = y[i] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i] = f(t[i], y[i])
        end
        
        # Основной цикл метода Адамса-Мультона
        for i in order:length(t) - 1
            # Сдвигаем массив производных
            for j in 1:(order-2)
                fs[j] = fs[j+1]
            end
            
            # Добавляем новую производную
            fs[order-1] = f(t[i], y[i])
            
            # Начальное приближение для y[i+1] (используем явный метод)
            y_prev = y[i] + step_size * (am_coeffs[1:order-1]' * fs[1:order-1])
            
            # Итерационный процесс для решения неявной схемы
            for iter in 1:max_iter
                f_new = f(t[i+1], y_prev)
                y_new = y[i] + step_size * (am_coeffs[1:order-1]' * fs[1:order-1] + am_coeffs[order] * f_new)
                
                # Проверка сходимости
                if abs(y_new - y_prev) < tol
                    y_prev = y_new
                    break
                end
                
                y_prev = y_new
            end
            
            y[i+1] = y_prev
        end
    else
        # Система ОДУ
        n = length(y0)
        y = zeros(length(t), n)
        y[1, :] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order, n)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order-1, length(t) - 1)
            k1 = f(t[i], y[i, :])
            k2 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i, :] + step_size * k3)
            
            y[i+1, :] = y[i, :] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i, :] = f(t[i], y[i, :])
        end
        
        # Основной цикл метода Адамса-Мультона
        for i in order:length(t) - 1
            # Сдвигаем массив производных
            for j in 1:(order-2)
                fs[j, :] = fs[j+1, :]
            end
            
            # Добавляем новую производную
            fs[order-1, :] = f(t[i], y[i, :])
            
            # Начальное приближение для y[i+1] (используем явный метод)
            y_prev = copy(y[i, :])
            for j in 1:n
                y_prev[j] += step_size * sum(am_coeffs[1:order-1] .* fs[1:order-1, j])
            end
            
            # Итерационный процесс для решения неявной схемы
            for iter in 1:max_iter
                f_new = f(t[i+1], y_prev)
                
                y_new = copy(y[i, :])
                for j in 1:n
                    y_new[j] += step_size * (sum(am_coeffs[1:order-1] .* fs[1:order-1, j]) + am_coeffs[order] * f_new[j])
                end
                
                # Проверка сходимости
                if norm(y_new - y_prev) < tol
                    y_prev = y_new
                    break
                end
                
                y_prev = y_new
            end
            
            y[i+1, :] = y_prev
        end
    end
    
    return t, y
end

"""
    adams_bashforth_moulton_solve(f, t_span, y0; step_size=0.01, order=4, tol=1e-6, max_iter=10)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
методом прогноза-коррекции Адамса-Башфорта-Мультона (предиктор-корректор).

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Башфорта-Мультона (от 1 до 5)
- `max_iter::Int=1`: Максимальное число итераций корректора

# Возвращает
- `t::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}` или `Matrix{Float64}`: Приближенное решение ОДУ
"""
function adams_bashforth_moulton_solve(f, t_span, y0; step_size=0.01, order=4, max_iter=1)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Создаем сетку точек
    t = collect(t0:step_size:tf)
    if t[end] != tf
        push!(t, tf)
    end
    
    # Получаем коэффициенты методов
    ab_coeffs = adams_bashforth_coefficients(order)
    am_coeffs = adams_moulton_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    if isa(y0, Number)
        # Скалярное ОДУ
        y = zeros(length(t))
        y[1] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order, length(t) - 1)
            k1 = f(t[i], y[i])
            k2 = f(t[i] + step_size/2, y[i] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i] + step_size * k3)
            
            y[i+1] = y[i] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i] = f(t[i], y[i])
        end
        
        # Основной цикл метода Адамса-Башфорта-Мультона
        for i in (order+1):length(t)
            # Сдвигаем массив производных
            for j in 1:(order-1)
                fs[j] = fs[j+1]
            end
            
            # Добавляем новую производную
            fs[order] = f(t[i-1], y[i-1])
            
            # Предиктор (метод Адамса-Башфорта)
            y_pred = y[i-1] + step_size * sum(ab_coeffs .* fs)
            
            # Корректор (метод Адамса-Мультона)
            y_corr = y_pred
            for iter in 1:max_iter
                f_new = f(t[i], y_corr)
                
                # Коэффициенты для корректора
                fs_corr = copy(fs)
                fs_corr[order] = f_new
                
                # Применяем корректор
                y_corr = y[i-1] + step_size * sum(am_coeffs .* fs_corr)
            end
            
            y[i] = y_corr
        end
    else
        # Система ОДУ
        n = length(y0)
        y = zeros(length(t), n)
        y[1, :] = y0
        
        # Массив для хранения производных на предыдущих шагах
        fs = zeros(order, n)
        
        # Используем РК4 для начальных значений
        for i in 1:min(order, length(t) - 1)
            k1 = f(t[i], y[i, :])
            k2 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k1)
            k3 = f(t[i] + step_size/2, y[i, :] + step_size/2 * k2)
            k4 = f(t[i] + step_size, y[i, :] + step_size * k3)
            
            y[i+1, :] = y[i, :] + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Сохраняем производную
            fs[i, :] = f(t[i], y[i, :])
        end
        
        # Основной цикл метода Адамса-Башфорта-Мультона
        for i in (order+1):length(t)
            # Сдвигаем массив производных
            for j in 1:(order-1)
                fs[j, :] = fs[j+1, :]
            end
            
            # Добавляем новую производную
            fs[order, :] = f(t[i-1], y[i-1, :])
            
            # Предиктор (метод Адамса-Башфорта)
            y_pred = copy(y[i-1, :])
            for j in 1:n
                y_pred[j] += step_size * sum(ab_coeffs .* fs[:, j])
            end
            
            # Корректор (метод Адамса-Мультона)
            y_corr = copy(y_pred)
            for iter in 1:max_iter
                f_new = f(t[i], y_corr)
                
                # Применяем корректор
                for j in 1:n
                    fs_corr = copy(fs[:, j])
                    fs_corr[order] = f_new[j]
                    
                    y_corr[j] = y[i-1, j] + step_size * sum(am_coeffs .* fs_corr)
                end
            end
            
            y[i, :] = y_corr
        end
    end
    
    return t, y
end

"""
    adams_bashforth_coefficients(order)

Возвращает коэффициенты явного метода Адамса-Башфорта заданного порядка.

# Аргументы
- `order::Int`: Порядок метода (от 1 до 5)

# Возвращает
- `coeffs::Vector{Float64}`: Вектор коэффициентов метода
"""
function adams_bashforth_coefficients(order)
    if order == 1
        # Явный метод Эйлера
        return [1.0]
    elseif order == 2
        return [3/2, -1/2]
    elseif order == 3
        return [23/12, -16/12, 5/12]
    elseif order == 4
        return [55/24, -59/24, 37/24, -9/24]
    elseif order == 5
        return [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
    else
        error("Неподдерживаемый порядок метода Адамса-Башфорта: $order")
    end
end

"""
    adams_moulton_coefficients(order)

Возвращает коэффициенты неявного метода Адамса-Мультона заданного порядка.

# Аргументы
- `order::Int`: Порядок метода (от 1 до 5)

# Возвращает
- `coeffs::Vector{Float64}`: Вектор коэффициентов метода
"""
function adams_moulton_coefficients(order)
    if order == 1
        # Неявный метод Эйлера
        return [1.0]
    elseif order == 2
        return [1/2, 1/2]
    elseif order == 3
        return [5/12, 8/12, -1/12]
    elseif order == 4
        return [9/24, 19/24, -5/24, 1/24]
    elseif order == 5
        return [251/720, 646/720, -264/720, 106/720, -19/720]
    else
        error("Неподдерживаемый порядок метода Адамса-Мультона: $order")
    end
end

end # module 