"""
# Численное интегрирование функций двух переменных

Данный модуль реализует методы численного интегрирования функций двух переменных,
включая двойные интегралы по прямоугольным и произвольным областям.
"""

"""
    double_integral(f, x_range, y_range; nx=100, ny=100)

Вычисляет двойной интеграл функции двух переменных `f(x, y)` по прямоугольной области.
Использует метод прямоугольников.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::Tuple{<:Real, <:Real}`: пределы интегрирования по x (a, b)
- `y_range::Tuple{<:Real, <:Real}`: пределы интегрирования по y (c, d)
- `nx::Int=100`: количество разбиений по оси x
- `ny::Int=100`: количество разбиений по оси y

# Возвращаемое значение
- `Float64`: значение двойного интеграла ∫∫ f(x, y) dx dy

# Примеры
```julia
f(x, y) = x^2 + y^2
result = double_integral(f, (0.0, 1.0), (0.0, 1.0))  # вычисляет ∫₀¹∫₀¹ (x² + y²) dx dy
```
"""
function double_integral(f::Function, x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                        nx::Int=100, ny::Int=100)
    # Проверка входных данных
    a, b = x_range
    c, d = y_range
    
    if a >= b || c >= d
        throw(ArgumentError("Пределы интегрирования должны быть упорядочены: a < b и c < d"))
    end
    
    if nx <= 0 || ny <= 0
        throw(ArgumentError("Количество разбиений должно быть положительным"))
    end
    
    # Шаги интегрирования
    dx = (b - a) / nx
    dy = (d - c) / ny
    
    # Площадь элементарного прямоугольника
    dA = dx * dy
    
    # Интегрирование методом средних прямоугольников
    result = 0.0
    for i in 1:nx
        x = a + (i - 0.5) * dx  # центр прямоугольника по x
        for j in 1:ny
            y = c + (j - 0.5) * dy  # центр прямоугольника по y
            result += f(x, y) * dA
        end
    end
    
    return result
end

"""
    double_integral_adaptive(f, x_range, y_range; tol=1e-6, max_depth=10)

Вычисляет двойной интеграл функции двух переменных `f(x, y)` по прямоугольной области
с использованием адаптивного алгоритма.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::Tuple{<:Real, <:Real}`: пределы интегрирования по x (a, b)
- `y_range::Tuple{<:Real, <:Real}`: пределы интегрирования по y (c, d)
- `tol::Real=1e-6`: допустимая погрешность
- `max_depth::Int=10`: максимальная глубина рекурсии

# Возвращаемое значение
- `Float64`: значение двойного интеграла ∫∫ f(x, y) dx dy

# Примеры
```julia
f(x, y) = sin(x) * cos(y)
result = double_integral_adaptive(f, (0.0, π), (0.0, π/2), tol=1e-8)
```
"""
function double_integral_adaptive(f::Function, x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                                 tol::Real=1e-6, max_depth::Int=10)
    # Реализуем рекурсивный адаптивный алгоритм
    function adaptive_quad(a, b, c, d, depth)
        # Вычисляем интеграл по всей области методом средних прямоугольников (одно разбиение)
        mid_x = (a + b) / 2
        mid_y = (c + d) / 2
        area = (b - a) * (d - c)
        I_whole = f(mid_x, mid_y) * area
        
        # Вычисляем интеграл по четырем подобластям
        mid_a_b = (a + b) / 2
        mid_c_d = (c + d) / 2
        
        q1_x = (a + mid_a_b) / 2
        q1_y = (c + mid_c_d) / 2
        q2_x = (mid_a_b + b) / 2
        q2_y = (c + mid_c_d) / 2
        q3_x = (a + mid_a_b) / 2
        q3_y = (mid_c_d + d) / 2
        q4_x = (mid_a_b + b) / 2
        q4_y = (mid_c_d + d) / 2
        
        area_quarter = area / 4
        I_q1 = f(q1_x, q1_y) * area_quarter
        I_q2 = f(q2_x, q2_y) * area_quarter
        I_q3 = f(q3_x, q3_y) * area_quarter
        I_q4 = f(q4_x, q4_y) * area_quarter
        
        I_refined = I_q1 + I_q2 + I_q3 + I_q4
        
        # Оцениваем погрешность
        error_estimate = abs(I_refined - I_whole)
        
        # Если погрешность в пределах допустимой или достигнута максимальная глубина
        if error_estimate < tol * area || depth >= max_depth
            return I_refined
        else
            # Рекурсивно вычисляем интеграл по четырем подобластям
            I1 = adaptive_quad(a, mid_a_b, c, mid_c_d, depth + 1)
            I2 = adaptive_quad(mid_a_b, b, c, mid_c_d, depth + 1)
            I3 = adaptive_quad(a, mid_a_b, mid_c_d, d, depth + 1)
            I4 = adaptive_quad(mid_a_b, b, mid_c_d, d, depth + 1)
            
            return I1 + I2 + I3 + I4
        end
    end
    
    # Запускаем рекурсивный алгоритм
    a, b = x_range
    c, d = y_range
    
    if a >= b || c >= d
        throw(ArgumentError("Пределы интегрирования должны быть упорядочены: a < b и c < d"))
    end
    
    return adaptive_quad(a, b, c, d, 0)
end

"""
    integrate_region(f, region_predicate, x_range, y_range; nx=100, ny=100)

Вычисляет интеграл функции двух переменных `f(x, y)` по произвольной области,
заданной предикатом `region_predicate`.

# Аргументы
- `f::Function`: функция двух переменных
- `region_predicate::Function`: функция предикат (x, y) -> Bool, определяющая принадлежность точки области
- `x_range::Tuple{<:Real, <:Real}`: диапазон по x, содержащий область
- `y_range::Tuple{<:Real, <:Real}`: диапазон по y, содержащий область
- `nx::Int=100`: количество разбиений по оси x
- `ny::Int=100`: количество разбиений по оси y

# Возвращаемое значение
- `Float64`: значение интеграла ∫∫ f(x, y) dx dy по заданной области

# Примеры
```julia
f(x, y) = x^2 + y^2
# Интеграл по кругу с центром в (0,0) и радиусом 1
circle(x, y) = x^2 + y^2 <= 1
result = integrate_region(f, circle, (-1.0, 1.0), (-1.0, 1.0), nx=200, ny=200)
```
"""
function integrate_region(f::Function, region_predicate::Function, 
                         x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                         nx::Int=100, ny::Int=100)
    # Проверка входных данных
    a, b = x_range
    c, d = y_range
    
    if a >= b || c >= d
        throw(ArgumentError("Пределы интегрирования должны быть упорядочены: a < b и c < d"))
    end
    
    if nx <= 0 || ny <= 0
        throw(ArgumentError("Количество разбиений должно быть положительным"))
    end
    
    # Шаги интегрирования
    dx = (b - a) / nx
    dy = (d - c) / ny
    
    # Площадь элементарного прямоугольника
    dA = dx * dy
    
    # Интегрирование методом средних прямоугольников
    result = 0.0
    for i in 1:nx
        x = a + (i - 0.5) * dx  # центр прямоугольника по x
        for j in 1:ny
            y = c + (j - 0.5) * dy  # центр прямоугольника по y
            
            # Проверяем, принадлежит ли точка области
            if region_predicate(x, y)
                result += f(x, y) * dA
            end
        end
    end
    
    return result
end

"""
    monte_carlo_integration(f, region_predicate, x_range, y_range; n_samples=10000)

Вычисляет интеграл функции двух переменных `f(x, y)` по произвольной области
методом Монте-Карло.

# Аргументы
- `f::Function`: функция двух переменных
- `region_predicate::Function`: функция предикат (x, y) -> Bool, определяющая принадлежность точки области
- `x_range::Tuple{<:Real, <:Real}`: диапазон по x, содержащий область
- `y_range::Tuple{<:Real, <:Real}`: диапазон по y, содержащий область
- `n_samples::Int=10000`: количество случайных точек

# Возвращаемое значение
- `Float64`: значение интеграла ∫∫ f(x, y) dx dy по заданной области

# Примеры
```julia
f(x, y) = exp(-(x^2 + y^2))
# Интеграл по кругу с центром в (0,0) и радиусом 2
circle(x, y) = x^2 + y^2 <= 4
result = monte_carlo_integration(f, circle, (-2.0, 2.0), (-2.0, 2.0), n_samples=100000)
```
"""
function monte_carlo_integration(f::Function, region_predicate::Function, 
                               x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                               n_samples::Int=10000)
    # Проверка входных данных
    a, b = x_range
    c, d = y_range
    
    if a >= b || c >= d
        throw(ArgumentError("Пределы интегрирования должны быть упорядочены: a < b и c < d"))
    end
    
    if n_samples <= 0
        throw(ArgumentError("Количество точек должно быть положительным"))
    end
    
    # Площадь прямоугольника, содержащего область
    rectangle_area = (b - a) * (d - c)
    
    # Генерируем случайные точки в прямоугольнике
    points_in_region = 0
    sum_f = 0.0
    
    for _ in 1:n_samples
        x = a + rand() * (b - a)
        y = c + rand() * (d - c)
        
        if region_predicate(x, y)
            points_in_region += 1
            sum_f += f(x, y)
        end
    end
    
    # Если не нашлось точек внутри области
    if points_in_region == 0
        return 0.0
    end
    
    # Оценка площади области
    region_area = rectangle_area * points_in_region / n_samples
    
    # Оценка среднего значения функции внутри области
    avg_f = sum_f / points_in_region
    
    # Интеграл = среднее значение функции * площадь области
    return avg_f * region_area
end

"""
    line_integral_path(f, path, t_range; n_points=1000)

Вычисляет криволинейный интеграл функции двух переменных `f(x, y)` 
по заданному пути `path(t) = (x(t), y(t))`.

# Аргументы
- `f::Function`: функция двух переменных
- `path::Function`: функция параметризации пути, возвращающая [x(t), y(t)]
- `t_range::Tuple{<:Real, <:Real}`: диапазон параметра t
- `n_points::Int=1000`: количество точек дискретизации пути

# Возвращаемое значение
- `Float64`: значение криволинейного интеграла ∫ f(x,y) ds

# Примеры
```julia
f(x, y) = x + y
# Интеграл по окружности радиуса 1 с центром в начале координат
circle_path(t) = [cos(t), sin(t)]
result = line_integral_path(f, circle_path, (0.0, 2π), n_points=1000)
```
"""
function line_integral_path(f::Function, path::Function, t_range::Tuple{<:Real, <:Real}; 
                           n_points::Int=1000)
    # Проверка входных данных
    t_start, t_end = t_range
    
    if t_start >= t_end
        throw(ArgumentError("Диапазон параметра должен быть упорядочен: t_start < t_end"))
    end
    
    if n_points <= 1
        throw(ArgumentError("Количество точек должно быть больше 1"))
    end
    
    # Шаг по параметру
    dt = (t_end - t_start) / (n_points - 1)
    
    # Вычисление интеграла методом трапеций
    result = 0.0
    
    # Предыдущая точка пути
    prev_point = path(t_start)
    prev_x, prev_y = prev_point
    
    for i in 2:n_points
        t = t_start + (i - 1) * dt
        
        # Текущая точка пути
        curr_point = path(t)
        curr_x, curr_y = curr_point
        
        # Длина элементарного участка пути (≈ расстояние между точками)
        ds = sqrt((curr_x - prev_x)^2 + (curr_y - prev_y)^2)
        
        # Среднее значение функции на участке
        avg_f = (f(prev_x, prev_y) + f(curr_x, curr_y)) / 2
        
        # Вклад в интеграл
        result += avg_f * ds
        
        # Обновляем предыдущую точку
        prev_x, prev_y = curr_x, curr_y
    end
    
    return result
end 