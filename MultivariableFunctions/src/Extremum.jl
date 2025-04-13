"""
# Анализ экстремумов функций двух переменных

Данный модуль предоставляет инструменты для поиска и классификации 
критических точек функций двух переменных (экстремумов и седловых точек).
"""

using LinearAlgebra
using Optim

"""
    find_critical_points(f, x_range, y_range; grid_size=20, threshold=1e-6)

Находит критические точки функции двух переменных `f` в заданной области.
Критические точки - это точки, в которых градиент функции равен нулю.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси x (min, max)
- `y_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси y (min, max)
- `grid_size::Int=20`: размер сетки для начального поиска
- `threshold::Real=1e-6`: порог для определения нулевого градиента

# Возвращаемое значение
- `Vector{Vector{Float64}}`: список найденных критических точек [x, y]

# Примеры
```julia
f(x, y) = x^2 + y^2 - 2x - 4y + 5
critical_points = find_critical_points(f, (-5.0, 5.0), (-5.0, 5.0))
```
"""
function find_critical_points(f::Function, x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                             grid_size::Int=20, threshold::Real=1e-6)
    # Создаем сетку начальных точек для поиска
    x_grid = range(x_range[1], x_range[2], length=grid_size)
    y_grid = range(y_range[1], y_range[2], length=grid_size)
    
    # Функция для минимизации - норма градиента
    grad_norm(point) = norm(gradient(f, point[1], point[2]))
    
    critical_points = Vector{Float64}[]
    
    # Проходим по всем точкам сетки как начальным приближениям
    for x0 in x_grid
        for y0 in y_grid
            # Используем оптимизатор для поиска ближайшей критической точки
            result = optimize(grad_norm, [x0, y0], BFGS())
            
            # Если найденная точка имеет нулевой градиент (с заданной точностью)
            if Optim.minimum(result) < threshold
                point = Optim.minimizer(result)
                
                # Проверяем, не нашли ли мы эту точку ранее
                is_new_point = true
                for existing_point in critical_points
                    if norm(point - existing_point) < threshold
                        is_new_point = false
                        break
                    end
                end
                
                # Добавляем только новые точки
                if is_new_point && point[1] >= x_range[1] && point[1] <= x_range[2] && 
                   point[2] >= y_range[1] && point[2] <= y_range[2]
                    push!(critical_points, point)
                end
            end
        end
    end
    
    return critical_points
end

"""
    classify_critical_point(f, point)

Классифицирует критическую точку функции двух переменных `f`.

# Аргументы
- `f::Function`: функция двух переменных
- `point::Vector{<:Real}`: критическая точка [x, y]

# Возвращаемое значение
- `Symbol`: тип критической точки (:minimum, :maximum, :saddle, :undefined)

# Примеры
```julia
f(x, y) = x^2 + y^2
point = [0.0, 0.0]
classify_critical_point(f, point)  # вернет :minimum
```
"""
function classify_critical_point(f::Function, point::Vector{<:Real})
    if length(point) != 2
        throw(ArgumentError("Точка должна иметь две координаты"))
    end
    
    # Вычисляем матрицу Гессе
    H = hessian(f, point[1], point[2])
    
    # Находим определитель и след
    det_H = det(H)
    trace_H = tr(H)
    
    # Классифицируем точку
    if det_H > 0 && trace_H > 0
        return :minimum
    elseif det_H > 0 && trace_H < 0
        return :maximum
    elseif det_H < 0
        return :saddle
    else
        # Вырожденный случай, требуется дополнительный анализ
        return :undefined
    end
end

"""
    find_and_classify_critical_points(f, x_range, y_range; kwargs...)

Находит и классифицирует критические точки функции двух переменных `f` в заданной области.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси x (min, max)
- `y_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси y (min, max)
- `kwargs...`: дополнительные аргументы для функции find_critical_points

# Возвращаемое значение
- `Vector{Tuple{Vector{Float64}, Symbol}}`: список пар (точка, тип)

# Примеры
```julia
f(x, y) = x^2 + y^2 - 2x - 4y + 5
points = find_and_classify_critical_points(f, (-5.0, 5.0), (-5.0, 5.0))
```
"""
function find_and_classify_critical_points(f::Function, x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; kwargs...)
    # Находим критические точки
    critical_points = find_critical_points(f, x_range, y_range; kwargs...)
    
    # Классифицируем каждую точку
    classified_points = Tuple{Vector{Float64}, Symbol}[]
    for point in critical_points
        point_type = classify_critical_point(f, point)
        push!(classified_points, (point, point_type))
    end
    
    return classified_points
end

"""
    find_constrained_extrema(f, g, x_range, y_range; constraint_value=0, grid_size=20)

Находит условные экстремумы функции двух переменных `f` при ограничении `g(x,y) = constraint_value`.
Использует метод множителей Лагранжа.

# Аргументы
- `f::Function`: функция двух переменных (целевая функция)
- `g::Function`: функция двух переменных (ограничение)
- `x_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси x (min, max)
- `y_range::Tuple{<:Real, <:Real}`: диапазон поиска по оси y (min, max)
- `constraint_value::Real=0`: значение ограничения g(x,y) = constraint_value
- `grid_size::Int=20`: размер сетки для начального поиска

# Возвращаемое значение
- `Vector{Vector{Float64}}`: список найденных точек условных экстремумов [x, y]

# Примеры
```julia
f(x, y) = x^2 + y^2  # Целевая функция
g(x, y) = x + y - 1   # Ограничение g(x,y) = 0
extrema = find_constrained_extrema(f, g, (-5.0, 5.0), (-5.0, 5.0))
```
"""
function find_constrained_extrema(f::Function, g::Function, x_range::Tuple{<:Real, <:Real}, y_range::Tuple{<:Real, <:Real}; 
                                 constraint_value::Real=0, grid_size::Int=20)
    # Создаем сетку начальных точек для поиска
    x_grid = range(x_range[1], x_range[2], length=grid_size)
    y_grid = range(y_range[1], y_range[2], length=grid_size)
    
    # Функция Лагранжа: L(x, y, λ) = f(x, y) - λ*(g(x, y) - constraint_value)
    function lagrangian_grad(point)
        x, y, λ = point
        
        # Вычисляем градиенты
        grad_f = gradient(f, x, y)
        grad_g = gradient(g, x, y)
        
        # Градиент функции Лагранжа
        grad_L = [
            grad_f[1] - λ * grad_g[1],            # ∂L/∂x
            grad_f[2] - λ * grad_g[2],            # ∂L/∂y
            -(g(x, y) - constraint_value)         # ∂L/∂λ
        ]
        
        return norm(grad_L)
    end
    
    extrema_points = Vector{Float64}[]
    
    # Проходим по всем точкам сетки как начальным приближениям
    for x0 in x_grid
        for y0 in y_grid
            # Начальное приближение для λ
            λ0 = 1.0
            
            # Используем оптимизатор для поиска стационарных точек функции Лагранжа
            result = optimize(lagrangian_grad, [x0, y0, λ0], BFGS())
            
            # Если найдена стационарная точка
            if Optim.minimum(result) < 1e-6
                point = Optim.minimizer(result)
                x, y = point[1], point[2]
                
                # Проверяем, удовлетворяет ли точка ограничению
                if abs(g(x, y) - constraint_value) < 1e-6
                    # Проверяем, находится ли точка в заданном диапазоне
                    if x >= x_range[1] && x <= x_range[2] && y >= y_range[1] && y <= y_range[2]
                        # Проверяем, не найдена ли уже эта точка
                        is_new_point = true
                        for existing_point in extrema_points
                            if norm([x, y] - existing_point) < 1e-6
                                is_new_point = false
                                break
                            end
                        end
                        
                        if is_new_point
                            push!(extrema_points, [x, y])
                        end
                    end
                end
            end
        end
    end
    
    return extrema_points
end 