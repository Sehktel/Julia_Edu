"""
# Линейная и квадратичная интерполяция

Данный модуль реализует методы линейной и квадратичной интерполяции для
приближения функций. Эти методы являются простейшими интерполяционными
техниками, которые используют полиномы низких степеней.
"""

using LinearAlgebra

"""
    linear_interpolation(x_points, y_points, x)

Выполняет линейную интерполяцию для нахождения значения в точке `x`.

Линейная интерполяция строит прямую линию между соседними точками данных и
использует её для оценки значения функции в промежуточной точке.

Для точки x на отрезке [x₁, x₂] интерполированное значение вычисляется как:
f(x) ≈ f(x₁) + (f(x₂) - f(x₁)) * (x - x₁) / (x₂ - x₁)

# Аргументы
- `x_points::Vector{<:Real}`: точки по оси X (узлы интерполяции)
- `y_points::Vector{<:Real}`: значения функции в узлах (f(x_points))
- `x::Real`: точка, в которой требуется найти приближенное значение

# Возвращаемое значение
- `Real`: интерполированное значение в точке `x`

# Исключения
- `ArgumentError`: если длины векторов `x_points` и `y_points` не совпадают
- `DomainError`: если точка `x` находится за пределами диапазона `x_points`

# Примеры
```julia
x_points = [0.0, 1.0, 2.0, 3.0]
y_points = [0.0, 1.0, 4.0, 9.0]  # значения f(x) = x²
linear_interpolation(x_points, y_points, 1.5)  # приближенное значение f(1.5)
```
"""
function linear_interpolation(x_points::Vector{<:Real}, y_points::Vector{<:Real}, x::Real)
    # Проверка размерностей входных данных
    if length(x_points) != length(y_points)
        throw(ArgumentError("Векторы x_points и y_points должны иметь одинаковую длину"))
    end
    
    # Проверка, что точка находится в допустимом диапазоне
    if x < minimum(x_points) || x > maximum(x_points)
        throw(DomainError(x, "Точка должна находиться в диапазоне [$(minimum(x_points)), $(maximum(x_points))]"))
    end
    
    # Находим индекс ближайшей слева точки
    i = findlast(p -> p <= x, x_points)
    
    # Если точка совпадает с узлом, возвращаем точное значение
    if x_points[i] == x
        return y_points[i]
    end
    
    # Если это последняя точка, используем предыдущий интервал
    if i == length(x_points)
        i = i - 1
    end
    
    # Линейная интерполяция между точками i и i+1
    x1, x2 = x_points[i], x_points[i+1]
    y1, y2 = y_points[i], y_points[i+1]
    
    # Интерполяционная формула: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
end

"""
    quadratic_interpolation(x_points, y_points, x)

Выполняет квадратичную интерполяцию для нахождения значения в точке `x`.

Квадратичная интерполяция строит параболу (полином второй степени), проходящую
через три ближайшие к `x` точки данных, и использует её для оценки значения
функции в заданной точке.

# Аргументы
- `x_points::Vector{<:Real}`: точки по оси X (узлы интерполяции)
- `y_points::Vector{<:Real}`: значения функции в узлах (f(x_points))
- `x::Real`: точка, в которой требуется найти приближенное значение

# Возвращаемое значение
- `Real`: интерполированное значение в точке `x`

# Исключения
- `ArgumentError`: если длины векторов `x_points` и `y_points` не совпадают
- `ArgumentError`: если количество точек меньше 3
- `DomainError`: если точка `x` находится за пределами диапазона `x_points`

# Примеры
```julia
x_points = [0.0, 1.0, 2.0, 3.0, 4.0]
y_points = [0.0, 1.0, 4.0, 9.0, 16.0]  # значения f(x) = x²
quadratic_interpolation(x_points, y_points, 1.5)  # приближенное значение f(1.5)
```
"""
function quadratic_interpolation(x_points::Vector{<:Real}, y_points::Vector{<:Real}, x::Real)
    # Проверка размерностей входных данных
    if length(x_points) != length(y_points)
        throw(ArgumentError("Векторы x_points и y_points должны иметь одинаковую длину"))
    end
    
    if length(x_points) < 3
        throw(ArgumentError("Для квадратичной интерполяции требуется минимум 3 точки"))
    end
    
    # Проверка, что точка находится в допустимом диапазоне
    if x < minimum(x_points) || x > maximum(x_points)
        throw(DomainError(x, "Точка должна находиться в диапазоне [$(minimum(x_points)), $(maximum(x_points))]"))
    end
    
    # Находим три ближайшие точки к x
    # Сначала находим ближайший индекс
    distances = abs.(x_points .- x)
    sorted_indices = sortperm(distances)
    
    # Используем три ближайшие точки, но сохраняем их в порядке возрастания x
    indices = sorted_indices[1:min(3, length(sorted_indices))]
    indices = sort(indices)
    
    # Если у нас меньше трех точек, используем все доступные
    if length(indices) < 3 && length(x_points) >= 3
        # Добавляем дополнительные точки для получения трех
        while length(indices) < 3
            if minimum(indices) > 1
                pushfirst!(indices, minimum(indices) - 1)
            elseif maximum(indices) < length(x_points)
                push!(indices, maximum(indices) + 1)
            else
                break # Не можем добавить больше точек
            end
        end
        indices = sort(indices)
    end
    
    # Извлекаем выбранные точки
    x_interp = x_points[indices]
    y_interp = y_points[indices]
    
    # Решаем систему уравнений для нахождения коэффициентов квадратичного полинома
    # p(x) = a*x^2 + b*x + c
    n = length(x_interp)
    A = [x_interp[i]^2 x_interp[i] 1 for i in 1:n]
    coeffs = A \ y_interp
    
    # Вычисляем значение полинома в точке x
    a, b, c = coeffs
    return a * x^2 + b * x + c
end

"""
    piecewise_quadratic_interpolation(x_points, y_points, x)

Выполняет кусочно-квадратичную интерполяцию для нахождения значения в точке `x`.

Этот метод строит отдельный квадратичный полином для каждого интервала между
точками данных, обеспечивая непрерывность функции и её первой производной
на всём интервале интерполяции.

# Аргументы
- `x_points::Vector{<:Real}`: точки по оси X (узлы интерполяции)
- `y_points::Vector{<:Real}`: значения функции в узлах (f(x_points))
- `x::Real`: точка, в которой требуется найти приближенное значение

# Возвращаемое значение
- `Real`: интерполированное значение в точке `x`

# Примеры
```julia
x_points = [0.0, 1.0, 2.0, 3.0, 4.0]
y_points = [0.0, 1.0, 4.0, 9.0, 16.0]  # значения f(x) = x²
piecewise_quadratic_interpolation(x_points, y_points, 1.5)
```
"""
function piecewise_quadratic_interpolation(x_points::Vector{<:Real}, y_points::Vector{<:Real}, x::Real)
    # Проверка размерностей входных данных
    if length(x_points) != length(y_points)
        throw(ArgumentError("Векторы x_points и y_points должны иметь одинаковую длину"))
    end
    
    if length(x_points) < 3
        throw(ArgumentError("Для кусочно-квадратичной интерполяции требуется минимум 3 точки"))
    end
    
    # Проверка, что точка находится в допустимом диапазоне
    if x < minimum(x_points) || x > maximum(x_points)
        throw(DomainError(x, "Точка должна находиться в диапазоне [$(minimum(x_points)), $(maximum(x_points))]"))
    end
    
    # Находим индекс ближайшей слева точки
    i = findlast(p -> p <= x, x_points)
    
    # Если точка совпадает с узлом, возвращаем точное значение
    if x_points[i] == x
        return y_points[i]
    end
    
    # Если это первая или последняя точка, используем соседний интервал
    if i == length(x_points)
        i = i - 1
    elseif i == 1
        i = 1
    end
    
    # Определяем три точки для интерполяции
    if i == 1
        indices = [1, 2, 3]
    elseif i == length(x_points) - 1
        indices = [i-1, i, i+1]
    else
        indices = [i-1, i, i+1]
    end
    
    # Извлекаем выбранные точки
    x_interp = x_points[indices]
    y_interp = y_points[indices]
    
    # Используем стандартную квадратичную интерполяцию для этих трех точек
    return quadratic_interpolation(x_interp, y_interp, x)
end 