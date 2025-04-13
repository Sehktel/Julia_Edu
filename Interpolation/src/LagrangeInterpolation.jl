"""
# Интерполяционный полином Лагранжа

Данный модуль реализует интерполяционный полином Лагранжа, который позволяет
построить полином степени n, проходящий через n+1 заданную точку.

Интерполяционный полином Лагранжа представляет собой линейную комбинацию
базисных полиномов Лагранжа, каждый из которых равен 1 в одной из интерполяционных
точек и 0 во всех остальных точках.
"""

"""
    lagrange_basis(x_points, i, x)

Вычисляет значение i-го базисного полинома Лагранжа в точке x.

Базисный полином Лагранжа l_i(x) определяется как:
l_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)

# Аргументы
- `x_points::Vector{<:Real}`: узлы интерполяции
- `i::Int`: индекс базисного полинома (1 ≤ i ≤ length(x_points))
- `x::Real`: точка, в которой вычисляется значение полинома

# Возвращаемое значение
- `Float64`: значение i-го базисного полинома в точке x

# Примеры
```julia
x_points = [0.0, 1.0, 2.0]
lagrange_basis(x_points, 2, 1.5)  # значение 2-го базисного полинома в точке x = 1.5
```
"""
function lagrange_basis(x_points::Vector{<:Real}, i::Int, x::Real)
    if i < 1 || i > length(x_points)
        throw(ArgumentError("Индекс i должен быть в диапазоне 1:$(length(x_points))"))
    end
    
    result = 1.0
    x_i = x_points[i]
    
    for j in 1:length(x_points)
        if j != i
            result *= (x - x_points[j]) / (x_i - x_points[j])
        end
    end
    
    return result
end

"""
    lagrange_polynomial(x_points, y_points)

Строит интерполяционный полином Лагранжа, проходящий через точки с координатами (x_points, y_points).

# Аргументы
- `x_points::Vector{<:Real}`: узлы интерполяции
- `y_points::Vector{<:Real}`: значения функции в узлах

# Возвращаемое значение
- `Function`: функция, представляющая полином Лагранжа, которую можно вызывать для любой точки

# Исключения
- `ArgumentError`: если длины векторов `x_points` и `y_points` не совпадают

# Примеры
```julia
x_points = [0.0, 1.0, 2.0]
y_points = [1.0, 3.0, 7.0]
p = lagrange_polynomial(x_points, y_points)
p(1.5)  # значение полинома в точке x = 1.5
```
"""
function lagrange_polynomial(x_points::Vector{<:Real}, y_points::Vector{<:Real})
    # Проверка размерностей входных данных
    if length(x_points) != length(y_points)
        throw(ArgumentError("Векторы x_points и y_points должны иметь одинаковую длину"))
    end
    
    # Возвращаем функцию, которая вычисляет значение полинома Лагранжа в точке x
    return function(x::Real)
        sum = 0.0
        for i in 1:length(x_points)
            sum += y_points[i] * lagrange_basis(x_points, i, x)
        end
        return sum
    end
end

"""
    lagrange_interpolation(x_points, y_points, x)

Вычисляет значение интерполяционного полинома Лагранжа в точке x.

# Аргументы
- `x_points::Vector{<:Real}`: узлы интерполяции
- `y_points::Vector{<:Real}`: значения функции в узлах
- `x::Real`: точка, в которой требуется найти значение

# Возвращаемое значение
- `Float64`: значение интерполяционного полинома в точке x

# Примеры
```julia
x_points = [0.0, 1.0, 2.0]
y_points = [1.0, 3.0, 7.0]
lagrange_interpolation(x_points, y_points, 1.5)  # значение полинома в точке x = 1.5
```
"""
function lagrange_interpolation(x_points::Vector{<:Real}, y_points::Vector{<:Real}, x::Real)
    p = lagrange_polynomial(x_points, y_points)
    return p(x)
end

"""
    barycentric_weights(x_points)

Вычисляет барицентрические веса для узлов интерполяции.

Барицентрические веса позволяют ускорить вычисление интерполяционного полинома Лагранжа.

# Аргументы
- `x_points::Vector{<:Real}`: узлы интерполяции

# Возвращаемое значение
- `Vector{Float64}`: барицентрические веса

# Примеры
```julia
x_points = [0.0, 1.0, 2.0, 3.0]
weights = barycentric_weights(x_points)
```
"""
function barycentric_weights(x_points::Vector{<:Real})
    n = length(x_points)
    weights = ones(n)
    
    for i in 1:n
        for j in 1:n
            if j != i
                weights[i] *= (x_points[i] - x_points[j])
            end
        end
        weights[i] = 1.0 / weights[i]
    end
    
    return weights
end

"""
    barycentric_lagrange_interpolation(x_points, y_points, weights, x)

Вычисляет значение интерполяционного полинома Лагранжа в точке x с использованием барицентрической формулы.

Барицентрическая формула позволяет значительно ускорить вычисления, особенно при многократном
вычислении полинома с одними и теми же узлами интерполяции.

# Аргументы
- `x_points::Vector{<:Real}`: узлы интерполяции
- `y_points::Vector{<:Real}`: значения функции в узлах
- `weights::Vector{<:Real}`: барицентрические веса
- `x::Real`: точка, в которой вычисляется значение полинома

# Возвращаемое значение
- `Float64`: значение интерполяционного полинома в точке x

# Примеры
```julia
x_points = [0.0, 1.0, 2.0]
y_points = [1.0, 3.0, 7.0]
weights = barycentric_weights(x_points)
barycentric_lagrange_interpolation(x_points, y_points, weights, 1.5)
```
"""
function barycentric_lagrange_interpolation(x_points::Vector{<:Real}, y_points::Vector{<:Real}, weights::Vector{<:Real}, x::Real)
    # Проверка размерностей входных данных
    if length(x_points) != length(y_points) || length(x_points) != length(weights)
        throw(ArgumentError("Векторы x_points, y_points и weights должны иметь одинаковую длину"))
    end
    
    # Проверка, если x совпадает с одним из узлов интерполяции
    for i in 1:length(x_points)
        if x ≈ x_points[i]
            return y_points[i]
        end
    end
    
    # Вычисление полинома по барицентрической формуле
    numerator = 0.0
    denominator = 0.0
    
    for i in 1:length(x_points)
        t = weights[i] / (x - x_points[i])
        numerator += t * y_points[i]
        denominator += t
    end
    
    return numerator / denominator
end

"""
    interpolation_error(f, x_points, x, n_derivative=nothing)

Оценивает погрешность интерполяции функции f в точке x.

При интерполяции функции f(x) полиномом степени n в узлах x₀, x₁, ..., xₙ
погрешность в точке x выражается формулой:
f(x) - p(x) = f^(n+1)(ξ) / (n+1)! * ∏_{i=0}^n (x - x_i)
где ξ - некоторая точка из интервала [min(x₀, x₁, ..., xₙ, x), max(x₀, x₁, ..., xₙ, x)].

Если производная n+1 порядка не задана (n_derivative=nothing), функция возвращает только
произведение ∏_{i=0}^n (x - x_i).

# Аргументы
- `f::Function`: интерполируемая функция
- `x_points::Vector{<:Real}`: узлы интерполяции
- `x::Real`: точка, в которой оценивается погрешность
- `n_derivative::Union{Function, Nothing}=nothing`: (n+1)-я производная функции f

# Возвращаемое значение
- `Float64`: оценка погрешности интерполяции в точке x

# Примеры
```julia
f(x) = sin(x)
f_derivative(x) = -sin(x)  # 5-я производная sin(x) при n=4
x_points = [0.0, 0.2, 0.4, 0.6, 0.8]
interpolation_error(f, x_points, 0.5, f_derivative)
```
"""
function interpolation_error(f::Function, x_points::Vector{<:Real}, x::Real, n_derivative::Union{Function, Nothing}=nothing)
    # Вычисляем произведение (x - x_i) для всех узлов интерполяции
    product = 1.0
    for x_i in x_points
        product *= (x - x_i)
    end
    
    # Если производная не задана, возвращаем только произведение
    if n_derivative === nothing
        return product
    end
    
    # Находим интервал, содержащий все точки
    a = min(minimum(x_points), x)
    b = max(maximum(x_points), x)
    
    # Оцениваем максимальное значение производной на интервале [a, b]
    # Для простоты используем равномерную сетку из 100 точек
    grid = range(a, b, length=100)
    max_derivative = maximum(abs.(n_derivative.(grid)))
    
    # Вычисляем факториал (n+1)!
    n = length(x_points) - 1
    factorial_n_plus_1 = factorial(n + 1)
    
    # Возвращаем оценку погрешности
    return abs(max_derivative * product / factorial_n_plus_1)
end

"""
    factorial(n)

Вычисляет факториал числа n.

# Аргументы
- `n::Int`: целое неотрицательное число

# Возвращаемое значение
- `Int`: факториал числа n (n!)

# Исключения
- `ArgumentError`: если n < 0

# Примеры
```julia
factorial(5)  # 120
```
"""
function factorial(n::Int)
    if n < 0
        throw(ArgumentError("Аргумент должен быть неотрицательным"))
    end
    
    result = 1
    for i in 2:n
        result *= i
    end
    
    return result
end

"""
    equidistant_nodes(a, b, n)

Создает вектор из n+1 равноотстоящих узлов на интервале [a, b].

# Аргументы
- `a::Real`: левая граница интервала
- `b::Real`: правая граница интервала
- `n::Int`: количество интервалов (узлов будет n+1)

# Возвращаемое значение
- `Vector{Float64}`: вектор равноотстоящих узлов

# Примеры
```julia
nodes = equidistant_nodes(0.0, 1.0, 4)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```
"""
function equidistant_nodes(a::Real, b::Real, n::Int)
    if n < 1
        throw(ArgumentError("Количество интервалов должно быть положительным"))
    end
    
    return range(a, b, length=n+1) |> collect
end

"""
    chebyshev_nodes(a, b, n)

Создает вектор из n+1 узлов Чебышева на интервале [a, b].

Узлы Чебышева минимизируют эффект Рунге и обеспечивают лучшую сходимость
интерполяционного полинома для гладких функций.

# Аргументы
- `a::Real`: левая граница интервала
- `b::Real`: правая граница интервала
- `n::Int`: количество интервалов (узлов будет n+1)

# Возвращаемое значение
- `Vector{Float64}`: вектор узлов Чебышева

# Примеры
```julia
nodes = chebyshev_nodes(0.0, 1.0, 4)
```
"""
function chebyshev_nodes(a::Real, b::Real, n::Int)
    if n < 1
        throw(ArgumentError("Количество интервалов должно быть положительным"))
    end
    
    nodes = zeros(n+1)
    for i in 0:n
        # Формула для узлов Чебышева на интервале [-1, 1]
        x = cos((2*i + 1) * π / (2*(n+1)))
        
        # Преобразование к интервалу [a, b]
        nodes[i+1] = 0.5 * ((b - a) * x + (b + a))
    end
    
    return nodes
end 