"""
# Постановка задачи о приближении функций

В вычислительной математике часто возникает задача о приближении функций.
Существуют два основных подхода:
1. Точечная аппроксимация - построение приближения в конечном наборе точек
2. Непрерывная аппроксимация - построение приближения на всем интервале

Данный модуль содержит реализации различных методов аппроксимации функций.
"""

"""
    point_approximation(f, points)

Выполняет точечную аппроксимацию функции `f` в заданном наборе точек `points`.

# Аргументы
- `f::Function`: аппроксимируемая функция
- `points::Vector{<:Real}`: набор точек, в которых выполняется аппроксимация

# Возвращаемое значение
- `Vector{<:Real}`: значения функции `f` в заданных точках

# Примеры
```julia
f(x) = sin(x)
points = [0.0, 0.5, 1.0, 1.5, 2.0]
values = point_approximation(f, points)
```
"""
function point_approximation(f::Function, points::Vector{<:Real})
    # Вычисляем значения функции в заданных точках
    return f.(points)
end

"""
    continuous_approximation(f, a, b, method, params...)

Выполняет непрерывную аппроксимацию функции `f` на интервале [a, b] с помощью выбранного метода.

# Аргументы
- `f::Function`: аппроксимируемая функция
- `a::Real`: левая граница интервала
- `b::Real`: правая граница интервала
- `method::Symbol`: метод аппроксимации (:linear, :quadratic, :spline, :lagrange)
- `params...`: дополнительные параметры для выбранного метода

# Возвращаемое значение
- `Function`: аппроксимирующая функция, которую можно вызывать для любой точки на интервале [a, b]

# Примеры
```julia
f(x) = sin(x)
approx_f = continuous_approximation(f, 0.0, π, :spline, 10)
approx_f(0.5) # Приближенное значение sin(0.5)
```
"""
function continuous_approximation(f::Function, a::Real, b::Real, method::Symbol, params...)
    # Определяем количество точек для аппроксимации
    n = length(params) > 0 ? params[1] : 10
    
    # Создаем равномерную сетку на интервале [a, b]
    points = range(a, b, length=n)
    values = f.(points)
    
    # Выбираем метод аппроксимации
    if method == :linear
        return x -> linear_interpolation(points, values, x)
    elseif method == :quadratic
        return x -> quadratic_interpolation(points, values, x)
    elseif method == :spline
        spline_coeffs = cubic_spline(points, values)
        return x -> evaluate_spline(spline_coeffs, points, x)
    elseif method == :lagrange
        return x -> lagrange_interpolation(points, values, x)
    else
        error("Неизвестный метод аппроксимации: $method")
    end
end

# Вспомогательная функция для оценки ошибки аппроксимации
"""
    approximation_error(f, approx_f, a, b, n=100)

Вычисляет максимальную абсолютную ошибку аппроксимации на интервале [a, b].

# Аргументы
- `f::Function`: исходная функция
- `approx_f::Function`: аппроксимирующая функция
- `a::Real`: левая граница интервала
- `b::Real`: правая граница интервала
- `n::Int=100`: количество точек для проверки

# Возвращаемое значение
- `Real`: максимальная абсолютная ошибка
"""
function approximation_error(f::Function, approx_f::Function, a::Real, b::Real, n::Int=100)
    # Создаем равномерную сетку для проверки ошибки
    check_points = range(a, b, length=n)
    
    # Вычисляем максимальную абсолютную ошибку
    max_error = 0.0
    for x in check_points
        error = abs(f(x) - approx_f(x))
        max_error = max(max_error, error)
    end
    
    return max_error
end 