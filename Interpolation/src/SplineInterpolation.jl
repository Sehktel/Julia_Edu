"""
# Сплайн-интерполяция

Данный модуль реализует методы сплайн-интерполяции, которые обеспечивают гладкое
приближение функций с помощью кусочно-полиномиальных функций (сплайнов).

Сплайн-интерполяция обеспечивает непрерывность функции и её производных,
что делает её особенно полезной для моделирования гладких кривых.
"""

using LinearAlgebra

"""
    Spline

Структура данных, представляющая кубический сплайн.

# Поля
- `x::Vector{Float64}`: узлы интерполяции
- `a::Vector{Float64}`: коэффициенты при (x - x_i)^0
- `b::Vector{Float64}`: коэффициенты при (x - x_i)^1
- `c::Vector{Float64}`: коэффициенты при (x - x_i)^2
- `d::Vector{Float64}`: коэффициенты при (x - x_i)^3

Сплайн на интервале [x_i, x_{i+1}] определяется как:
S(x) = a[i] + b[i]*(x - x[i]) + c[i]*(x - x[i])^2 + d[i]*(x - x[i])^3
"""
struct Spline
    x::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
end

"""
    natural_spline(x, y)

Строит естественный кубический сплайн, проходящий через точки с координатами (x, y).

Естественный сплайн имеет нулевые вторые производные на концах интервала интерполяции.

# Аргументы
- `x::Vector{<:Real}`: узлы интерполяции
- `y::Vector{<:Real}`: значения функции в узлах

# Возвращаемое значение
- `Spline`: структура, представляющая построенный сплайн

# Исключения
- `ArgumentError`: если длины векторов `x` и `y` не совпадают
- `ArgumentError`: если в `x` менее двух точек

# Примеры
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.0, 1.0, 4.0, 9.0, 16.0]  # значения f(x) = x²
spline = natural_spline(x, y)
```
"""
function natural_spline(x::Vector{<:Real}, y::Vector{<:Real})
    # Проверка размерностей входных данных
    if length(x) != length(y)
        throw(ArgumentError("Векторы x и y должны иметь одинаковую длину"))
    end
    
    if length(x) < 2
        throw(ArgumentError("Для построения сплайна требуется минимум 2 точки"))
    end
    
    # Конвертируем в Float64 для численной стабильности
    x_float = convert(Vector{Float64}, x)
    y_float = convert(Vector{Float64}, y)
    
    n = length(x_float)
    
    # Для случая с двумя точками возвращаем линейный сплайн
    if n == 2
        a = [y_float[1], y_float[2]]
        b = [(y_float[2] - y_float[1]) / (x_float[2] - x_float[1]), 0.0]
        c = [0.0, 0.0]
        d = [0.0, 0.0]
        return Spline(x_float, a, b, c, d)
    end
    
    # Вычисляем разности узлов и значений функции
    h = diff(x_float)
    delta = diff(y_float) ./ h
    
    # Строим трехдиагональную систему для коэффициентов сплайна
    A = zeros(n, n)
    b = zeros(n)
    
    # Заполняем матрицу системы
    for i in 2:n-1
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (delta[i] - delta[i-1])
    end
    
    # Граничные условия для естественного сплайна
    A[1, 1] = 1.0
    A[n, n] = 1.0
    b[1] = 0.0
    b[n] = 0.0
    
    # Решаем систему для нахождения вторых производных
    c_prime = A \ b
    
    # Вычисляем коэффициенты сплайна
    a = y_float[1:n-1]
    b = zeros(n-1)
    c = c_prime[1:n-1]
    d = zeros(n-1)
    
    for i in 1:n-1
        b[i] = delta[i] - h[i] * (2 * c[i] + c_prime[i+1]) / 3
        d[i] = (c_prime[i+1] - c[i]) / (3 * h[i])
    end
    
    return Spline(x_float, a, b, c, d)
end

"""
    clamped_spline(x, y, dy_start, dy_end)

Строит зажатый кубический сплайн с заданными значениями первой производной на концах.

# Аргументы
- `x::Vector{<:Real}`: узлы интерполяции
- `y::Vector{<:Real}`: значения функции в узлах
- `dy_start::Real`: значение первой производной в начальной точке
- `dy_end::Real`: значение первой производной в конечной точке

# Возвращаемое значение
- `Spline`: структура, представляющая построенный сплайн

# Примеры
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.0, 1.0, 4.0, 9.0, 16.0]  # значения f(x) = x²
spline = clamped_spline(x, y, 0.0, 8.0)  # f'(0) = 0, f'(4) = 8
```
"""
function clamped_spline(x::Vector{<:Real}, y::Vector{<:Real}, dy_start::Real, dy_end::Real)
    # Проверка размерностей входных данных
    if length(x) != length(y)
        throw(ArgumentError("Векторы x и y должны иметь одинаковую длину"))
    end
    
    if length(x) < 2
        throw(ArgumentError("Для построения сплайна требуется минимум 2 точки"))
    end
    
    # Конвертируем в Float64 для численной стабильности
    x_float = convert(Vector{Float64}, x)
    y_float = convert(Vector{Float64}, y)
    
    n = length(x_float)
    
    # Для случая с двумя точками возвращаем линейный сплайн
    if n == 2
        a = [y_float[1]]
        b = [dy_start]
        c = [0.0]
        d = [0.0]
        return Spline(x_float, a, b, c, d)
    end
    
    # Вычисляем разности узлов и значений функции
    h = diff(x_float)
    delta = diff(y_float) ./ h
    
    # Строим трехдиагональную систему для коэффициентов сплайна
    A = zeros(n, n)
    b = zeros(n)
    
    # Заполняем матрицу системы
    for i in 2:n-1
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (delta[i] - delta[i-1])
    end
    
    # Граничные условия для зажатого сплайна
    A[1, 1] = 2 * h[1]
    A[1, 2] = h[1]
    b[1] = 3 * (delta[1] - dy_start)
    
    A[n, n-1] = h[n-1]
    A[n, n] = 2 * h[n-1]
    b[n] = 3 * (dy_end - delta[n-1])
    
    # Решаем систему для нахождения вторых производных
    c_prime = A \ b
    
    # Вычисляем коэффициенты сплайна
    a = y_float[1:n-1]
    b = zeros(n-1)
    c = c_prime[1:n-1]
    d = zeros(n-1)
    
    for i in 1:n-1
        b[i] = delta[i] - h[i] * (2 * c[i] + c_prime[i+1]) / 3
        d[i] = (c_prime[i+1] - c[i]) / (3 * h[i])
    end
    
    return Spline(x_float, a, b, c, d)
end

"""
    cubic_spline(x, y)

Строит кубический сплайн, проходящий через точки с координатами (x, y).
По умолчанию создается естественный сплайн.

# Аргументы
- `x::Vector{<:Real}`: узлы интерполяции
- `y::Vector{<:Real}`: значения функции в узлах

# Возвращаемое значение
- `Spline`: структура, представляющая построенный сплайн

# Примеры
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.0, 1.0, 4.0, 9.0, 16.0]  # значения f(x) = x²
spline = cubic_spline(x, y)
```
"""
function cubic_spline(x::Vector{<:Real}, y::Vector{<:Real})
    return natural_spline(x, y)
end

"""
    evaluate_spline(spline, x)

Вычисляет значение сплайна в точке x.

# Аргументы
- `spline::Spline`: сплайн для вычисления
- `x::Real`: точка, в которой требуется найти значение

# Возвращаемое значение
- `Float64`: значение сплайна в точке x

# Исключения
- `DomainError`: если точка `x` находится за пределами диапазона сплайна

# Примеры
```julia
spline = cubic_spline([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
evaluate_spline(spline, 1.5)  # вычисление в точке x = 1.5
```
"""
function evaluate_spline(spline::Spline, x::Real)
    # Проверка, что точка находится в допустимом диапазоне
    if x < minimum(spline.x) || x > maximum(spline.x)
        throw(DomainError(x, "Точка должна находиться в диапазоне [$(minimum(spline.x)), $(maximum(spline.x))]"))
    end
    
    # Если точка совпадает с последним узлом, возвращаем значение в этом узле
    if x == spline.x[end]
        return spline.a[end] + spline.b[end] * 0 + spline.c[end] * 0^2 + spline.d[end] * 0^3
    end
    
    # Находим индекс интервала, содержащего точку x
    i = findlast(p -> p <= x, spline.x)
    
    # Вычисляем расстояние от x до левого узла интервала
    dx = x - spline.x[i]
    
    # Вычисляем значение сплайна по формуле
    # S(x) = a[i] + b[i]*(x - x[i]) + c[i]*(x - x[i])^2 + d[i]*(x - x[i])^3
    return spline.a[i] + spline.b[i] * dx + spline.c[i] * dx^2 + spline.d[i] * dx^3
end

"""
    evaluate_spline_derivative(spline, x, derivative_order=1)

Вычисляет значение производной сплайна в точке x.

# Аргументы
- `spline::Spline`: сплайн для вычисления
- `x::Real`: точка, в которой требуется найти производную
- `derivative_order::Int=1`: порядок производной (1, 2 или 3)

# Возвращаемое значение
- `Float64`: значение производной сплайна в точке x

# Исключения
- `DomainError`: если точка `x` находится за пределами диапазона сплайна
- `ArgumentError`: если порядок производной не в диапазоне [1, 3]

# Примеры
```julia
spline = cubic_spline([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
evaluate_spline_derivative(spline, 1.5)  # первая производная в точке x = 1.5
evaluate_spline_derivative(spline, 1.5, 2)  # вторая производная в точке x = 1.5
```
"""
function evaluate_spline_derivative(spline::Spline, x::Real, derivative_order::Int=1)
    # Проверка порядка производной
    if !(1 <= derivative_order <= 3)
        throw(ArgumentError("Порядок производной должен быть 1, 2 или 3"))
    end
    
    # Проверка, что точка находится в допустимом диапазоне
    if x < minimum(spline.x) || x > maximum(spline.x)
        throw(DomainError(x, "Точка должна находиться в диапазоне [$(minimum(spline.x)), $(maximum(spline.x))]"))
    end
    
    # Находим индекс интервала, содержащего точку x
    i = findlast(p -> p <= x, spline.x)
    
    # Если точка совпадает с последним узлом, используем предыдущий интервал
    if i == length(spline.x)
        i = i - 1
    end
    
    # Вычисляем расстояние от x до левого узла интервала
    dx = x - spline.x[i]
    
    # Вычисляем значение производной в зависимости от её порядка
    if derivative_order == 1
        # S'(x) = b[i] + 2*c[i]*(x - x[i]) + 3*d[i]*(x - x[i])^2
        return spline.b[i] + 2 * spline.c[i] * dx + 3 * spline.d[i] * dx^2
    elseif derivative_order == 2
        # S''(x) = 2*c[i] + 6*d[i]*(x - x[i])
        return 2 * spline.c[i] + 6 * spline.d[i] * dx
    else # derivative_order == 3
        # S'''(x) = 6*d[i]
        return 6 * spline.d[i]
    end
end 