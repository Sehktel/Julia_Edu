"""
    PolynomialLeastSquares

Модуль содержит функции для аппроксимации данных полиномиальными моделями
методом наименьших квадратов. Реализует стандартный и взвешенный МНК для полиномов
различной степени.
"""

using LinearAlgebra

"""
    polynomial_least_squares(x, y, degree)

Находит коэффициенты полинома заданной степени, который наилучшим образом 
(в смысле метода наименьших квадратов) аппроксимирует заданные данные.

# Аргументы
- `x::AbstractVector{<:Real}`: вектор независимой переменной
- `y::AbstractVector{<:Real}`: вектор зависимой переменной
- `degree::Int`: степень полинома (>= 0)

# Возвращает
- `::Vector{Float64}`: вектор коэффициентов полинома [a₀, a₁, a₂, ..., aₙ], 
  где n = degree, и полином имеет вид a₀ + a₁x + a₂x² + ... + aₙxⁿ
- `::Matrix{Float64}`: ковариационная матрица оценок коэффициентов

# Пример
```julia
# Генерируем тестовые данные
x = collect(-5.0:0.5:5.0)
y = 1.0 .+ 2.0 .* x .+ 3.0 .* x.^2 .+ 0.5 .* randn(length(x))  # Квадратичная функция с шумом
coeffs, cov_coeffs = polynomial_least_squares(x, y, 2)
println("Коэффициенты полинома: ", coeffs)
```
"""
function polynomial_least_squares(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, degree::Int)
    # Проверка размерностей
    n = length(x)
    if length(y) != n
        throw(DimensionMismatch("Длина x ($(n)) не соответствует длине y ($(length(y)))"))
    end
    
    # Проверка степени полинома
    if degree < 0
        throw(ArgumentError("Степень полинома должна быть неотрицательной"))
    end
    
    # Если степень полинома слишком высока по сравнению с количеством точек
    if degree > n - 1
        @warn "Степень полинома ($degree) слишком велика для числа точек ($n). Результат может быть ненадежным."
    end
    
    # Создаем матрицу регрессоров (матрицу Вандермонда)
    X = vandermonde_matrix(x, degree)
    
    # Используем линейный МНК для нахождения коэффициентов
    return linear_least_squares(X, y)
end

"""
    weighted_polynomial_least_squares(x, y, w, degree)

Находит коэффициенты полинома заданной степени, который наилучшим образом 
аппроксимирует заданные данные с учетом весов наблюдений.

# Аргументы
- `x::AbstractVector{<:Real}`: вектор независимой переменной
- `y::AbstractVector{<:Real}`: вектор зависимой переменной
- `w::AbstractVector{<:Real}`: вектор весов для наблюдений
- `degree::Int`: степень полинома (>= 0)

# Возвращает
- `::Vector{Float64}`: вектор коэффициентов полинома [a₀, a₁, a₂, ..., aₙ], 
  где n = degree, и полином имеет вид a₀ + a₁x + a₂x² + ... + aₙxⁿ
- `::Matrix{Float64}`: ковариационная матрица оценок коэффициентов

# Пример
```julia
# Генерируем тестовые данные
x = collect(-5.0:0.5:5.0)
y = 1.0 .+ 2.0 .* x .+ 3.0 .* x.^2 .+ 0.5 .* randn(length(x))  # Квадратичная функция с шумом
w = 1.0 ./ (1.0 .+ abs.(x))  # Веса обратно пропорциональны расстоянию от нуля
coeffs, cov_coeffs = weighted_polynomial_least_squares(x, y, w, 2)
println("Коэффициенты полинома с учетом весов: ", coeffs)
```
"""
function weighted_polynomial_least_squares(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, 
                                         w::AbstractVector{<:Real}, degree::Int)
    # Проверка размерностей
    n = length(x)
    if length(y) != n
        throw(DimensionMismatch("Длина x ($(n)) не соответствует длине y ($(length(y)))"))
    end
    if length(w) != n
        throw(DimensionMismatch("Длина x ($(n)) не соответствует длине w ($(length(w)))"))
    end
    
    # Проверка степени полинома
    if degree < 0
        throw(ArgumentError("Степень полинома должна быть неотрицательной"))
    end
    
    # Создаем матрицу регрессоров (матрицу Вандермонда)
    X = vandermonde_matrix(x, degree)
    
    # Используем взвешенный линейный МНК для нахождения коэффициентов
    return weighted_linear_least_squares(X, y, w)
end

"""
    vandermonde_matrix(x, degree)

Создает матрицу Вандермонда для вектора значений x и заданной степени полинома.

# Аргументы
- `x::AbstractVector{<:Real}`: вектор значений независимой переменной
- `degree::Int`: степень полинома (>= 0)

# Возвращает
- `::Matrix{Float64}`: матрица Вандермонда размером n×(degree+1), где n = длина x

# Пример
```julia
x = [1.0, 2.0, 3.0, 4.0]
V = vandermonde_matrix(x, 2)  # Матрица для полинома степени 2 (квадратичного)
# V будет 4×3 матрицей вида:
# [ 1.0  1.0  1.0 ]
# [ 1.0  2.0  4.0 ]
# [ 1.0  3.0  9.0 ]
# [ 1.0  4.0 16.0 ]
```
"""
function vandermonde_matrix(x::AbstractVector{<:Real}, degree::Int)
    n = length(x)
    V = zeros(n, degree + 1)
    
    for i in 1:n
        for j in 0:degree
            V[i, j+1] = x[i]^j
        end
    end
    
    return V
end

"""
    polynomial_value(coeffs, x)

Вычисляет значение полинома с заданными коэффициентами для заданного значения x.

# Аргументы
- `coeffs::AbstractVector{<:Real}`: вектор коэффициентов полинома [a₀, a₁, a₂, ..., aₙ]
- `x::Real`: значение независимой переменной

# Возвращает
- `::Float64`: значение полинома a₀ + a₁x + a₂x² + ... + aₙxⁿ в точке x

# Пример
```julia
coeffs = [1.0, 2.0, 3.0]  # Полином 1 + 2x + 3x²
x = 2.0
y = polynomial_value(coeffs, x)  # Вернет 1 + 2*2 + 3*2² = 1 + 4 + 12 = 17
```
"""
function polynomial_value(coeffs::AbstractVector{<:Real}, x::Real)
    result = 0.0
    degree = length(coeffs) - 1
    
    for i in 0:degree
        result += coeffs[i+1] * x^i
    end
    
    return result
end

"""
    polynomial_values(coeffs, x)

Вычисляет значения полинома с заданными коэффициентами для вектора значений x.

# Аргументы
- `coeffs::AbstractVector{<:Real}`: вектор коэффициентов полинома [a₀, a₁, a₂, ..., aₙ]
- `x::AbstractVector{<:Real}`: вектор значений независимой переменной

# Возвращает
- `::Vector{Float64}`: вектор значений полинома для каждого элемента x

# Пример
```julia
coeffs = [1.0, 2.0, 3.0]  # Полином 1 + 2x + 3x²
x = [0.0, 1.0, 2.0]
y = polynomial_values(coeffs, x)  # Вернет [1.0, 6.0, 17.0]
```
"""
function polynomial_values(coeffs::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return [polynomial_value(coeffs, xi) for xi in x]
end

"""
    polynomial_model(x, degree)

Создает функцию полиномиальной модели заданной степени для использования 
в нелинейном МНК или других контекстах.

# Аргументы
- `degree::Int`: степень полинома (>= 0)

# Возвращает
- `::Function`: функция f(x, p), где x - значение независимой переменной,
  p - вектор параметров полинома [a₀, a₁, a₂, ..., aₙ]

# Пример
```julia
model = polynomial_model(2)  # Создает модель полинома второй степени
x = 2.0
p = [1.0, 2.0, 3.0]  # Параметры полинома: 1 + 2x + 3x²
y = model(x, p)  # Вычисляет значение полинома при x = 2: 1 + 2*2 + 3*2² = 17
```
"""
function polynomial_model(degree::Int)
    # Возвращаем функцию, которая вычисляет полином заданной степени
    return (x, p) -> sum(p[i] * x^(i-1) for i in 1:(degree+1))
end

"""
    polynomial_fit(x, y, degree; weights=nothing)

Удобная функция для подбора полиномиальной модели к данным.
Возвращает структуру с коэффициентами и функцию для предсказания.

# Аргументы
- `x::AbstractVector{<:Real}`: вектор независимой переменной
- `y::AbstractVector{<:Real}`: вектор зависимой переменной
- `degree::Int`: степень полинома (>= 0)
- `weights::Union{AbstractVector{<:Real}, Nothing}=nothing`: опциональный вектор весов

# Возвращает
- `::NamedTuple`: структура с полями:
  - `coefficients::Vector{Float64}`: коэффициенты полинома
  - `cov_matrix::Matrix{Float64}`: ковариационная матрица коэффициентов
  - `predict::Function`: функция для предсказания значений на новых данных
  - `degree::Int`: степень полинома

# Пример
```julia
# Генерируем тестовые данные
x = collect(-5.0:0.5:5.0)
y = 1.0 .+ 2.0 .* x .+ 3.0 .* x.^2 .+ 0.5 .* randn(length(x))  # Квадратичная функция с шумом

# Подбираем полином
model = polynomial_fit(x, y, 2)
println("Коэффициенты полинома: ", model.coefficients)

# Предсказываем значения для новых данных
x_new = collect(-4.0:1.0:4.0)
y_pred = model.predict(x_new)
```
"""
function polynomial_fit(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, degree::Int;
                      weights::Union{AbstractVector{<:Real}, Nothing}=nothing)
    # Подбираем коэффициенты полинома
    if weights === nothing
        coeffs, cov_coeffs = polynomial_least_squares(x, y, degree)
    else
        coeffs, cov_coeffs = weighted_polynomial_least_squares(x, y, weights, degree)
    end
    
    # Создаем функцию для предсказания
    predict_func = (x_new) -> polynomial_values(coeffs, x_new)
    
    return (
        coefficients = coeffs,
        cov_matrix = cov_coeffs,
        predict = predict_func,
        degree = degree
    )
end 