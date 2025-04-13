"""
    LinearLeastSquares

Модуль содержит функции для решения задачи линейной аппроксимации методом наименьших квадратов.
Реализует стандартный и взвешенный МНК для линейных моделей вида y = X * β, где:
- y - вектор зависимой переменной
- X - матрица независимых переменных (регрессоров)
- β - вектор коэффициентов, который требуется определить
"""

using LinearAlgebra

"""
    linear_least_squares(X, y)

Находит решение задачи линейной регрессии методом наименьших квадратов.
Минимизирует сумму квадратов отклонений между наблюдаемыми значениями y 
и значениями, предсказанными моделью X * β.

# Аргументы
- `X::AbstractMatrix{<:Real}`: матрица независимых переменных (регрессоров), 
   размером n×p, где n - число наблюдений, p - число параметров
- `y::AbstractVector{<:Real}`: вектор зависимой переменной длиной n

# Возвращает
- `::Vector{Float64}`: оптимальный вектор коэффициентов β
- `::Matrix{Float64}`: ковариационная матрица оценок коэффициентов

# Пример
```julia
# Генерируем тестовые данные
X = [ones(10) collect(1:10)]  # Матрица с константой и одним регрессором
y = 2.0 .+ 3.0 .* (1:10) .+ randn(10)  # Зависимая переменная с шумом
β, cov_β = linear_least_squares(X, y)
println("Оценка коэффициентов: ", β)
```
"""
function linear_least_squares(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    # Проверка размерностей
    n, p = size(X)
    if length(y) != n
        throw(DimensionMismatch("Количество строк в X ($(n)) не соответствует длине y ($(length(y)))"))
    end
    
    # Вычисляем X'X
    XtX = X' * X
    
    # Вычисляем β = (X'X)^(-1) * X'y
    β = try
        XtX \ (X' * y)
    catch e
        # Если матрица плохо обусловлена, используем псевдообратную матрицу
        pinv(XtX) * (X' * y)
    end
    
    # Оцениваем дисперсию ошибок
    ŷ = X * β
    residuals = y - ŷ
    σ² = sum(residuals.^2) / (n - p)
    
    # Вычисляем ковариационную матрицу оценок коэффициентов
    cov_β = try
        σ² * inv(XtX)
    catch e
        σ² * pinv(XtX)
    end
    
    return β, cov_β
end

"""
    weighted_linear_least_squares(X, y, w)

Находит решение задачи взвешенной линейной регрессии методом наименьших квадратов.
Минимизирует взвешенную сумму квадратов отклонений между наблюдаемыми значениями y 
и значениями, предсказанными моделью X * β.

# Аргументы
- `X::AbstractMatrix{<:Real}`: матрица независимых переменных (регрессоров), 
   размером n×p, где n - число наблюдений, p - число параметров
- `y::AbstractVector{<:Real}`: вектор зависимой переменной длиной n
- `w::AbstractVector{<:Real}`: вектор весов для наблюдений длиной n

# Возвращает
- `::Vector{Float64}`: оптимальный вектор коэффициентов β
- `::Matrix{Float64}`: ковариационная матрица оценок коэффициентов

# Пример
```julia
# Генерируем тестовые данные
X = [ones(10) collect(1:10)]  # Матрица с константой и одним регрессором
y = 2.0 .+ 3.0 .* (1:10) .+ randn(10)  # Зависимая переменная с шумом
w = 1.0 ./ (1:10)  # Веса обратно пропорциональны номеру наблюдения
β, cov_β = weighted_linear_least_squares(X, y, w)
println("Оценка коэффициентов с учетом весов: ", β)
```
"""
function weighted_linear_least_squares(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, 
                                     w::AbstractVector{<:Real})
    # Проверка размерностей
    n, p = size(X)
    if length(y) != n
        throw(DimensionMismatch("Количество строк в X ($(n)) не соответствует длине y ($(length(y)))"))
    end
    if length(w) != n
        throw(DimensionMismatch("Количество строк в X ($(n)) не соответствует длине w ($(length(w)))"))
    end
    
    # Проверка, что все веса положительны
    if any(w .<= 0)
        throw(ArgumentError("Все веса должны быть положительными"))
    end
    
    # Создаем диагональную матрицу весов
    W = Diagonal(w)
    
    # Вычисляем X'WX
    XtWX = X' * W * X
    
    # Вычисляем β = (X'WX)^(-1) * X'Wy
    β = try
        XtWX \ (X' * W * y)
    catch e
        # Если матрица плохо обусловлена, используем псевдообратную матрицу
        pinv(XtWX) * (X' * W * y)
    end
    
    # Оцениваем дисперсию ошибок
    ŷ = X * β
    weighted_residuals = sqrt.(w) .* (y - ŷ)
    σ² = sum(weighted_residuals.^2) / (n - p)
    
    # Вычисляем ковариационную матрицу оценок коэффициентов
    cov_β = try
        σ² * inv(XtWX)
    catch e
        σ² * pinv(XtWX)
    end
    
    return β, cov_β
end

"""
    add_constant(X)

Добавляет столбец с константой (единицы) в начало матрицы регрессоров X.

# Аргументы
- `X::AbstractMatrix{<:Real}`: исходная матрица независимых переменных (регрессоров)

# Возвращает
- `::Matrix{Float64}`: матрица X с добавленным первым столбцом из единиц

# Пример
```julia
X = rand(10, 2)  # Матрица с двумя регрессорами
X_with_const = add_constant(X)  # Теперь матрица имеет размер 10×3, первый столбец - единицы
```
"""
function add_constant(X::AbstractMatrix{<:Real})
    n, p = size(X)
    return [ones(n) X]
end

"""
    linear_model(X, β)

Вычисляет значения линейной модели y = X * β для заданных значений X и коэффициентов β.

# Аргументы
- `X::AbstractMatrix{<:Real}`: матрица независимых переменных (регрессоров)
- `β::AbstractVector{<:Real}`: вектор коэффициентов модели

# Возвращает
- `::Vector{Float64}`: вектор предсказанных значений ŷ

# Пример
```julia
X = [ones(10) collect(1:10)]  # Матрица с константой и одним регрессором
β = [2.0, 3.0]  # Коэффициенты модели
ŷ = linear_model(X, β)  # Предсказанные значения
```
"""
function linear_model(X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
    # Проверка размерностей
    n, p = size(X)
    if length(β) != p
        throw(DimensionMismatch("Количество столбцов в X ($(p)) не соответствует длине β ($(length(β)))"))
    end
    
    # Вычисляем предсказанные значения
    return X * β
end

"""
    predict(model, X_new)

Предсказывает значения зависимой переменной для новых данных, используя обученную модель.

# Аргументы
- `model::NamedTuple`: структура с обученной моделью, содержащая поля:
   - `coefficients::Vector{Float64}`: коэффициенты модели
   - `add_constant::Bool`: флаг, указывающий, нужно ли добавлять столбец с константой
- `X_new::AbstractMatrix{<:Real}`: матрица с новыми данными для предсказания

# Возвращает
- `::Vector{Float64}`: вектор предсказанных значений

# Пример
```julia
# Генерируем тестовые данные
X = rand(10, 2)  # Матрица с двумя регрессорами
X_with_const = add_constant(X)  # Добавляем константу
y = 2.0 .+ 3.0 .* X[:, 1] .+ 1.5 .* X[:, 2] .+ randn(10)  # Зависимая переменная с шумом

# Обучаем модель
β, _ = linear_least_squares(X_with_const, y)
model = (coefficients = β, add_constant = false)  # Константа уже добавлена

# Предсказываем для новых данных
X_new = rand(5, 2)
X_new_with_const = add_constant(X_new)
ŷ_new = predict(model, X_new_with_const)
```
"""
function predict(model::NamedTuple, X_new::AbstractMatrix{<:Real})
    # Извлекаем коэффициенты и флаг для константы
    β = model.coefficients
    add_const = get(model, :add_constant, false)
    
    # Добавляем константу, если требуется
    X_predict = add_const ? add_constant(X_new) : X_new
    
    # Проверка размерностей
    _, p = size(X_predict)
    if length(β) != p
        throw(DimensionMismatch("Количество столбцов в X_new (с учетом константы: $(p)) не соответствует длине β ($(length(β)))"))
    end
    
    # Вычисляем предсказанные значения
    return X_predict * β
end 