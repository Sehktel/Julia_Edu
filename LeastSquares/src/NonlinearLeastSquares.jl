"""
    NonlinearLeastSquares

Модуль содержит функции для решения задач нелинейной аппроксимации
методом наименьших квадратов. Включает в себя алгоритм Левенберга-Марквардта
и другие методы оптимизации для нелинейных моделей.
"""

using LinearAlgebra
using Statistics

"""
    numerical_jacobian(f, p, x, y; eps=1e-6)

Вычисляет численно матрицу Якоби (матрицу частных производных) функции f
по параметрам p в точках x для модели y = f(x, p).

# Аргументы
- `f::Function`: функция модели вида f(x, p) -> y
- `p::AbstractVector{<:Real}`: текущие значения параметров модели
- `x::AbstractVector{<:Real}`: вектор значений независимой переменной
- `y::AbstractVector{<:Real}`: вектор значений зависимой переменной
- `eps::Real=1e-6`: шаг для численного дифференцирования

# Возвращает
- `::Matrix{Float64}`: матрица Якоби размером n×m, где n = длина x, m = длина p
- `::Vector{Float64}`: вектор невязок (residuals) размером n

# Пример
```julia
# Модель: y = a * exp(b * x)
model(x, p) = p[1] * exp(p[2] * x)

# Данные
x = collect(0.0:0.1:1.0)
true_params = [2.0, -1.5]
y = model.(x, Ref(true_params)) .+ 0.01 .* randn(length(x))

# Начальное приближение
initial_params = [1.0, -1.0]

# Вычисляем якобиан
J, residuals = numerical_jacobian(model, initial_params, x, y)
```
"""
function numerical_jacobian(f::Function, p::AbstractVector{<:Real}, 
                          x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                          eps::Real=1e-6)
    n = length(x)
    m = length(p)
    
    J = zeros(n, m)
    y_model = zeros(n)
    
    # Вычисляем значения модели при текущих параметрах
    for i in 1:n
        y_model[i] = f(x[i], p)
    end
    
    # Вычисляем якобиан
    for j in 1:m
        p_plus = copy(p)
        p_plus[j] += eps
        
        for i in 1:n
            # Приближение производной методом конечных разностей
            J[i, j] = (f(x[i], p_plus) - y_model[i]) / eps
        end
    end
    
    # Вычисляем невязки
    residuals = y_model - y
    
    return J, residuals
end

"""
    levenberg_marquardt(f, x, y, p0; max_iter=100, lambda_init=0.1, 
                     lambda_factor=10.0, tolerance=1e-8, verbose=false)

Реализует алгоритм Левенберга-Марквардта для нелинейного метода наименьших квадратов.

# Аргументы
- `f::Function`: функция модели вида f(x, p) -> y
- `x::AbstractVector{<:Real}`: вектор значений независимой переменной
- `y::AbstractVector{<:Real}`: вектор значений зависимой переменной
- `p0::AbstractVector{<:Real}`: начальное приближение параметров
- `max_iter::Int=100`: максимальное число итераций
- `lambda_init::Float64=0.1`: начальное значение параметра регуляризации
- `lambda_factor::Float64=10.0`: множитель для изменения параметра регуляризации
- `tolerance::Float64=1e-8`: точность сходимости (по изменению суммы квадратов невязок)
- `verbose::Bool=false`: вывод подробной информации об итерациях

# Возвращает
- `::NamedTuple`: структура с полями:
  - `parameters::Vector{Float64}`: найденные оптимальные параметры
  - `cov_matrix::Matrix{Float64}`: ковариационная матрица параметров
  - `iterations::Int`: число выполненных итераций
  - `residuals::Vector{Float64}`: вектор невязок на последней итерации
  - `sum_of_squares::Float64`: сумма квадратов невязок на последней итерации

# Пример
```julia
# Модель: y = a * exp(b * x)
model(x, p) = p[1] * exp(p[2] * x)

# Генерируем данные
x = collect(0.0:0.1:1.0)
true_params = [2.0, -1.5]
y = model.(x, Ref(true_params)) .+ 0.01 .* randn(length(x))

# Начальное приближение
initial_params = [1.0, -1.0]

# Применяем алгоритм Левенберга-Марквардта
result = levenberg_marquardt(model, x, y, initial_params, verbose=true)
println("Найденные параметры: ", result.parameters)
println("Ковариационная матрица: ", result.cov_matrix)
```
"""
function levenberg_marquardt(f::Function, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, 
                          p0::AbstractVector{<:Real}; max_iter::Int=100, lambda_init::Float64=0.1, 
                          lambda_factor::Float64=10.0, tolerance::Float64=1e-8, verbose::Bool=false)
    # Инициализация
    p = copy(p0)
    n = length(x)
    m = length(p)
    lambda = lambda_init
    
    # Получаем начальный якобиан и невязки
    J, residuals = numerical_jacobian(f, p, x, y)
    sum_squares = sum(residuals.^2)
    
    if verbose
        println("Начальная сумма квадратов невязок: $sum_squares")
    end
    
    # Итерационный процесс
    iter = 0
    for iter in 1:max_iter
        # Формируем нормальное уравнение с регуляризацией
        H = J' * J
        g = J' * residuals
        
        # Добавляем регуляризацию Левенберга-Марквардта
        H_reg = H + lambda * Diagonal(diag(H))
        
        # Решаем систему уравнений для получения шага
        delta_p = H_reg \ (-g)
        
        # Проверяем новые параметры
        p_new = p + delta_p
        
        # Вычисляем новые невязки
        residuals_new = zeros(n)
        for i in 1:n
            residuals_new[i] = f(x[i], p_new) - y[i]
        end
        sum_squares_new = sum(residuals_new.^2)
        
        # Решаем, принять новый шаг или изменить lambda
        if sum_squares_new < sum_squares
            # Шаг успешный, уменьшаем lambda
            lambda = max(lambda / lambda_factor, 1e-10)
            
            # Обновляем параметры и невязки
            p = p_new
            sum_squares = sum_squares_new
            
            # Обновляем якобиан для новых параметров
            J, residuals = numerical_jacobian(f, p, x, y)
            
            if verbose
                println("Итерация $iter: сумма квадратов = $sum_squares, lambda = $lambda")
                println("  Параметры: $p")
            end
            
            # Проверяем условие сходимости
            if abs(sum_squares_new - sum_squares) < tolerance
                if verbose
                    println("Сходимость достигнута на итерации $iter")
                end
                break
            end
        else
            # Шаг неуспешный, увеличиваем lambda
            lambda *= lambda_factor
            
            if verbose
                println("Итерация $iter: шаг отклонен, lambda = $lambda")
            end
            
            if lambda > 1e10
                if verbose
                    println("Lambda слишком большая, прекращаем итерации")
                end
                break
            end
        end
    end
    
    # Оценка ковариационной матрицы параметров
    sigma_squared = sum_squares / (n - m)  # Оценка дисперсии
    cov_matrix = sigma_squared * inv(J' * J)
    
    return (
        parameters = p,
        cov_matrix = cov_matrix,
        iterations = iter,
        residuals = residuals,
        sum_of_squares = sum_squares
    )
end

"""
    nlls_fit(model, x, y, p0; method=:levenberg_marquardt, kwargs...)

Удобная функция для подбора нелинейной модели к данным методом наименьших квадратов.

# Аргументы
- `model::Function`: функция модели вида model(x, p) -> y
- `x::AbstractVector{<:Real}`: вектор значений независимой переменной
- `y::AbstractVector{<:Real}`: вектор значений зависимой переменной
- `p0::AbstractVector{<:Real}`: начальное приближение параметров
- `method::Symbol=:levenberg_marquardt`: метод оптимизации (пока поддерживается только `:levenberg_marquardt`)
- `kwargs...`: дополнительные аргументы для метода оптимизации

# Возвращает
- `::NamedTuple`: структура с полями:
  - `parameters::Vector{Float64}`: найденные оптимальные параметры
  - `cov_matrix::Matrix{Float64}`: ковариационная матрица параметров
  - `predict::Function`: функция для предсказания значений на новых данных
  - `model::Function`: исходная функция модели
  - А также другие поля, специфичные для выбранного метода оптимизации

# Пример
```julia
# Определяем модель: y = a * exp(b * x)
function exp_model(x, p)
    a, b = p
    return a * exp(b * x)
end

# Генерируем тестовые данные
x = collect(0.0:0.1:1.0)
true_params = [2.0, -1.5]
y = exp_model.(x, Ref(true_params)) .+ 0.01 .* randn(length(x))

# Начальное приближение
initial_params = [1.0, -1.0]

# Подбираем модель
fit_result = nlls_fit(exp_model, x, y, initial_params, verbose=true)
println("Параметры модели: ", fit_result.parameters)

# Используем модель для предсказания на новых данных
x_new = collect(0.0:0.05:1.2)
y_pred = fit_result.predict(x_new)
```
"""
function nlls_fit(model::Function, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, 
                p0::AbstractVector{<:Real}; method::Symbol=:levenberg_marquardt, kwargs...)
    # Проверка выбранного метода
    if method != :levenberg_marquardt
        throw(ArgumentError("Метод '$method' не поддерживается. Доступный метод: :levenberg_marquardt"))
    end
    
    # Применяем выбранный метод
    result = levenberg_marquardt(model, x, y, p0; kwargs...)
    
    # Создаем функцию для предсказания
    predict_func = (x_new) -> [model(xi, result.parameters) for xi in x_new]
    
    # Возвращаем результат с добавленной функцией предсказания
    return merge(result, (
        predict = predict_func,
        model = model
    ))
end

"""
    confidence_intervals(fit_result, alpha=0.05)

Вычисляет доверительные интервалы для параметров модели.

# Аргументы
- `fit_result::NamedTuple`: результат подбора модели функцией `nlls_fit`
- `alpha::Float64=0.05`: уровень значимости (по умолчанию 5%)

# Возвращает
- `::Matrix{Float64}`: матрица размером n×2, где n - число параметров, 
  содержащая нижнюю и верхнюю границы доверительных интервалов

# Пример
```julia
# После подбора модели
fit_result = nlls_fit(model, x, y, initial_params)

# Вычисляем 95% доверительные интервалы
ci = confidence_intervals(fit_result)
for i in 1:length(fit_result.parameters)
    println("Параметр $(i): $(fit_result.parameters[i]) ∈ [$(ci[i,1]), $(ci[i,2])]")
end

# Вычисляем 99% доверительные интервалы
ci_99 = confidence_intervals(fit_result, 0.01)
```
"""
function confidence_intervals(fit_result::NamedTuple, alpha::Float64=0.05)
    # Получаем параметры из результата
    p = fit_result.parameters
    cov_matrix = fit_result.cov_matrix
    n_params = length(p)
    
    # Получаем стандартные отклонения параметров
    std_errors = sqrt.(diag(cov_matrix))
    
    # Аппроксимация квантили t-распределения для заданного alpha
    # В реальном приложении лучше использовать статистические библиотеки для точных значений
    t_quantile = 1.96  # Приближение для 95% доверительного интервала нормального распределения
    
    if alpha == 0.01
        t_quantile = 2.576  # Приближение для 99% доверительного интервала
    elseif alpha == 0.1
        t_quantile = 1.645  # Приближение для 90% доверительного интервала
    end
    
    # Вычисляем доверительные интервалы
    ci = zeros(n_params, 2)
    for i in 1:n_params
        ci[i, 1] = p[i] - t_quantile * std_errors[i]  # Нижняя граница
        ci[i, 2] = p[i] + t_quantile * std_errors[i]  # Верхняя граница
    end
    
    return ci
end 