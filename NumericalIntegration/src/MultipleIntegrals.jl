"""
    MultipleIntegrals

Модуль содержит реализации методов для вычисления многомерных интегралов.
"""
module MultipleIntegrals

using ..BasicMethods: IntegrationResult, simpson_rule, trapezoidal_rule

export double_integral, triple_integral, multidimensional_integral

"""
    double_integral(f, a1, b1, a2, b2; method=:simpson, n1=30, n2=30, return_details=false)

Вычисляет двойной интеграл функции `f(x, y)` в прямоугольной области `[a1, b1] × [a2, b2]`.

# Аргументы
- `f::Function`: интегрируемая функция двух переменных
- `a1::Real`: нижняя граница интегрирования по первой переменной
- `b1::Real`: верхняя граница интегрирования по первой переменной
- `a2::Real`: нижняя граница интегрирования по второй переменной
- `b2::Real`: верхняя граница интегрирования по второй переменной
- `method::Symbol=:simpson`: метод интегрирования (:simpson или :trapezoidal)
- `n1::Int=30`: число шагов по первой переменной
- `n2::Int=30`: число шагов по второй переменной
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Вычисление двойного интеграла ∫∫ x*y dxdy по области [0,1]×[0,1]
result = double_integral((x, y) -> x * y, 0, 1, 0, 1)  # Ожидаемый результат: 0.25
```
"""
function double_integral(f::Function, a1::Real, b1::Real, a2::Real, b2::Real;
                         method::Symbol=:simpson, n1::Int=30, n2::Int=30,
                         return_details::Bool=false)
    # Проверка аргументов
    if a1 > b1 || a2 > b2
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    if n1 <= 0 || n2 <= 0
        throw(ArgumentError("Число шагов должно быть положительным"))
    end
    
    # Выбор функции интегрирования
    integration_func = method == :simpson ? simpson_rule : trapezoidal_rule
    
    # Шаги интегрирования
    h1 = (b1 - a1) / n1
    h2 = (b2 - a2) / n2
    
    # Создаем сетку точек
    x_points = [a1 + i * h1 for i in 0:n1]
    y_points = [a2 + j * h2 for j in 0:n2]
    
    # Для каждого y вычисляем интеграл по x
    y_integrals = zeros(length(y_points))
    
    for (j, y) in enumerate(y_points)
        # Создаем функцию одной переменной, фиксируя y
        f_x = x -> f(x, y)
        
        # Интегрируем по x
        y_integrals[j] = integration_func(f_x, a1, b1, n1, return_details=false)
    end
    
    # Интегрируем результаты по y
    g = y -> begin
        # Находим ближайший индекс в массиве y_points
        idx = argmin(abs.(y_points .- y))
        return y_integrals[idx]
    end
    
    result = integration_func(g, a2, b2, n2, return_details=false)
    
    if return_details
        n_evals = (n1 + 1) * (n2 + 1)  # Приблизительное число вычислений функции
        error_est = max(h1^4, h2^4) # Грубая оценка ошибки для метода Симпсона
        if method == :trapezoidal
            error_est = max(h1^2, h2^2) # Грубая оценка ошибки для метода трапеций
        end
        method_name = method == :simpson ? "Метод Симпсона" : "Метод трапеций"
        
        return IntegrationResult(result, error_est, n_evals, "Двойной интеграл ($(method_name))")
    else
        return result
    end
end

"""
    triple_integral(f, a1, b1, a2, b2, a3, b3; method=:simpson, n1=20, n2=20, n3=20, return_details=false)

Вычисляет тройной интеграл функции `f(x, y, z)` в прямоугольной области `[a1, b1] × [a2, b2] × [a3, b3]`.

# Аргументы
- `f::Function`: интегрируемая функция трех переменных
- `a1::Real`, `b1::Real`: границы интегрирования по первой переменной
- `a2::Real`, `b2::Real`: границы интегрирования по второй переменной
- `a3::Real`, `b3::Real`: границы интегрирования по третьей переменной
- `method::Symbol=:simpson`: метод интегрирования (:simpson или :trapezoidal)
- `n1::Int=20`, `n2::Int=20`, `n3::Int=20`: число шагов по каждой переменной
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Тройной интеграл ∫∫∫ x*y*z dxdydz по области [0,1]³
result = triple_integral((x, y, z) -> x * y * z, 0, 1, 0, 1, 0, 1)  # Ожидаемый результат: 0.125
```
"""
function triple_integral(f::Function, a1::Real, b1::Real, a2::Real, b2::Real, a3::Real, b3::Real;
                         method::Symbol=:simpson, n1::Int=20, n2::Int=20, n3::Int=20,
                         return_details::Bool=false)
    # Проверка аргументов
    if a1 > b1 || a2 > b2 || a3 > b3
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    if n1 <= 0 || n2 <= 0 || n3 <= 0
        throw(ArgumentError("Число шагов должно быть положительным"))
    end
    
    # Выбор функции интегрирования
    integration_func = method == :simpson ? simpson_rule : trapezoidal_rule
    
    # Шаги интегрирования
    h1 = (b1 - a1) / n1
    h2 = (b2 - a2) / n2
    h3 = (b3 - a3) / n3
    
    # Создаем сетку точек
    x_points = [a1 + i * h1 for i in 0:n1]
    y_points = [a2 + j * h2 for j in 0:n2]
    z_points = [a3 + k * h3 for k in 0:n3]
    
    # Интегрирование по z для каждой пары (x, y)
    xy_integrals = zeros(length(x_points), length(y_points))
    
    for (i, x) in enumerate(x_points), (j, y) in enumerate(y_points)
        # Создаем функцию одной переменной, фиксируя x и y
        f_z = z -> f(x, y, z)
        
        # Интегрируем по z
        xy_integrals[i, j] = integration_func(f_z, a3, b3, n3, return_details=false)
    end
    
    # Интегрирование по y для каждого x
    x_integrals = zeros(length(x_points))
    
    for (i, x) in enumerate(x_points)
        # Создаем функцию одной переменной, фиксируя x
        f_y = y -> begin
            # Находим ближайший индекс в массиве y_points
            j = argmin(abs.(y_points .- y))
            return xy_integrals[i, j]
        end
        
        # Интегрируем по y
        x_integrals[i] = integration_func(f_y, a2, b2, n2, return_details=false)
    end
    
    # Интегрирование по x
    g = x -> begin
        # Находим ближайший индекс в массиве x_points
        i = argmin(abs.(x_points .- x))
        return x_integrals[i]
    end
    
    result = integration_func(g, a1, b1, n1, return_details=false)
    
    if return_details
        n_evals = (n1 + 1) * (n2 + 1) * (n3 + 1)  # Приблизительное число вычислений функции
        error_est = max(h1^4, h2^4, h3^4) # Грубая оценка ошибки для метода Симпсона
        if method == :trapezoidal
            error_est = max(h1^2, h2^2, h3^2) # Грубая оценка ошибки для метода трапеций
        end
        method_name = method == :simpson ? "Метод Симпсона" : "Метод трапеций"
        
        return IntegrationResult(result, error_est, n_evals, "Тройной интеграл ($(method_name))")
    else
        return result
    end
end

"""
    multidimensional_integral(f, bounds; method=:simpson, n_steps=20, return_details=false)

Вычисляет многомерный интеграл функции `f(x)` в прямоугольной области, заданной границами `bounds`.

# Аргументы
- `f::Function`: интегрируемая функция, принимающая вектор координат
- `bounds::Vector{Tuple{Real, Real}}`: список пар (нижняя граница, верхняя граница) для каждого измерения
- `method::Symbol=:simpson`: метод интегрирования (:simpson или :trapezoidal)
- `n_steps::Int=20`: число шагов по каждой переменной
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Многомерный интеграл функции f(x, y, z, w) = x*y*z*w по области [0,1]⁴
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
result = multidimensional_integral(x -> prod(x), bounds)
```
"""
function multidimensional_integral(f::Function, bounds::Vector{Tuple{Real, Real}};
                                  method::Symbol=:simpson, n_steps::Int=20,
                                  return_details::Bool=false)
    # Проверка аргументов
    dimensions = length(bounds)
    if dimensions == 0
        throw(ArgumentError("Должно быть указано хотя бы одно измерение"))
    end
    
    # Для одномерного интеграла
    if dimensions == 1
        a, b = bounds[1]
        integration_func = method == :simpson ? simpson_rule : trapezoidal_rule
        return integration_func(f, a, b, n_steps, return_details=return_details)
    end
    
    # Разделение границ
    current_dim_bounds = bounds[dimensions]
    a, b = current_dim_bounds
    remaining_bounds = bounds[1:dimensions-1]
    
    # Выбор функции интегрирования
    integration_func = method == :simpson ? simpson_rule : trapezoidal_rule
    
    # Шаг интегрирования
    h = (b - a) / n_steps
    
    # Создаем сетку точек для текущего измерения
    points = [a + i * h for i in 0:n_steps]
    
    # Вычисляем интегралы по оставшимся измерениям для каждой точки текущего измерения
    dim_integrals = zeros(length(points))
    
    for (i, point) in enumerate(points)
        # Создаем функцию с уменьшенной размерностью
        g = x -> f([x..., point])
        
        # Вычисляем интеграл по оставшимся измерениям
        dim_integrals[i] = multidimensional_integral(g, remaining_bounds, 
                                                   method=method, n_steps=n_steps, 
                                                   return_details=false)
    end
    
    # Интегрируем результаты по текущему измерению
    result_func = x -> begin
        # Находим ближайший индекс в массиве points
        idx = argmin(abs.(points .- x))
        return dim_integrals[idx]
    end
    
    result = integration_func(result_func, a, b, n_steps, return_details=false)
    
    if return_details
        # Оценка числа вычислений функции
        n_evals = n_steps^dimensions
        
        # Грубая оценка ошибки
        h_values = [(b - a) / n_steps for (a, b) in bounds]
        if method == :simpson
            error_est = maximum(h_values) ^ 4  # Метод Симпсона имеет порядок O(h^4)
        else
            error_est = maximum(h_values) ^ 2  # Метод трапеций имеет порядок O(h^2)
        end
        
        method_name = method == :simpson ? "Метод Симпсона" : "Метод трапеций"
        
        return IntegrationResult(result, error_est, n_evals, "$(dimensions)-мерный интеграл ($(method_name))")
    else
        return result
    end
end

"""
    iterated_integral(f, bounds; method=:simpson, n_steps=100, return_details=false)

Вычисляет многомерный интеграл функции `f` с использованием итеративного подхода.
Функция `f` должна принимать все аргументы отдельно, например, `f(x, y, z)`.

# Аргументы
- `f::Function`: интегрируемая функция
- `bounds::Vector{Tuple{Real, Real}}`: список пар (нижняя граница, верхняя граница) для каждого измерения
- `method::Symbol=:simpson`: метод интегрирования (:simpson или :trapezoidal)
- `n_steps::Int=100`: число шагов по каждой переменной
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Вычисление итерированного интеграла ∫₀¹∫₀¹∫₀¹ x*y*z dz dy dx
bounds = [(0, 1), (0, 1), (0, 1)]
f(x, y, z) = x * y * z
result = iterated_integral(f, bounds)  # Ожидаемый результат: 0.125
```
"""
function iterated_integral(f::Function, bounds::Vector{Tuple{Real, Real}};
                          method::Symbol=:simpson, n_steps::Int=100,
                          return_details::Bool=false)
    # Проверка аргументов
    dimensions = length(bounds)
    if dimensions == 0
        throw(ArgumentError("Должно быть указано хотя бы одно измерение"))
    end
    
    # Преобразование функции к виду, принимающему вектор координат
    function vectorized_f(x)
        return Base.invokelatest(f, x...)
    end
    
    # Вызов основной функции интегрирования
    return multidimensional_integral(vectorized_f, bounds, 
                                   method=method, n_steps=n_steps, 
                                   return_details=return_details)
end

end # module MultipleIntegrals 