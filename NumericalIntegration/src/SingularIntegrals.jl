"""
    SingularIntegrals

Модуль содержит методы для вычисления интегралов функций, имеющих особенности.
Используется для функций с сингулярностями, разрывами и быстро осциллирующих функций.
"""
module SingularIntegrals

using ..BasicMethods: IntegrationResult
using ..AdaptiveMethods: adaptive_integration

export singularity_subtraction, singularity_transformation, 
       dblexp_rule, tanh_sinh_quadrature, log_weighted_integration,
       cauchy_principal_value

"""
    singularity_subtraction(f, singularity, a, b; n_points=100, return_details=false)

Метод вычитания особенности для интегрирования функций с известной особенностью.

# Аргументы
- `f::Function`: интегрируемая функция с особенностью
- `singularity::Function`: функция, содержащая особенность, интеграл которой известен аналитически
- `analytical_integral::Function`: аналитический интеграл функции `singularity`
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n_points::Int=100`: число точек для численного интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Интегрирование log(x) на [0,1]
f(x) = log(x)
# Особенность и её известный интеграл
singularity(x) = log(x)
analytical_integral(a, b) = b*log(b) - b - (a*log(a) - a)
result = singularity_subtraction(f, singularity, analytical_integral, 0, 1)
```
"""
function singularity_subtraction(f::Function, singularity::Function, analytical_integral::Function,
                                a::Real, b::Real; n_points::Int=100, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    
    # Вычисляем разность f - singularity (гладкая функция)
    diff_func = x -> f(x) - singularity(x)
    
    # Интегрируем гладкую часть численно
    smooth_part, error_est = adaptive_integration(diff_func, a, b, tol=1e-10, max_depth=15)
    
    # Добавляем аналитический интеграл особенности
    singular_part = analytical_integral(a, b)
    
    # Окончательный результат
    result = smooth_part + singular_part
    
    if return_details
        return IntegrationResult(
            result,
            error_est,
            n_points,
            "Метод вычитания особенности"
        )
    else
        return result
    end
end

"""
    singularity_transformation(f, transformation, a, b; n_points=100, return_details=false)

Метод преобразования особенности для интегрирования функций с особенностями.

# Аргументы
- `f::Function`: интегрируемая функция с особенностью
- `transformation::Function`: функция преобразования переменных t -> x(t)
- `transformation_derivative::Function`: производная преобразования dx/dt
- `t_a::Real`: нижняя граница интегрирования после преобразования
- `t_b::Real`: верхняя граница интегрирования после преобразования
- `n_points::Int=100`: число точек для численного интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Интегрирование 1/sqrt(x) на [0,1]
f(x) = 1/sqrt(x)
# Преобразование: x = t^2, dx = 2t dt
transform(t) = t^2
transform_deriv(t) = 2*t
# После преобразования интеграл станет: ∫(0,1) 1/sqrt(t^2) * 2t dt = ∫(0,1) 2 dt = 2
result = singularity_transformation(f, transform, transform_deriv, 0, 1)
```
"""
function singularity_transformation(f::Function, transformation::Function, 
                                   transformation_derivative::Function,
                                   t_a::Real, t_b::Real; 
                                   n_points::Int=100, return_details::Bool=false)
    # Проверка аргументов
    if t_a > t_b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    
    # Создаем преобразованную функцию без особенностей
    transformed_func = t -> f(transformation(t)) * transformation_derivative(t)
    
    # Интегрируем преобразованную функцию численно
    result, error_est = adaptive_integration(transformed_func, t_a, t_b, tol=1e-10)
    
    if return_details
        return IntegrationResult(
            result,
            error_est,
            n_points,
            "Метод преобразования особенности"
        )
    else
        return result
    end
end

"""
    tanh_sinh_quadrature(f, a, b; n_points=30, return_details=false)

Квадратурная формула tanh-sinh для интегрирования функций с особенностями на концах интервала.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n_points::Int=30`: число точек квадратуры в одну сторону от центра
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Интегрирование 1/sqrt(x) на [0,1]
f(x) = 1/sqrt(x)
result = tanh_sinh_quadrature(f, 0, 1)  # Ожидаемый результат: 2.0
```
"""
function tanh_sinh_quadrature(f::Function, a::Real, b::Real; 
                             n_points::Int=30, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    
    # Параметр квадратурной формулы
    h = 2.0 / n_points
    
    # Преобразование к стандартному интервалу [-1, 1]
    mid = (a + b) / 2
    half_length = (b - a) / 2
    
    # Функция замены переменных для интеграла на [-1, 1]
    g = t -> f(mid + half_length * t) * half_length
    
    # Основное преобразование tanh-sinh: t = tanh(π/2 * sinh(x))
    # Интегрируем на [-n_points*h/2, n_points*h/2] с шагом h
    result = 0.0
    n_evals = 0
    
    for i in -n_points:n_points
        t = i * h
        
        # Преобразование tanh-sinh
        x = tanh(π/2 * sinh(t))
        
        # Вес для текущей точки
        weight = π/2 * cosh(t) / (cosh(π/2 * sinh(t)))^2 * half_length
        
        # Добавляем вклад точки в интеграл
        result += weight * f(mid + half_length * x)
        n_evals += 1
    end
    
    # Приближенная оценка погрешности (грубая)
    # Используем половину числа точек для сравнения
    half_result = 0.0
    for i in -n_points:2:n_points
        t = i * h
        x = tanh(π/2 * sinh(t))
        weight = π/2 * cosh(t) / (cosh(π/2 * sinh(t)))^2 * half_length
        half_result += weight * f(mid + half_length * x)
    end
    
    error_est = abs(result - half_result * 2) / 3
    
    if return_details
        return IntegrationResult(
            result,
            error_est,
            n_evals,
            "Квадратурная формула tanh-sinh"
        )
    else
        return result
    end
end

"""
    dblexp_rule(f, a, b; n_points=30, return_details=false)

Квадратурная формула Double Exponential (DE) для интегрирования функций с особенностями.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n_points::Int=30`: число точек квадратуры в одну сторону от центра
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Интегрирование log(x) на [0,1]
f(x) = log(x)
result = dblexp_rule(f, 0, 1)  # Ожидаемый результат: -1.0
```
"""
function dblexp_rule(f::Function, a::Real, b::Real; 
                   n_points::Int=30, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    
    # Параметр квадратурной формулы
    h = 3.0 / n_points
    
    # Преобразование к стандартному интервалу [-1, 1]
    mid = (a + b) / 2
    half_length = (b - a) / 2
    
    # Функция замены переменных для интеграла на [-1, 1]
    g = t -> f(mid + half_length * t) * half_length
    
    # Double Exponential преобразование: t = tanh(sinh(x))
    # Интегрируем на [-n_points*h, n_points*h] с шагом h
    result = 0.0
    n_evals = 0
    
    for i in -n_points:n_points
        t = i * h
        
        # Double Exponential преобразование
        x = tanh(sinh(t))
        
        # Вес для текущей точки
        weight = cosh(t) / (cosh(sinh(t)))^2 * half_length
        
        # Добавляем вклад точки в интеграл
        result += weight * f(mid + half_length * x)
        n_evals += 1
    end
    
    # Приближенная оценка погрешности
    # Используем половину числа точек для сравнения
    half_result = 0.0
    for i in -n_points:2:n_points
        t = i * h
        x = tanh(sinh(t))
        weight = cosh(t) / (cosh(sinh(t)))^2 * half_length
        half_result += weight * f(mid + half_length * x)
    end
    
    error_est = abs(result - half_result * 2) / 3
    
    if return_details
        return IntegrationResult(
            result,
            error_est,
            n_evals,
            "Квадратурная формула Double Exponential"
        )
    else
        return result
    end
end

"""
    log_weighted_integration(f, a, b; n_points=50, return_details=false)

Вычисляет интеграл с логарифмическим весом: ∫(a,b) f(x) * log(x-a) dx или ∫(a,b) f(x) * log(b-x) dx.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `singularity::Symbol=:left`: расположение логарифмической особенности 
  (:left для log(x-a) или :right для log(b-x))
- `n_points::Int=50`: число точек для квадратуры
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Интегрирование x * log(x) на [0,1]
f(x) = x
result = log_weighted_integration(f, 0, 1, singularity=:left)  # -0.25
```
"""
function log_weighted_integration(f::Function, a::Real, b::Real; 
                                 singularity::Symbol=:left,
                                 n_points::Int=50, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    
    if singularity != :left && singularity != :right
        throw(ArgumentError("Параметр singularity должен быть :left или :right"))
    end
    
    # Разбиение интервала
    h = (b - a) / n_points
    
    # Узлы Гаусса-Лежандра для интегрирования с логарифмическим весом
    # Для упрощения используем узлы адаптивной квадратуры
    
    result = 0.0
    error_est = 0.0
    n_evals = 0
    
    if singularity == :left
        # Интеграл с весом log(x-a)
        # Используем замену переменных y = sqrt(x-a) для сглаживания особенности
        transformed_f = y -> f(a + y^2) * 2 * y * log(y^2)
        result, error_est = adaptive_integration(transformed_f, 0, sqrt(b-a), tol=1e-8)
        n_evals = n_points  # Приблизительное число вызовов функции
    else
        # Интеграл с весом log(b-x)
        # Используем замену переменных y = sqrt(b-x) для сглаживания особенности
        transformed_f = y -> f(b - y^2) * 2 * y * log(y^2)
        result, error_est = adaptive_integration(transformed_f, 0, sqrt(b-a), tol=1e-8)
        n_evals = n_points  # Приблизительное число вызовов функции
    end
    
    if return_details
        method_name = singularity == :left ? 
            "Интегрирование с весом log(x-a)" : 
            "Интегрирование с весом log(b-x)"
        
        return IntegrationResult(
            result,
            error_est,
            n_evals,
            method_name
        )
    else
        return result
    end
end

"""
    cauchy_principal_value(f, a, b, c; n_points=100, ε=1e-6, return_details=false)

Вычисляет главное значение по Коши интеграла с сингулярностью вида 1/(x-c).

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `c::Real`: точка сингулярности (должна находиться внутри интервала [a, b])
- `n_points::Int=100`: число точек для квадратуры
- `ε::Real=1e-6`: параметр регуляризации
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
# Вычисление главного значения ∫(-1,1) 1/x dx = 0
f(x) = 1
result = cauchy_principal_value(f, -1, 1, 0)  # Ожидаемый результат: 0
```
"""
function cauchy_principal_value(f::Function, a::Real, b::Real, c::Real; 
                              n_points::Int=100, ε::Real=1e-6, 
                              return_details::Bool=false)
    # Проверка аргументов
    if a > b
        throw(ArgumentError("Верхняя граница должна быть больше нижней"))
    end
    if c < a || c > b
        throw(ArgumentError("Точка сингулярности c должна лежать внутри интервала [a, b]"))
    end
    
    # Разбиваем интеграл на две части: [a, c-ε] и [c+ε, b]
    result1, err1 = adaptive_integration(x -> f(x)/(x-c), a, c-ε, tol=1e-8)
    result2, err2 = adaptive_integration(x -> f(x)/(x-c), c+ε, b, tol=1e-8)
    
    # Суммарный результат
    result = result1 + result2
    error_est = err1 + err2
    
    if return_details
        return IntegrationResult(
            result,
            error_est,
            n_points,
            "Главное значение по Коши"
        )
    else
        return result
    end
end

end # module SingularIntegrals 