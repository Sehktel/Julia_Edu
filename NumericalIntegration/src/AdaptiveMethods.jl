"""
    AdaptiveMethods

Модуль содержит реализации адаптивных методов численного интегрирования.

В адаптивных методах шаг интегрирования автоматически подстраивается
в зависимости от локальной сложности функции, обеспечивая заданную точность
при минимально необходимом числе вычислений функции.
"""
module AdaptiveMethods

export adaptive_integration, adaptive_simpson, adaptive_quadrature
export IntegrationResult

"""
    IntegrationResult

Структура, содержащая результат численного интегрирования и дополнительную информацию.

# Поля
- `value::Float64`: приближенное значение интеграла
- `error_estimate::Float64`: оценка погрешности (если доступна)
- `n_function_calls::Int`: количество вызовов интегрируемой функции
- `n_subdivisions::Int`: количество подразбиений интервала (для адаптивных методов)
- `convergence::Bool`: флаг, указывающий, сошелся ли метод к заданной точности
- `details::Dict{Symbol,Any}`: дополнительные детали, специфичные для конкретного метода
"""
struct IntegrationResult
    value::Float64
    error_estimate::Float64
    n_function_calls::Int
    n_subdivisions::Int
    convergence::Bool
    details::Dict{Symbol,Any}
    
    # Конструктор с значениями по умолчанию
    function IntegrationResult(value::Float64;
                               error_estimate::Float64=NaN,
                               n_function_calls::Int=0,
                               n_subdivisions::Int=0,
                               convergence::Bool=true,
                               details::Dict{Symbol,Any}=Dict{Symbol,Any}())
        new(value, error_estimate, n_function_calls, n_subdivisions, convergence, details)
    end
end

"""
    adaptive_integration(f, a, b; tol=1e-8, max_depth=20, method=:simpson, return_details=false)

Выполняет адаптивное интегрирование функции `f` на интервале `[a, b]` с заданной точностью.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `tol::Real=1e-8`: требуемая абсолютная точность
- `max_depth::Int=20`: максимальная глубина рекурсии
- `method::Symbol=:simpson`: метод для вычисления приближения 
  (`:trapezoid` или `:simpson`)
- `return_details::Bool=false`: если `true`, возвращает структуру с подробной информацией

# Возвращает
- `result::Float64` или `::IntegrationResult`: приближенное значение интеграла или 
  структуру с результатом и дополнительной информацией

# Пример
```julia
# Интегрирование функции sin(x) от 0 до π с точностью 1e-10
result = adaptive_integration(sin, 0, π, tol=1e-10)

# Получение дополнительной информации о процессе интегрирования
details = adaptive_integration(x -> 1/sqrt(x), 0, 1, tol=1e-8, return_details=true)
println("Число вызовов функции: \$(details.n_function_calls)")
println("Число подразбиений: \$(details.n_subdivisions)")
```
"""
function adaptive_integration(f::Function, a::Real, b::Real; 
                             tol::Real=1e-8, 
                             max_depth::Int=20,
                             method::Symbol=:simpson,
                             return_details::Bool=false)
    
    # Структуры для отслеживания статистики
    n_calls = Ref(0)
    n_subdivisions = Ref(0)
    
    # Оберточная функция для подсчета вызовов
    function f_with_count(x)
        n_calls[] += 1
        return f(x)
    end
    
    # Выбор базового метода в зависимости от параметра
    if method == :trapezoid
        base_method_low = (f, a, b) -> (b - a) * (f(a) + f(b)) / 2
        base_method_high = (f, a, b) -> (b - a) * (f(a) + 2*f((a+b)/2) + f(b)) / 4
    elseif method == :simpson
        base_method_low = (f, a, b) -> (b - a) * (f(a) + 4*f((a+b)/2) + f(b)) / 6
        function base_method_high(f, a, b)
            h = (b - a) / 4
            return (b - a) * (f(a) + 4*f(a+h) + 2*f(a+2*h) + 4*f(a+3*h) + f(b)) / 12
        end
    else
        throw(ArgumentError("Неизвестный базовый метод: $method"))
    end
    
    # Рекурсивная функция адаптивного интегрирования
    function adaptive_recursive(f, a, b, tol, depth)
        # Вычисляем приближения разной точности
        I_low = base_method_low(f, a, b)
        I_high = base_method_high(f, a, b)
        
        # Оценка погрешности
        error_est = abs(I_high - I_low)
        
        # Проверка условия остановки
        if error_est <= tol || depth >= max_depth
            n_subdivisions[] += 1
            return I_high, error_est, depth >= max_depth
        end
        
        # Разбиваем интервал пополам
        mid = (a + b) / 2
        I_left, err_left, nconv_left = adaptive_recursive(f, a, mid, tol/2, depth+1)
        I_right, err_right, nconv_right = adaptive_recursive(f, mid, b, tol/2, depth+1)
        
        # Объединяем результаты
        I_total = I_left + I_right
        err_total = err_left + err_right
        not_converged = nconv_left || nconv_right
        
        return I_total, err_total, not_converged
    end
    
    # Вызываем рекурсивную функцию
    integral, error_est, not_converged = adaptive_recursive(f_with_count, a, b, tol, 0)
    
    # Возвращаем результат в запрошенном формате
    if return_details
        return IntegrationResult(
            float(integral),
            error_estimate=float(error_est),
            n_function_calls=n_calls[],
            n_subdivisions=n_subdivisions[],
            convergence=!not_converged,
            details=Dict{Symbol,Any}(:method => method)
        )
    else
        return float(integral)
    end
end

"""
    adaptive_simpson(f, a, b; tol=1e-8, max_depth=20, return_details=false)

Адаптивный метод Симпсона для численного интегрирования.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `tol::Real=1e-8`: требуемая абсолютная точность
- `max_depth::Int=20`: максимальная глубина рекурсии
- `return_details::Bool=false`: если `true`, возвращает структуру с подробной информацией

# Возвращает
- `result::Float64` или `::IntegrationResult`: приближенное значение интеграла или 
  структуру с результатом и дополнительной информацией

# Пример
```julia
# Интегрирование функции sin(x) от 0 до π с точностью 1e-10
result = adaptive_simpson(sin, 0, π, tol=1e-10)
```
"""
function adaptive_simpson(f::Function, a::Real, b::Real; 
                         tol::Real=1e-8, 
                         max_depth::Int=20,
                         return_details::Bool=false)
    return adaptive_integration(f, a, b, 
                               tol=tol, 
                               max_depth=max_depth, 
                               method=:simpson,
                               return_details=return_details)
end

"""
    adaptive_quadrature(f, a, b; tol=1e-8, max_subdivisions=1000, return_details=false)

Адаптивная квадратурная формула с автоматическим выбором подынтервалов.
Использует комбинацию методов Гаусса и метода Симпсона.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `tol::Real=1e-8`: требуемая относительная точность
- `max_subdivisions::Int=1000`: максимальное число подразбиений
- `return_details::Bool=false`: если `true`, возвращает структуру с подробной информацией

# Возвращает
- `result::Float64` или `::IntegrationResult`: приближенное значение интеграла или 
  структуру с результатом и дополнительной информацией

# Пример
```julia
# Интегрирование функции с особенностью
result = adaptive_quadrature(x -> 1/sqrt(x), 0, 1, tol=1e-8)
```
"""
function adaptive_quadrature(f::Function, a::Real, b::Real; 
                            tol::Real=1e-8, 
                            max_subdivisions::Int=1000,
                            return_details::Bool=false)
    # Узлы и веса для 5-точечной квадратуры Гаусса-Лежандра
    gauss_nodes = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
    gauss_weights = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]
    
    # Счетчики
    n_calls = Ref(0)
    n_subdivisions = Ref(0)
    
    # Оберточная функция для подсчета вызовов
    function f_with_count(x)
        n_calls[] += 1
        return f(x)
    end
    
    # Интегрирование на отрезке методом Гаусса-Лежандра
    function gauss_quadrature(f, a, b)
        mid = (a + b) / 2
        h = (b - a) / 2
        
        result = 0.0
        for i in 1:length(gauss_nodes)
            x = mid + h * gauss_nodes[i]
            result += gauss_weights[i] * f(x)
        end
        
        return h * result
    end
    
    # Правило Симпсона на отрезке
    function simpson_rule(f, a, b)
        h = (b - a) / 2
        mid = (a + b) / 2
        return (h / 3) * (f(a) + 4 * f(mid) + f(b))
    end
    
    # Очередь интервалов для обработки
    intervals = [(a, b)]
    integrals = Float64[]
    errors = Float64[]
    
    total_integral = 0.0
    total_error = 0.0
    
    while !isempty(intervals) && n_subdivisions[] < max_subdivisions
        # Извлекаем следующий интервал
        a_i, b_i = pop!(intervals)
        
        # Вычисляем приближения разными методами
        I_gauss = gauss_quadrature(f_with_count, a_i, b_i)
        I_simpson = simpson_rule(f_with_count, a_i, b_i)
        
        # Оценка погрешности
        error_est = abs(I_gauss - I_simpson)
        
        # Проверка условия точности
        if error_est <= tol * (abs(I_gauss) + 1e-10) * (b_i - a_i) / (b - a)
            # Достаточная точность, добавляем к результату
            push!(integrals, I_gauss)
            push!(errors, error_est)
            total_integral += I_gauss
            total_error += error_est
            n_subdivisions[] += 1
        else
            # Разбиваем интервал пополам
            mid = (a_i + b_i) / 2
            push!(intervals, (a_i, mid))
            push!(intervals, (mid, b_i))
        end
    end
    
    # Проверка сходимости
    converged = n_subdivisions[] < max_subdivisions
    
    # Возвращаем результат в запрошенном формате
    if return_details
        return IntegrationResult(
            float(total_integral),
            error_estimate=float(total_error),
            n_function_calls=n_calls[],
            n_subdivisions=n_subdivisions[],
            convergence=converged,
            details=Dict{Symbol,Any}(
                :subdivisions => n_subdivisions[],
                :interval_integrals => integrals,
                :interval_errors => errors
            )
        )
    else
        return float(total_integral)
    end
end

end # module AdaptiveMethods 