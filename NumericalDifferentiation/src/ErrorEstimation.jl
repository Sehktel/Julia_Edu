"""
    ErrorEstimation

Модуль, предоставляющий методы для оценки погрешности численного дифференцирования.
"""
module ErrorEstimation

using ..DifferentiationStructures
using ..BasicMethods

export estimate_error, adaptive_differentiation, optimal_step

"""
    estimate_error(f::Function, x::Real, h::Real, method::DifferentiationMethod=central, order::Int=1)

Оценивает погрешность численного дифференцирования функции `f` в точке `x`
с использованием метода Рунге.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `h::Real`: шаг дифференцирования
- `method::DifferentiationMethod=central`: используемый метод дифференцирования
- `order::Int=1`: порядок вычисляемой производной

# Возвращаемое значение
- `error_estimate::Float64`: оценка абсолютной погрешности

# Пример
```julia
f(x) = sin(x)
error_val = estimate_error(f, π/4, 0.01)
```

# Примечания
- Использует метод Рунге, вычисляя производную с шагами h и h/2
- Погрешность оценивается по формуле |f'_h - f'_{h/2}|/(2^p - 1), где p - порядок точности метода
"""
function estimate_error(
    f::Function, 
    x::Real, 
    h::Real, 
    method::DifferentiationMethod=central, 
    order::Int=1
)
    # Вычисляем производную с исходным шагом
    deriv_h = differentiate(f, x, method=method, h=h, order=order)
    
    # Вычисляем производную с уменьшенным вдвое шагом
    deriv_h_half = differentiate(f, x, method=method, h=h/2, order=order)
    
    # Определяем порядок точности метода
    p = (method == central) ? 2 : 1
    
    # Формула оценки погрешности по методу Рунге
    error_estimate = abs(deriv_h - deriv_h_half) / (2^p - 1)
    
    return error_estimate
end

"""
    optimal_step(f::Function, x::Real, method::DifferentiationMethod=central, order::Int=1)

Оценивает оптимальный шаг для численного дифференцирования функции с минимальной погрешностью.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `method::DifferentiationMethod=central`: используемый метод дифференцирования
- `order::Int=1`: порядок вычисляемой производной

# Возвращаемое значение
- `h_opt::Float64`: оптимальный шаг дифференцирования

# Пример
```julia
f(x) = sin(x)
h_optimal = optimal_step(f, π/4)
```

# Примечания
- Предполагает баланс между ошибкой усечения и ошибкой округления
- Для метода центральных разностей оптимальный шаг h ~ ε^(1/3)
- Для методов односторонних разностей h ~ ε^(1/2)
- Учитывает масштаб функции для автоматической настройки шага
"""
function optimal_step(
    f::Function, 
    x::Real, 
    method::DifferentiationMethod=central, 
    order::Int=1
)
    # Машинная точность для типа значения x
    eps_x = eps(typeof(float(x)))
    
    # Масштаб функции в окрестности точки x
    f_scale = max(abs(f(x)), 1.0)
    
    # Определяем базовый шаг в зависимости от метода
    if method == central
        # Для центральной разности h ~ ε^(1/3)
        h_base = cbrt(eps_x)
    else
        # Для других методов h ~ ε^(1/2)
        h_base = sqrt(eps_x)
    end
    
    # Корректируем шаг с учетом масштаба функции и порядка производной
    h_opt = h_base * max(abs(x), 1.0) / f_scale^(1/order)
    
    return h_opt
end

"""
    adaptive_differentiation(f::Function, x::Real; 
                           method::DifferentiationMethod=central,
                           order::Int=1,
                           tol::Real=1e-6,
                           max_iters::Int=10)

Вычисляет производную функции `f` с адаптивным выбором шага для достижения заданной точности.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `method::DifferentiationMethod=central`: метод дифференцирования
- `order::Int=1`: порядок производной
- `tol::Real=1e-6`: требуемая относительная точность
- `max_iters::Int=10`: максимальное количество итераций

# Возвращаемое значение
- `result::DifferentiationResult`: структура с результатом дифференцирования

# Пример
```julia
f(x) = sin(x)
result = adaptive_differentiation(f, π/4)
```
"""
function adaptive_differentiation(
    f::Function, 
    x::Real; 
    method::DifferentiationMethod=central,
    order::Int=1,
    tol::Real=1e-6,
    max_iters::Int=10
)
    # Начальное приближение шага
    h = optimal_step(f, x, method, order)
    
    # Первое вычисление производной
    deriv = differentiate(f, x, method=method, h=h, order=order)
    error_est = estimate_error(f, x, h, method, order)
    
    # Итеративный процесс уточнения шага
    iter = 1
    while error_est > tol * abs(deriv) && iter < max_iters
        # Уменьшаем шаг вдвое
        h /= 2.0
        
        # Перевычисляем производную и оценку погрешности
        deriv = differentiate(f, x, method=method, h=h, order=order)
        error_est = estimate_error(f, x, h, method, order)
        
        iter += 1
    end
    
    # Создаем структуру с результатом
    result = DifferentiationResult(deriv, error_est, h, method)
    
    return result
end

end # module ErrorEstimation 