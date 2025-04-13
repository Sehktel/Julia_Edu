"""
    FiniteDifferences

Реализация методов конечных разностей для численного дифференцирования 
функций одной переменной.
"""

"""
    DifferentiationResult

Структура для хранения результатов численного дифференцирования.

# Поля
- `value::Float64`: Значение производной
- `error_estimate::Float64`: Оценка погрешности вычисления
- `step_size::Float64`: Использованный шаг дифференцирования

# Пример
```julia
result = forward_difference(sin, 1.0, error_estimate=true)
println("f'(1.0) ≈ \$(result.value) ± \$(result.error_estimate)")
```
"""
struct DifferentiationResult
    value::Float64
    error_estimate::Float64
    step_size::Float64
end

Base.show(io::IO, r::DifferentiationResult) = 
    print(io, "$(r.value) ± $(r.error_estimate) (шаг = $(r.step_size))")

"""
    forward_difference(f, x; step=1e-6, error_estimate=false)

Вычисляет первую производную функции `f` в точке `x` используя
метод конечных разностей с шагом вперед.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `step::Real=1e-6`: Шаг дифференцирования
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Теоретическая информация
Формула разностной производной вперёд:
    f'(x) ≈ [f(x + h) - f(x)] / h
    
Погрешность метода имеет порядок O(h).

# Пример
```julia
f(x) = x^2
df_dx = forward_difference(f, 2.0)  # Должно быть близко к 4.0
```
"""
function forward_difference(f::Function, x::Real; step::Real=1e-6, error_estimate::Bool=false)
    # Вычисление производной методом конечных разностей
    h = Float64(step)
    df = (f(x + h) - f(x)) / h
    
    if error_estimate
        # Оценка погрешности: O(h) для метода вперёд
        error_est = abs(h)
        return DifferentiationResult(df, error_est, h)
    else
        return df
    end
end

"""
    backward_difference(f, x; step=1e-6, error_estimate=false)

Вычисляет первую производную функции `f` в точке `x` используя
метод конечных разностей с шагом назад.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `step::Real=1e-6`: Шаг дифференцирования
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Теоретическая информация
Формула разностной производной назад:
    f'(x) ≈ [f(x) - f(x - h)] / h
    
Погрешность метода имеет порядок O(h).

# Пример
```julia
f(x) = x^2
df_dx = backward_difference(f, 2.0)  # Должно быть близко к 4.0
```
"""
function backward_difference(f::Function, x::Real; step::Real=1e-6, error_estimate::Bool=false)
    # Вычисление производной методом конечных разностей
    h = Float64(step)
    df = (f(x) - f(x - h)) / h
    
    if error_estimate
        # Оценка погрешности: O(h) для метода назад
        error_est = abs(h)
        return DifferentiationResult(df, error_est, h)
    else
        return df
    end
end

"""
    central_difference(f, x; step=1e-6, error_estimate=false)

Вычисляет первую производную функции `f` в точке `x` используя
центральный метод конечных разностей.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `step::Real=1e-6`: Шаг дифференцирования
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Теоретическая информация
Формула центральной разностной производной:
    f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    
Погрешность метода имеет порядок O(h²), что лучше, чем у односторонних методов.

# Пример
```julia
f(x) = x^2
df_dx = central_difference(f, 2.0)  # Должно быть близко к 4.0
```
"""
function central_difference(f::Function, x::Real; step::Real=1e-6, error_estimate::Bool=false)
    # Вычисление производной центральным методом конечных разностей
    h = Float64(step)
    df = (f(x + h) - f(x - h)) / (2h)
    
    if error_estimate
        # Оценка погрешности: O(h²) для центрального метода
        error_est = h^2
        return DifferentiationResult(df, error_est, h)
    else
        return df
    end
end

"""
    second_derivative(f, x; step=1e-4, error_estimate=false)

Вычисляет вторую производную функции `f` в точке `x` используя
метод центральных конечных разностей.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `step::Real=1e-4`: Шаг дифференцирования
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение второй производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Теоретическая информация
Формула для второй производной:
    f''(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h²
    
Погрешность метода имеет порядок O(h²).

# Пример
```julia
f(x) = x^3
d2f_dx2 = second_derivative(f, 2.0)  # Должно быть близко к 12.0
```
"""
function second_derivative(f::Function, x::Real; step::Real=1e-4, error_estimate::Bool=false)
    # Вычисление второй производной методом центральных разностей
    h = Float64(step)
    d2f = (f(x + h) - 2f(x) + f(x - h)) / (h^2)
    
    if error_estimate
        # Оценка погрешности: O(h²) для данной формулы второй производной
        error_est = h^2
        return DifferentiationResult(d2f, error_est, h)
    else
        return d2f
    end
end

"""
    higher_order_derivative(f, x, order; step=1e-4, error_estimate=false)

Вычисляет производную заданного порядка для функции `f` в точке `x`.
Использует рекурсивное применение метода центральных разностей.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `order::Integer`: Порядок производной (1, 2, 3, ...)
- `step::Real=1e-4`: Шаг дифференцирования
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Пример
```julia
f(x) = x^4
d3f_dx3 = higher_order_derivative(f, 2.0, 3)  # Должно быть близко к 144.0
```
"""
function higher_order_derivative(f::Function, x::Real, order::Integer; 
                                 step::Real=1e-4, error_estimate::Bool=false)
    # Проверка аргумента order
    if order < 1
        throw(ArgumentError("Порядок производной должен быть положительным числом"))
    end
    
    # Для первой и второй производных используем стандартные функции
    if order == 1
        return central_difference(f, x, step=step, error_estimate=error_estimate)
    elseif order == 2
        return second_derivative(f, x, step=step, error_estimate=error_estimate)
    end
    
    # Для высших производных используем рекурсивный подход
    h = Float64(step)
    
    # Создаем функцию для производной на порядок ниже
    df_prev(t) = higher_order_derivative(f, t, order-1, step=step, error_estimate=false)
    
    # Вычисляем производную с помощью центрального метода
    dⁿf = central_difference(df_prev, x, step=h, error_estimate=false)
    
    if error_estimate
        # Оценка погрешности: при рекурсивном применении ошибка накапливается
        # Порядок погрешности O(h²) для центрального метода
        error_est = h^2 * order
        return DifferentiationResult(dⁿf, error_est, h)
    else
        return dⁿf
    end
end 