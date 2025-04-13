"""
# Основные операции с функциями двух переменных

Данный модуль реализует базовые операции для работы с функциями двух переменных,
включая вычисление значений функции, градиента и частных производных.
"""

using LinearAlgebra

"""
    evaluate(f, x, y)

Вычисляет значение функции двух переменных `f` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x::Real`: координата x
- `y::Real`: координата y

# Возвращаемое значение
- `Real`: значение функции `f(x, y)`

# Примеры
```julia
f(x, y) = x^2 + y^2
evaluate(f, 1.0, 2.0)  # вернет 5.0
```
"""
function evaluate(f::Function, x::Real, y::Real)
    return f(x, y)
end

"""
    evaluate(f, point)

Вычисляет значение функции двух переменных `f` в точке `point`.

# Аргументы
- `f::Function`: функция двух переменных
- `point::Vector{<:Real}`: точка [x, y]

# Возвращаемое значение
- `Real`: значение функции `f(point[1], point[2])`

# Примеры
```julia
f(x, y) = x^2 + y^2
evaluate(f, [1.0, 2.0])  # вернет 5.0
```
"""
function evaluate(f::Function, point::Vector{<:Real})
    if length(point) != 2
        throw(ArgumentError("Точка должна иметь две координаты"))
    end
    return f(point[1], point[2])
end

"""
    partial_derivative(f, var, x, y; h=1e-6)

Вычисляет частную производную функции двух переменных `f` по переменной `var` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `var::Symbol`: переменная, по которой берется производная (`:x` или `:y`)
- `x::Real`: координата x
- `y::Real`: координата y
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Real`: значение частной производной в точке `(x, y)`

# Примеры
```julia
f(x, y) = x^2 + y^2
partial_derivative(f, :x, 1.0, 2.0)  # вернет ≈ 2.0 (∂f/∂x = 2x)
partial_derivative(f, :y, 1.0, 2.0)  # вернет ≈ 4.0 (∂f/∂y = 2y)
```
"""
function partial_derivative(f::Function, var::Symbol, x::Real, y::Real; h::Real=1e-6)
    if var == :x
        # Центральная разностная схема для ∂f/∂x
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elseif var == :y
        # Центральная разностная схема для ∂f/∂y
        return (f(x, y + h) - f(x, y - h)) / (2 * h)
    else
        throw(ArgumentError("Переменная должна быть :x или :y"))
    end
end

"""
    partial_derivative(f, var, point; h=1e-6)

Вычисляет частную производную функции двух переменных `f` по переменной `var` в точке `point`.

# Аргументы
- `f::Function`: функция двух переменных
- `var::Symbol`: переменная, по которой берется производная (`:x` или `:y`)
- `point::Vector{<:Real}`: точка [x, y]
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Real`: значение частной производной в точке `point`

# Примеры
```julia
f(x, y) = x^2 + y^2
partial_derivative(f, :x, [1.0, 2.0])  # вернет ≈ 2.0 (∂f/∂x = 2x)
```
"""
function partial_derivative(f::Function, var::Symbol, point::Vector{<:Real}; h::Real=1e-6)
    if length(point) != 2
        throw(ArgumentError("Точка должна иметь две координаты"))
    end
    return partial_derivative(f, var, point[1], point[2], h=h)
end

"""
    gradient(f, x, y; h=1e-6)

Вычисляет градиент функции двух переменных `f` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x::Real`: координата x
- `y::Real`: координата y
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Vector{Float64}`: вектор градиента [∂f/∂x, ∂f/∂y]

# Примеры
```julia
f(x, y) = x^2 + y^2
gradient(f, 1.0, 2.0)  # вернет ≈ [2.0, 4.0]
```
"""
function gradient(f::Function, x::Real, y::Real; h::Real=1e-6)
    df_dx = partial_derivative(f, :x, x, y, h=h)
    df_dy = partial_derivative(f, :y, x, y, h=h)
    return [df_dx, df_dy]
end

"""
    gradient(f, point; h=1e-6)

Вычисляет градиент функции двух переменных `f` в точке `point`.

# Аргументы
- `f::Function`: функция двух переменных
- `point::Vector{<:Real}`: точка [x, y]
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Vector{Float64}`: вектор градиента [∂f/∂x, ∂f/∂y]

# Примеры
```julia
f(x, y) = x^2 + y^2
gradient(f, [1.0, 2.0])  # вернет ≈ [2.0, 4.0]
```
"""
function gradient(f::Function, point::Vector{<:Real}; h::Real=1e-6)
    if length(point) != 2
        throw(ArgumentError("Точка должна иметь две координаты"))
    end
    return gradient(f, point[1], point[2], h=h)
end

"""
    gradient_magnitude(f, x, y; h=1e-6)

Вычисляет величину градиента функции двух переменных `f` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x::Real`: координата x
- `y::Real`: координата y
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Float64`: величина градиента (норма вектора градиента)

# Примеры
```julia
f(x, y) = x^2 + y^2
gradient_magnitude(f, 1.0, 2.0)  # вернет ≈ √(2.0^2 + 4.0^2) ≈ 4.47
```
"""
function gradient_magnitude(f::Function, x::Real, y::Real; h::Real=1e-6)
    grad = gradient(f, x, y, h=h)
    return norm(grad)
end

"""
    directional_derivative(f, x, y, direction; h=1e-6)

Вычисляет производную по направлению функции двух переменных `f` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x::Real`: координата x
- `y::Real`: координата y
- `direction::Vector{<:Real}`: вектор направления [dx, dy] (не обязательно единичный)
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Float64`: производная по направлению (проекция градиента на единичный вектор направления)

# Примеры
```julia
f(x, y) = x^2 + y^2
directional_derivative(f, 1.0, 2.0, [1.0, 0.0])  # производная по x, вернет ≈ 2.0
directional_derivative(f, 1.0, 2.0, [0.0, 1.0])  # производная по y, вернет ≈ 4.0
directional_derivative(f, 1.0, 2.0, [1.0, 1.0])  # по диагонали, вернет ≈ 4.24
```
"""
function directional_derivative(f::Function, x::Real, y::Real, direction::Vector{<:Real}; h::Real=1e-6)
    if length(direction) != 2
        throw(ArgumentError("Вектор направления должен иметь две компоненты"))
    end
    
    # Нормализуем вектор направления
    dir_norm = norm(direction)
    if dir_norm ≈ 0
        throw(ArgumentError("Вектор направления не может быть нулевым"))
    end
    unit_direction = direction ./ dir_norm
    
    # Вычисляем градиент и проекцию на направление
    grad = gradient(f, x, y, h=h)
    return dot(grad, unit_direction)
end

"""
    hessian(f, x, y; h=1e-6)

Вычисляет матрицу Гессе (вторых производных) функции двух переменных `f` в точке `(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x::Real`: координата x
- `y::Real`: координата y
- `h::Real=1e-6`: шаг для численного дифференцирования

# Возвращаемое значение
- `Matrix{Float64}`: матрица Гессе размера 2×2 [∂²f/∂x² ∂²f/∂x∂y; ∂²f/∂y∂x ∂²f/∂y²]

# Примеры
```julia
f(x, y) = x^2 + y^2
hessian(f, 1.0, 2.0)  # вернет ≈ [2.0 0.0; 0.0 2.0]
```
"""
function hessian(f::Function, x::Real, y::Real; h::Real=1e-6)
    # Вторая производная по x
    d2f_dx2 = (f(x + h, y) - 2*f(x, y) + f(x - h, y)) / (h^2)
    
    # Вторая производная по y
    d2f_dy2 = (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / (h^2)
    
    # Смешанная производная ∂²f/∂x∂y
    d2f_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h^2)
    
    return [d2f_dx2 d2f_dxdy; d2f_dxdy d2f_dy2]
end 