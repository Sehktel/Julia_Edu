"""
    BasicMethods

Модуль, реализующий основные методы численного дифференцирования функций одной переменной.
"""
module BasicMethods

using ..DifferentiationStructures

export forward_difference, backward_difference, central_difference, richardson_extrapolation

"""
    forward_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)

Вычисляет приближённое значение производной функции `f` порядка `order` в точке `x` 
с использованием метода конечной разности вперёд.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `h::Real=1e-8`: шаг дифференцирования
- `order::Int=1`: порядок производной

# Возвращаемое значение
- Приближённое значение производной порядка `order`

# Примеры
```julia
f(x) = x^2
forward_difference(f, 2.0)  # Приблизительно 4.0
```

# Математическое описание
Для первой производной:
f'(x) ≈ [f(x+h) - f(x)] / h + O(h)

Для второй производной:
f''(x) ≈ [f(x+2h) - 2f(x+h) + f(x)] / h^2 + O(h)
"""
function forward_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)
    if order < 1
        throw(ArgumentError("Порядок производной должен быть положительным целым числом"))
    end
    
    if order == 1
        return (f(x + h) - f(x)) / h
    elseif order == 2
        return (f(x + 2*h) - 2*f(x + h) + f(x)) / h^2
    else
        # Для производных высших порядков используем рекурсию
        return (forward_difference(f, x + h, h, order - 1) - 
                forward_difference(f, x, h, order - 1)) / h
    end
end

"""
    backward_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)

Вычисляет приближённое значение производной функции `f` порядка `order` в точке `x` 
с использованием метода конечной разности назад.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `h::Real=1e-8`: шаг дифференцирования
- `order::Int=1`: порядок производной

# Возвращаемое значение
- Приближённое значение производной порядка `order`

# Примеры
```julia
f(x) = x^2
backward_difference(f, 2.0)  # Приблизительно 4.0
```

# Математическое описание
Для первой производной:
f'(x) ≈ [f(x) - f(x-h)] / h + O(h)

Для второй производной:
f''(x) ≈ [f(x) - 2f(x-h) + f(x-2h)] / h^2 + O(h)
"""
function backward_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)
    if order < 1
        throw(ArgumentError("Порядок производной должен быть положительным целым числом"))
    end
    
    if order == 1
        return (f(x) - f(x - h)) / h
    elseif order == 2
        return (f(x) - 2*f(x - h) + f(x - 2*h)) / h^2
    else
        # Для производных высших порядков используем рекурсию
        return (backward_difference(f, x, h, order - 1) - 
                backward_difference(f, x - h, h, order - 1)) / h
    end
end

"""
    central_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)

Вычисляет приближённое значение производной функции `f` порядка `order` в точке `x` 
с использованием метода центральной разности.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `h::Real=1e-8`: шаг дифференцирования
- `order::Int=1`: порядок производной

# Возвращаемое значение
- Приближённое значение производной порядка `order`

# Примеры
```julia
f(x) = x^2
central_difference(f, 2.0)  # Приблизительно 4.0
```

# Математическое описание
Для первой производной:
f'(x) ≈ [f(x+h) - f(x-h)] / (2h) + O(h^2)

Для второй производной:
f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2 + O(h^2)
"""
function central_difference(f::Function, x::Real, h::Real=1e-8, order::Int=1)
    if order < 1
        throw(ArgumentError("Порядок производной должен быть положительным целым числом"))
    end
    
    if order == 1
        return (f(x + h) - f(x - h)) / (2*h)
    elseif order == 2
        return (f(x + h) - 2*f(x) + f(x - h)) / h^2
    elseif order == 3
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2*h^3)
    elseif order == 4
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / h^4
    else
        # Для производных очень высоких порядков используем рекурсию
        # Это менее точно, но более общий метод
        return (central_difference(f, x + h/2, h, order - 1) - 
                central_difference(f, x - h/2, h, order - 1)) / h
    end
end

"""
    richardson_extrapolation(f::Function, x::Real; order::Int=1, initial_h::Real=0.1, levels::Int=4)

Вычисляет производную функции `f` в точке `x` с использованием экстраполяции Ричардсона
для повышения точности численного дифференцирования.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `order::Int=1`: порядок вычисляемой производной
- `initial_h::Real=0.1`: начальный шаг дифференцирования
- `levels::Int=4`: количество уровней экстраполяции

# Возвращаемое значение
- `derivative::Float64`: уточненное значение производной

# Пример
```julia
f(x) = sin(x)
df_dx = richardson_extrapolation(f, π/4) # Уточненное значение cos(π/4)
```

# Примечания
- Алгоритм использует таблицу Романовского для экстраполяции значений,
  последовательно уменьшая шаг дифференцирования и применяя формулу экстраполяции
- Погрешность метода может быть существенно меньше, чем у простых разностных методов
"""
function richardson_extrapolation(
    f::Function, 
    x::Real; 
    order::Int=1, 
    initial_h::Real=0.1, 
    levels::Int=4
)
    # Создаем таблицу для хранения приближений
    extrapolation_table = zeros(levels, levels)
    
    # Заполняем первый столбец значениями на разных шагах
    h = initial_h
    for i in 1:levels
        extrapolation_table[i, 1] = central_difference(f, x, h, order)
        h /= 2.0  # Уменьшаем шаг вдвое для следующей итерации
    end
    
    # Выполняем экстраполяцию по формуле Ричардсона
    for j in 2:levels
        for i in 1:(levels-j+1)
            # Формула экстраполяции для метода центральных разностей (порядок O(h²))
            factor = 4^(j-1)
            extrapolation_table[i, j] = (factor * extrapolation_table[i+1, j-1] - 
                                        extrapolation_table[i, j-1]) / (factor - 1)
        end
    end
    
    # Возвращаем наилучшее приближение
    return extrapolation_table[1, levels]
end

"""
    differentiate(f::Function, x::Real; 
                 method::DifferentiationMethod=central, 
                 h::Real=0.0, 
                 order::Int=1)

Унифицированный интерфейс для вычисления производной функции `f` в точке `x`
с использованием указанного метода численного дифференцирования.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `method::DifferentiationMethod=central`: метод дифференцирования
- `h::Real=0.0`: шаг дифференцирования (если 0, будет выбран автоматически)
- `order::Int=1`: порядок производной

# Возвращаемое значение
- `derivative::Float64`: приближенное значение производной

# Пример
```julia
f(x) = sin(x)
df_dx = differentiate(f, π/4, method=central)
```
"""
function differentiate(
    f::Function, 
    x::Real; 
    method::DifferentiationMethod=central, 
    h::Real=0.0, 
    order::Int=1
)
    # Автоматический выбор шага, если не задан
    if h <= 0.0
        if method == central
            h = cbrt(eps(typeof(x)))  # Для центральной разности оптимально h ~ ε^(1/3)
        else
            h = sqrt(eps(typeof(x)))  # Для остальных методов h ~ ε^(1/2)
        end
    end
    
    # Выбор метода дифференцирования
    if method == forward
        return forward_difference(f, x, h, order)
    elseif method == backward
        return backward_difference(f, x, h, order)
    elseif method == central
        return central_difference(f, x, h, order)
    elseif method == richardson
        return richardson_extrapolation(f, x, order=order, initial_h=h, levels=4)
    else
        throw(ArgumentError("Неизвестный метод дифференцирования"))
    end
end

# Экспортируем унифицированную функцию
export differentiate

# Обобщенная функция для производной второго порядка
"""
    second_derivative(f::Function, x::Real, h::Real=1e-6, method::Symbol=:central)

Вычисляет приближённое значение второй производной функции `f` в точке `x` 
используя указанный метод.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется вторая производная
- `h::Real=1e-6`: шаг дифференцирования
- `method::Symbol=:central`: метод дифференцирования (:forward, :backward, :central)

# Возвращаемое значение
- Приближённое значение второй производной
"""
function second_derivative(f::Function, x::Real, h::Real=1e-6, method::Symbol=:central)
    if method == :forward
        return forward_difference(f, x, h, 2)
    elseif method == :backward
        return backward_difference(f, x, h, 2)
    elseif method == :central
        return central_difference(f, x, h, 2)
    else
        throw(ArgumentError("Неизвестный метод: $method. Используйте :forward, :backward или :central"))
    end
end

# Функция для вычисления производной произвольного порядка
"""
    higher_derivative(f::Function, x::Real, order::Int, h::Real=1e-6, method::Symbol=:central)

Вычисляет приближённое значение производной функции `f` произвольного порядка `order` в точке `x`
используя указанный метод.

# Аргументы
- `f::Function`: дифференцируемая функция
- `x::Real`: точка, в которой вычисляется производная
- `order::Int`: порядок производной (должен быть положительным)
- `h::Real=1e-6`: шаг дифференцирования
- `method::Symbol=:central`: метод дифференцирования (:forward, :backward, :central)

# Возвращаемое значение
- Приближённое значение производной указанного порядка
"""
function higher_derivative(f::Function, x::Real, order::Int, h::Real=1e-6, method::Symbol=:central)
    if order < 1
        throw(ArgumentError("Порядок производной должен быть положительным целым числом"))
    end
    
    if method == :forward
        return forward_difference(f, x, h, order)
    elseif method == :backward
        return backward_difference(f, x, h, order)
    elseif method == :central
        return central_difference(f, x, h, order)
    else
        throw(ArgumentError("Неизвестный метод: $method. Используйте :forward, :backward или :central"))
    end
end

end # module BasicMethods 