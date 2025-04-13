"""
    BasicMethods

Модуль с базовыми методами численного интегрирования: прямоугольников, трапеций, Симпсона и др.
Методы в этом модуле представляют простые формулы интегрирования без разбиения на подынтервалы.
"""
module BasicMethods

export rectangle_method, midpoint_method, trapezoid_method, simpson_method
export simpson38_method, boole_method, IntegrationResult

"""
    IntegrationResult

Структура для хранения результатов численного интегрирования.

# Поля
- `value::Float64`: значение интеграла
- `error_estimate::Float64`: оценка погрешности (если доступна)
- `n_evaluations::Int`: количество вычислений функции
- `method::String`: название использованного метода

# Пример
```julia
result = IntegrationResult(2.0, 1e-8, 21, "Метод Симпсона")
println("Интеграл: \$(result.value), погрешность: \$(result.error_estimate)")
```
"""
struct IntegrationResult
    value::Float64
    error_estimate::Float64
    n_evaluations::Int
    method::String
end

# Переопределение вывода структуры IntegrationResult
import Base: show
function show(io::IO, result::IntegrationResult)
    println(io, "Результат численного интегрирования:")
    println(io, "  Метод: $(result.method)")
    println(io, "  Значение: $(result.value)")
    println(io, "  Оценка погрешности: $(result.error_estimate)")
    println(io, "  Число вычислений функции: $(result.n_evaluations)")
end

"""
    rectangle_method(f, a, b; position=:left, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом прямоугольников.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `position::Symbol=:left`: положение точки в интервале для вычисления значения функции
  (:left - в левой границе, :right - в правой границе)
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод прямоугольников аппроксимирует подынтегральную функцию константой на всём интервале:
- Для левых прямоугольников: ∫[a,b] f(x) dx ≈ (b-a) * f(a)
- Для правых прямоугольников: ∫[a,b] f(x) dx ≈ (b-a) * f(b)

Погрешность метода имеет порядок O(h), где h = b-a.

# Пример
```julia
f(x) = x^2
result = rectangle_method(f, 0, 1)  # ≈ 0.0, погрешность для этой функции велика
```
"""
function rectangle_method(f::Function, a::Real, b::Real; 
                        position::Symbol=:left, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return rectangle_method(f, b, a, position=position, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Значение функции в зависимости от выбранного положения
    if position == :left
        x = a
        method_name = "Метод левых прямоугольников"
    elseif position == :right
        x = b
        method_name = "Метод правых прямоугольников"
    else
        throw(ArgumentError("Неизвестное положение: $position. Используйте :left или :right"))
    end
    
    # Вычисление интеграла методом прямоугольников
    result = h * f(x)
    
    # Приближенная оценка погрешности
    # Используем разницу между правым и левым прямоугольником как оценку
    error_est = abs(f(b) - f(a)) * h
    
    if return_details
        return IntegrationResult(result, error_est, 1, method_name)
    else
        return result
    end
end

"""
    midpoint_method(f, a, b; return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом средней точки (средних прямоугольников).

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод средней точки аппроксимирует подынтегральную функцию константой на всём интервале,
используя значение функции в середине интервала:
∫[a,b] f(x) dx ≈ (b-a) * f((a+b)/2)

Погрешность метода имеет порядок O(h²), где h = b-a, что лучше, чем у метода прямоугольников.

# Пример
```julia
f(x) = x^2
result = midpoint_method(f, 0, 1)  # ≈ 0.25, точнее метода прямоугольников
```
"""
function midpoint_method(f::Function, a::Real, b::Real; return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return midpoint_method(f, b, a, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Средняя точка интервала
    x_mid = (a + b) / 2
    
    # Вычисление интеграла методом средней точки
    result = h * f(x_mid)
    
    # Приближенная оценка погрешности
    # Сравниваем с методом трапеций для оценки
    trapezoid_result = (h/2) * (f(a) + f(b))
    error_est = abs(result - trapezoid_result) / 3
    
    if return_details
        return IntegrationResult(result, error_est, 1, "Метод средней точки")
    else
        return result
    end
end

"""
    trapezoid_method(f, a, b; return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом трапеций.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод трапеций аппроксимирует подынтегральную функцию линейной функцией:
∫[a,b] f(x) dx ≈ (b-a) * (f(a) + f(b)) / 2

Погрешность метода имеет порядок O(h²), где h = b-a.

# Пример
```julia
f(x) = x^2
result = trapezoid_method(f, 0, 1)  # ≈ 0.5, что отличается от точного значения 1/3
```
"""
function trapezoid_method(f::Function, a::Real, b::Real; return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return trapezoid_method(f, b, a, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Вычисление интеграла методом трапеций
    result = (h / 2) * (f(a) + f(b))
    
    # Сравниваем с методом Симпсона для оценки погрешности
    x_mid = (a + b) / 2
    simpson_result = (h / 6) * (f(a) + 4*f(x_mid) + f(b))
    error_est = abs(result - simpson_result) / 15
    
    if return_details
        return IntegrationResult(result, error_est, 2, "Метод трапеций")
    else
        return result
    end
end

"""
    simpson_method(f, a, b; return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом Симпсона (параболический метод).

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод Симпсона аппроксимирует подынтегральную функцию квадратичной функцией (параболой):
∫[a,b] f(x) dx ≈ (b-a)/6 * [f(a) + 4f((a+b)/2) + f(b)]

Погрешность метода имеет порядок O(h⁴), где h = b-a, что значительно точнее метода трапеций.

# Пример
```julia
f(x) = x^2
result = simpson_method(f, 0, 1)  # ≈ 0.33333, что близко к точному значению 1/3
```
"""
function simpson_method(f::Function, a::Real, b::Real; return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return simpson_method(f, b, a, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Средняя точка
    x_mid = (a + b) / 2
    
    # Вычисление интеграла методом Симпсона
    result = (h / 6) * (f(a) + 4*f(x_mid) + f(b))
    
    # Для оценки погрешности используем более точный метод Симпсона 3/8
    x1 = a + h/3
    x2 = a + 2*h/3
    simpson38_result = (h / 8) * (f(a) + 3*f(x1) + 3*f(x2) + f(b))
    error_est = abs(result - simpson38_result) / 15
    
    if return_details
        return IntegrationResult(result, error_est, 3, "Метод Симпсона")
    else
        return result
    end
end

"""
    simpson38_method(f, a, b; return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом Симпсона 3/8.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод Симпсона 3/8 - это расширение стандартного метода Симпсона с использованием 
дополнительных точек для более точной аппроксимации:
∫[a,b] f(x) dx ≈ (b-a)/8 * [f(a) + 3f(a+h/3) + 3f(a+2h/3) + f(b)]

Погрешность метода имеет порядок O(h⁴), как и у обычного метода Симпсона, 
но в некоторых случаях даёт более точные результаты.

# Пример
```julia
f(x) = x^2
result = simpson38_method(f, 0, 1)  # ≈ 0.33333, что близко к точному значению 1/3
```
"""
function simpson38_method(f::Function, a::Real, b::Real; return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return simpson38_method(f, b, a, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Промежуточные точки
    x1 = a + h/3
    x2 = a + 2*h/3
    
    # Вычисление интеграла методом Симпсона 3/8
    result = (h / 8) * (f(a) + 3*f(x1) + 3*f(x2) + f(b))
    
    # Оценка погрешности
    # Используем метод Буля (Boole's rule) для сравнения
    x_mid = (a + b) / 2
    x1_mid = (a + x_mid) / 2
    x2_mid = (x_mid + b) / 2
    boole_result = (h / 90) * (7*f(a) + 32*f(x1_mid) + 12*f(x_mid) + 32*f(x2_mid) + 7*f(b))
    error_est = abs(result - boole_result) / 15
    
    if return_details
        return IntegrationResult(result, error_est, 4, "Метод Симпсона 3/8")
    else
        return result
    end
end

"""
    boole_method(f, a, b; return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом Буля (Boole's rule).

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод Буля (Boole's rule) - это высокоточный метод численного интегрирования,
использующий полином 4-й степени для аппроксимации:
∫[a,b] f(x) dx ≈ (b-a)/90 * [7f(a) + 32f(a+h/4) + 12f(a+h/2) + 32f(a+3h/4) + 7f(b)]

Погрешность метода имеет порядок O(h⁶), где h = b-a, что делает его значительно 
точнее метода Симпсона для гладких функций.

# Пример
```julia
f(x) = x^2
result = boole_method(f, 0, 1)  # ≈ 0.33333, очень близко к точному значению 1/3
```
"""
function boole_method(f::Function, a::Real, b::Real; return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return boole_method(f, b, a, return_details=return_details)
    end
    
    # Ширина интервала
    h = b - a
    
    # Промежуточные точки
    x_mid = (a + b) / 2
    x1 = a + h/4
    x2 = a + 3*h/4
    
    # Вычисление интеграла методом Буля
    result = (h / 90) * (7*f(a) + 32*f(x1) + 12*f(x_mid) + 32*f(x2) + 7*f(b))
    
    # Для оценки погрешности используем метод с меньшим размером шага
    h_half = h / 2
    a_mid = a + h_half
    result1 = (h_half / 90) * (7*f(a) + 32*f(a + h_half/4) + 12*f(a + h_half/2) + 32*f(a + 3*h_half/4) + 7*f(a_mid))
    result2 = (h_half / 90) * (7*f(a_mid) + 32*f(a_mid + h_half/4) + 12*f(a_mid + h_half/2) + 32*f(a_mid + 3*h_half/4) + 7*f(b))
    error_est = abs(result - (result1 + result2)) / 63
    
    if return_details
        return IntegrationResult(result, error_est, 5, "Метод Буля")
    else
        return result
    end
end

end # module BasicMethods 