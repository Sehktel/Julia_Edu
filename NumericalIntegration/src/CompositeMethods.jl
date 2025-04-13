"""
    CompositeMethods

Модуль с составными методами численного интегрирования,
которые разбивают исходный интервал на подынтервалы для повышения точности.
"""
module CompositeMethods

using ..BasicMethods: IntegrationResult

export composite_rectangle, composite_midpoint, composite_trapezoid, composite_simpson, romberg_method

"""
    composite_rectangle(f, a, b; n=100, position=:left, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] составным методом прямоугольников
с разбиением на `n` подынтервалов.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n::Int=100`: число подынтервалов разбиения
- `position::Symbol=:left`: положение прямоугольников (:left, :right, :midpoint)
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Составной метод прямоугольников разбивает интервал [`a`, `b`] на `n` равных подынтервалов
и применяет метод прямоугольников к каждому из них:

- Для левых прямоугольников: 
  ∫[a,b] f(x) dx ≈ h * ∑(i=0 to n-1) f(a + i*h)
  
- Для правых прямоугольников:
  ∫[a,b] f(x) dx ≈ h * ∑(i=1 to n) f(a + i*h)
  
- Для средних прямоугольников:
  ∫[a,b] f(x) dx ≈ h * ∑(i=0 to n-1) f(a + (i+0.5)*h)

где h = (b-a)/n - ширина каждого подынтервала.

Погрешность составного метода с `n` подынтервалами имеет порядок O(h), O(h^2) или O(h^3)
в зависимости от положения точек (left, right или midpoint).

# Пример
```julia
f(x) = sin(x)
result = composite_rectangle(f, 0, π, n=1000, position=:midpoint)  # Хорошее приближение к 2.0
```
"""
function composite_rectangle(f::Function, a::Real, b::Real; 
                          n::Int=100, position::Symbol=:left, 
                          return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return composite_rectangle(f, b, a, n=n, position=position, return_details=return_details)
    end
    if n <= 0
        throw(ArgumentError("Число подынтервалов должно быть положительным"))
    end
    
    # Ширина подынтервала
    h = (b - a) / n
    
    # Вычисление интеграла методом прямоугольников
    # в зависимости от выбранного положения
    if position == :left
        # Левые прямоугольники
        result = sum(f(a + i*h) for i in 0:(n-1)) * h
        method_name = "Составной метод левых прямоугольников"
        # Для оценки погрешности используем метод с большим числом подынтервалов
        error_est = abs(result - composite_rectangle(f, a, b, n=2*n, position=position)) / 3
        n_evals = n
    elseif position == :right
        # Правые прямоугольники
        result = sum(f(a + i*h) for i in 1:n) * h
        method_name = "Составной метод правых прямоугольников"
        # Для оценки погрешности используем метод с большим числом подынтервалов
        error_est = abs(result - composite_rectangle(f, a, b, n=2*n, position=position)) / 3
        n_evals = n
    elseif position == :midpoint
        # Средние прямоугольники (средние точки)
        result = sum(f(a + (i+0.5)*h) for i in 0:(n-1)) * h
        method_name = "Составной метод средних прямоугольников"
        # Для оценки погрешности сравниваем с методом трапеций
        trapz_result = composite_trapezoid(f, a, b, n=n)
        error_est = abs(result - trapz_result) / 3
        n_evals = n
    else
        throw(ArgumentError("Неизвестное положение: $position. Используйте :left, :right или :midpoint"))
    end
    
    if return_details
        return IntegrationResult(result, error_est, n_evals, method_name)
    else
        return result
    end
end

"""
    composite_midpoint(f, a, b; n=100, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] составным методом средних прямоугольников
с разбиением на `n` подынтервалов.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n::Int=100`: число подынтервалов разбиения
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Пример
```julia
f(x) = x^2
result = composite_midpoint(f, 0, 1, n=100)  # Приближение к 1/3 ≈ 0.333...
```
"""
function composite_midpoint(f::Function, a::Real, b::Real; 
                          n::Int=100, return_details::Bool=false)
    return composite_rectangle(f, a, b, n=n, position=:midpoint, return_details=return_details)
end

"""
    composite_trapezoid(f, a, b; n=100, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] составным методом трапеций
с разбиением на `n` подынтервалов.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n::Int=100`: число подынтервалов разбиения
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Составной метод трапеций разбивает интервал [`a`, `b`] на `n` равных подынтервалов
и применяет метод трапеций к каждому из них:
∫[a,b] f(x) dx ≈ h/2 * [f(a) + 2∑(i=1 to n-1)f(a + i*h) + f(b)]

где h = (b-a)/n - ширина каждого подынтервала.

Погрешность составного метода трапеций имеет порядок O(h²).

# Пример
```julia
f(x) = sin(x)
result = composite_trapezoid(f, 0, π, n=100)  # Хорошее приближение к 2.0
```
"""
function composite_trapezoid(f::Function, a::Real, b::Real; 
                           n::Int=100, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return composite_trapezoid(f, b, a, n=n, return_details=return_details)
    end
    if n <= 0
        throw(ArgumentError("Число подынтервалов должно быть положительным"))
    end
    
    # Ширина подынтервала
    h = (b - a) / n
    
    # Краевые значения
    sum_val = f(a) + f(b)
    
    # Внутренние точки (каждая учитывается дважды)
    for i in 1:(n-1)
        sum_val += 2 * f(a + i*h)
    end
    
    # Результат
    result = (h / 2) * sum_val
    
    # Оценка погрешности
    if return_details
        # Сравниваем с методом Симпсона для оценки
        simpson_result = composite_simpson(f, a, b, n=n)
        error_est = abs(result - simpson_result) / 15
        method_name = "Составной метод трапеций"
        n_evals = n + 1
        
        return IntegrationResult(result, error_est, n_evals, method_name)
    else
        return result
    end
end

"""
    composite_simpson(f, a, b; n=100, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] составным методом Симпсона
с разбиением на `n` подынтервалов.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `n::Int=100`: число подынтервалов разбиения (должно быть чётным)
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Составной метод Симпсона разбивает интервал [`a`, `b`] на `n` равных подынтервалов
и применяет метод Симпсона к каждому из них:
∫[a,b] f(x) dx ≈ h/3 * [f(a) + 4∑(i=0 to n/2-1)f(a + (2i+1)*h) + 2∑(i=1 to n/2-1)f(a + 2i*h) + f(b)]

где h = (b-a)/n - ширина каждого подынтервала.

Погрешность составного метода Симпсона имеет порядок O(h⁴).

# Пример
```julia
f(x) = x^2
result = composite_simpson(f, 0, 1, n=100)  # Очень точное приближение к 1/3
```
"""
function composite_simpson(f::Function, a::Real, b::Real; 
                         n::Int=100, return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return composite_simpson(f, b, a, n=n, return_details=return_details)
    end
    if n <= 0
        throw(ArgumentError("Число подынтервалов должно быть положительным"))
    end
    
    # Обеспечиваем чётное число подынтервалов для метода Симпсона
    if n % 2 != 0
        n = n + 1
    end
    
    # Ширина подынтервала
    h = (b - a) / n
    
    # Краевые значения
    sum_val = f(a) + f(b)
    
    # Точки с коэффициентом 4 (нечётные индексы)
    for i in 1:2:(n-1)
        sum_val += 4 * f(a + i*h)
    end
    
    # Точки с коэффициентом 2 (чётные индексы, кроме крайних)
    for i in 2:2:(n-2)
        sum_val += 2 * f(a + i*h)
    end
    
    # Результат
    result = (h / 3) * sum_val
    
    # Оценка погрешности
    if return_details
        # Используем метод с большим числом подынтервалов для оценки ошибки
        refined_result = composite_simpson(f, a, b, n=2*n)
        error_est = abs(result - refined_result) / 15
        method_name = "Составной метод Симпсона"
        n_evals = n + 1
        
        return IntegrationResult(result, error_est, n_evals, method_name)
    else
        return result
    end
end

"""
    romberg_method(f, a, b; max_steps=10, tol=1e-10, return_details=false)

Вычисляет интеграл функции `f` на интервале [`a`, `b`] методом Ромберга.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `max_steps::Int=10`: максимальное число шагов уточнения
- `tol::Real=1e-10`: требуемая точность (для досрочного завершения)
- `return_details::Bool=false`: возвращать ли подробную информацию о результате

# Возвращает
- `::Float64`: результат интегрирования, если `return_details=false`
- `::IntegrationResult`: структуру с результатом и дополнительной информацией,
  если `return_details=true`

# Математическое описание
Метод Ромберга основан на экстраполяции Ричардсона и методе трапеций. 
Он строит таблицу значений, где каждый столбец представляет экстраполяцию более 
высокого порядка, устраняющую ошибки предыдущего столбца:

R(i,0) = составной метод трапеций с 2^(i-1) интервалами
R(i,j) = [4^j * R(i,j-1) - R(i-1,j-1)] / (4^j - 1) для j ≥ 1

Погрешность метода Ромберга при использовании k столбцов имеет порядок O(h^(2k)).

# Пример
```julia
f(x) = sin(x)
result = romberg_method(f, 0, π)  # Очень точное приближение к 2.0
```
"""
function romberg_method(f::Function, a::Real, b::Real; 
                      max_steps::Int=10, tol::Real=1e-10, 
                      return_details::Bool=false)
    # Проверка аргументов
    if a > b
        return romberg_method(f, b, a, max_steps=max_steps, tol=tol, return_details=return_details)
    end
    if max_steps <= 0
        throw(ArgumentError("Максимальное число шагов должно быть положительным"))
    end
    
    # Инициализируем таблицу для интегрирования методом Ромберга
    R = zeros(max_steps, max_steps)
    h = b - a
    
    # Вычисляем первое приближение (метод трапеций с одним интервалом)
    R[1, 1] = (h / 2) * (f(a) + f(b))
    
    # Счетчик вычислений функции
    n_evals = 2
    
    # Строим таблицу Ромберга
    for i in 2:max_steps
        # Шаг разбиения
        h = h / 2
        
        # Вычисляем составной метод трапеций с 2^(i-1) интервалами
        # Используем ранее вычисленные значения, добавляя только новые точки
        sum_val = 0.0
        for j in 1:2^(i-2)
            sum_val += f(a + (2*j - 1) * h)
            n_evals += 1
        end
        
        # Первый столбец таблицы: составной метод трапеций
        R[i, 1] = R[i-1, 1] / 2 + h * sum_val
        
        # Экстраполяция Ричардсона для улучшения приближения
        for j in 2:i
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4^(j-1) - 1)
        end
        
        # Проверка на сходимость (если две последние диагональные точки достаточно близки)
        if i > 1 && abs(R[i, i] - R[i-1, i-1]) < tol
            # Достигнута требуемая точность, возвращаем результат
            result = R[i, i]
            error_est = abs(R[i, i] - R[i-1, i-1])
            
            if return_details
                return IntegrationResult(result, error_est, n_evals, "Метод Ромберга")
            else
                return result
            end
        end
    end
    
    # Если не достигнута требуемая точность, возвращаем лучшее приближение
    result = R[max_steps, max_steps]
    
    # Оценка погрешности как разность между последними двумя приближениями
    error_est = abs(R[max_steps, max_steps] - R[max_steps-1, max_steps-1])
    
    if return_details
        return IntegrationResult(result, error_est, n_evals, "Метод Ромберга (не достигнута требуемая точность)")
    else
        return result
    end
end

end # module CompositeMethods 