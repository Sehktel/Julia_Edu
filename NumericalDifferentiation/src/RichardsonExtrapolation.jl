"""
    RichardsonExtrapolation

Реализация метода экстраполяции Ричардсона для повышения точности 
численного дифференцирования.
"""

"""
    richardson_extrapolation(f, x; base_step=1e-2, levels=3, method=:central, error_estimate=false)

Вычисляет производную функции `f` в точке `x` с применением метода экстраполяции Ричардсона,
что значительно повышает точность по сравнению с обычными методами.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `base_step::Real=1e-2`: Начальный шаг дифференцирования
- `levels::Integer=3`: Количество уровней экстраполяции
- `method::Symbol=:central`: Метод дифференцирования (:forward, :backward, :central)
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Теоретическая информация
Метод экстраполяции Ричардсона использует последовательность вычислений с уменьшающимся шагом
и комбинирует их, устраняя члены погрешности более низкого порядка.

Для центральной разности, порядок точности формулы экстраполяции Ричардсона O(h^(2k)),
где k - число уровней экстраполяции.

# Пример
```julia
f(x) = sin(x)
df_dx = richardson_extrapolation(f, π/4)  # Должно быть близко к cos(π/4) ≈ 0.7071
```
"""
function richardson_extrapolation(f::Function, x::Real; 
                                 base_step::Real=1e-2, 
                                 levels::Integer=3, 
                                 method::Symbol=:central,
                                 error_estimate::Bool=false)
    # Проверка аргументов
    if levels < 1
        throw(ArgumentError("Количество уровней должно быть не менее 1"))
    end
    
    # Создаем таблицу для экстраполяции
    table = zeros(Float64, levels, levels)
    
    # Вычисляем производные с различными шагами
    for i in 1:levels
        h = base_step / (2^(i-1))
        
        # Выбираем соответствующий метод конечных разностей
        if method == :forward
            from_FiniteDifferences = Base.invokelatest(Main.NumericalDifferentiation.FiniteDifferences.forward_difference, f, x, step=h)
            table[i, 1] = from_FiniteDifferences
        elseif method == :backward
            from_FiniteDifferences = Base.invokelatest(Main.NumericalDifferentiation.FiniteDifferences.backward_difference, f, x, step=h)
            table[i, 1] = from_FiniteDifferences
        elseif method == :central
            from_FiniteDifferences = Base.invokelatest(Main.NumericalDifferentiation.FiniteDifferences.central_difference, f, x, step=h)
            table[i, 1] = from_FiniteDifferences
        else
            throw(ArgumentError("Неизвестный метод: $method. Используйте :forward, :backward или :central"))
        end
    end
    
    # Определяем порядок точности используемого метода
    order = method == :central ? 2 : 1
    
    # Строим таблицу экстраполяции Ричардсона
    for j in 2:levels
        for i in j:levels
            # Формула экстраполяции Ричардсона
            factor = 4^((j-1)/order)
            table[i, j] = (factor * table[i, j-1] - table[i-1, j-1]) / (factor - 1)
        end
    end
    
    # Результат — значение с наивысшей точностью
    result = table[levels, levels]
    
    if error_estimate
        # Оценка погрешности как разность между последними двумя приближениями
        if levels > 1
            error_est = abs(table[levels, levels] - table[levels-1, levels-1])
        else
            # Если только один уровень, используем простую оценку
            h = base_step
            error_est = method == :central ? h^2 : h
        end
        
        return DifferentiationResult(result, error_est, base_step)
    else
        return result
    end
end

"""
    richardson_second_derivative(f, x; base_step=1e-2, levels=3, error_estimate=false)

Вычисляет вторую производную функции `f` в точке `x` с применением метода 
экстраполяции Ричардсона для повышения точности.

# Аргументы
- `f::Function`: Функция, для которой вычисляется вторая производная
- `x::Real`: Точка, в которой вычисляется производная
- `base_step::Real=1e-2`: Начальный шаг дифференцирования
- `levels::Integer=3`: Количество уровней экстраполяции
- `error_estimate::Bool=false`: Вычислять ли оценку погрешности

# Возвращаемое значение
- Если `error_estimate=false`: Значение второй производной (`Float64`)
- Если `error_estimate=true`: Объект `DifferentiationResult`

# Пример
```julia
f(x) = sin(x)
d2f_dx2 = richardson_second_derivative(f, π/4)  # Должно быть близко к -sin(π/4) ≈ -0.7071
```
"""
function richardson_second_derivative(f::Function, x::Real; 
                                     base_step::Real=1e-2, 
                                     levels::Integer=3, 
                                     error_estimate::Bool=false)
    # Проверка аргументов
    if levels < 1
        throw(ArgumentError("Количество уровней должно быть не менее 1"))
    end
    
    # Создаем таблицу для экстраполяции
    table = zeros(Float64, levels, levels)
    
    # Вычисляем вторые производные с различными шагами
    for i in 1:levels
        h = base_step / (2^(i-1))
        
        # Используем стандартную формулу для второй производной
        from_FiniteDifferences = Base.invokelatest(Main.NumericalDifferentiation.FiniteDifferences.second_derivative, f, x, step=h)
        table[i, 1] = from_FiniteDifferences
    end
    
    # Строим таблицу экстраполяции Ричардсона
    # Для второй производной порядок точности равен 2
    for j in 2:levels
        for i in j:levels
            # Формула экстраполяции Ричардсона
            factor = 4^(j-1)
            table[i, j] = (factor * table[i, j-1] - table[i-1, j-1]) / (factor - 1)
        end
    end
    
    # Результат — значение с наивысшей точностью
    result = table[levels, levels]
    
    if error_estimate
        # Оценка погрешности как разность между последними двумя приближениями
        if levels > 1
            error_est = abs(table[levels, levels] - table[levels-1, levels-1])
        else
            # Если только один уровень, используем простую оценку
            error_est = base_step^2
        end
        
        return DifferentiationResult(result, error_est, base_step)
    else
        return result
    end
end

"""
    adaptive_richardson_extrapolation(f, x; tol=1e-8, max_levels=10, method=:central)

Вычисляет производную функции `f` в точке `x`, адаптивно определяя необходимое
количество уровней экстраполяции Ричардсона для достижения заданной точности.

# Аргументы
- `f::Function`: Функция, для которой вычисляется производная
- `x::Real`: Точка, в которой вычисляется производная
- `tol::Real=1e-8`: Требуемая точность (погрешность)
- `max_levels::Integer=10`: Максимальное число уровней экстраполяции
- `method::Symbol=:central`: Метод дифференцирования (:forward, :backward, :central)

# Возвращаемое значение
- Объект `DifferentiationResult` с вычисленной производной, оценкой погрешности и использованным шагом

# Пример
```julia
f(x) = exp(x)
result = adaptive_richardson_extrapolation(f, 1.0, tol=1e-10)
println("f'(1.0) ≈ \$(result.value) с погрешностью \$(result.error_estimate)")
```
"""
function adaptive_richardson_extrapolation(f::Function, x::Real; 
                                          tol::Real=1e-8, 
                                          max_levels::Integer=10,
                                          method::Symbol=:central)
    # Начальный шаг
    base_step = 0.1
    
    # Начинаем с двух уровней и увеличиваем их число, пока не достигнем требуемой точности
    for levels in 2:max_levels
        # Вычисляем производную с текущим числом уровней, запрашивая оценку погрешности
        result = richardson_extrapolation(f, x, base_step=base_step, levels=levels, 
                                         method=method, error_estimate=true)
        
        # Если достигли требуемой точности, возвращаем результат
        if result.error_estimate <= tol
            return result
        end
    end
    
    # Если не удалось достичь требуемой точности, возвращаем последний результат
    @warn "Не удалось достичь требуемой точности $tol. Возвращаем результат с максимальным числом уровней."
    return richardson_extrapolation(f, x, base_step=base_step, levels=max_levels, 
                                  method=method, error_estimate=true)
end 