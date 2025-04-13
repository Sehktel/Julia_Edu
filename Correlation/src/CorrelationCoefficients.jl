"""
    CorrelationCoefficients

Модуль содержит функции для вычисления различных коэффициентов корреляции и корреляционных отношений.
Позволяет оценить силу и направление связи между переменными.
"""

using Statistics
using StatsBase
using LinearAlgebra

"""
    pearson_correlation(x, y)

Вычисляет коэффициент корреляции Пирсона между двумя векторами данных.

# Аргументы
- `x::AbstractVector{<:Real}`: первый вектор данных
- `y::AbstractVector{<:Real}`: второй вектор данных

# Возвращает
- `::Float64`: значение коэффициента корреляции Пирсона в диапазоне [-1, 1]

# Пример
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.2, 1.8, 3.2, 4.1, 4.8]
r = pearson_correlation(x, y)  # Вернёт примерно 0.997
```
"""
function pearson_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("Векторы должны иметь одинаковую длину"))
    end
    
    n = length(x)
    if n < 2
        throw(ArgumentError("Для расчёта корреляции необходимо как минимум 2 точки"))
    end
    
    # Вычисляем средние значения
    x_mean = mean(x)
    y_mean = mean(y)
    
    # Вычисляем числитель и знаменатель
    numerator = sum((x .- x_mean) .* (y .- y_mean))
    denominator = sqrt(sum((x .- x_mean).^2) * sum((y .- y_mean).^2))
    
    # Проверка на деление на ноль
    if denominator ≈ 0
        return 0.0
    end
    
    return numerator / denominator
end

"""
    spearman_correlation(x, y)

Вычисляет ранговый коэффициент корреляции Спирмена между двумя векторами данных.
Оценивает монотонность зависимости между переменными без предположения о линейности.

# Аргументы
- `x::AbstractVector{<:Real}`: первый вектор данных
- `y::AbstractVector{<:Real}`: второй вектор данных

# Возвращает
- `::Float64`: значение коэффициента корреляции Спирмена в диапазоне [-1, 1]

# Пример
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 3.0, 4.0, 6.0, 10.0]  # нелинейно растущие значения
r = spearman_correlation(x, y)  # Вернёт 1.0, так как порядок сохраняется
```
"""
function spearman_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("Векторы должны иметь одинаковую длину"))
    end
    
    n = length(x)
    if n < 2
        throw(ArgumentError("Для расчёта корреляции необходимо как минимум 2 точки"))
    end
    
    # Вычисляем ранги
    x_ranks = ordinalrank(x)
    y_ranks = ordinalrank(y)
    
    # Вычисляем корреляцию Пирсона между рангами
    return pearson_correlation(x_ranks, y_ranks)
end

"""
    kendall_correlation(x, y)

Вычисляет коэффициент ранговой корреляции Кендалла (тау) между двумя векторами данных.
Оценивает согласованность порядка значений в векторах.

# Аргументы
- `x::AbstractVector{<:Real}`: первый вектор данных
- `y::AbstractVector{<:Real}`: второй вектор данных

# Возвращает
- `::Float64`: значение коэффициента корреляции Кендалла в диапазоне [-1, 1]

# Пример
```julia
x = [1.0, 2.0, 3.0, 4.0]
y = [2.0, 3.0, 1.0, 4.0]
tau = kendall_correlation(x, y)  # Вернёт примерно 0.67
```
"""
function kendall_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("Векторы должны иметь одинаковую длину"))
    end
    
    n = length(x)
    if n < 2
        throw(ArgumentError("Для расчёта корреляции необходимо как минимум 2 точки"))
    end
    
    # Считаем согласованные и несогласованные пары
    concordant = 0
    discordant = 0
    
    for i in 1:(n-1)
        for j in (i+1):n
            # Вычисляем произведение разностей
            sign_diff = sign((x[i] - x[j]) * (y[i] - y[j]))
            
            if sign_diff > 0
                concordant += 1
            elseif sign_diff < 0
                discordant += 1
            end
            # При sign_diff = 0 имеем связи, которые не учитываются
        end
    end
    
    # Вычисляем тау Кендалла
    return (concordant - discordant) / (concordant + discordant)
end

"""
    correlation_ratio(x, y)

Вычисляет корреляционное отношение (коэффициент детерминации) между независимой 
переменной x и зависимой переменной y. Используется для оценки нелинейных зависимостей.

# Аргументы
- `x::AbstractVector{<:Real}`: вектор независимой переменной (фактор)
- `y::AbstractVector{<:Real}`: вектор зависимой переменной

# Возвращает
- `::Float64`: значение корреляционного отношения в диапазоне [0, 1]

# Пример
```julia
x = [1, 1, 2, 2, 3, 3, 4, 4]  # Категории/группы
y = [5, 6, 7, 8, 10, 11, 12, 13]
eta = correlation_ratio(x, y)  # Оценка нелинейной связи
```
"""
function correlation_ratio(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("Векторы должны иметь одинаковую длину"))
    end
    
    n = length(x)
    if n < 2
        throw(ArgumentError("Для расчёта корреляционного отношения необходимо как минимум 2 точки"))
    end
    
    # Вычисляем общее среднее
    y_mean = mean(y)
    
    # Группируем значения y по уникальным значениям x
    unique_x = unique(x)
    grouped_y = Dict(val => Float64[] for val in unique_x)
    
    for i in 1:n
        push!(grouped_y[x[i]], y[i])
    end
    
    # Вычисляем внутригрупповую и общую дисперсию
    between_group_var = 0.0
    
    for (_, group) in grouped_y
        group_mean = mean(group)
        group_size = length(group)
        between_group_var += group_size * (group_mean - y_mean)^2
    end
    
    total_var = sum((y .- y_mean).^2)
    
    # Проверка на деление на ноль
    if total_var ≈ 0
        return 0.0
    end
    
    return sqrt(between_group_var / total_var)
end

"""
    point_biserial_correlation(x, y)

Вычисляет точечно-бисериальный коэффициент корреляции между дихотомической 
переменной x и количественной переменной y.

# Аргументы
- `x::AbstractVector{Bool}`: бинарный вектор
- `y::AbstractVector{<:Real}`: вектор количественных данных

# Возвращает
- `::Float64`: значение точечно-бисериального коэффициента корреляции

# Пример
```julia
x = [true, false, true, false, true]
y = [5.2, 3.1, 4.8, 2.9, 5.1]
r_pb = point_biserial_correlation(x, y)
```
"""
function point_biserial_correlation(x::AbstractVector{Bool}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("Векторы должны иметь одинаковую длину"))
    end
    
    n = length(x)
    if n < 2
        throw(ArgumentError("Для расчёта корреляции необходимо как минимум 2 точки"))
    end
    
    # Выделяем группы
    y_1 = y[x]
    y_0 = y[.!x]
    
    n_1 = length(y_1)
    n_0 = length(y_0)
    
    if n_1 == 0 || n_0 == 0
        throw(ArgumentError("Обе группы должны содержать данные"))
    end
    
    # Средние значения групп
    mean_1 = mean(y_1)
    mean_0 = mean(y_0)
    
    # Стандартное отклонение всех значений y
    std_y = std(y)
    
    # Формула точечно-бисериальной корреляции
    p_1 = n_1 / n
    p_0 = n_0 / n
    
    return (mean_1 - mean_0) / std_y * sqrt(p_1 * p_0)
end 