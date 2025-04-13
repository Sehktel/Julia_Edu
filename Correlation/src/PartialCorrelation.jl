"""
    PartialCorrelation

Модуль содержит функции для вычисления и анализа частных корреляций, 
которые измеряют связь между двумя переменными при контроле влияния других переменных.
"""

module PartialCorrelation

using LinearAlgebra
using Statistics
using ..CorrelationMatrix: correlation_matrix

export partial_correlation, partial_correlation_matrix, partial_correlation_test

"""
    partial_correlation(data, var1_idx, var2_idx, control_indices)

Вычисляет частный коэффициент корреляции между двумя переменными, 
исключая влияние контрольных переменных.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `var1_idx::Int`: индекс первой переменной
- `var2_idx::Int`: индекс второй переменной
- `control_indices::Vector{Int}`: индексы контрольных переменных

# Возвращает
- `::Float64`: частный коэффициент корреляции

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
# Корреляция между 1-й и 2-й переменными при контроле 3-й
r_partial = partial_correlation(data, 1, 2, [3])
```
"""
function partial_correlation(data::AbstractMatrix{<:Real}, var1_idx::Int, var2_idx::Int, control_indices::Vector{Int})
    n_vars = size(data, 2)
    
    # Проверка индексов
    for idx in [var1_idx, var2_idx, control_indices...]
        if idx < 1 || idx > n_vars
            throw(ArgumentError("Индекс переменной должен быть между 1 и $n_vars"))
        end
    end
    
    # Проверка на уникальность индексов
    if var1_idx == var2_idx || var1_idx in control_indices || var2_idx in control_indices
        throw(ArgumentError("Индексы переменных должны быть уникальными"))
    end
    
    if isempty(control_indices)
        # Если нет контрольных переменных, это обычная корреляция
        corr_mat = correlation_matrix(data)
        return corr_mat[var1_idx, var2_idx]
    end
    
    # Вычисляем корреляционную матрицу
    corr_mat = correlation_matrix(data)
    
    # Если только одна контрольная переменная, используем простую формулу
    if length(control_indices) == 1
        control_idx = control_indices[1]
        r_12 = corr_mat[var1_idx, var2_idx]
        r_13 = corr_mat[var1_idx, control_idx]
        r_23 = corr_mat[var2_idx, control_idx]
        
        # Формула для частной корреляции
        numerator = r_12 - r_13 * r_23
        denominator = sqrt((1 - r_13^2) * (1 - r_23^2))
        
        # Проверка на деление на ноль
        if abs(denominator) < 1e-10
            return 0.0
        end
        
        return numerator / denominator
    else
        # Для нескольких контрольных переменных используем инверсию корреляционной матрицы
        # Создаем индексы для соответствующей подматрицы
        indices = [var1_idx, var2_idx, control_indices...]
        
        # Получаем подматрицу
        submatrix = corr_mat[indices, indices]
        
        # Инвертируем матрицу
        try
            precision_matrix = inv(submatrix)
            
            # Частная корреляция через элементы матрицы точности
            # (нормированная отрицательная ковариация)
            return -precision_matrix[1, 2] / sqrt(precision_matrix[1, 1] * precision_matrix[2, 2])
        catch e
            # Если матрица сингулярная, возвращаем NaN
            @warn "Не удалось вычислить частную корреляцию: $(e)"
            return NaN
        end
    end
end

"""
    partial_correlation_matrix(data, control_indices)

Вычисляет матрицу частных корреляций между всеми парами переменных,
исключая влияние контрольных переменных.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `control_indices::Vector{Int}`: индексы контрольных переменных

# Возвращает
- `::Matrix{Float64}`: матрица частных корреляций

# Пример
```julia
data = [1.0 2.0 3.0 4.0; 2.0 3.0 3.5 4.5; 3.0 4.0 4.0 5.0; 4.0 5.0 6.0 7.0; 5.0 6.0 7.0 8.0]
# Матрица частных корреляций при контроле 3-й и 4-й переменных
partial_corr_mat = partial_correlation_matrix(data, [3, 4])
```
"""
function partial_correlation_matrix(data::AbstractMatrix{<:Real}, control_indices::Vector{Int})
    n_vars = size(data, 2)
    
    # Проверка индексов
    for idx in control_indices
        if idx < 1 || idx > n_vars
            throw(ArgumentError("Индекс переменной должен быть между 1 и $n_vars"))
        end
    end
    
    # Создаем индексы для переменных, которые не являются контрольными
    variable_indices = [i for i in 1:n_vars if !(i in control_indices)]
    
    if isempty(variable_indices)
        throw(ArgumentError("Все переменные указаны как контрольные"))
    end
    
    # Создаем матрицу частных корреляций
    n_variables = length(variable_indices)
    partial_corr = Matrix{Float64}(I, n_variables, n_variables)
    
    # Вычисляем частные корреляции для всех пар переменных
    for i in 1:n_variables
        for j in i+1:n_variables
            var1_idx = variable_indices[i]
            var2_idx = variable_indices[j]
            r_partial = partial_correlation(data, var1_idx, var2_idx, control_indices)
            partial_corr[i, j] = r_partial
            partial_corr[j, i] = r_partial
        end
    end
    
    return partial_corr
end

"""
    partial_correlation_test(data, var1_idx, var2_idx, control_indices; alpha=0.05)

Проводит статистический тест для проверки значимости частной корреляции.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `var1_idx::Int`: индекс первой переменной
- `var2_idx::Int`: индекс второй переменной
- `control_indices::Vector{Int}`: индексы контрольных переменных
- `alpha::Float64=0.05`: уровень значимости (по умолчанию 0.05)

# Возвращает
- `::Tuple{Float64, Float64, Bool}`: кортеж из (коэффициент частной корреляции, p-значение, флаг значимости)

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
# Тест значимости частной корреляции между 1-й и 2-й переменными при контроле 3-й
r_partial, p_value, is_significant = partial_correlation_test(data, 1, 2, [3])
```
"""
function partial_correlation_test(data::AbstractMatrix{<:Real}, var1_idx::Int, var2_idx::Int, 
                                control_indices::Vector{Int}; alpha::Float64=0.05)
    n_obs = size(data, 1)
    df = n_obs - 2 - length(control_indices)
    
    if df < 1
        throw(ArgumentError("Недостаточно наблюдений для проведения теста"))
    end
    
    # Вычисляем частную корреляцию
    r_partial = partial_correlation(data, var1_idx, var2_idx, control_indices)
    
    # Вычисляем t-статистику
    t_stat = r_partial * sqrt(df) / sqrt(1 - r_partial^2)
    
    # Вычисляем p-значение
    # Примечание: для точного расчета p-значения необходим пакет Distributions.jl
    # Здесь используем приближение
    p_value = 2 * (1 - cdf_t(abs(t_stat), df))
    
    # Определяем значимость
    is_significant = p_value < alpha
    
    return (r_partial, p_value, is_significant)
end

"""
Аппроксимация кумулятивной функции распределения t-распределения.
"""
function cdf_t(t::Float64, df::Int)
    # Приближение кумулятивной функции распределения
    x = df / (t^2 + df)
    if df >= 2
        # Используем бета-функцию для вычисления CDF
        return 1.0 - 0.5 * x^(df/2)
    else
        # Для малых df используем приближение
        return 0.5 + atan(t/sqrt(df)) / π
    end
end

end # module 