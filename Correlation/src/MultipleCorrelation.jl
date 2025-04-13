"""
    MultipleCorrelation

Модуль содержит функции для анализа множественной корреляции и коэффициентов детерминации.
"""

module MultipleCorrelation

using LinearAlgebra
using Statistics
using ..CorrelationMatrix: correlation_matrix

export multiple_correlation, adjusted_r_squared, variance_inflation_factors

"""
    multiple_correlation(data, dependent_var_index)

Вычисляет коэффициент множественной корреляции между зависимой переменной 
(указанной по индексу) и всеми остальными переменными.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `dependent_var_index::Int`: индекс столбца с зависимой переменной

# Возвращает
- `::Float64`: коэффициент множественной корреляции R

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
R = multiple_correlation(data, 1)  # Корреляция между 1-й переменной и остальными
```
"""
function multiple_correlation(data::AbstractMatrix{<:Real}, dependent_var_index::Int)
    n_vars = size(data, 2)
    
    if dependent_var_index < 1 || dependent_var_index > n_vars
        throw(ArgumentError("Индекс зависимой переменной должен быть между 1 и $n_vars"))
    end
    
    # Вычисляем корреляционную матрицу
    corr_matrix = correlation_matrix(data)
    
    # Извлекаем корреляции зависимой переменной с предикторами
    r_yx = [corr_matrix[dependent_var_index, j] for j in 1:n_vars if j != dependent_var_index]
    
    # Выделяем подматрицу корреляций между предикторами
    predictor_indices = [j for j in 1:n_vars if j != dependent_var_index]
    r_xx = corr_matrix[predictor_indices, predictor_indices]
    
    # Решаем систему для вычисления R^2
    # R^2 = r_yx' * inv(r_xx) * r_yx
    try
        r_squared = dot(r_yx, inv(r_xx) * r_yx)
        # Ограничиваем результат в пределах [0, 1]
        r_squared = min(max(r_squared, 0.0), 1.0)
        return sqrt(r_squared)
    catch e
        # В случае сингулярной матрицы используем альтернативный метод
        # На основе коэффициента детерминации модели регрессии
        y = data[:, dependent_var_index]
        X = data[:, predictor_indices]
        
        # Среднее y
        y_mean = mean(y)
        
        # Общая сумма квадратов
        SST = sum((y .- y_mean).^2)
        
        # Для регрессии используем псевдообратную матрицу
        X_centered = X .- mean(X, dims=1)
        y_centered = y .- y_mean
        
        # Вычисляем коэффициенты при помощи псевдоинверсии
        beta = pinv(X_centered' * X_centered) * (X_centered' * y_centered)
        
        # Прогнозные значения
        y_pred = X_centered * beta .+ y_mean
        
        # Сумма квадратов остатков
        SSE = sum((y .- y_pred).^2)
        
        # Коэффициент детерминации
        r_squared = 1.0 - SSE / SST
        
        # Ограничиваем результат в пределах [0, 1]
        r_squared = min(max(r_squared, 0.0), 1.0)
        return sqrt(r_squared)
    end
end

"""
    adjusted_r_squared(r_squared, n_samples, n_predictors)

Вычисляет скорректированный коэффициент детерминации (adjusted R²).
Это модификация R², которая учитывает количество предикторов и объем выборки.

# Аргументы
- `r_squared::Real`: коэффициент детерминации (R²)
- `n_samples::Int`: количество наблюдений
- `n_predictors::Int`: количество предикторов (независимых переменных)

# Возвращает
- `::Float64`: скорректированный коэффициент детерминации

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
R = multiple_correlation(data, 1)
R_squared = R^2
adj_R_squared = adjusted_r_squared(R_squared, size(data, 1), size(data, 2) - 1)
```
"""
function adjusted_r_squared(r_squared::Real, n_samples::Int, n_predictors::Int)
    if n_samples <= n_predictors + 1
        throw(ArgumentError("Количество наблюдений должно быть больше количества предикторов + 1"))
    end
    
    # Формула скорректированного R²
    return 1.0 - (1.0 - r_squared) * (n_samples - 1) / (n_samples - n_predictors - 1)
end

"""
    variance_inflation_factors(data)

Вычисляет факторы инфляции дисперсии (VIF) для каждой переменной.
VIF показывает, насколько увеличивается дисперсия коэффициента регрессии 
из-за мультиколлинеарности.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную

# Возвращает
- `::Vector{Float64}`: вектор факторов инфляции дисперсии для каждой переменной

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
vifs = variance_inflation_factors(data)
```
"""
function variance_inflation_factors(data::AbstractMatrix{<:Real})
    n_vars = size(data, 2)
    vifs = zeros(n_vars)
    
    for i in 1:n_vars
        # Для каждой переменной вычисляем R² с остальными переменными в качестве предикторов
        R = multiple_correlation(data, i)
        R_squared = R^2
        
        # VIF = 1 / (1 - R²)
        # Если R² близок к 1, то VIF будет очень большим, указывая на сильную мультиколлинеарность
        if R_squared >= 0.999
            vifs[i] = Inf  # Избегаем деления на очень маленькое число
        else
            vifs[i] = 1.0 / (1.0 - R_squared)
        end
    end
    
    return vifs
end

"""
    stepwise_variable_selection(data, dependent_var_index; 
                              direction=:forward, p_value_threshold=0.05, 
                              max_iterations=nothing)

Выполняет пошаговый отбор переменных для модели множественной регрессии.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `dependent_var_index::Int`: индекс столбца с зависимой переменной
- `direction::Symbol=:forward`: направление отбора (:forward, :backward или :both)
- `p_value_threshold::Float64=0.05`: пороговое значение p-value для включения/исключения
- `max_iterations::Union{Int, Nothing}=nothing`: максимальное число итераций (по умолчанию число предикторов)

# Возвращает
- `::Vector{Int}`: индексы выбранных предикторов
- `::Float64`: итоговый скорректированный R²

# Пример
```julia
data = [1.0 2.0 3.0 4.0; 2.0 3.0 3.5 5.0; 3.0 4.0 4.0 6.0; 4.0 5.0 6.0 7.0; 5.0 6.0 7.0 8.0]
selected_vars, adj_r2 = stepwise_variable_selection(data, 1)
```
"""
function stepwise_variable_selection(data::AbstractMatrix{<:Real}, dependent_var_index::Int;
                                     direction::Symbol=:forward, 
                                     p_value_threshold::Float64=0.05,
                                     max_iterations::Union{Int, Nothing}=nothing)
    n_obs, n_vars = size(data)
    
    if dependent_var_index < 1 || dependent_var_index > n_vars
        throw(ArgumentError("Индекс зависимой переменной должен быть между 1 и $n_vars"))
    end
    
    # Множество всех возможных предикторов
    all_predictors = [i for i in 1:n_vars if i != dependent_var_index]
    
    # Инициализация множества выбранных предикторов
    selected_predictors = Int[]
    
    # Инициализация множества доступных предикторов
    if direction == :forward || direction == :both
        available_predictors = copy(all_predictors)
        current_predictors = Int[]
    elseif direction == :backward
        available_predictors = Int[]
        current_predictors = copy(all_predictors)
    else
        throw(ArgumentError("Неизвестное направление: $direction. Поддерживаются: :forward, :backward, :both"))
    end
    
    # Устанавливаем максимальное количество итераций
    if max_iterations === nothing
        max_iterations = length(all_predictors)
    end
    
    # Извлекаем зависимую переменную
    y = data[:, dependent_var_index]
    y_mean = mean(y)
    SST = sum((y .- y_mean).^2)
    
    # Лучшая модель
    best_adj_r2 = -Inf
    best_predictors = Int[]
    
    # Функция для вычисления F-статистики для новой переменной
    function f_statistic_for_new_var(X_current, X_new, y)
        n = length(y)
        
        # Модель только с текущими предикторами
        if isempty(X_current)
            # Если нет текущих предикторов, модель только с интерцептом
            SSE_current = SST
            df_current = n - 1
        else
            # Центрируем данные
            X_current_centered = X_current .- mean(X_current, dims=1)
            
            # Вычисляем коэффициенты
            beta_current = pinv(X_current_centered' * X_current_centered) * (X_current_centered' * (y .- y_mean))
            
            # Прогнозные значения
            y_pred_current = X_current_centered * beta_current .+ y_mean
            
            # Остаточная сумма квадратов
            SSE_current = sum((y .- y_pred_current).^2)
            df_current = n - size(X_current, 2) - 1
        end
        
        # Модель с добавлением новой переменной
        X_combined = isempty(X_current) ? X_new : hcat(X_current, X_new)
        X_combined_centered = X_combined .- mean(X_combined, dims=1)
        
        # Вычисляем коэффициенты
        beta_combined = pinv(X_combined_centered' * X_combined_centered) * (X_combined_centered' * (y .- y_mean))
        
        # Прогнозные значения
        y_pred_combined = X_combined_centered * beta_combined .+ y_mean
        
        # Остаточная сумма квадратов
        SSE_combined = sum((y .- y_pred_combined).^2)
        df_combined = n - size(X_combined, 2) - 1
        
        # Изменение в SSE
        delta_SSE = SSE_current - SSE_combined
        
        # F-статистика
        # F = (Delta SSE / Delta df) / (SSE_combined / df_combined)
        if delta_SSE <= 0
            return 0.0  # Если добавление переменной не улучшает модель
        end
        
        F = (delta_SSE / 1) / (SSE_combined / df_combined)
        
        return F
    end
    
    # Функция для получения p-value из F-статистики
    # Примечание: для точного вычисления p-value требуется пакет Distributions.jl
    function p_value_from_f(F, df1, df2)
        # Приблизительное вычисление p-value
        # Для малых значений F, p-value близко к 1
        if F <= 0
            return 1.0
        end
        
        # Приблизительные критические значения F для разных уровней значимости
        # Это очень грубое приближение
        if F > 10.0
            return 0.001
        elseif F > 6.0
            return 0.01
        elseif F > 4.0
            return 0.05
        elseif F > 2.7
            return 0.1
        else
            return 0.2
        end
    end
    
    # Пошаговый отбор
    iteration = 0
    continue_selection = true
    
    while continue_selection && iteration < max_iterations
        iteration += 1
        continue_selection = false
        
        if direction == :forward || direction == :both
            # Forward selection: добавление переменной
            best_predictor = nothing
            best_f_stat = 0.0
            best_p_value = 1.0
            
            for predictor in available_predictors
                # Вычисляем F-статистику для добавления этой переменной
                X_current = isempty(current_predictors) ? Matrix{Float64}(undef, n_obs, 0) : data[:, current_predictors]
                X_new = data[:, predictor:predictor]
                
                f_stat = f_statistic_for_new_var(X_current, X_new, y)
                p_value = p_value_from_f(f_stat, 1, n_obs - length(current_predictors) - 2)
                
                if p_value < p_value_threshold && f_stat > best_f_stat
                    best_predictor = predictor
                    best_f_stat = f_stat
                    best_p_value = p_value
                end
            end
            
            # Добавляем лучший предиктор, если он найден
            if best_predictor !== nothing
                push!(current_predictors, best_predictor)
                deleteat!(available_predictors, findfirst(==(best_predictor), available_predictors))
                continue_selection = true
                
                # Пересчитываем R^2 и adj_R^2
                X = data[:, current_predictors]
                X_centered = X .- mean(X, dims=1)
                beta = pinv(X_centered' * X_centered) * (X_centered' * (y .- y_mean))
                y_pred = X_centered * beta .+ y_mean
                SSE = sum((y .- y_pred).^2)
                R_squared = 1.0 - SSE / SST
                adj_R_squared = adjusted_r_squared(R_squared, n_obs, length(current_predictors))
                
                if adj_R_squared > best_adj_r2
                    best_adj_r2 = adj_R_squared
                    best_predictors = copy(current_predictors)
                end
            end
        end
        
        if direction == :backward || direction == :both
            # Backward elimination: удаление переменной
            if !isempty(current_predictors)
                worst_predictor = nothing
                worst_f_stat = Inf
                highest_p_value = 0.0
                
                for (i, predictor) in enumerate(current_predictors)
                    # Вычисляем статистику для модели без этого предиктора
                    remaining_predictors = [p for p in current_predictors if p != predictor]
                    
                    if isempty(remaining_predictors)
                        # Если это последний предиктор, сравниваем с нулевой моделью
                        X_current = Matrix{Float64}(undef, n_obs, 0)
                        X_removed = data[:, predictor:predictor]
                    else
                        # Иначе сравниваем с моделью без этого предиктора
                        X_current = data[:, remaining_predictors]
                        X_removed = data[:, predictor:predictor]
                    end
                    
                    # F-статистика для удаления этой переменной (обратный знак, т.к. мы рассматриваем удаление)
                    f_stat = f_statistic_for_new_var(X_current, X_removed, y)
                    p_value = p_value_from_f(f_stat, 1, n_obs - length(remaining_predictors) - 2)
                    
                    # Для backward elimination мы удаляем переменную с наименьшей значимостью (наибольшим p-value)
                    if p_value > p_value_threshold && p_value > highest_p_value
                        worst_predictor = i
                        worst_f_stat = f_stat
                        highest_p_value = p_value
                    end
                end
                
                # Удаляем наименее значимый предиктор, если он найден
                if worst_predictor !== nothing
                    predictor_to_remove = current_predictors[worst_predictor]
                    push!(available_predictors, predictor_to_remove)
                    deleteat!(current_predictors, worst_predictor)
                    continue_selection = true
                    
                    # Пересчитываем R^2 и adj_R^2
                    if isempty(current_predictors)
                        # Нулевая модель
                        adj_R_squared = 0.0
                    else
                        X = data[:, current_predictors]
                        X_centered = X .- mean(X, dims=1)
                        beta = pinv(X_centered' * X_centered) * (X_centered' * (y .- y_mean))
                        y_pred = X_centered * beta .+ y_mean
                        SSE = sum((y .- y_pred).^2)
                        R_squared = 1.0 - SSE / SST
                        adj_R_squared = adjusted_r_squared(R_squared, n_obs, length(current_predictors))
                    end
                    
                    if adj_R_squared > best_adj_r2
                        best_adj_r2 = adj_R_squared
                        best_predictors = copy(current_predictors)
                    end
                end
            end
        end
    end
    
    return best_predictors, best_adj_r2
end

end # module 