"""
    CorrelationMatrix

Модуль содержит функции для создания и анализа корреляционных матриц,
включая вычисление матрицы коэффициентов корреляции и оценку их значимости.
"""

module CorrelationMatrix

using LinearAlgebra
using Statistics

export correlation_matrix, correlation_significance, correlation_heatmap,
       correlation_with_confidence_intervals, correlation_distance_matrix

"""
    correlation_matrix(data; method="pearson")

Вычисляет матрицу корреляции для набора данных.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `method::String="pearson"`: метод вычисления корреляции ("pearson", "spearman", или "kendall")

# Возвращает
- `::Matrix{Float64}`: матрица корреляции

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
corr_mat = correlation_matrix(data)
```
"""
function correlation_matrix(data::AbstractMatrix{<:Real}; method::String="pearson")
    n_vars = size(data, 2)
    
    if method == "pearson"
        # Вычисляем матрицу корреляции Пирсона
        # Нормализуем данные
        centered_data = similar(data)
        for j in 1:n_vars
            centered_data[:, j] = data[:, j] .- mean(data[:, j])
        end
        
        # Вычисляем корреляцию через ковариационную матрицу
        cov_mat = (centered_data' * centered_data) / (size(data, 1) - 1)
        
        # Вычисляем стандартные отклонения
        std_devs = [sqrt(cov_mat[j, j]) for j in 1:n_vars]
        
        # Создаем матрицу корреляции
        corr_mat = similar(cov_mat)
        for i in 1:n_vars
            for j in 1:n_vars
                corr_mat[i, j] = cov_mat[i, j] / (std_devs[i] * std_devs[j])
            end
        end
        
        return corr_mat
    elseif method == "spearman"
        # Вычисляем матрицу ранговой корреляции Спирмена
        # Ранжируем данные
        ranked_data = similar(data)
        for j in 1:n_vars
            # Вычисляем ранги
            ranks = zeros(size(data, 1))
            sorted_indices = sortperm(data[:, j])
            for (rank, idx) in enumerate(sorted_indices)
                ranks[idx] = rank
            end
            # Обрабатываем связи (одинаковые значения)
            for i in 1:size(data, 1)
                same_val_indices = findall(x -> x == data[i, j], data[:, j])
                if length(same_val_indices) > 1
                    ranks[same_val_indices] .= mean(ranks[same_val_indices])
                end
            end
            ranked_data[:, j] = ranks
        end
        
        # Теперь вычисляем корреляцию Пирсона для рангов
        return correlation_matrix(ranked_data, method="pearson")
    elseif method == "kendall"
        # Вычисляем тау Кендалла для каждой пары переменных
        corr_mat = zeros(n_vars, n_vars)
        
        for i in 1:n_vars
            # Диагональные элементы - единицы
            corr_mat[i, i] = 1.0
            
            for j in i+1:n_vars
                # Вычисляем тау Кендалла для пары (i, j)
                x = data[:, i]
                y = data[:, j]
                n = length(x)
                
                # Подсчитываем конкордантные и дискордантные пары
                concordant = 0
                discordant = 0
                
                for k in 1:n
                    for l in k+1:n
                        # Проверяем согласованность упорядочения
                        if (x[k] < x[l] && y[k] < y[l]) || (x[k] > x[l] && y[k] > y[l])
                            concordant += 1
                        elseif (x[k] < x[l] && y[k] > y[l]) || (x[k] > x[l] && y[k] < y[l])
                            discordant += 1
                        end
                        # Игнорируем связи (когда x[k] == x[l] или y[k] == y[l])
                    end
                end
                
                # Вычисляем тау Кендалла
                tau = (concordant - discordant) / (0.5 * n * (n - 1))
                
                # Заполняем матрицу (она симметрична)
                corr_mat[i, j] = tau
                corr_mat[j, i] = tau
            end
        end
        
        return corr_mat
    else
        throw(ArgumentError("Неподдерживаемый метод корреляции: $method. Используйте 'pearson', 'spearman' или 'kendall'."))
    end
end

"""
    correlation_significance(data, alpha=0.05; method="pearson")

Вычисляет матрицу значимости корреляции для набора данных.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `alpha::Float64=0.05`: уровень значимости (по умолчанию 0.05)
- `method::String="pearson"`: метод вычисления корреляции ("pearson", "spearman", или "kendall")

# Возвращает
- `::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Bool}}`: кортеж из (матрица корреляции, 
   матрица p-значений, булева матрица значимости)

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
corr_mat, p_values, significance = correlation_significance(data)
```
"""
function correlation_significance(data::AbstractMatrix{<:Real}, alpha::Float64=0.05; method::String="pearson")
    n_vars = size(data, 2)
    n_obs = size(data, 1)
    
    if n_obs < 3
        throw(ArgumentError("Для вычисления значимости корреляции необходимо не менее 3 наблюдений"))
    end
    
    # Вычисляем матрицу корреляции
    corr_mat = correlation_matrix(data, method=method)
    
    # Создаем матрицы для p-значений и флагов значимости
    p_values = zeros(n_vars, n_vars)
    significance = falses(n_vars, n_vars)
    
    # Вычисляем p-значения и значимость
    for i in 1:n_vars
        # Диагональные элементы - единицы, p = 0, всегда значимы
        p_values[i, i] = 0.0
        significance[i, i] = true
        
        for j in i+1:n_vars
            # Вычисляем p-значение в зависимости от метода
            if method == "pearson" || method == "spearman"
                # Для корреляции Пирсона и Спирмена используем t-распределение
                r = corr_mat[i, j]
                # Степени свободы
                df = n_obs - 2
                
                # Вычисляем t-статистику
                t_stat = r * sqrt(df) / sqrt(1 - r^2)
                
                # Вычисляем p-значение
                # Примечание: для точного расчета p-значения необходим пакет Distributions.jl
                # Здесь используем приближение
                p = 2 * (1 - cdf_t(abs(t_stat), df))
            elseif method == "kendall"
                # Для тау Кендалла используем нормальное приближение
                r = corr_mat[i, j]
                n = n_obs
                
                # Вычисляем стандартную ошибку
                se = sqrt((4 * n + 10) / (9 * n * (n - 1)))
                
                # Вычисляем z-статистику
                z_stat = r / se
                
                # Вычисляем p-значение
                # Примечание: для точного расчета p-значения необходим пакет Distributions.jl
                # Здесь используем приближение
                p = 2 * (1 - cdf_normal(abs(z_stat)))
            else
                throw(ArgumentError("Неподдерживаемый метод корреляции: $method"))
            end
            
            # Заполняем матрицы (они симметричны)
            p_values[i, j] = p
            p_values[j, i] = p
            
            # Определяем значимость
            is_significant = p < alpha
            significance[i, j] = is_significant
            significance[j, i] = is_significant
        end
    end
    
    return (corr_mat, p_values, significance)
end

"""
    correlation_with_confidence_intervals(data, confidence_level=0.95; method="pearson")

Вычисляет матрицу корреляции с доверительными интервалами для каждого коэффициента.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `confidence_level::Float64=0.95`: уровень доверия (по умолчанию 0.95)
- `method::String="pearson"`: метод вычисления корреляции ("pearson", "spearman", или "kendall")

# Возвращает
- `::Array{Tuple{Float64, Float64, Float64}, 2}`: матрица кортежей (корреляция, нижняя граница, верхняя граница)

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
corr_with_ci = correlation_with_confidence_intervals(data)
```
"""
function correlation_with_confidence_intervals(data::AbstractMatrix{<:Real}, 
                                              confidence_level::Float64=0.95; 
                                              method::String="pearson")
    n_vars = size(data, 2)
    n_obs = size(data, 1)
    
    if n_obs < 3
        throw(ArgumentError("Для вычисления доверительных интервалов необходимо не менее 3 наблюдений"))
    end
    
    # Вычисляем матрицу корреляции
    corr_mat = correlation_matrix(data, method=method)
    
    # Создаем матрицу для хранения кортежей (корреляция, нижняя граница, верхняя граница)
    result = Array{Tuple{Float64, Float64, Float64}, 2}(undef, n_vars, n_vars)
    
    # Z-значение для заданного уровня доверия
    z = quantile_normal(1 - (1 - confidence_level) / 2)
    
    for i in 1:n_vars
        # Диагональные элементы
        result[i, i] = (1.0, 1.0, 1.0)
        
        for j in i+1:n_vars
            r = corr_mat[i, j]
            
            if method == "pearson" || method == "spearman"
                # Преобразование Фишера
                z_r = atanh(r)
                
                # Стандартная ошибка
                se = 1 / sqrt(n_obs - 3)
                
                # Доверительный интервал в z-шкале
                lower_z = z_r - z * se
                upper_z = z_r + z * se
                
                # Обратное преобразование в r-шкалу
                lower_r = tanh(lower_z)
                upper_r = tanh(upper_z)
            elseif method == "kendall"
                # Приближение для тау Кендалла
                # Стандартная ошибка
                se = sqrt((4 * n_obs + 10) / (9 * n_obs * (n_obs - 1)))
                
                # Доверительный интервал
                lower_r = r - z * se
                upper_r = r + z * se
                
                # Ограничиваем интервал в пределах [-1, 1]
                lower_r = max(-1.0, lower_r)
                upper_r = min(1.0, upper_r)
            else
                throw(ArgumentError("Неподдерживаемый метод корреляции: $method"))
            end
            
            # Заполняем матрицу (она симметрична)
            result[i, j] = (r, lower_r, upper_r)
            result[j, i] = (r, lower_r, upper_r)
        end
    end
    
    return result
end

"""
    correlation_distance_matrix(data; method="pearson")

Вычисляет матрицу расстояний на основе коэффициентов корреляции.
Расстояние определяется как 1 - |r|, где r - коэффициент корреляции.

# Аргументы
- `data::AbstractMatrix{<:Real}`: матрица данных, где каждый столбец представляет одну переменную
- `method::String="pearson"`: метод вычисления корреляции ("pearson", "spearman", или "kendall")

# Возвращает
- `::Matrix{Float64}`: матрица расстояний

# Пример
```julia
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
dist_mat = correlation_distance_matrix(data)
```
"""
function correlation_distance_matrix(data::AbstractMatrix{<:Real}; method::String="pearson")
    # Вычисляем матрицу корреляции
    corr_mat = correlation_matrix(data, method=method)
    
    # Преобразуем корреляции в расстояния
    n_vars = size(corr_mat, 1)
    dist_mat = zeros(n_vars, n_vars)
    
    for i in 1:n_vars
        for j in 1:n_vars
            # Расстояние = 1 - |r|
            dist_mat[i, j] = 1.0 - abs(corr_mat[i, j])
        end
    end
    
    return dist_mat
end

"""
    correlation_heatmap(corr_mat; labels=nothing)

Подготавливает данные для визуализации матрицы корреляции в виде тепловой карты.
Примечание: для фактического отображения необходим пакет визуализации (Plots, Makie и т.д.).

# Аргументы
- `corr_mat::Matrix{<:Real}`: матрица корреляции
- `labels::Vector{String}=nothing`: необязательные метки для переменных

# Возвращает
- `::Dict`: словарь с данными для создания тепловой карты

# Пример
```julia
using Plots
data = [1.0 2.0 3.0; 2.0 3.0 3.5; 3.0 4.0 4.0; 4.0 5.0 6.0; 5.0 6.0 7.0]
corr_mat = correlation_matrix(data)
heatmap_data = correlation_heatmap(corr_mat, labels=["X", "Y", "Z"])
heatmap(heatmap_data["x"], heatmap_data["y"], heatmap_data["z"], 
        c=:RdBu, clims=(-1,1), aspect_ratio=:equal, 
        annotations=heatmap_data["annotations"])
```
"""
function correlation_heatmap(corr_mat::Matrix{<:Real}; labels::Union{Vector{String}, Nothing}=nothing)
    n_vars = size(corr_mat, 1)
    
    # Подготавливаем данные для тепловой карты
    if labels === nothing
        labels = ["Var$i" for i in 1:n_vars]
    elseif length(labels) != n_vars
        throw(ArgumentError("Количество меток должно соответствовать размеру матрицы корреляции"))
    end
    
    # Создаем аннотации (метки с коэффициентами корреляции)
    annotations = []
    for i in 1:n_vars
        for j in 1:n_vars
            push!(annotations, (j, i, @sprintf("%.2f", corr_mat[i, j])))
        end
    end
    
    # Создаем словарь с данными для тепловой карты
    return Dict(
        "x" => 1:n_vars,
        "y" => 1:n_vars,
        "z" => corr_mat,
        "labels" => labels,
        "annotations" => annotations
    )
end

"""
Аппроксимация кумулятивной функции распределения нормального распределения.
"""
function cdf_normal(x::Float64)
    # Приближение кумулятивной функции распределения
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
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

"""
Аппроксимация квантили нормального распределения.
"""
function quantile_normal(p::Float64)
    # Приближение квантили нормального распределения
    if p <= 0.0
        return -Inf
    elseif p >= 1.0
        return Inf
    elseif p == 0.5
        return 0.0
    end
    
    # Приближение Абрамовица и Стегуна
    if p > 0.5
        q = 1.0 - p
    else
        q = p
    end
    
    t = sqrt(-2.0 * log(q))
    x = t - (2.515517 + 0.802853*t + 0.010328*t^2) / 
              (1.0 + 1.432788*t + 0.189269*t^2 + 0.001308*t^3)
    
    return p > 0.5 ? -x : x
end

end # module 