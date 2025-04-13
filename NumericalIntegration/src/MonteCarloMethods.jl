"""
    MonteCarloMethods

Модуль содержит реализации методов Монте-Карло для численного интегрирования.

Методы Монте-Карло основаны на стохастическом подходе к вычислению интегралов
и особенно эффективны для многомерных интегралов и сложных областей интегрирования.
"""
module MonteCarloMethods

using Random, Statistics
using ..BasicMethods: IntegrationResult

export monte_carlo_integration, monte_carlo_importance_sampling
export monte_carlo_stratified, monte_carlo_quasi_random

"""
    monte_carlo_integration(f, a, b, n_samples=10000; dims=1, rng=Random.GLOBAL_RNG)

Вычисляет интеграл функции `f` методом Монте-Карло.

# Аргументы
- `f::Function`: интегрируемая функция
- `a`: нижняя граница (или вектор границ для многомерного случая)
- `b`: верхняя граница (или вектор границ для многомерного случая)
- `n_samples::Int=10000`: число точек для генерации
- `dims::Int=1`: размерность интеграла
- `rng=Random.GLOBAL_RNG`: генератор случайных чисел

# Возвращает
- `result::Float64`: приближенное значение интеграла
- `error::Float64`: оценка стандартной ошибки

# Примеры
```julia
# Интегрирование sin(x) от 0 до π
result, error = monte_carlo_integration(sin, 0, π, 100000)

# Многомерный интеграл: ∫∫∫ x*y*z dV в кубе [0,1]³
f(x) = x[1] * x[2] * x[3]
result, error = monte_carlo_integration(f, zeros(3), ones(3), 100000, dims=3)
```
"""
function monte_carlo_integration(f::Function, a, b, n_samples::Int=10000; 
                                dims::Int=1, rng=Random.GLOBAL_RNG)
    if dims == 1
        # Одномерный случай
        volume = b - a
        samples = a .+ (b - a) .* rand(rng, n_samples)
        
        # Вычисляем значения функции в точках
        f_values = [f(x) for x in samples]
        
        # Оцениваем интеграл и ошибку
        mean_f = mean(f_values)
        std_f = std(f_values, corrected=true)
        
        # Интеграл и стандартная ошибка
        integral = volume * mean_f
        error = volume * std_f / sqrt(n_samples)
        
        return integral, error
    else
        # Многомерный случай
        if !isa(a, AbstractVector) || !isa(b, AbstractVector)
            throw(ArgumentError("Для многомерного интеграла границы a и b должны быть векторами"))
        end
        
        if length(a) != dims || length(b) != dims
            throw(ArgumentError("Длина векторов a и b должна соответствовать размерности dims"))
        end
        
        # Объем области интегрирования
        volume = prod(b - a)
        
        # Генерируем случайные точки
        points = [a .+ (b - a) .* rand(rng, dims) for _ in 1:n_samples]
        
        # Вычисляем значения функции
        f_values = [f(x) for x in points]
        
        # Оцениваем интеграл и ошибку
        mean_f = mean(f_values)
        std_f = std(f_values, corrected=true)
        
        # Интеграл и стандартная ошибка
        integral = volume * mean_f
        error = volume * std_f / sqrt(n_samples)
        
        return integral, error
    end
end

"""
    monte_carlo_importance_sampling(f, a, b, pdf, sampler, n_samples=10000)

Метод Монте-Карло с выборкой по важности (importance sampling).

# Аргументы
- `f::Function`: интегрируемая функция
- `a`: нижняя граница 
- `b`: верхняя граница
- `pdf::Function`: функция плотности вероятности для выборки
- `sampler::Function`: функция для генерации случайных чисел по заданному распределению
- `n_samples::Int=10000`: число точек для генерации

# Возвращает
- `result::Float64`: приближенное значение интеграла
- `error::Float64`: оценка стандартной ошибки

# Пример
```julia
# Функция с пиком около x = 0
f(x) = 1 / (0.1 + x^2)

# Выборка по распределению Коши
cauchy_pdf(x) = 1 / (π * (1 + x^2))
cauchy_sampler(n) = tan.(π * (rand(n) .- 0.5))

result, error = monte_carlo_importance_sampling(f, -10, 10, cauchy_pdf, cauchy_sampler, 10000)
```
"""
function monte_carlo_importance_sampling(f::Function, a, b, pdf::Function, 
                                        sampler::Function, n_samples::Int=10000)
    # Генерируем выборку
    samples = sampler(n_samples)
    
    # Фильтруем точки в пределах интервала [a, b]
    in_range = findall(x -> a <= x <= b, samples)
    samples = samples[in_range]
    
    # Если слишком мало точек попало в интервал, генерируем еще
    while length(samples) < 0.1 * n_samples
        new_samples = sampler(n_samples)
        in_range = findall(x -> a <= x <= b, new_samples)
        append!(samples, new_samples[in_range])
    end
    
    # Ограничиваем число точек
    if length(samples) > n_samples
        samples = samples[1:n_samples]
    end
    
    # Вычисляем отношение f/pdf для каждой точки
    ratios = [f(x) / pdf(x) for x in samples]
    
    # Среднее и стандартное отклонение
    mean_ratio = mean(ratios)
    std_ratio = std(ratios, corrected=true)
    
    # Интеграл и стандартная ошибка
    integral = mean_ratio
    error = std_ratio / sqrt(length(samples))
    
    return integral, error
end

"""
    monte_carlo_stratified(f, a, b, n_strata; n_samples_per_stratum=10)

Метод Монте-Карло со стратифицированной выборкой.

# Аргументы
- `f::Function`: интегрируемая функция
- `a`: нижняя граница (или вектор границ для многомерного случая)
- `b`: верхняя граница (или вектор границ для многомерного случая)
- `n_strata::Int`: число страт (разбиений) для каждого измерения
- `n_samples_per_stratum::Int=10`: число точек в каждой страте

# Возвращает
- `result::Float64`: приближенное значение интеграла
- `error::Float64`: оценка стандартной ошибки

# Пример
```julia
# Интегрирование с разбиением на 20 страт
result, error = monte_carlo_stratified(sin, 0, π, 20)

# Двумерный случай с 10x10 стратами
f(x) = x[1]^2 + x[2]^2
result, error = monte_carlo_stratified(f, [0, 0], [1, 1], 10)
```
"""
function monte_carlo_stratified(f::Function, a, b, n_strata::Int; 
                               n_samples_per_stratum::Int=10)
    if isa(a, AbstractVector) && isa(b, AbstractVector)
        # Многомерный случай
        dims = length(a)
        
        if length(b) != dims
            throw(ArgumentError("Длина векторов a и b должна быть одинаковой"))
        end
        
        # Общий объем области
        volume = prod(b - a)
        
        # Размер каждой страты
        stratum_sizes = (b - a) ./ n_strata
        
        # Объем одной страты
        stratum_volume = volume / (n_strata^dims)
        
        # Создаем индексы для каждой страты
        stratum_indices = [CartesianIndices(ntuple(_ -> n_strata, dims))...]
        
        # Инициализируем массив для результатов
        stratum_integrals = zeros(length(stratum_indices))
        stratum_variances = zeros(length(stratum_indices))
        
        # Для каждой страты
        for (i, idx) in enumerate(stratum_indices)
            # Вычисляем границы страты
            stratum_a = [a[d] + (idx[d] - 1) * stratum_sizes[d] for d in 1:dims]
            stratum_b = [a[d] + idx[d] * stratum_sizes[d] for d in 1:dims]
            
            # Генерируем точки в страте
            points = [stratum_a .+ (stratum_b - stratum_a) .* rand(dims) for _ in 1:n_samples_per_stratum]
            
            # Вычисляем значения функции
            f_values = [f(x) for x in points]
            
            # Оцениваем интеграл и дисперсию для этой страты
            stratum_integrals[i] = mean(f_values) * stratum_volume
            stratum_variances[i] = var(f_values, corrected=true) * (stratum_volume^2) / n_samples_per_stratum
        end
        
        # Общий интеграл и ошибка
        integral = sum(stratum_integrals)
        error = sqrt(sum(stratum_variances))
        
        return integral, error
    else
        # Одномерный случай
        stratum_size = (b - a) / n_strata
        
        # Инициализируем массивы для результатов
        stratum_integrals = zeros(n_strata)
        stratum_variances = zeros(n_strata)
        
        # Для каждой страты
        for i in 1:n_strata
            stratum_a = a + (i - 1) * stratum_size
            stratum_b = a + i * stratum_size
            
            # Генерируем точки в страте
            points = stratum_a .+ (stratum_b - stratum_a) .* rand(n_samples_per_stratum)
            
            # Вычисляем значения функции
            f_values = [f(x) for x in points]
            
            # Оцениваем интеграл и дисперсию для этой страты
            stratum_integrals[i] = mean(f_values) * stratum_size
            stratum_variances[i] = var(f_values, corrected=true) * (stratum_size^2) / n_samples_per_stratum
        end
        
        # Общий интеграл и ошибка
        integral = sum(stratum_integrals)
        error = sqrt(sum(stratum_variances))
        
        return integral, error
    end
end

"""
    monte_carlo_quasi_random(f, a, b, n_samples=1000; sequence=:sobol)

Квази-случайный метод Монте-Карло с использованием низкодисперсионных последовательностей.

# Аргументы
- `f::Function`: интегрируемая функция
- `a`: нижняя граница (или вектор границ для многомерного случая)
- `b`: верхняя граница (или вектор границ для многомерного случая)
- `n_samples::Int=1000`: число точек
- `sequence::Symbol=:sobol`: тип последовательности (:sobol, :halton)

# Возвращает
- `result::Float64`: приближенное значение интеграла

# Примечания
- Для использования метода требуется установить пакет QuasiMonteCarlo.jl

# Пример
```julia
using QuasiMonteCarlo
result = monte_carlo_quasi_random(sin, 0, π, 1000, sequence=:sobol)
```
"""
function monte_carlo_quasi_random(f::Function, a, b, n_samples::Int=1000; sequence::Symbol=:sobol)
    @warn "Для использования этой функции необходимо установить пакет QuasiMonteCarlo.jl"
    @warn "Установите его командой: using Pkg; Pkg.add(\"QuasiMonteCarlo\")"
    
    error("Функция требует пакета QuasiMonteCarlo.jl")
    
    # # Этот код будет работать при наличии QuasiMonteCarlo.jl
    # if sequence == :sobol
    #     qrng = QuasiMonteCarlo.SobolSample()
    # elseif sequence == :halton
    #     qrng = QuasiMonteCarlo.HaltonSample()
    # else
    #     throw(ArgumentError("Неизвестный тип последовательности: $sequence"))
    # end
    # 
    # if isa(a, AbstractVector) && isa(b, AbstractVector)
    #     # Многомерный случай
    #     dims = length(a)
    #     volume = prod(b - a)
    #     
    #     # Генерируем квази-случайные точки
    #     points = QuasiMonteCarlo.sample(n_samples, a, b, qrng)
    #     
    #     # Вычисляем значения функции
    #     f_values = [f(points[:, i]) for i in 1:n_samples]
    # else
    #     # Одномерный случай
    #     dims = 1
    #     volume = b - a
    #     
    #     # Генерируем квази-случайные точки
    #     points = QuasiMonteCarlo.sample(n_samples, [a], [b], qrng)
    #     
    #     # Вычисляем значения функции
    #     f_values = [f(points[1, i]) for i in 1:n_samples]
    # end
    # 
    # # Вычисляем интеграл
    # integral = volume * mean(f_values)
    # 
    # return integral
end

end # module MonteCarloMethods 