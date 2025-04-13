"""
    NumericalIntegration

Модуль, предоставляющий различные методы численного интегрирования.

Основная функция:
- `integrate(f, a, b, method=:trapezoid; kwargs...)`: универсальный метод для численного интегрирования

Доступные методы:
- Базовые методы: `:rectangle`, `:midpoint`, `:trapezoid`, `:simpson`, `:simpson38`, `:boole`
- Составные методы: `:composite_rectangle`, `:composite_midpoint`, `:composite_trapezoid`, `:composite_simpson`
- Адаптивные методы: `:adaptive`, `:adaptive_simpson`, `:adaptive_quadrature`
- Методы Монте-Карло: `:monte_carlo`, `:importance_sampling`, `:monte_carlo_stratified`
- Кратные интегралы: `:double_rectangle`, `:double_trapezoid`, `:double_simpson`,
  `:triple_rectangle`, `:triple_trapezoid`, `:triple_simpson`, `:monte_carlo_multidim`
- Интегралы с особенностями: `:singularity_subtraction`, `:singularity_transformation`, 
  `:tanh_sinh`, `:double_exponential`, `:log_weighted`, `:cauchy_principal_value`
"""
module NumericalIntegration

# Экспорт основных функций
export integrate
# Структуры
export IntegrationResult
# Базовые методы
export rectangle_method, midpoint_method, trapezoid_method
export simpson_method, simpson38_method, boole_method
# Составные методы
export composite_rectangle, composite_midpoint, composite_trapezoid
export composite_simpson, romberg_method
# Адаптивные методы
export adaptive_integration, adaptive_simpson, adaptive_quadrature
# Методы Монте-Карло
export monte_carlo_integration, monte_carlo_importance_sampling
export monte_carlo_stratified, monte_carlo_quasi_random
# Кратные интегралы
export double_integral, triple_integral, multidimensional_integral, iterated_integral
# Интегралы с особенностями
export singularity_subtraction, singularity_transformation
export tanh_sinh_quadrature, dblexp_rule, log_weighted_integration
export cauchy_principal_value

# Включаем необходимые модули из стандартной библиотеки
using Random, Statistics

# Подключаем базовые методы
include("BasicMethods.jl")
using .BasicMethods

# Подключаем составные методы
include("CompositeMethods.jl")
using .CompositeMethods

# Подключаем адаптивные методы
include("AdaptiveMethods.jl")
using .AdaptiveMethods

# Подключаем методы Монте-Карло
include("MonteCarloMethods.jl")
using .MonteCarloMethods

# Подключаем методы для кратных интегралов
include("MultipleIntegrals.jl")
using .MultipleIntegrals

# Подключаем методы для интегралов с особенностями
include("SingularIntegrals.jl")
using .SingularIntegrals

"""
    integrate(f, a, b, method=:trapezoid; kwargs...)

Универсальная функция для численного интегрирования функции `f` на интервале `[a, b]`.

# Аргументы
- `f::Function`: интегрируемая функция
- `a::Real`: нижняя граница интегрирования
- `b::Real`: верхняя граница интегрирования
- `method::Symbol=:trapezoid`: метод интегрирования
- `kwargs...`: дополнительные параметры, специфичные для выбранного метода

# Доступные методы
- Базовые методы: `:rectangle`, `:midpoint`, `:trapezoid`, `:simpson`, `:simpson38`, `:boole`
- Составные методы: `:composite_rectangle`, `:composite_midpoint`, `:composite_trapezoid`, `:composite_simpson`
- Адаптивные методы: `:adaptive`, `:adaptive_simpson`, `:adaptive_quadrature`
- Методы Монте-Карло: `:monte_carlo`, `:monte_carlo_importance`, `:monte_carlo_stratified`
- Интегралы с особенностями: `:tanh_sinh`, `:double_exponential`, `:log_weighted`, `:cauchy_principal_value`

# Возвращает
- `::Float64` или `::IntegrationResult`: результат интегрирования или структуру с результатом и дополнительной информацией

# Пример
```julia
# Интегрирование sin(x) от 0 до π методом трапеций
result = integrate(sin, 0, π, :trapezoid)  # ≈ 2.0

# Использование составного метода Симпсона с 100 подынтервалами
result = integrate(x -> x^2, 0, 1, :composite_simpson, n=100)

# Адаптивное интегрирование с заданной точностью
result = integrate(exp, 0, 1, :adaptive, tol=1e-10)

# Метод Монте-Карло с 10000 точками
result = integrate(x -> x^3, 0, 1, :monte_carlo, n_samples=10000)

# Интегрирование функции с логарифмической особенностью
result = integrate(log, 0, 1, :tanh_sinh, n_points=50)
```
"""
function integrate(f::Function, a::Real, b::Real, method::Symbol=:trapezoid; kwargs...)
    if a > b
        # Если границы указаны в обратном порядке, меняем знак результата
        return -integrate(f, b, a, method; kwargs...)
    end
    
    # Базовые методы
    if method == :rectangle || method == :left_rectangle
        return rectangle_method(f, a, b, position=:left; kwargs...)
    elseif method == :right_rectangle
        return rectangle_method(f, a, b, position=:right; kwargs...)
    elseif method == :midpoint
        return midpoint_method(f, a, b; kwargs...)
    elseif method == :trapezoid
        return trapezoid_method(f, a, b; kwargs...)
    elseif method == :simpson
        return simpson_method(f, a, b; kwargs...)
    elseif method == :simpson38
        return simpson38_method(f, a, b; kwargs...)
    elseif method == :boole
        return boole_method(f, a, b; kwargs...)
    
    # Составные методы
    elseif method == :composite_rectangle || method == :composite_left_rectangle
        return composite_rectangle(f, a, b, position=:left; kwargs...)
    elseif method == :composite_right_rectangle
        return composite_rectangle(f, a, b, position=:right; kwargs...)
    elseif method == :composite_midpoint
        return composite_midpoint(f, a, b; kwargs...)
    elseif method == :composite_trapezoid
        return composite_trapezoid(f, a, b; kwargs...)
    elseif method == :composite_simpson
        return composite_simpson(f, a, b; kwargs...)
    elseif method == :romberg
        return romberg_method(f, a, b; kwargs...)
    
    # Адаптивные методы
    elseif method == :adaptive || method == :adaptive_integration
        return adaptive_integration(f, a, b; kwargs...)
    elseif method == :adaptive_simpson
        return adaptive_simpson(f, a, b; kwargs...)
    elseif method == :adaptive_quadrature
        return adaptive_quadrature(f, a, b; kwargs...)
    
    # Методы Монте-Карло
    elseif method == :monte_carlo
        return monte_carlo_integration(f, a, b; kwargs...)[1]
    elseif method == :monte_carlo_importance
        if !haskey(kwargs, :pdf) || !haskey(kwargs, :sampler)
            throw(ArgumentError("Для метода monte_carlo_importance необходимо указать pdf и sampler"))
        end
        return monte_carlo_importance_sampling(f, a, b, kwargs[:pdf], kwargs[:sampler]; 
                                             [(k => v) for (k, v) in kwargs if k != :pdf && k != :sampler]...)[1]
    elseif method == :monte_carlo_stratified
        return monte_carlo_stratified(f, a, b; kwargs...)[1]
    
    # Методы для интегралов с особенностями
    elseif method == :tanh_sinh
        return tanh_sinh_quadrature(f, a, b; kwargs...)
    elseif method == :double_exponential
        return dblexp_rule(f, a, b; kwargs...)
    elseif method == :log_weighted
        return log_weighted_integration(f, a, b; kwargs...)
    elseif method == :cauchy_principal_value
        if !haskey(kwargs, :c)
            throw(ArgumentError("Для метода cauchy_principal_value необходимо указать точку особенности c"))
        end
        return cauchy_principal_value(f, a, b, kwargs[:c]; 
                                    [(k => v) for (k, v) in kwargs if k != :c]...)
    
    else
        throw(ArgumentError("Неизвестный метод интегрирования: $method"))
    end
end

end # module NumericalIntegration 