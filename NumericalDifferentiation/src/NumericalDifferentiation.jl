"""
    NumericalDifferentiation

Модуль для численного дифференцирования функций одной и нескольких переменных.
Включает различные методы аппроксимации производных и оценки погрешности.
"""
module NumericalDifferentiation

using LinearAlgebra

# Экспорт основных методов численного дифференцирования
export forward_difference, backward_difference, central_difference
export second_derivative, higher_derivative, mixed_derivative
export richardson_extrapolation, adaptive_differentiation
export estimate_error, optimal_step_size
export differentiate_data, smooth_differentiate

# Экспорт структур и типов
export DifferentiationMethod, ForwardDifference, BackwardDifference, CentralDifference
export HigherOrderMethod, RichardsonMethod, AdaptiveMethod

# Включение файлов с реализациями
include("DifferentiationStructures.jl")  # Структуры и типы для методов дифференцирования
include("BasicMethods.jl")               # Базовые методы: вперед, назад, центральный
include("FiniteDifferences.jl")          # Методы конечных разностей высокого порядка
include("ErrorEstimation.jl")            # Оценка ошибки и выбор оптимального шага
include("RichardsonExtrapolation.jl")    # Экстраполяция Ричардсона для повышения точности

# Функция для конвертации аргументов в соответствующий тип метода
function differentiate(f::Function, x::Real, method::Symbol=:central; 
                     step::Real=1e-8, order::Int=1, kwargs...)
    if method == :forward
        return forward_difference(f, x, step, order; kwargs...)
    elseif method == :backward
        return backward_difference(f, x, step, order; kwargs...)
    elseif method == :central
        return central_difference(f, x, step, order; kwargs...)
    elseif method == :richardson
        return richardson_extrapolation(f, x, step, order; kwargs...)
    elseif method == :adaptive
        return adaptive_differentiation(f, x; kwargs...)
    else
        throw(ArgumentError("Неизвестный метод дифференцирования: $method"))
    end
end

end # module 