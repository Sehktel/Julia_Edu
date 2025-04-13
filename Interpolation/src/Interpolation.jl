module Interpolation

# Экспортируем публичные функции и структуры
export 
    # Функции для точечной и непрерывной аппроксимации
    point_approximation, continuous_approximation,
    
    # Линейная и квадратичная интерполяция
    linear_interpolation, quadratic_interpolation,
    
    # Сплайн-интерполяция
    cubic_spline, natural_spline, clamped_spline,
    
    # Полином Лагранжа
    lagrange_polynomial, lagrange_interpolation, interpolation_error

# Подключаем реализации
include("ApproximationProblem.jl")
include("LinearQuadraticInterpolation.jl")
include("SplineInterpolation.jl")
include("LagrangeInterpolation.jl")

end # module Interpolation 