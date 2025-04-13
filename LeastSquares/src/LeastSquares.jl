module LeastSquares

# Экспортируем основные функции и структуры
export 
    # Линейный МНК
    linear_least_squares, weighted_linear_least_squares,
    
    # Полиномиальный МНК
    polynomial_least_squares, weighted_polynomial_least_squares,
    
    # Нелинейный МНК
    nonlinear_least_squares, weighted_nonlinear_least_squares,
    
    # Оценка качества аппроксимации
    residuals, sum_of_squared_residuals, r_squared, adjusted_r_squared, 
    mean_squared_error, root_mean_squared_error, 
    mean_absolute_error, max_absolute_error,
    
    # Доверительные интервалы
    confidence_intervals

# Подключаем реализации
include("LinearLeastSquares.jl")
include("PolynomialLeastSquares.jl")
include("NonlinearLeastSquares.jl")
include("GoodnessOfFit.jl")

end # module LeastSquares 