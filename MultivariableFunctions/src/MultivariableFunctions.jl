module MultivariableFunctions

# Экспортируем основные функции и структуры
export 
    # Основные операции с функциями двух переменных
    evaluate, gradient, partial_derivative,
    
    # Визуализация
    plot_function, plot_contour, plot_gradient_field,
    
    # Экстремумы функций
    find_critical_points, classify_critical_point,
    
    # Интегрирование
    double_integral, integrate_region

# Подключаем реализации
include("BasicOperations.jl")
include("Visualization.jl")
include("Extremum.jl")
include("Integration.jl")

end # module MultivariableFunctions 