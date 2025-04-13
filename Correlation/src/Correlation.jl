module Correlation

# Экспортируем основные функции и структуры
export 
    # Точечные оценки корреляции
    pearson_correlation, spearman_correlation, kendall_correlation,
    correlation_ratio,
    
    # Матрицы корреляции
    correlation_matrix, partial_correlation_matrix,
    
    # Множественная корреляция
    multiple_correlation, multiple_determination,
    
    # Регрессионный анализ
    linear_regression, polynomial_regression, exponential_regression,
    power_regression, logarithmic_regression,
    
    # Подбор параметров
    fit_model, goodness_of_fit, predict

# Подключаем реализации
include("CorrelationCoefficients.jl")
include("CorrelationMatrix.jl")
include("MultipleCorrelation.jl")
include("Regression.jl")
include("ModelFitting.jl")

end # module Correlation 