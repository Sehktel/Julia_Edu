"""
    JacobianMatrix

Модуль, реализующий методы для работы с матрицей Якоби.
"""
module JacobianMatrix

export jacobian_matrix, jacobian_determinant, condition_number, 
       matrix_eigenvalues, matrix_eigenvectors, 
       analyze_critical_point, is_invertible

using LinearAlgebra

"""
    jacobian_matrix(f, x; h=1e-6)

Вычисляет матрицу Якоби для вектор-функции `f` в точке `x` с использованием 
метода конечных разностей.

# Матрица Якоби

Матрица Якоби для вектор-функции `f: ℝⁿ → ℝᵐ` в точке `x` определяется как:

```math
J(x) = \\begin{pmatrix}
\\frac{\\partial f_1}{\\partial x_1} & \\frac{\\partial f_1}{\\partial x_2} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\
\\frac{\\partial f_2}{\\partial x_1} & \\frac{\\partial f_2}{\\partial x_2} & \\cdots & \\frac{\\partial f_2}{\\partial x_n} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
\\frac{\\partial f_m}{\\partial x_1} & \\frac{\\partial f_m}{\\partial x_2} & \\cdots & \\frac{\\partial f_m}{\\partial x_n}
\\end{pmatrix}
```

Элемент `J_{ij}` матрицы представляет частную производную `∂f_i/∂x_j`.

# Аргументы
- `f::Function`: Вектор-функция, принимающая вектор `x` и возвращающая вектор того же или другого размера
- `x::Vector{<:Real}`: Точка, в которой вычисляется матрица Якоби
- `h::Float64=1e-6`: Шаг для конечно-разностной аппроксимации производных

# Возвращает
- `J::Matrix{Float64}`: Матрица Якоби размера m×n, где m - размерность выходного вектора f(x), 
  n - размерность входного вектора x

# Примеры
```julia
# Вектор-функция размерности 2×2
f(x) = [x[1]^2 + x[2], sin(x[1]) * x[2]]
x = [1.0, 2.0]
J = jacobian_matrix(f, x)
```
"""
function jacobian_matrix(f, x; h=1e-6)
    # Вычисляем базовое значение f(x)
    f0 = f(x)
    
    # Определяем размерности входного и выходного векторов
    n = length(x)     # Размерность входного вектора
    m = length(f0)    # Размерность выходного вектора
    
    # Создаем матрицу Якоби размера m×n
    J = zeros(m, n)
    
    # Вычисляем частные производные по каждой переменной
    for j in 1:n
        # Создаем копию вектора x с возмущением j-й компоненты
        x_plus = copy(x)
        x_plus[j] += h
        
        # Вычисляем f(x + h*e_j)
        f_plus = f(x_plus)
        
        # Аппроксимация производной: ∂f_i/∂x_j ≈ (f_i(x + h*e_j) - f_i(x)) / h
        J[:, j] = (f_plus - f0) / h
    end
    
    return J
end

"""
    jacobian_matrix(f, x, df; validate=true)

Вычисляет матрицу Якоби для вектор-функции `f` в точке `x` с использованием
аналитической функции производной `df`.

# Аргументы
- `f::Function`: Вектор-функция
- `x::Vector{<:Real}`: Точка, в которой вычисляется матрица Якоби
- `df::Function`: Функция, возвращающая матрицу Якоби аналитически (должна принимать x и возвращать матрицу)
- `validate::Bool=true`: Если true, проверяет аналитическое решение численным

# Возвращает
- `J::Matrix{Float64}`: Матрица Якоби

# Примеры
```julia
# Вектор-функция размерности 2×2
f(x) = [x[1]^2 + x[2], sin(x[1]) * x[2]]
# Аналитическая матрица Якоби
df(x) = [2*x[1] 1; cos(x[1])*x[2] sin(x[1])]
x = [1.0, 2.0]
J = jacobian_matrix(f, x, df)
```
"""
function jacobian_matrix(f, x, df; validate=true)
    # Вычисляем матрицу Якоби аналитически
    J_analytical = df(x)
    
    # Если нужна проверка, сравниваем с численным расчетом
    if validate
        J_numerical = jacobian_matrix(f, x)
        rel_error = norm(J_analytical - J_numerical) / norm(J_numerical)
        
        if rel_error > 1e-3
            @warn "Относительная ошибка между аналитическим и численным Якобианом: $rel_error"
        end
    end
    
    return J_analytical
end

"""
    jacobian_determinant(J)

Вычисляет определитель матрицы Якоби.

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби

# Возвращает
- `det_J::Float64`: Определитель матрицы Якоби

# Примечание
Определитель матрицы Якоби имеет важное значение в различных приложениях:
- В теории динамических систем определитель характеризует сжатие или расширение фазового объема
- При замене переменных в кратных интегралах, модуль определителя представляет коэффициент масштабирования
- Нулевой определитель указывает на особую точку (точку бифуркации)
"""
function jacobian_determinant(J)
    return det(J)
end

"""
    is_invertible(J; tol=1e-10)

Проверяет, является ли матрица Якоби обратимой (невырожденной).

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби
- `tol::Float64=1e-10`: Порог для определения нулевого определителя

# Возвращает
- `invertible::Bool`: true, если матрица обратима, иначе false
"""
function is_invertible(J; tol=1e-10)
    return abs(det(J)) > tol
end

"""
    condition_number(J)

Вычисляет число обусловленности матрицы Якоби.

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби

# Возвращает
- `cond_J::Float64`: Число обусловленности матрицы Якоби

# Примечание
Число обусловленности характеризует чувствительность решения системы уравнений к изменениям в исходных данных.
Большое число обусловленности указывает на плохо обусловленную матрицу, что может привести к численной нестабильности.
"""
function condition_number(J)
    return cond(J)
end

"""
    matrix_eigenvalues(J)

Вычисляет собственные значения матрицы Якоби.

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби

# Возвращает
- `λ::Vector{ComplexF64}`: Вектор собственных значений матрицы Якоби
"""
function matrix_eigenvalues(J)
    return eigvals(J)
end

"""
    matrix_eigenvectors(J)

Вычисляет собственные векторы матрицы Якоби.

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби

# Возвращает
- `λ::Vector{ComplexF64}`: Вектор собственных значений
- `V::Matrix{ComplexF64}`: Матрица собственных векторов, где i-й столбец соответствует i-му собственному значению
"""
function matrix_eigenvectors(J)
    λ, V = eigen(J)
    return λ, V
end

"""
    analyze_critical_point(J)

Анализирует устойчивость критической точки динамической системы по матрице Якоби.

# Аргументы
- `J::Matrix{<:Real}`: Матрица Якоби в критической точке

# Возвращает
- `stable::Bool`: true, если точка асимптотически устойчива
- `λ::Vector{ComplexF64}`: Собственные значения матрицы Якоби
- `type::String`: Тип критической точки (устойчивый/неустойчивый узел/фокус/седло/центр)
"""
function analyze_critical_point(J)
    # Вычисляем собственные значения
    λ = matrix_eigenvalues(J)
    
    # Проверяем действительные и мнимые части собственных значений
    real_parts = real.(λ)
    imag_parts = imag.(λ)
    
    # Система устойчива, если все действительные части отрицательны
    stable = all(real_parts .< 0)
    
    # Определяем тип устойчивости
    if all(real_parts .< 0)
        if all(imag_parts .≈ 0)
            stability_type = "устойчивый узел"
        else
            stability_type = "устойчивый фокус"
        end
    elseif all(real_parts .> 0)
        if all(imag_parts .≈ 0)
            stability_type = "неустойчивый узел"
        else
            stability_type = "неустойчивый фокус"
        end
    elseif any(real_parts .> 0) && any(real_parts .< 0)
        stability_type = "седло"
    elseif all(real_parts .≈ 0) && any(imag_parts .!= 0)
        stability_type = "центр"
    else
        stability_type = "нейтральный"
    end
    
    return stable, λ, stability_type
end

end # module 