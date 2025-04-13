"""
    NewtonMethod

Модуль, реализующий метод Ньютона для решения систем нелинейных уравнений.
"""
module NewtonMethod

export newton_solve, 
       compute_jacobian, 
       modified_newton_solve

"""
    compute_jacobian(f, x, tol=1e-8)

Вычисляет матрицу Якоби функции f в точке x методом конечных разностей.

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x::Vector{Float64}`: Точка, в которой вычисляется матрица Якоби
- `tol::Float64=1e-8`: Малая величина для численного дифференцирования

# Возвращает
- `J::Matrix{Float64}`: Матрица Якоби в точке x
"""
function compute_jacobian(f, x, tol=1e-8)
    n = length(x)
    # Вычисляем значение функции в текущей точке
    f_x = f(x)
    if length(f_x) != n
        throw(DimensionMismatch("Размерность функции f должна соответствовать размерности вектора x"))
    end
    
    # Матрица Якоби для f
    J = zeros(n, n)
    
    # Численное дифференцирование
    for i in 1:n
        x_plus = copy(x)
        x_plus[i] += tol
        
        # Аппроксимация частных производных
        J[:, i] = (f(x_plus) - f_x) / tol
    end
    
    return J
end

"""
    newton_solve(f, x₀; max_iter=100, tol=1e-6, jacobian_tol=1e-8)

Решает систему нелинейных уравнений f(x) = 0 методом Ньютона.
Метод Ньютона использует линейное приближение функции в окрестности текущей точки
и находит следующее приближение, решая линейную систему J(x_k)·Δx = -f(x_k).

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x₀::Vector{Float64}`: Начальное приближение
- `max_iter::Int=100`: Максимальное число итераций
- `tol::Float64=1e-6`: Допустимая погрешность
- `jacobian_tol::Float64=1e-8`: Точность для вычисления матрицы Якоби

# Возвращает
- `x::Vector{Float64}`: Найденное решение
- `converged::Bool`: Флаг сходимости метода
- `iter_count::Int`: Число выполненных итераций
- `errors::Vector{Float64}`: История ошибок на каждой итерации
"""
function newton_solve(f, x₀; max_iter=100, tol=1e-6, jacobian_tol=1e-8)
    # Инициализация
    x = copy(x₀)
    errors = Float64[]
    
    for iter in 1:max_iter
        # Вычисление значения функции в текущей точке
        f_val = f(x)
        
        # Проверка сходимости по значению функции
        f_norm = norm(f_val)
        push!(errors, f_norm)
        
        if f_norm < tol
            return x, true, iter, errors
        end
        
        # Вычисление матрицы Якоби
        J = compute_jacobian(f, x, jacobian_tol)
        
        # Решение линейной системы J(x_k)·Δx = -f(x_k)
        try
            # Используем решение системы линейных уравнений
            Δx = J \ (-f_val)
            
            # Обновление приближения: x_{k+1} = x_k + Δx
            x += Δx
        catch e
            # Если матрица Якоби близка к сингулярной
            if isa(e, LinearAlgebra.SingularException)
                @warn "Матрица Якоби близка к сингулярной на итерации $iter"
                # Можно использовать регуляризацию или псевдообратную матрицу
                Δx = pinv(J) * (-f_val)
                x += Δx
            else
                rethrow(e)
            end
        end
    end
    
    # Если метод не сошелся за максимальное число итераций
    return x, false, max_iter, errors
end

"""
    modified_newton_solve(f, x₀; max_iter=100, tol=1e-6, jacobian_tol=1e-8, jacobian_update_freq=5)

Решает систему нелинейных уравнений f(x) = 0 модифицированным методом Ньютона.
Отличается от классического метода Ньютона тем, что матрица Якоби не пересчитывается 
на каждой итерации, а обновляется через заданное количество итераций.

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x₀::Vector{Float64}`: Начальное приближение
- `max_iter::Int=100`: Максимальное число итераций
- `tol::Float64=1e-6`: Допустимая погрешность
- `jacobian_tol::Float64=1e-8`: Точность для вычисления матрицы Якоби
- `jacobian_update_freq::Int=5`: Частота обновления матрицы Якоби

# Возвращает
- `x::Vector{Float64}`: Найденное решение
- `converged::Bool`: Флаг сходимости метода
- `iter_count::Int`: Число выполненных итераций
- `errors::Vector{Float64}`: История ошибок на каждой итерации
"""
function modified_newton_solve(f, x₀; max_iter=100, tol=1e-6, jacobian_tol=1e-8, jacobian_update_freq=5)
    # Инициализация
    x = copy(x₀)
    errors = Float64[]
    
    # Вычисление начальной матрицы Якоби
    J = compute_jacobian(f, x, jacobian_tol)
    
    for iter in 1:max_iter
        # Вычисление значения функции в текущей точке
        f_val = f(x)
        
        # Проверка сходимости по значению функции
        f_norm = norm(f_val)
        push!(errors, f_norm)
        
        if f_norm < tol
            return x, true, iter, errors
        end
        
        # Обновление матрицы Якоби только через заданное количество итераций
        if iter % jacobian_update_freq == 1
            J = compute_jacobian(f, x, jacobian_tol)
        end
        
        # Решение линейной системы J(x_k)·Δx = -f(x_k)
        try
            # Используем решение системы линейных уравнений
            Δx = J \ (-f_val)
            
            # Обновление приближения: x_{k+1} = x_k + Δx
            x += Δx
        catch e
            # Если матрица Якоби близка к сингулярной
            if isa(e, LinearAlgebra.SingularException)
                @warn "Матрица Якоби близка к сингулярной на итерации $iter"
                # Пересчитываем матрицу Якоби
                J = compute_jacobian(f, x, jacobian_tol)
                Δx = pinv(J) * (-f_val)
                x += Δx
            else
                rethrow(e)
            end
        end
    end
    
    # Если метод не сошелся за максимальное число итераций
    return x, false, max_iter, errors
end

end # module 