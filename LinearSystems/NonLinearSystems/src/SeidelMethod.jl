"""
    SeidelMethod

Модуль, реализующий метод Зейделя для решения систем нелинейных уравнений.
"""
module SeidelMethod

export seidel_solve, 
       check_convergence_seidel

"""
    check_convergence_seidel(f, x₀, tol=1e-8)

Проверяет условие сходимости метода Зейделя в окрестности точки x₀.
Метод сходится, если диагонально доминирующая матрица частных производных.

# Аргументы
- `f::Function`: Функция системы уравнений
- `x₀::Vector{Float64}`: Текущее приближение
- `tol::Float64=1e-8`: Малая величина для численного дифференцирования

# Возвращает
- `converges::Bool`: Флаг, показывающий, выполняется ли условие сходимости
- `jacobian::Matrix{Float64}`: Матрица Якоби в точке x₀
"""
function check_convergence_seidel(f, x₀, tol=1e-8)
    n = length(x₀)
    # Матрица Якоби для f
    J = zeros(n, n)
    
    # Вычисляем значение функции в текущей точке
    f_x0 = f(x₀)
    if length(f_x0) != n
        throw(DimensionMismatch("Размерность функции f должна соответствовать размерности вектора x₀"))
    end
    
    # Численное дифференцирование для заполнения матрицы Якоби
    for i in 1:n
        x_plus = copy(x₀)
        x_plus[i] += tol
        
        # Аппроксимация частных производных
        J[:, i] = (f(x_plus) - f_x0) / tol
    end
    
    # Проверка условия диагонального преобладания
    converges = true
    for i in 1:n
        # Сумма абсолютных значений недиагональных элементов в i-й строке
        row_sum = sum(abs.(J[i,j]) for j in 1:n if j != i)
        # Условие диагонального преобладания: |a_ii| > sum(|a_ij|) для j≠i
        if abs(J[i,i]) <= row_sum
            converges = false
            break
        end
    end
    
    return converges, J
end

"""
    seidel_solve(f, x₀; max_iter=1000, tol=1e-6, check_condition=true)

Решает систему нелинейных уравнений f(x) = 0 методом Зейделя.
Метод Зейделя является модификацией метода простой итерации, где каждая компонента
решения сразу используется при вычислении следующих компонент.

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x₀::Vector{Float64}`: Начальное приближение
- `max_iter::Int=1000`: Максимальное число итераций
- `tol::Float64=1e-6`: Допустимая погрешность
- `check_condition::Bool=true`: Проверять ли условие сходимости
- `relaxation::Float64=1.0`: Параметр релаксации (ω)

# Возвращает
- `x::Vector{Float64}`: Найденное решение
- `converged::Bool`: Флаг сходимости метода
- `iter_count::Int`: Число выполненных итераций
- `errors::Vector{Float64}`: История ошибок на каждой итерации
"""
function seidel_solve(f, x₀; max_iter=1000, tol=1e-6, check_condition=true, relaxation=1.0)
    # Проверка условия сходимости
    if check_condition
        converges, J = check_convergence_seidel(f, x₀)
        if !converges
            @warn "Матрица системы не является диагонально доминирующей, сходимость метода Зейделя не гарантирована"
        end
    end
    
    # Инициализация
    n = length(x₀)
    x = copy(x₀)
    x_new = copy(x)
    errors = Float64[]
    
    # Определяем функцию для решения i-го уравнения относительно x[i]
    # f_i(x) = 0 => x[i] = x[i] - f_i(x) / df_i/dx_i
    function solve_for_component(f, x, i, tol=1e-8)
        # Вычисляем значение функции
        f_val = f(x)[i]
        
        # Если значение функции близко к нулю, компонента уже решена
        if abs(f_val) < tol
            return x[i]
        end
        
        # Вычисляем частную производную по i-й компоненте
        x_plus = copy(x)
        x_plus[i] += tol
        df_i = (f(x_plus)[i] - f_val) / tol
        
        # Предотвращаем деление на очень маленькие значения
        if abs(df_i) < tol
            df_i = sign(df_i) * tol
        end
        
        # Новое значение x[i]
        return x[i] - f_val / df_i
    end
    
    for iter in 1:max_iter
        # Применяем метод Зейделя: сразу используем обновленные компоненты
        for i in 1:n
            # Решаем i-е уравнение относительно x[i] с обновленными компонентами
            new_val = solve_for_component(f, x_new, i)
            
            # Применяем параметр релаксации: x_new[i] = (1-ω)*x[i] + ω*new_val
            x_new[i] = (1.0 - relaxation) * x[i] + relaxation * new_val
        end
        
        # Вычисление ошибки
        error = norm(x_new - x) / max(norm(x), 1.0)
        push!(errors, error)
        
        # Проверка сходимости
        if error < tol
            return x_new, true, iter, errors
        end
        
        # Обновление текущего приближения
        x .= x_new
    end
    
    # Если метод не сошелся за максимальное число итераций
    return x, false, max_iter, errors
end

end # module 