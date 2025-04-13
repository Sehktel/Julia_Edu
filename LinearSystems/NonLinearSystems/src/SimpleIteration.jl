"""
    SimpleIteration

Модуль, реализующий метод простой итерации для решения систем нелинейных уравнений.
"""
module SimpleIteration

export simple_iteration_solve, 
       prepare_iteration_function, 
       check_convergence_simple_iteration

"""
    prepare_iteration_function(f, x₀, α=0.01)

Подготавливает итерационную функцию для метода простой итерации из системы уравнений f(x) = 0.
Используется преобразование x_{k+1} = x_k - α*f(x_k), где α - параметр релаксации.

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x₀::Vector{Float64}`: Начальное приближение
- `α::Float64=0.01`: Параметр релаксации

# Возвращает
- `g::Function`: Итерационная функция
"""
function prepare_iteration_function(f, x₀, α=0.01)
    # Проверяем значение параметра релаксации
    if α <= 0 || α >= 2
        throw(ArgumentError("Параметр релаксации α должен быть в диапазоне (0, 2)"))
    end
    
    # Возвращаем итерационную функцию: x_{k+1} = x_k - α*f(x_k)
    return x -> x .- α .* f(x)
end

"""
    check_convergence_simple_iteration(g, x, tol=1e-8)

Проверяет условие сходимости метода простой итерации в окрестности точки x.

# Аргументы
- `g::Function`: Итерационная функция
- `x::Vector{Float64}`: Текущее приближение
- `tol::Float64=1e-8`: Малая величина для численного дифференцирования

# Возвращает
- `converges::Bool`: Флаг, показывающий, выполняется ли условие сходимости
- `max_deriv::Float64`: Максимальная норма производной в точке x
"""
function check_convergence_simple_iteration(g, x, tol=1e-8)
    n = length(x)
    # Матрица Якоби для g
    J = zeros(n, n)
    
    # Численное дифференцирование
    for i in 1:n
        x_plus = copy(x)
        x_plus[i] += tol
        
        # Аппроксимация частных производных
        J[:, i] = (g(x_plus) - g(x)) / tol
    end
    
    # Норма матрицы Якоби (используем спектральную норму)
    # Для метода простой итерации условие сходимости: ||J|| < 1
    eigvals = eigen(J).values
    max_deriv = maximum(abs.(eigvals))
    
    return max_deriv < 1, max_deriv
end

"""
    simple_iteration_solve(f, x₀; α=0.01, max_iter=1000, tol=1e-6, check_condition=true)

Решает систему нелинейных уравнений f(x) = 0 методом простой итерации.

# Аргументы
- `f::Function`: Функция системы уравнений f(x) = 0
- `x₀::Vector{Float64}`: Начальное приближение
- `α::Float64=0.01`: Параметр релаксации 
- `max_iter::Int=1000`: Максимальное число итераций
- `tol::Float64=1e-6`: Допустимая погрешность
- `check_condition::Bool=true`: Проверять ли условие сходимости

# Возвращает
- `x::Vector{Float64}`: Найденное решение
- `converged::Bool`: Флаг сходимости метода
- `iter_count::Int`: Число выполненных итераций
- `errors::Vector{Float64}`: История ошибок на каждой итерации
"""
function simple_iteration_solve(f, x₀; α=0.01, max_iter=1000, tol=1e-6, check_condition=true)
    # Подготовка итерационной функции
    g = prepare_iteration_function(f, x₀, α)
    
    # Проверка условия сходимости
    if check_condition
        converges, norm_val = check_convergence_simple_iteration(g, x₀)
        if !converges
            @warn "Нарушено условие сходимости. Норма производной: $norm_val > 1"
        end
    end
    
    # Инициализация
    x = copy(x₀)
    errors = Float64[]
    
    for iter in 1:max_iter
        # Вычисление нового приближения
        x_new = g(x)
        
        # Вычисление ошибки
        error = norm(x_new - x) / max(norm(x), 1.0)
        push!(errors, error)
        
        # Проверка сходимости
        if error < tol
            return x_new, true, iter, errors
        end
        
        # Обновление текущего приближения
        x = x_new
    end
    
    # Если метод не сошелся за максимальное число итераций
    return x, false, max_iter, errors
end

end # module 