"""
    BoundaryValueProblems

Модуль, реализующий методы для решения краевых задач для обыкновенных дифференциальных уравнений.
"""
module BoundaryValueProblems

export shooting_solve, finite_difference_solve

using LinearAlgebra
include("RungeKuttaMethod.jl")

"""
    shooting_solve(f, a, b, alpha, beta; 
                   initial_guess=alpha, tol=1e-6, max_iter=100, step_size=0.01)

Решает краевую задачу для ОДУ первого порядка методом стрельбы.

# Задача
y' = f(x, y)
y(a) = alpha
y(b) = beta

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(x, y)
- `a::Float64`: Левая граница интервала
- `b::Float64`: Правая граница интервала
- `alpha::Float64`: Значение функции на левой границе y(a) = alpha
- `beta::Float64`: Значение функции на правой границе y(b) = beta
- `initial_guess::Float64=alpha`: Начальное приближение для y'(a)
- `tol::Float64=1e-6`: Допустимая погрешность
- `max_iter::Int=100`: Максимальное число итераций
- `step_size::Float64=0.01`: Шаг интегрирования для решения задачи Коши

# Возвращает
- `x::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}`: Приближенное решение ОДУ
"""
function shooting_solve(f, a, b, alpha, beta; 
                        initial_guess=alpha, tol=1e-6, max_iter=100, step_size=0.01)
    # Создаем вспомогательную функцию для решения задачи Коши
    function ivp_solve(s)
        # Функция для системы из двух уравнений первого порядка
        # y1' = y2
        # y2' = f(x, y1, y2)
        function system(x, y)
            y1, y2 = y
            return [y2, f(x, y1, y2)]
        end
        
        # Начальные условия для системы
        y0 = [alpha, s]
        
        # Решаем задачу Коши методом Рунге-Кутта 4-го порядка
        t, y = RungeKuttaMethod.runge_kutta4_solve(system, (a, b), y0, step_size=step_size)
        
        return t, y
    end
    
    # Функция невязки: разница между полученным значением y(b) и целевым beta
    function residual(s)
        _, y = ivp_solve(s)
        return y[end, 1] - beta
    end
    
    # Метод секущих для нахождения значения s, при котором y(b) = beta
    # Начальные приближения
    s0 = initial_guess
    s1 = s0 * 1.1  # Небольшое возмущение начального приближения
    
    # Вычисляем невязки для начальных приближений
    r0 = residual(s0)
    r1 = residual(s1)
    
    # Итерационный процесс метода секущих
    iter = 0
    while abs(r1) > tol && iter < max_iter
        # Вычисляем новое приближение методом секущих
        s_new = s1 - r1 * (s1 - s0) / (r1 - r0)
        
        # Обновляем приближения
        s0, s1 = s1, s_new
        r0, r1 = r1, residual(s1)
        
        iter += 1
    end
    
    # Проверяем сходимость метода
    if iter == max_iter && abs(r1) > tol
        @warn "Метод стрельбы не сошелся за максимальное число итераций. Погрешность: $(abs(r1))"
    end
    
    # Получаем решение для найденного значения параметра стрельбы
    x, y = ivp_solve(s1)
    
    return x, y[:, 1]
end

"""
    finite_difference_solve(f, a, b, alpha, beta; 
                          n=100, max_iter=100, tol=1e-6)

Решает краевую задачу для ОДУ второго порядка методом конечных разностей.

# Задача
y'' = f(x, y, y')
y(a) = alpha
y(b) = beta

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y'' = f(x, y, y')
- `a::Float64`: Левая граница интервала
- `b::Float64`: Правая граница интервала
- `alpha::Float64`: Значение функции на левой границе y(a) = alpha
- `beta::Float64`: Значение функции на правой границе y(b) = beta
- `n::Int=100`: Число интервалов разбиения
- `max_iter::Int=100`: Максимальное число итераций для нелинейных задач
- `tol::Float64=1e-6`: Допустимая погрешность для нелинейных задач

# Возвращает
- `x::Vector{Float64}`: Сетка значений аргумента
- `y::Vector{Float64}`: Приближенное решение ОДУ
"""
function finite_difference_solve(f, a, b, alpha, beta; 
                               n=100, max_iter=100, tol=1e-6)
    # Создаем сетку
    h = (b - a) / n
    x = collect(range(a, b, length=n+1))
    
    # Проверяем, является ли задача линейной или нелинейной
    # Для простоты считаем, что если функция f принимает только x и y (без y'),
    # то задача линейная, иначе - нелинейная
    is_linear = methods(f)[1].nargs == 3  # Метод с двумя аргументами: x и y
    
    # Инициализируем массив решения
    y = zeros(n+1)
    y[1] = alpha   # Граничное условие на левом конце
    y[n+1] = beta   # Граничное условие на правом конце
    
    # Для линейной задачи
    if is_linear
        # Формируем СЛАУ для внутренних точек
        A = zeros(n-1, n-1)
        b_vec = zeros(n-1)
        
        # Заполняем матрицу и вектор правой части
        for i in 1:n-1
            # Индекс узла в сетке
            j = i + 1
            x_j = x[j]
            
            # Диагональные и внедиагональные элементы
            A[i, i] = -2/h^2
            
            if i > 1
                A[i, i-1] = 1/h^2
            end
            
            if i < n-1
                A[i, i+1] = 1/h^2
            end
            
            # Правая часть
            b_vec[i] = f(x_j, 0)  # Для линейной задачи y'' = p(x)y' + q(x)y + r(x)
            
            # Учитываем граничные условия
            if i == 1
                b_vec[i] -= alpha/h^2
            end
            
            if i == n-1
                b_vec[i] -= beta/h^2
            end
        end
        
        # Решаем СЛАУ
        middle_values = A \ b_vec
        
        # Заполняем массив решения
        y[2:n] = middle_values
    else
        # Для нелинейной задачи используем метод Ньютона
        
        # Инициализируем начальное приближение (линейная интерполяция)
        for i in 2:n
            y[i] = alpha + (beta - alpha) * (x[i] - a) / (b - a)
        end
        
        # Итерационный процесс метода Ньютона
        for iter in 1:max_iter
            # Формируем СЛАУ для приращений
            A = zeros(n-1, n-1)
            b_vec = zeros(n-1)
            
            # Вычисляем производные и невязки
            for i in 1:n-1
                j = i + 1
                x_j = x[j]
                
                # Аппроксимация производных в точке x_j
                if j == 2
                    y_prime_j = (y[j+1] - alpha) / (2*h)
                elseif j == n
                    y_prime_j = (beta - y[j-1]) / (2*h)
                else
                    y_prime_j = (y[j+1] - y[j-1]) / (2*h)
                end
                
                # Вторая производная (для невязки)
                y_second_j = (y[j+1] - 2*y[j] + y[j-1]) / h^2
                
                # Невязка
                residual = y_second_j - f(x_j, y[j], y_prime_j)
                b_vec[i] = -residual
                
                # Якобиан (частные производные по y[j-1], y[j], y[j+1])
                df_dy_prev = -1/h^2 - df_dyprime(x_j, y[j], y_prime_j) / (2*h)
                df_dy_curr = 2/h^2 - df_dy(x_j, y[j], y_prime_j)
                df_dy_next = -1/h^2 + df_dyprime(x_j, y[j], y_prime_j) / (2*h)
                
                # Заполняем матрицу Якоби
                if i > 1
                    A[i, i-1] = df_dy_prev
                end
                
                A[i, i] = df_dy_curr
                
                if i < n-1
                    A[i, i+1] = df_dy_next
                end
            end
            
            # Решаем СЛАУ для приращений
            delta_y = A \ b_vec
            
            # Обновляем приближение
            y[2:n] += delta_y
            
            # Проверяем условие сходимости
            if norm(delta_y) < tol
                break
            end
            
            # Проверяем достижение максимального числа итераций
            if iter == max_iter
                @warn "Метод конечных разностей не сошелся за максимальное число итераций."
            end
        end
    end
    
    return x, y
end

"""
    df_dy(x, y, y_prime)

Приближенное вычисление частной производной функции f по y.
"""
function df_dy(x, y, y_prime)
    # Приближение частной производной по y через конечные разности
    h = 1e-6
    return (f(x, y + h, y_prime) - f(x, y - h, y_prime)) / (2*h)
end

"""
    df_dyprime(x, y, y_prime)

Приближенное вычисление частной производной функции f по y'.
"""
function df_dyprime(x, y, y_prime)
    # Приближение частной производной по y' через конечные разности
    h = 1e-6
    return (f(x, y, y_prime + h) - f(x, y, y_prime - h)) / (2*h)
end

end # module 