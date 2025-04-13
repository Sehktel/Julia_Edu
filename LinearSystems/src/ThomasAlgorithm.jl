"""
Модуль с реализацией метода прогонки (алгоритма Томаса) для решения трехдиагональных систем.
"""

"""
    thomas_algorithm(a, b, c, d)

Решает трехдиагональную систему уравнений методом прогонки (алгоритмом Томаса).
Система имеет вид:
    b₁x₁ + c₁x₂ = d₁
    a₂x₁ + b₂x₂ + c₂x₃ = d₂
    ...
    aₙxₙ₋₁ + bₙxₙ = dₙ

# Аргументы
- `a`: вектор коэффициентов под главной диагональю (a[1] не используется)
- `b`: вектор коэффициентов на главной диагонали
- `c`: вектор коэффициентов над главной диагональю (c[n] не используется)
- `d`: вектор правой части

# Возвращает
- Вектор решения x
"""
function thomas_algorithm(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector)
    n = length(d)
    
    # Проверки размерностей
    if length(a) != n || length(b) != n || length(c) != n
        error("Все входные векторы должны иметь одинаковую длину")
    end
    
    # Создаем копии для вычислений
    c_prime = zeros(n)
    d_prime = zeros(n)
    x = zeros(n)
    
    # Проверка устойчивости метода
    for i = 1:n
        if i > 1 && abs(b[i]) < abs(a[i]) + abs(c[i])
            @warn "Система не является диагонально доминирующей, метод может быть неустойчивым"
            break
        end
    end
    
    # Прямой ход прогонки
    # Вычисляем коэффициенты c' и d'
    c_prime[1] = c[1] / b[1]
    d_prime[1] = d[1] / b[1]
    
    for i = 2:n
        # Проверка деления на ноль
        denominator = b[i] - a[i] * c_prime[i-1]
        if abs(denominator) < eps()
            error("Деление на ноль в строке $i при прямом ходе")
        end
        
        c_prime[i] = i < n ? c[i] / denominator : 0
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
    end
    
    # Обратный ход прогонки
    x[n] = d_prime[n]
    
    for i = n-1:-1:1
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    end
    
    return x
end

"""
    thomas_algorithm(A, b)

Решает трехдиагональную систему уравнений методом прогонки, принимая матрицу A и вектор b.
Матрица A должна быть трехдиагональной.

# Аргументы
- `A`: трехдиагональная матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function thomas_algorithm(A::AbstractMatrix, b::AbstractVector)
    n = length(b)
    
    # Проверяем, что матрица квадратная
    if size(A, 1) != n || size(A, 2) != n
        error("Матрица должна быть квадратной размера n×n, где n - длина вектора правой части")
    end
    
    # Извлекаем диагонали
    a = zeros(n)  # поддиагональ
    b_diag = zeros(n)  # главная диагональ
    c = zeros(n)  # наддиагональ
    
    # Заполняем вектор главной диагонали
    for i = 1:n
        b_diag[i] = A[i, i]
    end
    
    # Заполняем вектор поддиагонали
    for i = 2:n
        a[i] = A[i, i-1]
    end
    
    # Заполняем вектор наддиагонали
    for i = 1:n-1
        c[i] = A[i, i+1]
    end
    
    # Проверяем, что матрица действительно трехдиагональная
    for i = 1:n
        for j = 1:n
            if abs(A[i, j]) > eps() && abs(i - j) > 1
                error("Матрица не является трехдиагональной")
            end
        end
    end
    
    # Вызываем основную функцию метода прогонки
    return thomas_algorithm(a, b_diag, c, b)
end

# Пример использования метода прогонки
function example_thomas_algorithm()
    println("Пример использования метода прогонки (алгоритма Томаса):")
    
    # Создаем трехдиагональную систему
    n = 5
    A = zeros(n, n)
    
    # Заполняем главную диагональ
    for i = 1:n
        A[i, i] = 2.0
    end
    
    # Заполняем поддиагональ
    for i = 2:n
        A[i, i-1] = -1.0
    end
    
    # Заполняем наддиагональ
    for i = 1:n-1
        A[i, i+1] = -1.0
    end
    
    # Создаем правую часть
    b = zeros(n)
    b[1] = 1.0
    b[n] = 1.0
    
    println("Трехдиагональная матрица A:")
    display(A)
    println()
    
    println("Вектор правой части b:")
    display(b)
    println()
    
    # Извлекаем диагонали для метода прогонки
    a = zeros(n)  # поддиагональ (a[1] не используется)
    b_diag = zeros(n)  # главная диагональ
    c = zeros(n)  # наддиагональ (c[n] не используется)
    
    for i = 2:n
        a[i] = A[i, i-1]
    end
    
    for i = 1:n
        b_diag[i] = A[i, i]
    end
    
    for i = 1:n-1
        c[i] = A[i, i+1]
    end
    
    # Решаем систему методом прогонки (оба варианта)
    x_thomas = thomas_algorithm(a, b_diag, c, b)
    x_thomas_matrix = thomas_algorithm(A, b)
    
    println("Решение методом прогонки (через векторы a, b, c, d):")
    display(x_thomas)
    println()
    
    println("Решение методом прогонки (через матрицу A и вектор b):")
    display(x_thomas_matrix)
    println()
    
    # Проверяем результат
    println("Проверка решения: ||Ax - b|| = ", norm(A * x_thomas - b))
    
    # Пример с классической задачей дискретизации дифференциального уравнения
    println("\nПример задачи дискретизации дифференциального уравнения:")
    
    # Решаем u'' = f на отрезке [0, 1] с граничными условиями u(0) = u(1) = 0
    # Дискретизация: -u[i-1] + 2u[i] - u[i+1] = h²f[i]
    
    n = 10  # количество внутренних точек
    h = 1.0 / (n + 1)  # шаг сетки
    
    # Создаем трехдиагональную матрицу для дискретизации
    A_diff = zeros(n, n)
    
    # Заполняем матрицу
    for i = 1:n
        A_diff[i, i] = 2.0
        if i > 1
            A_diff[i, i-1] = -1.0
        end
        if i < n
            A_diff[i, i+1] = -1.0
        end
    end
    
    # Создаем правую часть для f(x) = sin(πx)
    b_diff = zeros(n)
    for i = 1:n
        x_i = i * h
        b_diff[i] = h^2 * sin(π * x_i)
    end
    
    println("Матрица дискретизации A:")
    display(A_diff)
    println()
    
    println("Вектор правой части b:")
    display(b_diff)
    println()
    
    # Решаем систему методом прогонки
    u = thomas_algorithm(A_diff, b_diff)
    
    println("Численное решение u:")
    display(u)
    println()
    
    # Вычисляем точное решение для сравнения
    u_exact = zeros(n)
    for i = 1:n
        x_i = i * h
        u_exact[i] = sin(π * x_i) / (π^2)
    end
    
    println("Точное решение u_exact:")
    display(u_exact)
    println()
    
    println("Погрешность (максимальная): ", maximum(abs.(u - u_exact)))
end 