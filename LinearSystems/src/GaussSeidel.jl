"""
Модуль с реализацией метода Гаусса-Зейделя для решения систем линейных уравнений.
"""

"""
    is_diagonally_dominant(A)

Проверяет, является ли матрица A диагонально доминирующей.
Матрица диагонально доминирующая, если для каждой строки модуль диагонального элемента
больше или равен сумме модулей остальных элементов строки.

# Аргументы
- `A`: матрица для проверки

# Возвращает
- `true`, если матрица диагонально доминирующая, иначе `false`
"""
function is_diagonally_dominant(A::AbstractMatrix)
    n = size(A, 1)
    
    for i = 1:n
        diagonal_element = abs(A[i, i])
        sum_of_others = sum(abs.(A[i, :]) - diagonal_element)
        
        if diagonal_element < sum_of_others
            return false
        end
    end
    
    return true
end

"""
    gauss_seidel(A, b, x0=zeros(length(b)), tol=1e-6, max_iter=1000)

Решает систему линейных уравнений Ax = b методом Гаусса-Зейделя.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части
- `x0`: начальное приближение (по умолчанию нулевой вектор)
- `tol`: допустимая погрешность для критерия остановки
- `max_iter`: максимальное число итераций

# Возвращает
- Вектор решения x
- Количество выполненных итераций
- Вектор невязок на каждой итерации
"""
function gauss_seidel(A::AbstractMatrix, b::AbstractVector, 
                     x0::AbstractVector=zeros(length(b)), 
                     tol::Real=1e-6, max_iter::Integer=1000)
    n = length(b)
    
    # Проверки размерностей
    if size(A, 1) != n || size(A, 2) != n
        error("Матрица A должна быть квадратной размера n×n")
    end
    
    if length(x0) != n
        error("Начальное приближение x0 должно иметь длину n")
    end
    
    # Проверка диагонального доминирования
    if !is_diagonally_dominant(A)
        @warn "Матрица не является диагонально доминирующей. Сходимость метода Гаусса-Зейделя не гарантирована."
    end
    
    # Проверка нулевых элементов на диагонали
    for i = 1:n
        if abs(A[i, i]) < eps()
            error("Нулевой элемент на диагонали в позиции ($i, $i)")
        end
    end
    
    # Инициализация
    x = copy(x0)
    residuals = Float64[]
    
    # Итерационный процесс
    for iter = 1:max_iter
        x_prev = copy(x)
        
        # Один шаг метода Гаусса-Зейделя
        for i = 1:n
            # Вычисляем сумму для j < i (используем уже обновленные значения x)
            sum1 = 0.0
            for j = 1:i-1
                sum1 += A[i, j] * x[j]
            end
            
            # Вычисляем сумму для j > i (используем старые значения x)
            sum2 = 0.0
            for j = i+1:n
                sum2 += A[i, j] * x_prev[j]
            end
            
            # Обновляем x[i]
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        end
        
        # Вычисляем невязку
        residual = norm(A * x - b) / norm(b)
        push!(residuals, residual)
        
        # Проверяем критерий остановки
        if residual < tol
            return x, iter, residuals
        end
    end
    
    @warn "Метод Гаусса-Зейделя не сошелся за $max_iter итераций"
    return x, max_iter, residuals
end

"""
    gauss_seidel_matrix_form(A, b, x0=zeros(length(b)), tol=1e-6, max_iter=1000)

Решает систему линейных уравнений Ax = b методом Гаусса-Зейделя в матричной форме.
Использует разложение A = L + D + U, где L - нижняя треугольная часть, 
D - диагональная часть и U - верхняя треугольная часть.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части
- `x0`: начальное приближение (по умолчанию нулевой вектор)
- `tol`: допустимая погрешность для критерия остановки
- `max_iter`: максимальное число итераций

# Возвращает
- Вектор решения x
- Количество выполненных итераций
- Вектор невязок на каждой итерации
"""
function gauss_seidel_matrix_form(A::AbstractMatrix, b::AbstractVector, 
                                 x0::AbstractVector=zeros(length(b)), 
                                 tol::Real=1e-6, max_iter::Integer=1000)
    n = length(b)
    
    # Проверки размерностей
    if size(A, 1) != n || size(A, 2) != n
        error("Матрица A должна быть квадратной размера n×n")
    end
    
    if length(x0) != n
        error("Начальное приближение x0 должно иметь длину n")
    end
    
    # Создаем матрицы L, D и U
    L = zeros(n, n)
    D = zeros(n, n)
    U = zeros(n, n)
    
    for i = 1:n
        for j = 1:n
            if i > j
                L[i, j] = A[i, j]
            elseif i == j
                D[i, j] = A[i, j]
            else  # i < j
                U[i, j] = A[i, j]
            end
        end
    end
    
    # Инициализация
    x = copy(x0)
    residuals = Float64[]
    
    # Итерационный процесс
    for iter = 1:max_iter
        # Один шаг метода Гаусса-Зейделя в матричной форме
        # x = (D + L)^(-1) * (b - U * x)
        x = (D + L) \ (b - U * x)
        
        # Вычисляем невязку
        residual = norm(A * x - b) / norm(b)
        push!(residuals, residual)
        
        # Проверяем критерий остановки
        if residual < tol
            return x, iter, residuals
        end
    end
    
    @warn "Метод Гаусса-Зейделя не сошелся за $max_iter итераций"
    return x, max_iter, residuals
end

# Пример использования метода Гаусса-Зейделя
function example_gauss_seidel()
    println("Пример использования метода Гаусса-Зейделя:")
    
    # Создаем диагонально доминирующую систему
    A = [10.0 -1.0 2.0; -1.0 11.0 -1.0; 2.0 -1.0 10.0]
    b = [6.0, 25.0, -11.0]
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    println("Проверка на диагональное доминирование: ", is_diagonally_dominant(A))
    
    # Решаем систему методом Гаусса-Зейделя
    x0 = zeros(length(b))
    x_gs, iter_gs, residuals_gs = gauss_seidel(A, b, x0)
    
    println("Решение методом Гаусса-Зейделя:")
    display(x_gs)
    println()
    
    println("Количество итераций: ", iter_gs)
    println("Итоговая невязка: ", residuals_gs[end])
    println("Проверка решения: ||Ax - b|| = ", norm(A * x_gs - b))
    
    # Решаем ту же систему методом Гаусса-Зейделя в матричной форме
    x_gsm, iter_gsm, residuals_gsm = gauss_seidel_matrix_form(A, b, x0)
    
    println("\nРешение методом Гаусса-Зейделя (матричная форма):")
    display(x_gsm)
    println()
    
    println("Количество итераций: ", iter_gsm)
    println("Итоговая невязка: ", residuals_gsm[end])
    println("Проверка решения: ||Ax - b|| = ", norm(A * x_gsm - b))
    
    # Визуализация сходимости (можно раскомментировать при наличии пакета Plots)
    # using Plots
    # plot(1:iter_gs, residuals_gs, xlabel="Итерация", ylabel="Невязка", 
    #      yscale=:log10, label="Поэлементная форма", legend=:topright)
    # plot!(1:iter_gsm, residuals_gsm, label="Матричная форма")
    # title!("Сходимость метода Гаусса-Зейделя")
    
    # Пример с ненулевым начальным приближением
    println("\nПример с ненулевым начальным приближением:")
    x0_nonzero = ones(length(b))
    x_gs_nonzero, iter_gs_nonzero, residuals_gs_nonzero = gauss_seidel(A, b, x0_nonzero)
    
    println("Начальное приближение x0:")
    display(x0_nonzero)
    println()
    
    println("Решение методом Гаусса-Зейделя:")
    display(x_gs_nonzero)
    println()
    
    println("Количество итераций: ", iter_gs_nonzero)
    
    # Пример с недиагонально доминирующей матрицей
    println("\nПример с недиагонально доминирующей матрицей:")
    A_nondom = [1.0 2.0; 3.0 4.0]
    b_nondom = [5.0, 6.0]
    
    println("Матрица A (недиагонально доминирующая):")
    display(A_nondom)
    println()
    
    println("Вектор b:")
    display(b_nondom)
    println()
    
    println("Проверка на диагональное доминирование: ", is_diagonally_dominant(A_nondom))
    
    # Решаем недиагонально доминирующую систему
    x_gs_nondom, iter_gs_nondom, residuals_gs_nondom = gauss_seidel(A_nondom, b_nondom, zeros(2), 1e-6, 100)
    
    println("Решение методом Гаусса-Зейделя:")
    display(x_gs_nondom)
    println()
    
    println("Количество итераций: ", iter_gs_nondom)
    println("Итоговая невязка: ", residuals_gs_nondom[end])
    println("Проверка решения: ||Ax - b|| = ", norm(A_nondom * x_gs_nondom - b_nondom))
    
    # Сравнение с точным решением
    x_exact = A_nondom \ b_nondom
    println("Точное решение:")
    display(x_exact)
    println()
    
    println("Погрешность: ", norm(x_gs_nondom - x_exact))
end 