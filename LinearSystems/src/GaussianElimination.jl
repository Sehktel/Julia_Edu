"""
Модуль с реализацией метода Гаусса для решения систем линейных уравнений.
"""

"""
    gauss_elimination(A, b)

Решает систему линейных уравнений Ax = b методом Гаусса без выбора главного элемента.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function gauss_elimination(A::AbstractMatrix, b::AbstractVector)
    # Создаем копии входных данных, чтобы не изменять оригиналы
    n = size(A, 1)
    A_copy = copy(A)
    b_copy = copy(b)
    
    # Прямой ход метода Гаусса (приведение к верхнетреугольному виду)
    for k = 1:n-1
        for i = k+1:n
            # Проверка деления на ноль
            if abs(A_copy[k, k]) < eps()
                error("Нулевой ведущий элемент в строке $k")
            end
            
            # Вычисляем множитель
            factor = A_copy[i, k] / A_copy[k, k]
            
            # Вычитаем k-ю строку из i-й строки с множителем
            for j = k:n
                A_copy[i, j] -= factor * A_copy[k, j]
            end
            
            # Корректируем правую часть
            b_copy[i] -= factor * b_copy[k]
        end
    end
    
    # Обратный ход (решение верхнетреугольной системы)
    x = zeros(n)
    for i = n:-1:1
        # Вычисляем значение x[i]
        sum_val = 0.0
        for j = i+1:n
            sum_val += A_copy[i, j] * x[j]
        end
        
        # Проверка деления на ноль
        if abs(A_copy[i, i]) < eps()
            error("Нулевой диагональный элемент при обратном ходе в строке $i")
        end
        
        x[i] = (b_copy[i] - sum_val) / A_copy[i, i]
    end
    
    return x
end

"""
    gauss_elimination_pivot(A, b)

Решает систему линейных уравнений Ax = b методом Гаусса с выбором главного элемента по столбцу.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function gauss_elimination_pivot(A::AbstractMatrix, b::AbstractVector)
    # Создаем копии входных данных
    n = size(A, 1)
    A_copy = copy(A)
    b_copy = copy(b)
    
    # Вектор перестановок
    p = collect(1:n)
    
    # Прямой ход метода Гаусса с выбором главного элемента
    for k = 1:n-1
        # Находим максимальный элемент в текущем столбце
        max_val = abs(A_copy[k, k])
        max_idx = k
        
        for i = k+1:n
            if abs(A_copy[i, k]) > max_val
                max_val = abs(A_copy[i, k])
                max_idx = i
            end
        end
        
        # Проверка на сингулярность
        if max_val < eps()
            error("Матрица близка к сингулярной или сингулярна")
        end
        
        # Меняем местами строки, если нашли лучший ведущий элемент
        if max_idx != k
            # Меняем строки в матрице A
            for j = 1:n
                A_copy[k, j], A_copy[max_idx, j] = A_copy[max_idx, j], A_copy[k, j]
            end
            
            # Меняем элементы в векторе b
            b_copy[k], b_copy[max_idx] = b_copy[max_idx], b_copy[k]
            
            # Запоминаем перестановку
            p[k], p[max_idx] = p[max_idx], p[k]
        end
        
        # Исключение переменных (как в обычном методе Гаусса)
        for i = k+1:n
            factor = A_copy[i, k] / A_copy[k, k]
            
            for j = k:n
                A_copy[i, j] -= factor * A_copy[k, j]
            end
            
            b_copy[i] -= factor * b_copy[k]
        end
    end
    
    # Обратный ход (решение верхнетреугольной системы)
    x = zeros(n)
    for i = n:-1:1
        sum_val = 0.0
        for j = i+1:n
            sum_val += A_copy[i, j] * x[j]
        end
        
        # Проверка деления на ноль
        if abs(A_copy[i, i]) < eps()
            error("Нулевой диагональный элемент при обратном ходе в строке $i")
        end
        
        x[i] = (b_copy[i] - sum_val) / A_copy[i, i]
    end
    
    # Создаем вектор решения с учетом перестановок
    x_original = zeros(n)
    for i = 1:n
        x_original[p[i]] = x[i]
    end
    
    return x
end

# Пример использования метода Гаусса
function example_gaussian_elimination()
    println("Пример использования метода Гаусса:")
    
    # Создаем тестовую систему
    A = [2.0 1.0 -1.0; -3.0 -1.0 2.0; -2.0 1.0 2.0]
    b = [8.0, -11.0, -3.0]
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Решаем систему методом Гаусса без выбора главного элемента
    x_gauss = gauss_elimination(A, b)
    
    println("Решение (метод Гаусса без выбора главного элемента):")
    display(x_gauss)
    println()
    
    # Решаем систему методом Гаусса с выбором главного элемента
    x_gauss_pivot = gauss_elimination_pivot(A, b)
    
    println("Решение (метод Гаусса с выбором главного элемента):")
    display(x_gauss_pivot)
    println()
    
    # Проверяем результат
    println("Проверка решения: ||Ax - b|| = ", norm(A * x_gauss - b))
    println("Проверка решения с выбором главного элемента: ||Ax - b|| = ", norm(A * x_gauss_pivot - b))
    
    # Пример неустойчивой системы
    println("\nПример неустойчивой системы:")
    A_unstable = [1e-10 1.0; 1.0 1.0]
    b_unstable = [1.0, 2.0]
    
    println("Матрица A (неустойчивая):")
    display(A_unstable)
    println()
    
    println("Вектор b:")
    display(b_unstable)
    println()
    
    # Решаем неустойчивую систему обоими методами
    try
        x_unstable = gauss_elimination(A_unstable, b_unstable)
        println("Решение (метод Гаусса без выбора главного элемента):")
        display(x_unstable)
        println("Проверка решения: ||Ax - b|| = ", norm(A_unstable * x_unstable - b_unstable))
    catch e
        println("Ошибка при решении без выбора главного элемента: ", e.msg)
    end
    
    try
        x_unstable_pivot = gauss_elimination_pivot(A_unstable, b_unstable)
        println("Решение (метод Гаусса с выбором главного элемента):")
        display(x_unstable_pivot)
        println("Проверка решения: ||Ax - b|| = ", norm(A_unstable * x_unstable_pivot - b_unstable))
    catch e
        println("Ошибка при решении с выбором главного элемента: ", e.msg)
    end
end 