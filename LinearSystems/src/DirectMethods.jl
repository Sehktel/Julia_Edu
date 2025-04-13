"""
Модуль с прямыми методами решения систем линейных уравнений.
"""

"""
    solve_direct(A, b)

Решает систему линейных уравнений Ax = b с использованием оператора \\.
Это стандартный метод, который выбирает подходящий алгоритм на основе свойств матрицы A.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function solve_direct(A::AbstractMatrix, b::AbstractVector)
    # Решаем систему с использованием оператора \
    x = A \ b
    return x
end

"""
    solve_lu(A, b)

Решает систему линейных уравнений Ax = b с использованием LU-разложения.

# Аргументы
- `A`: матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function solve_lu(A::AbstractMatrix, b::AbstractVector)
    # LU-разложение матрицы A
    lu_factorization = lu(A)
    
    # Решение системы с использованием LU-разложения
    x = lu_factorization \ b
    
    return x
end

"""
    solve_cholesky(A, b)

Решает систему линейных уравнений Ax = b с использованием разложения Холецкого.
Матрица A должна быть симметричной и положительно определенной.

# Аргументы
- `A`: симметричная положительно определенная матрица коэффициентов
- `b`: вектор правой части

# Возвращает
- Вектор решения x
"""
function solve_cholesky(A::AbstractMatrix, b::AbstractVector)
    # Проверяем, что матрица симметрична
    if !issymmetric(A)
        error("Матрица должна быть симметричной для разложения Холецкого")
    end
    
    # Разложение Холецкого
    cholesky_factorization = cholesky(A)
    
    # Решение системы с использованием разложения Холецкого
    x = cholesky_factorization \ b
    
    return x
end

# Пример использования прямых методов
function example_direct_methods()
    println("Пример использования прямых методов:")
    
    # Создаем тестовую систему
    A = [4.0 1.0 1.0; 1.0 3.0 -1.0; 2.0 -1.0 5.0]
    b = [6.0, 3.0, 7.0]
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Решаем систему разными методами
    x_direct = solve_direct(A, b)
    x_lu = solve_lu(A, b)
    x_cholesky = solve_cholesky(A, b)
    
    println("Решение (стандартный метод):")
    display(x_direct)
    println()
    
    println("Решение (LU-разложение):")
    display(x_lu)
    println()
    
    println("Решение (разложение Холецкого):")
    display(x_cholesky)
    println()
    
    # Проверяем результат
    println("Проверка решения: ||Ax - b|| = ", norm(A * x_direct - b))
end 