#!/usr/bin/env julia

"""
Примеры использования методов решения систем линейных уравнений.
Запустите этот скрипт, чтобы увидеть работу всех реализованных методов.
"""

# Подключаем модуль
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using LinearSystems
using LinearAlgebra

# Функция для вывода разделителя
function print_separator()
    println("\n", "="^80, "\n")
end

function main()
    println("ПРИМЕРЫ МЕТОДОВ РЕШЕНИЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ В JULIA")
    print_separator()
    
    # Запускаем примеры прямых методов
    println("1. ПРЯМЫЕ МЕТОДЫ РЕШЕНИЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ")
    print_separator()
    example_direct_methods()
    
    print_separator()
    
    # Запускаем примеры метода Гаусса
    println("2. МЕТОД ГАУССА")
    print_separator()
    example_gaussian_elimination()
    
    print_separator()
    
    # Запускаем примеры метода прогонки
    println("3. МЕТОД ПРОГОНКИ (АЛГОРИТМ ТОМАСА)")
    print_separator()
    example_thomas_algorithm()
    
    print_separator()
    
    # Запускаем примеры метода Гаусса-Зейделя
    println("4. МЕТОД ГАУССА-ЗЕЙДЕЛЯ")
    print_separator()
    example_gauss_seidel()
    
    print_separator()
    
    println("СРАВНЕНИЕ ВСЕХ МЕТОДОВ НА ОДНОЙ ЗАДАЧЕ")
    print_separator()
    
    # Создаем тестовую систему
    n = 5
    A = zeros(n, n)
    
    # Заполняем матрицу
    for i = 1:n
        A[i, i] = 4.0
        if i > 1
            A[i, i-1] = -1.0
        end
        if i < n
            A[i, i+1] = -1.0
        end
    end
    
    # Создаем правую часть
    b = ones(n)
    
    println("Тестовая система:")
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Решаем систему разными методами
    println("\nРешение разными методами:")
    
    # Прямой метод
    x_direct = solve_direct(A, b)
    println("1. Прямой метод (A \\ b):")
    display(x_direct)
    println("Невязка: ", norm(A * x_direct - b))
    println()
    
    # Метод Гаусса
    x_gauss = gauss_elimination(A, b)
    println("2. Метод Гаусса:")
    display(x_gauss)
    println("Невязка: ", norm(A * x_gauss - b))
    println()
    
    # Метод прогонки
    x_thomas = thomas_algorithm(A, b)
    println("3. Метод прогонки:")
    display(x_thomas)
    println("Невязка: ", norm(A * x_thomas - b))
    println()
    
    # Метод Гаусса-Зейделя
    x_gs, iter_gs, _ = gauss_seidel(A, b)
    println("4. Метод Гаусса-Зейделя:")
    display(x_gs)
    println("Невязка: ", norm(A * x_gs - b))
    println("Количество итераций: ", iter_gs)
    println()
    
    # Сравнение с точным решением
    println("Максимальная разница между методами: ", 
           maximum([
               norm(x_direct - x_gauss, Inf),
               norm(x_direct - x_thomas, Inf),
               norm(x_direct - x_gs, Inf)
           ]))
           
    print_separator()
    println("ЗАВЕРШЕНО")
end

# Запускаем основную функцию
main() 