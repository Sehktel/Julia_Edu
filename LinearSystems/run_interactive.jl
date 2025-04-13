#!/usr/bin/env julia

"""
Интерактивное исследование методов решения систем линейных уравнений.
Запустите этот скрипт и следуйте инструкциям для изучения различных методов.
"""

# Подключаем модуль
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using LinearSystems
using LinearAlgebra

# Функция для вывода разделителя
function print_separator()
    println("\n", "="^80, "\n")
end

# Функция для паузы и ожидания нажатия клавиши Enter
function pause()
    println("\nНажмите Enter для продолжения...")
    readline()
end

# Функция для создания примера системы
function create_system()
    println("Выберите тип системы:")
    println("1. Общая 3x3 система")
    println("2. Диагонально доминирующая система")
    println("3. Трехдиагональная система")
    println("4. Плохо обусловленная система")
    println("5. Ввести свою систему")
    
    choice = parse(Int, readline())
    
    if choice == 1
        # Общая 3x3 система
        A = [4.0 1.0 1.0; 1.0 3.0 -1.0; 2.0 -1.0 5.0]
        b = [6.0, 3.0, 7.0]
        return A, b, "Общая 3x3 система"
    elseif choice == 2
        # Диагонально доминирующая система
        A = [10.0 -1.0 2.0; -1.0 11.0 -1.0; 2.0 -1.0 10.0]
        b = [6.0, 25.0, -11.0]
        return A, b, "Диагонально доминирующая система"
    elseif choice == 3
        # Трехдиагональная система
        A = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]
        b = [1.0, 0.0, 1.0]
        return A, b, "Трехдиагональная система"
    elseif choice == 4
        # Плохо обусловленная система
        A = [1e-10 1.0; 1.0 1.0]
        b = [1.0, 2.0]
        return A, b, "Плохо обусловленная система"
    elseif choice == 5
        # Ввод своей системы
        println("\nВведите размерность системы:")
        n = parse(Int, readline())
        
        println("\nВведите матрицу A по строкам (разделяйте элементы пробелами):")
        A = zeros(n, n)
        for i = 1:n
            println("Строка $i:")
            row = parse.(Float64, split(readline()))
            if length(row) != n
                error("Строка должна содержать $n элементов")
            end
            A[i, :] = row
        end
        
        println("\nВведите вектор правой части b (разделяйте элементы пробелами):")
        b_input = parse.(Float64, split(readline()))
        if length(b_input) != n
            error("Вектор b должен содержать $n элементов")
        end
        b = b_input
        
        return A, b, "Пользовательская система размера $(n)x$(n)"
    else
        error("Неверный выбор")
    end
end

# Функция для демонстрации прямых методов
function demo_direct_methods(A, b)
    println("ДЕМОНСТРАЦИЯ ПРЯМЫХ МЕТОДОВ")
    print_separator()
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Решаем систему стандартным методом
    x_direct = solve_direct(A, b)
    println("\nРешение стандартным методом (A \\ b):")
    display(x_direct)
    println("\nНевязка: ||Ax - b|| = ", norm(A * x_direct - b))
    
    pause()
    
    # Решаем систему методом LU-разложения
    try
        x_lu = solve_lu(A, b)
        println("\nРешение методом LU-разложения:")
        display(x_lu)
        println("\nНевязка: ||Ax - b|| = ", norm(A * x_lu - b))
        
        # Выводим LU-разложение
        lu_factor = lu(A)
        println("\nLU-разложение матрицы A:")
        println("L:")
        display(lu_factor.L)
        println("\nU:")
        display(lu_factor.U)
        println("\nP:")
        display(lu_factor.p)
    catch e
        println("\nОшибка при решении методом LU-разложения: ", e.msg)
    end
    
    pause()
    
    # Решаем систему методом разложения Холецкого
    if issymmetric(A) && isposdef(A)
        try
            x_chol = solve_cholesky(A, b)
            println("\nРешение методом разложения Холецкого:")
            display(x_chol)
            println("\nНевязка: ||Ax - b|| = ", norm(A * x_chol - b))
            
            # Выводим разложение Холецкого
            chol_factor = cholesky(A)
            println("\nРазложение Холецкого матрицы A:")
            println("L:")
            display(chol_factor.L)
            println("\nL':")
            display(chol_factor.U)
        catch e
            println("\nОшибка при решении методом разложения Холецкого: ", e.msg)
        end
    else
        println("\nМатрица не является симметричной положительно определенной,")
        println("поэтому разложение Холецкого не может быть применено напрямую.")
        
        println("\nВместо этого применим разложение Холецкого к нормальной системе A'Ax = A'b:")
        try
            A_sym = A'A
            b_sym = A'b
            
            x_chol = cholesky(A_sym) \ b_sym
            println("\nРешение методом разложения Холецкого через нормальную систему:")
            display(x_chol)
            println("\nНевязка: ||Ax - b|| = ", norm(A * x_chol - b))
        catch e
            println("\nОшибка при решении методом разложения Холецкого: ", e.msg)
        end
    end
    
    println("\nПрямые методы завершены.")
end

# Функция для демонстрации метода Гаусса
function demo_gaussian_elimination(A, b)
    println("ДЕМОНСТРАЦИЯ МЕТОДА ГАУССА")
    print_separator()
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Решаем систему методом Гаусса без выбора главного элемента
    try
        println("\nПрименяем метод Гаусса без выбора главного элемента")
        x_gauss = gauss_elimination(A, b)
        println("\nРешение:")
        display(x_gauss)
        println("\nНевязка: ||Ax - b|| = ", norm(A * x_gauss - b))
    catch e
        println("\nОшибка при решении методом Гаусса без выбора главного элемента: ", e.msg)
    end
    
    pause()
    
    # Решаем систему методом Гаусса с выбором главного элемента
    try
        println("\nПрименяем метод Гаусса с выбором главного элемента")
        x_gauss_pivot = gauss_elimination_pivot(A, b)
        println("\nРешение:")
        display(x_gauss_pivot)
        println("\nНевязка: ||Ax - b|| = ", norm(A * x_gauss_pivot - b))
    catch e
        println("\nОшибка при решении методом Гаусса с выбором главного элемента: ", e.msg)
    end
    
    println("\nМетод Гаусса завершен.")
end

# Функция для демонстрации метода прогонки
function demo_thomas_algorithm(A, b)
    println("ДЕМОНСТРАЦИЯ МЕТОДА ПРОГОНКИ (АЛГОРИТМА ТОМАСА)")
    print_separator()
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Проверяем, является ли матрица трехдиагональной
    n = size(A, 1)
    is_tridiagonal = true
    for i = 1:n
        for j = 1:n
            if abs(A[i, j]) > eps() && abs(i - j) > 1
                is_tridiagonal = false
                break
            end
        end
    end
    
    if is_tridiagonal
        # Решаем систему методом прогонки
        try
            println("\nПрименяем метод прогонки")
            x_thomas = thomas_algorithm(A, b)
            println("\nРешение:")
            display(x_thomas)
            println("\nНевязка: ||Ax - b|| = ", norm(A * x_thomas - b))
            
            # Извлекаем диагонали для наглядности
            a = zeros(n)  # поддиагональ
            d = zeros(n)  # главная диагональ
            c = zeros(n)  # наддиагональ
            
            for i = 1:n
                d[i] = A[i, i]
                if i > 1
                    a[i] = A[i, i-1]
                end
                if i < n
                    c[i] = A[i, i+1]
                end
            end
            
            println("\nДиагонали трехдиагональной матрицы:")
            println("a (поддиагональ): ", a)
            println("d (главная диагональ): ", d)
            println("c (наддиагональ): ", c)
        catch e
            println("\nОшибка при решении методом прогонки: ", e.msg)
        end
    else
        println("\nМатрица не является трехдиагональной, метод прогонки не применим.")
        println("Для демонстрации метода прогонки выберите трехдиагональную систему.")
        
        # Создаем трехдиагональную матрицу для демонстрации
        println("\nСоздаем трехдиагональную систему для демонстрации:")
        n_demo = size(A, 1)
        A_tridiag = zeros(n_demo, n_demo)
        
        for i = 1:n_demo
            A_tridiag[i, i] = 2.0
            if i > 1
                A_tridiag[i, i-1] = -1.0
            end
            if i < n_demo
                A_tridiag[i, i+1] = -1.0
            end
        end
        
        b_tridiag = ones(n_demo)
        
        println("\nТрехдиагональная матрица A_tridiag:")
        display(A_tridiag)
        println()
        
        println("Вектор b_tridiag:")
        display(b_tridiag)
        println()
        
        # Решаем трехдиагональную систему методом прогонки
        try
            x_thomas = thomas_algorithm(A_tridiag, b_tridiag)
            println("\nРешение:")
            display(x_thomas)
            println("\nНевязка: ||A_tridiag * x - b_tridiag|| = ", norm(A_tridiag * x_thomas - b_tridiag))
        catch e
            println("\nОшибка при решении методом прогонки: ", e.msg)
        end
    end
    
    println("\nМетод прогонки завершен.")
end

# Функция для демонстрации метода Гаусса-Зейделя
function demo_gauss_seidel(A, b)
    println("ДЕМОНСТРАЦИЯ МЕТОДА ГАУССА-ЗЕЙДЕЛЯ")
    print_separator()
    
    println("Матрица A:")
    display(A)
    println()
    
    println("Вектор b:")
    display(b)
    println()
    
    # Проверяем диагональное доминирование
    is_diag_dom = is_diagonally_dominant(A)
    println("\nМатрица", is_diag_dom ? " " : " не ", "является диагонально доминирующей.")
    
    # Решаем систему методом Гаусса-Зейделя
    try
        println("\nПрименяем метод Гаусса-Зейделя")
        x0 = zeros(length(b))
        x_gs, iter_gs, residuals_gs = gauss_seidel(A, b, x0)
        
        println("\nНачальное приближение x0:")
        display(x0)
        println()
        
        println("Решение:")
        display(x_gs)
        println("\nКоличество итераций: ", iter_gs)
        println("\nИтоговая невязка: ", residuals_gs[end])
        println("\nНевязка: ||Ax - b|| = ", norm(A * x_gs - b))
        
        # Выводим график сходимости (если есть пакет Plots)
        try
            using Plots
            plot(1:iter_gs, residuals_gs, xlabel="Итерация", ylabel="Невязка", 
                 yscale=:log10, label="Невязка", legend=:topright)
            title!("Сходимость метода Гаусса-Зейделя")
            display(plot!)
        catch
            println("\nДля отображения графика сходимости установите пакет Plots:")
            println("using Pkg; Pkg.add(\"Plots\")")
            
            # Выводим первые 10 итераций и последние 5 итераций
            if iter_gs <= 15
                println("\nНевязки на каждой итерации:")
                for i = 1:iter_gs
                    println("Итерация $i: ", residuals_gs[i])
                end
            else
                println("\nНевязки на первых 10 итерациях:")
                for i = 1:10
                    println("Итерация $i: ", residuals_gs[i])
                end
                println("...")
                println("\nНевязки на последних 5 итерациях:")
                for i = iter_gs-4:iter_gs
                    println("Итерация $i: ", residuals_gs[i])
                end
            end
        end
    catch e
        println("\nОшибка при решении методом Гаусса-Зейделя: ", e.msg)
    end
    
    pause()
    
    # Решаем систему методом Гаусса-Зейделя с другим начальным приближением
    try
        println("\nПрименяем метод Гаусса-Зейделя с другим начальным приближением")
        x0_ones = ones(length(b))
        x_gs_ones, iter_gs_ones, residuals_gs_ones = gauss_seidel(A, b, x0_ones)
        
        println("\nНачальное приближение x0:")
        display(x0_ones)
        println()
        
        println("Решение:")
        display(x_gs_ones)
        println("\nКоличество итераций: ", iter_gs_ones)
        println("\nИтоговая невязка: ", residuals_gs_ones[end])
        println("\nНевязка: ||Ax - b|| = ", norm(A * x_gs_ones - b))
    catch e
        println("\nОшибка при решении методом Гаусса-Зейделя: ", e.msg)
    end
    
    println("\nМетод Гаусса-Зейделя завершен.")
end

# Главная функция
function main()
    println("ИНТЕРАКТИВНОЕ ИССЛЕДОВАНИЕ МЕТОДОВ РЕШЕНИЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ В JULIA")
    print_separator()
    
    # Создаем систему уравнений
    A, b, system_name = create_system()
    
    print_separator()
    println("Выбрана система: ", system_name)
    
    while true
        println("\nВыберите метод для демонстрации:")
        println("1. Прямые методы")
        println("2. Метод Гаусса")
        println("3. Метод прогонки")
        println("4. Метод Гаусса-Зейделя")
        println("5. Выбрать другую систему")
        println("0. Выход")
        
        choice = parse(Int, readline())
        
        if choice == 0
            break
        elseif choice == 1
            demo_direct_methods(A, b)
        elseif choice == 2
            demo_gaussian_elimination(A, b)
        elseif choice == 3
            demo_thomas_algorithm(A, b)
        elseif choice == 4
            demo_gauss_seidel(A, b)
        elseif choice == 5
            A, b, system_name = create_system()
            print_separator()
            println("Выбрана система: ", system_name)
        else
            println("Неверный выбор. Попробуйте снова.")
        end
        
        print_separator()
    end
    
    println("Спасибо за использование интерактивного исследования методов решения СЛАУ!")
end

# Запускаем основную функцию
main() 