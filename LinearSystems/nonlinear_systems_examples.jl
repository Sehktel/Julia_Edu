using LinearSystems.SimpleIteration
using LinearSystems.SeidelMethod
using LinearSystems.NewtonMethod
using Plots
using LinearAlgebra

# Пример 1: Простая система нелинейных уравнений
# {
#   x^2 + y^2 = 4
#   x*y = 1
# }
function example_system1(x)
    return [
        x[1]^2 + x[2]^2 - 4,  # x^2 + y^2 - 4 = 0
        x[1]*x[2] - 1         # x*y - 1 = 0
    ]
end

# Пример 2: Более сложная система нелинейных уравнений
# {
#   sin(x) + y^2 = 1
#   e^x + y = 2
# }
function example_system2(x)
    return [
        sin(x[1]) + x[2]^2 - 1,  # sin(x) + y^2 - 1 = 0
        exp(x[1]) + x[2] - 2     # e^x + y - 2 = 0
    ]
end

# Пример 3: Система из 3 уравнений
# {
#   x^2 + y^2 + z^2 = 3
#   x*y*z = 1
#   x + y - z = 0
# }
function example_system3(x)
    return [
        x[1]^2 + x[2]^2 + x[3]^2 - 3,  # x^2 + y^2 + z^2 - 3 = 0
        x[1]*x[2]*x[3] - 1,            # x*y*z - 1 = 0
        x[1] + x[2] - x[3]             # x + y - z = 0
    ]
end

function print_results(method_name, x, converged, iter_count, errors)
    println("\n$method_name:")
    println("Решение: $x")
    println("Сходимость: $converged")
    println("Количество итераций: $iter_count")
    println("Конечная ошибка: $(errors[end])")
end

function plot_convergence(errors_list, method_names)
    p = plot(xlabel="Итерации", ylabel="Ошибка (log scale)",
             title="Сходимость методов", legend=:topright, 
             yscale=:log10)
    
    for (i, errors) in enumerate(errors_list)
        plot!(p, 1:length(errors), errors, label=method_names[i])
    end
    
    return p
end

function main()
    println("Пример 1: Система {x^2 + y^2 = 4, x*y = 1}")
    initial_guess = [1.0, 1.0]  # Начальное приближение
    
    # Метод простой итерации
    result_simple = simple_iteration_solve(example_system1, initial_guess, α=0.1)
    print_results("Метод простой итерации", result_simple...)
    
    # Метод Зейделя
    result_seidel = seidel_solve(example_system1, initial_guess)
    print_results("Метод Зейделя", result_seidel...)
    
    # Метод Ньютона
    result_newton = newton_solve(example_system1, initial_guess)
    print_results("Метод Ньютона", result_newton...)
    
    # Сравнение скорости сходимости
    p1 = plot_convergence(
        [result_simple[4], result_seidel[4], result_newton[4]],
        ["Простая итерация", "Зейдель", "Ньютон"]
    )
    savefig(p1, "convergence_example1.png")
    
    println("\nПример 2: Система {sin(x) + y^2 = 1, e^x + y = 2}")
    initial_guess = [0.5, 0.5]  # Начальное приближение
    
    # Метод простой итерации
    result_simple = simple_iteration_solve(example_system2, initial_guess, α=0.05)
    print_results("Метод простой итерации", result_simple...)
    
    # Метод Зейделя
    result_seidel = seidel_solve(example_system2, initial_guess)
    print_results("Метод Зейделя", result_seidel...)
    
    # Метод Ньютона
    result_newton = newton_solve(example_system2, initial_guess)
    print_results("Метод Ньютона", result_newton...)
    
    # Сравнение скорости сходимости
    p2 = plot_convergence(
        [result_simple[4], result_seidel[4], result_newton[4]],
        ["Простая итерация", "Зейдель", "Ньютон"]
    )
    savefig(p2, "convergence_example2.png")
    
    println("\nПример 3: Система из 3 уравнений")
    initial_guess = [1.0, 1.0, 1.0]  # Начальное приближение
    
    # Метод Ньютона
    result_newton = newton_solve(example_system3, initial_guess)
    print_results("Метод Ньютона", result_newton...)
    
    # Модифицированный метод Ньютона
    result_mod_newton = modified_newton_solve(example_system3, initial_guess)
    print_results("Модифицированный метод Ньютона", result_mod_newton...)
    
    # Сравнение скорости сходимости между классическим и модифицированным методом Ньютона
    p3 = plot_convergence(
        [result_newton[4], result_mod_newton[4]],
        ["Ньютон", "Модифицированный Ньютон"]
    )
    savefig(p3, "convergence_example3.png")
    
    println("\nСравнение точности решений на примере 1:")
    solutions = [result_simple[1], result_seidel[1], result_newton[1]]
    for (i, sol) in enumerate(solutions)
        method_name = ["Простая итерация", "Зейдель", "Ньютон"][i]
        residual = norm(example_system1(sol))
        println("$method_name: Остаточная ошибка = $residual")
    end
end

# Выполнение примеров
main() 