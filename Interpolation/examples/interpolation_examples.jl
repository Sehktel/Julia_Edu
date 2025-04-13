"""
Примеры использования методов интерполяции

Данный скрипт демонстрирует применение различных методов интерполяции,
реализованных в модуле Interpolation.
"""

using Pkg
Pkg.activate("..")

using Interpolation
using Plots

# Определим тестовую функцию для интерполяции
f(x) = sin(2π * x)
f_derivative(x) = 2π * cos(2π * x)  # первая производная

# Функция для визуализации результатов интерполяции
function plot_interpolation(f, x_points, interp_func; title="Сравнение методов интерполяции", 
                          x_range=nothing, show_points=true)
    # Если диапазон не задан, определяем его автоматически
    if x_range === nothing
        a, b = minimum(x_points), maximum(x_points)
        # Добавляем небольшие отступы
        margin = 0.05 * (b - a)
        x_range = range(a - margin, b + margin, length=500)
    end
    
    # Вычисляем значения исходной функции и интерполяции
    y_exact = f.(x_range)
    y_interp = [interp_func(x) for x in x_range]
    
    # Создаем график
    p = plot(x_range, y_exact, label="Точная функция", linewidth=2, legend=:topleft)
    plot!(x_range, y_interp, label="Интерполяция", linestyle=:dash, linewidth=2)
    
    if show_points
        # Добавляем точки, через которые проходит интерполяция
        y_points = f.(x_points)
        scatter!(x_points, y_points, label="Узлы интерполяции", markersize=5)
    end
    
    # Добавляем заголовок и подписи осей
    title!(title)
    xlabel!("x")
    ylabel!("y")
    
    return p
end

# Пример 1: Линейная интерполяция
function example_linear_interpolation()
    println("Пример 1: Линейная интерполяция")
    
    # Создаем узлы интерполяции
    x_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_points = f.(x_points)
    
    # Создаем функцию интерполяции
    interp_func = x -> linear_interpolation(x_points, y_points, x)
    
    # Оцениваем точность в заданной точке
    test_point = 0.33
    exact_value = f(test_point)
    interp_value = interp_func(test_point)
    error = abs(exact_value - interp_value)
    
    println("Интерполируемая функция: f(x) = sin(2π*x)")
    println("Точка тестирования: x = $test_point")
    println("Точное значение: f($test_point) = $exact_value")
    println("Интерполированное значение: $interp_value")
    println("Абсолютная ошибка: $error")
    
    # Визуализируем результаты
    p = plot_interpolation(f, x_points, interp_func, 
                         title="Линейная интерполяция функции sin(2π*x)",
                         x_range=range(0.0, 1.0, length=500))
    
    display(p)
    println("\n")
    
    return p
end

# Пример 2: Квадратичная интерполяция
function example_quadratic_interpolation()
    println("Пример 2: Квадратичная интерполяция")
    
    # Создаем узлы интерполяции
    x_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_points = f.(x_points)
    
    # Создаем функцию интерполяции
    interp_func = x -> quadratic_interpolation(x_points, y_points, x)
    
    # Оцениваем точность в заданной точке
    test_point = 0.33
    exact_value = f(test_point)
    interp_value = interp_func(test_point)
    error = abs(exact_value - interp_value)
    
    println("Интерполируемая функция: f(x) = sin(2π*x)")
    println("Точка тестирования: x = $test_point")
    println("Точное значение: f($test_point) = $exact_value")
    println("Интерполированное значение: $interp_value")
    println("Абсолютная ошибка: $error")
    
    # Визуализируем результаты
    p = plot_interpolation(f, x_points, interp_func, 
                         title="Квадратичная интерполяция функции sin(2π*x)",
                         x_range=range(0.0, 1.0, length=500))
    
    display(p)
    println("\n")
    
    return p
end

# Пример 3: Сплайн-интерполяция
function example_spline_interpolation()
    println("Пример 3: Сплайн-интерполяция")
    
    # Создаем узлы интерполяции
    x_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_points = f.(x_points)
    
    # Создаем естественный сплайн
    natural_spline_obj = natural_spline(x_points, y_points)
    natural_interp_func = x -> evaluate_spline(natural_spline_obj, x)
    
    # Создаем зажатый сплайн с заданными производными на концах
    dy_start = f_derivative(x_points[1])
    dy_end = f_derivative(x_points[end])
    clamped_spline_obj = clamped_spline(x_points, y_points, dy_start, dy_end)
    clamped_interp_func = x -> evaluate_spline(clamped_spline_obj, x)
    
    # Оцениваем точность в заданной точке
    test_point = 0.33
    exact_value = f(test_point)
    natural_value = natural_interp_func(test_point)
    clamped_value = clamped_interp_func(test_point)
    natural_error = abs(exact_value - natural_value)
    clamped_error = abs(exact_value - clamped_value)
    
    println("Интерполируемая функция: f(x) = sin(2π*x)")
    println("Точка тестирования: x = $test_point")
    println("Точное значение: f($test_point) = $exact_value")
    println("Значение естественного сплайна: $natural_value (ошибка: $natural_error)")
    println("Значение зажатого сплайна: $clamped_value (ошибка: $clamped_error)")
    
    # Визуализируем результаты
    x_range = range(0.0, 1.0, length=500)
    y_exact = f.(x_range)
    y_natural = [natural_interp_func(x) for x in x_range]
    y_clamped = [clamped_interp_func(x) for x in x_range]
    
    p = plot(x_range, y_exact, label="Точная функция", linewidth=2, legend=:topleft)
    plot!(x_range, y_natural, label="Естественный сплайн", linestyle=:dash, linewidth=2)
    plot!(x_range, y_clamped, label="Зажатый сплайн", linestyle=:dashdot, linewidth=2)
    scatter!(x_points, y_points, label="Узлы интерполяции", markersize=5)
    title!("Сплайн-интерполяция функции sin(2π*x)")
    xlabel!("x")
    ylabel!("y")
    
    display(p)
    println("\n")
    
    return p
end

# Пример 4: Интерполяционный полином Лагранжа
function example_lagrange_interpolation()
    println("Пример 4: Интерполяционный полином Лагранжа")
    
    # Создаем узлы интерполяции
    n = 10  # количество узлов
    
    # a) Равноотстоящие узлы
    x_equidistant = equidistant_nodes(0.0, 1.0, n-1)
    y_equidistant = f.(x_equidistant)
    
    # b) Узлы Чебышева
    x_chebyshev = chebyshev_nodes(0.0, 1.0, n-1)
    y_chebyshev = f.(x_chebyshev)
    
    # Создаем функции интерполяции
    equidistant_interp_func = lagrange_polynomial(x_equidistant, y_equidistant)
    chebyshev_interp_func = lagrange_polynomial(x_chebyshev, y_chebyshev)
    
    # Оценка точности
    test_point = 0.33
    exact_value = f(test_point)
    equidistant_value = equidistant_interp_func(test_point)
    chebyshev_value = chebyshev_interp_func(test_point)
    equidistant_error = abs(exact_value - equidistant_value)
    chebyshev_error = abs(exact_value - chebyshev_value)
    
    println("Интерполируемая функция: f(x) = sin(2π*x)")
    println("Количество узлов: $n")
    println("Точка тестирования: x = $test_point")
    println("Точное значение: f($test_point) = $exact_value")
    println("Значение с равноотстоящими узлами: $equidistant_value (ошибка: $equidistant_error)")
    println("Значение с узлами Чебышева: $chebyshev_value (ошибка: $chebyshev_error)")
    
    # Визуализируем результаты
    x_range = range(0.0, 1.0, length=500)
    y_exact = f.(x_range)
    y_equidistant = [equidistant_interp_func(x) for x in x_range]
    y_chebyshev = [chebyshev_interp_func(x) for x in x_range]
    
    p = plot(x_range, y_exact, label="Точная функция", linewidth=2, legend=:topleft)
    plot!(x_range, y_equidistant, label="Полином Лагранжа (равном.)", linestyle=:dash, linewidth=2)
    plot!(x_range, y_chebyshev, label="Полином Лагранжа (Чебышев)", linestyle=:dashdot, linewidth=2)
    scatter!(x_equidistant, y_equidistant, label="Равноотстоящие узлы", markersize=5)
    scatter!(x_chebyshev, y_chebyshev, label="Узлы Чебышева", markersize=5)
    title!("Интерполяционный полином Лагранжа для sin(2π*x)")
    xlabel!("x")
    ylabel!("y")
    
    display(p)
    println("\n")
    
    return p
end

# Пример 5: Эффект Рунге и сравнение различных методов
function example_runge_phenomenon()
    println("Пример 5: Эффект Рунге и сравнение различных методов")
    
    # Функция Рунге
    runge(x) = 1 / (1 + 25x^2)
    
    # Создаем узлы интерполяции
    n = 11  # количество узлов
    
    # a) Равноотстоящие узлы
    x_equidistant = equidistant_nodes(-1.0, 1.0, n-1)
    y_equidistant = runge.(x_equidistant)
    
    # b) Узлы Чебышева
    x_chebyshev = chebyshev_nodes(-1.0, 1.0, n-1)
    y_chebyshev = runge.(x_chebyshev)
    
    # Создаем функции интерполяции
    lagrange_equidistant = lagrange_polynomial(x_equidistant, y_equidistant)
    lagrange_chebyshev = lagrange_polynomial(x_chebyshev, y_chebyshev)
    spline = cubic_spline(x_equidistant, y_equidistant)
    spline_func = x -> evaluate_spline(spline, x)
    
    # Визуализируем результаты
    x_range = range(-1.0, 1.0, length=500)
    y_exact = runge.(x_range)
    y_lagrange_equidistant = [lagrange_equidistant(x) for x in x_range]
    y_lagrange_chebyshev = [lagrange_chebyshev(x) for x in x_range]
    y_spline = [spline_func(x) for x in x_range]
    
    p = plot(x_range, y_exact, label="Функция Рунге", linewidth=2, legend=:topleft)
    plot!(x_range, y_lagrange_equidistant, label="Полином Лагранжа (равном.)", linestyle=:dash, linewidth=2)
    plot!(x_range, y_lagrange_chebyshev, label="Полином Лагранжа (Чебышев)", linestyle=:dashdot, linewidth=2)
    plot!(x_range, y_spline, label="Кубический сплайн", linestyle=:dot, linewidth=2)
    scatter!(x_equidistant, y_equidistant, label="Равноотстоящие узлы", markersize=5)
    scatter!(x_chebyshev, y_chebyshev, label="Узлы Чебышева", markersize=5)
    title!("Интерполяция функции Рунге (n = $n узлов)")
    xlabel!("x")
    ylabel!("y")
    
    # Вычисляем максимальную ошибку для каждого метода
    max_error_lagrange_equidistant = maximum([abs(runge(x) - lagrange_equidistant(x)) for x in x_range])
    max_error_lagrange_chebyshev = maximum([abs(runge(x) - lagrange_chebyshev(x)) for x in x_range])
    max_error_spline = maximum([abs(runge(x) - spline_func(x)) for x in x_range])
    
    println("Максимальная ошибка для полинома Лагранжа (равноотст.): $max_error_lagrange_equidistant")
    println("Максимальная ошибка для полинома Лагранжа (Чебышев): $max_error_lagrange_chebyshev")
    println("Максимальная ошибка для кубического сплайна: $max_error_spline")
    
    display(p)
    println("\n")
    
    return p
end

# Запускаем примеры
function run_all_examples()
    p1 = example_linear_interpolation()
    p2 = example_quadratic_interpolation()
    p3 = example_spline_interpolation()
    p4 = example_lagrange_interpolation()
    p5 = example_runge_phenomenon()
    
    # Отображаем все графики вместе
    display(plot(p1, p2, p3, p4, p5, layout=(3, 2), size=(900, 800)))
end

# Если скрипт запущен напрямую, выполняем все примеры
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end 