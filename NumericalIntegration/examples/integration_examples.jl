"""
Примеры использования модуля NumericalIntegration
"""

using NumericalIntegration

println("Примеры численного интегрирования в Julia")
println("==========================================\n")

# Функция для сравнения различных методов интегрирования
function compare_methods(f, a, b, exact_value; title="Сравнение методов интегрирования")
    println("\n", title)
    println("-" ^ length(title), "\n")
    println("Интегрирование функции на интервале [$a, $b]")
    println("Точное значение: $exact_value\n")
    
    methods = [
        (:rectangle, "Метод прямоугольников"),
        (:midpoint, "Метод средней точки"),
        (:trapezoid, "Метод трапеций"),
        (:simpson, "Метод Симпсона"),
        (:composite_trapezoid, "Составной метод трапеций (n=100)"),
        (:composite_simpson, "Составной метод Симпсона (n=100)"),
        (:adaptive, "Адаптивное интегрирование"),
        (:romberg, "Метод Ромберга")
    ]
    
    println("| Метод | Результат | Погрешность |")
    println("|-------|-----------|-------------|")
    
    for (method_sym, method_name) in methods
        # Для составных методов устанавливаем число подынтервалов
        if method_sym == :composite_trapezoid || method_sym == :composite_simpson
            result = integrate(f, a, b, method_sym, n=100, return_details=true)
        else
            result = integrate(f, a, b, method_sym, return_details=true)
        end
        
        absolute_error = abs(result.value - exact_value)
        
        println("| $(rpad(method_name, 30)) | $(lpad(round(result.value, digits=10), 15)) | $(lpad(scientific_format(absolute_error), 15)) |")
    end
    println()
end

# Функция для форматирования чисел в научной нотации
function scientific_format(x)
    if x == 0
        return "0"
    end
    exponent = floor(Int, log10(abs(x)))
    mantissa = x / 10.0^exponent
    return @sprintf("%.2fe%d", mantissa, exponent)
end

# Пример 1: Элементарная функция
println("Пример 1: Интегрирование полинома")
f1(x) = x^2
a1, b1 = 0, 1
exact1 = 1/3
compare_methods(f1, a1, b1, exact1, title="Интегрирование f(x) = x²")

# Пример 2: Тригонометрическая функция
println("Пример 2: Интегрирование тригонометрической функции")
f2(x) = sin(x)
a2, b2 = 0, π
exact2 = 2.0
compare_methods(f2, a2, b2, exact2, title="Интегрирование f(x) = sin(x)")

# Пример 3: Функция с быстро изменяющимся поведением
println("Пример 3: Функция с быстро изменяющимся поведением")
f3(x) = 1 / (1 + 25*x^2)
a3, b3 = -1, 1
exact3 = 0.4 * atan(5)
compare_methods(f3, a3, b3, exact3, title="Интегрирование f(x) = 1/(1+25x²)")

# Пример 4: Интегрирование функции с особенностью
println("Пример 4: Интегрирование функции с особенностью")
println("Функция log(x) на интервале [0, 1] (имеет особенность в x=0)")
println("Точное значение: -1.0\n")

methods_singular = [
    (:tanh_sinh, "Tanh-Sinh квадратура", Dict()),
    (:double_exponential, "Double Exponential метод", Dict()),
    (:adaptive, "Адаптивное интегрирование (начиная с 0.0001)", Dict{Symbol,Any}(:a => 0.0001))
]

println("| Метод | Результат | Погрешность |")
println("|-------|-----------|-------------|")

for (method_sym, method_name, kwargs) in methods_singular
    # Интегрирование с особенностью в точке x=0
    if method_sym == :adaptive
        # Для адаптивного метода начинаем с 0.0001 вместо 0
        result = integrate(log, 0.0001, 1.0, method_sym, return_details=true)
    else
        result = integrate(log, 0, 1, method_sym, return_details=true)
    end
    
    absolute_error = abs(result.value - (-1.0))
    
    println("| $(rpad(method_name, 30)) | $(lpad(round(result.value, digits=10), 15)) | $(lpad(scientific_format(absolute_error), 15)) |")
end
println()

# Пример 5: Двойной интеграл
println("Пример 5: Вычисление двойного интеграла")
println("Интегрирование f(x,y) = x*y на квадрате [0,1]×[0,1]")
println("Точное значение: 0.25\n")

f5(x, y) = x * y
result5 = double_integral(f5, 0, 1, 0, 1, return_details=true)
println("Результат: $(result5.value)")
println("Оценка погрешности: $(result5.error_estimate)")
println("Число вычислений функции: $(result5.n_evaluations)")
println()

# Пример 6: Метод Монте-Карло
println("Пример 6: Интегрирование методом Монте-Карло")
println("Вычисление объема единичного шара в трехмерном пространстве")
println("Точное значение: $(4/3 * π)\n")

function ball_indicator(x)
    # Функция-индикатор единичного шара
    return sum(x.^2) <= 1 ? 1.0 : 0.0
end

# Интегрируем по кубу [-1,1]³
bounds6 = [(-1, 1), (-1, 1), (-1, 1)]
n_samples = 100000

# Используем функцию из модуля MultipleIntegrals
result6, error6 = monte_carlo_integration(ball_indicator, -ones(3), ones(3), n_samples, dims=3)

volume = result6 * 8  # Умножаем на объем куба [-1,1]³
error = error6 * 8

println("Метод Монте-Карло с $(n_samples) точками:")
println("Приближенное значение: $volume")
println("Стандартная ошибка: $error")
println("Относительная погрешность: $(abs(volume - 4/3*π) / (4/3*π) * 100)%")
println()

# Пример 7: Интегрирование с весом 1/sqrt(x) на [0,1]
println("Пример 7: Интегрирование с весом 1/sqrt(x)")
println("Функция f(x) = x/sqrt(x) = sqrt(x) на интервале [0, 1]")
println("Точное значение: 2/3\n")

# Преобразование переменных для обработки особенности
f7(x) = x
transformation(t) = t^2
transformation_deriv(t) = 2*t

result7 = singularity_transformation(f7, transformation, transformation_deriv, 0, 1, return_details=true)

println("Метод преобразования особенности:")
println("Приближенное значение: $(result7.value)")
println("Оценка погрешности: $(result7.error_estimate)")
println("Абсолютная погрешность: $(abs(result7.value - 2/3))")
println()

println("Завершение примеров") 