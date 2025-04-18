# Методы Адамса для решения обыкновенных дифференциальных уравнений

## Теоретическое описание

Методы Адамса — семейство линейных многошаговых методов для численного решения обыкновенных дифференциальных уравнений (ОДУ). Они используют информацию о решении в нескольких предыдущих точках для вычисления следующего значения. В отличие от одношаговых методов (например, методов Рунге-Кутта), многошаговые методы более эффективны по вычислительным затратам, но требуют дополнительных процедур для получения начальных значений.

### Задача Коши

Рассмотрим задачу Коши для ОДУ первого порядка:

$$\begin{cases}
\frac{dy}{dt} = f(t, y) \\
y(t_0) = y_0
\end{cases}$$

где $f(t, y)$ - правая часть ОДУ, $t_0$ - начальный момент времени, $y_0$ - начальное условие.

### Методы Адамса-Башфорта (явные)

Явные методы Адамса-Башфорта определяются следующей формулой:

$$y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j f(t_{n-j}, y_{n-j})$$

где $h$ - шаг интегрирования, $k$ - порядок метода, $\beta_j$ - коэффициенты метода.

Для методов различных порядков коэффициенты $\beta_j$ имеют следующие значения:

**1-й порядок (метод Эйлера):**
$$y_{n+1} = y_n + h f(t_n, y_n)$$
$\beta_0 = 1$

**2-й порядок:**
$$y_{n+1} = y_n + h \left( \frac{3}{2} f(t_n, y_n) - \frac{1}{2} f(t_{n-1}, y_{n-1}) \right)$$
$\beta_0 = \frac{3}{2}$, $\beta_1 = -\frac{1}{2}$

**3-й порядок:**
$$y_{n+1} = y_n + h \left( \frac{23}{12} f(t_n, y_n) - \frac{16}{12} f(t_{n-1}, y_{n-1}) + \frac{5}{12} f(t_{n-2}, y_{n-2}) \right)$$
$\beta_0 = \frac{23}{12}$, $\beta_1 = -\frac{16}{12}$, $\beta_2 = \frac{5}{12}$

**4-й порядок:**
$$y_{n+1} = y_n + h \left( \frac{55}{24} f(t_n, y_n) - \frac{59}{24} f(t_{n-1}, y_{n-1}) + \frac{37}{24} f(t_{n-2}, y_{n-2}) - \frac{9}{24} f(t_{n-3}, y_{n-3}) \right)$$
$\beta_0 = \frac{55}{24}$, $\beta_1 = -\frac{59}{24}$, $\beta_2 = \frac{37}{24}$, $\beta_3 = -\frac{9}{24}$

### Методы Адамса-Мультона (неявные)

Неявные методы Адамса-Мультона определяются следующей формулой:

$$y_{n+1} = y_n + h \sum_{j=-1}^{k-2} \gamma_j f(t_{n-j}, y_{n-j})$$

где $\gamma_j$ - коэффициенты метода.

Для методов различных порядков коэффициенты $\gamma_j$ имеют следующие значения:

**1-й порядок (неявный метод Эйлера):**
$$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$
$\gamma_{-1} = 1$

**2-й порядок (метод трапеций):**
$$y_{n+1} = y_n + h \left( \frac{1}{2} f(t_{n+1}, y_{n+1}) + \frac{1}{2} f(t_n, y_n) \right)$$
$\gamma_{-1} = \frac{1}{2}$, $\gamma_0 = \frac{1}{2}$

**3-й порядок:**
$$y_{n+1} = y_n + h \left( \frac{5}{12} f(t_{n+1}, y_{n+1}) + \frac{8}{12} f(t_n, y_n) - \frac{1}{12} f(t_{n-1}, y_{n-1}) \right)$$
$\gamma_{-1} = \frac{5}{12}$, $\gamma_0 = \frac{8}{12}$, $\gamma_1 = -\frac{1}{12}$

**4-й порядок:**
$$y_{n+1} = y_n + h \left( \frac{9}{24} f(t_{n+1}, y_{n+1}) + \frac{19}{24} f(t_n, y_n) - \frac{5}{24} f(t_{n-1}, y_{n-1}) + \frac{1}{24} f(t_{n-2}, y_{n-2}) \right)$$
$\gamma_{-1} = \frac{9}{24}$, $\gamma_0 = \frac{19}{24}$, $\gamma_1 = -\frac{5}{24}$, $\gamma_2 = \frac{1}{24}$

### Методы предиктор-корректор (Адамса-Башфорта-Мультона)

Методы предиктор-корректор объединяют явные и неявные методы для повышения эффективности:

1. **Предиктор**: Используется явный метод Адамса-Башфорта для получения предварительного значения $\tilde{y}_{n+1}$.
2. **Корректор**: Используется неявный метод Адамса-Мультона, в котором $f(t_{n+1}, y_{n+1})$ аппроксимируется через $f(t_{n+1}, \tilde{y}_{n+1})$.

Общая схема метода предиктор-корректор порядка $k$:

**Предиктор (Адамс-Башфорт):**
$$\tilde{y}_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j f(t_{n-j}, y_{n-j})$$

**Корректор (Адамс-Мультон):**
$$y_{n+1} = y_n + h \left( \gamma_{-1} f(t_{n+1}, \tilde{y}_{n+1}) + \sum_{j=0}^{k-2} \gamma_j f(t_{n-j}, y_{n-j}) \right)$$

## Реализация на Julia

В нашей реализации представлены три метода Адамса:
1. Явные методы Адамса-Башфорта
2. Неявные методы Адамса-Мультона
3. Методы предиктор-корректор Адамса-Башфорта-Мультона

### Адамс-Башфорт (явные методы)

```julia
"""
    adams_bashforth_solve(f, t_span, y0; step_size=0.01, order=4)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
явным методом Адамса-Башфорта заданного порядка.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Башфорта (от 1 до 5)
"""
function adams_bashforth_solve(f, t_span, y0; step_size=0.01, order=4)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Распаковываем временной интервал
    t0, tf = t_span
    
    # Получаем коэффициенты метода Адамса-Башфорта
    ab_coeffs = adams_bashforth_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    # ...
    
    # Основной цикл метода Адамса-Башфорта
    for i in (order+1):length(t)
        # Сдвигаем массив производных
        # ...
        
        # Вычисляем следующее значение
        y[i] = y[i-1] + step_size * sum(ab_coeffs .* fs)
    end
    
    return t, y
end

"""
    adams_bashforth_coefficients(order)

Возвращает коэффициенты явного метода Адамса-Башфорта заданного порядка.
"""
function adams_bashforth_coefficients(order)
    if order == 1
        # Явный метод Эйлера
        return [1.0]
    elseif order == 2
        return [3/2, -1/2]
    elseif order == 3
        return [23/12, -16/12, 5/12]
    elseif order == 4
        return [55/24, -59/24, 37/24, -9/24]
    elseif order == 5
        return [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
    else
        error("Неподдерживаемый порядок метода Адамса-Башфорта: $order")
    end
end
```

### Адамс-Мультон (неявные методы)

```julia
"""
    adams_moulton_solve(f, t_span, y0; step_size=0.01, order=4, tol=1e-6, max_iter=10)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
неявным методом Адамса-Мультона заданного порядка.

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Мультона (от 1 до 5)
- `tol::Float64=1e-6`: Допустимая погрешность для итерационного процесса
- `max_iter::Int=10`: Максимальное число итераций для неявной схемы
"""
function adams_moulton_solve(f, t_span, y0; step_size=0.01, order=4, tol=1e-6, max_iter=10)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Получаем коэффициенты метода Адамса-Мультона
    am_coeffs = adams_moulton_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    # ...
    
    # Основной цикл метода Адамса-Мультона
    for i in order:length(t) - 1
        # Начальное приближение для y[i+1]
        y_prev = y[i] + step_size * (am_coeffs[1:order-1]' * fs[1:order-1])
        
        # Итерационный процесс для решения неявной схемы
        for iter in 1:max_iter
            f_new = f(t[i+1], y_prev)
            y_new = y[i] + step_size * (am_coeffs[1:order-1]' * fs[1:order-1] + am_coeffs[order] * f_new)
            
            # Проверка сходимости
            if abs(y_new - y_prev) < tol
                y_prev = y_new
                break
            end
            
            y_prev = y_new
        end
        
        y[i+1] = y_prev
    end
    
    return t, y
end

"""
    adams_moulton_coefficients(order)

Возвращает коэффициенты неявного метода Адамса-Мультона заданного порядка.
"""
function adams_moulton_coefficients(order)
    if order == 1
        # Неявный метод Эйлера
        return [1.0]
    elseif order == 2
        return [1/2, 1/2]
    elseif order == 3
        return [5/12, 8/12, -1/12]
    elseif order == 4
        return [9/24, 19/24, -5/24, 1/24]
    elseif order == 5
        return [251/720, 646/720, -264/720, 106/720, -19/720]
    else
        error("Неподдерживаемый порядок метода Адамса-Мультона: $order")
    end
end
```

### Адамс-Башфорт-Мультон (предиктор-корректор)

```julia
"""
    adams_bashforth_moulton_solve(f, t_span, y0; step_size=0.01, order=4, max_iter=1)

Решает задачу Коши для обыкновенного дифференциального уравнения первого порядка 
методом прогноза-коррекции Адамса-Башфорта-Мультона (предиктор-корректор).

# Аргументы
- `f::Function`: Правая часть ОДУ в форме y' = f(t, y)
- `t_span::Tuple{Float64, Float64}`: Интервал интегрирования [t0, tf]
- `y0::Union{Float64, Vector{Float64}}`: Начальное условие y(t0) = y0
- `step_size::Float64=0.01`: Размер шага интегрирования
- `order::Int=4`: Порядок метода Адамса-Башфорта-Мультона (от 1 до 5)
- `max_iter::Int=1`: Максимальное число итераций корректора
"""
function adams_bashforth_moulton_solve(f, t_span, y0; step_size=0.01, order=4, max_iter=1)
    # Ограничиваем порядок метода
    order = min(max(order, 1), 5)
    
    # Получаем коэффициенты методов
    ab_coeffs = adams_bashforth_coefficients(order)
    am_coeffs = adams_moulton_coefficients(order)
    
    # Используем метод Рунге-Кутта 4 порядка для получения начальных значений
    # ...
    
    # Основной цикл метода Адамса-Башфорта-Мультона
    for i in (order+1):length(t)
        # Предиктор (метод Адамса-Башфорта)
        y_pred = y[i-1] + step_size * sum(ab_coeffs .* fs)
        
        # Корректор (метод Адамса-Мультона)
        y_corr = y_pred
        for iter in 1:max_iter
            f_new = f(t[i], y_corr)
            
            # Применяем корректор
            fs_corr = copy(fs)
            fs_corr[order] = f_new
            
            y_corr = y[i-1] + step_size * sum(am_coeffs .* fs_corr)
        end
        
        y[i] = y_corr
    end
    
    return t, y
end
```

## Примеры использования

### Пример 1: Решение скалярного ОДУ методами Адамса различных порядков

Рассмотрим задачу Коши:
$$y' = -2y, \quad y(0) = 1$$

Точное решение этой задачи: $y(t) = e^{-2t}$.

```julia
using OrdinaryDifferentialEquations
using Plots

# Определим правую часть ОДУ
f(t, y) = -2 * y

# Параметры интегрирования
t0, tf = 0.0, 2.0
y0 = 1.0
step_size = 0.1

# Решаем задачу методами Адамса различных порядков
t_ab4, y_ab4 = adams_bashforth_solve(f, (t0, tf), y0, step_size=step_size, order=4)
t_am4, y_am4 = adams_moulton_solve(f, (t0, tf), y0, step_size=step_size, order=4)
t_abm4, y_abm4 = adams_bashforth_moulton_solve(f, (t0, tf), y0, step_size=step_size, order=4)

# Вычисляем точное решение
exact_solution(t) = exp(-2 * t)
t_exact = t0:0.01:tf
y_exact = exact_solution.(t_exact)

# Визуализируем результаты
plot(t_exact, y_exact, label="Точное решение", linewidth=2, color=:black)
scatter!(t_ab4, y_ab4, label="Адамс-Башфорт (4)", markersize=4, color=:blue)
scatter!(t_am4, y_am4, label="Адамс-Мультон (4)", markersize=4, color=:green)
scatter!(t_abm4, y_abm4, label="Предиктор-корректор (4)", markersize=4, color=:red)
xlabel!("t")
ylabel!("y(t)")
title!("Сравнение методов Адамса")
```

### Пример 2: Сравнение эффективности методов Адамса и Рунге-Кутта

```julia
using OrdinaryDifferentialEquations
using Plots
using BenchmarkTools

# Определим систему ОДУ (осциллятор Ван дер Поля)
function van_der_pol(t, x, μ=1.0)
    return [
        x[2],
        μ * (1 - x[1]^2) * x[2] - x[1]
    ]
end

# Параметры интегрирования
t0, tf = 0.0, 20.0
x0 = [2.0, 0.0]
step_size = 0.05

# Решаем задачу различными методами
t_rk4, x_rk4 = @btime runge_kutta4_solve(t -> van_der_pol(t, t, 1.0), (t0, tf), x0, step_size=step_size)
t_abm4, x_abm4 = @btime adams_bashforth_moulton_solve(t -> van_der_pol(t, t, 1.0), (t0, tf), x0, step_size=step_size, order=4)

# Визуализируем фазовые портреты
plot(x_rk4[:, 1], x_rk4[:, 2], label="RK4", linewidth=2)
plot!(x_abm4[:, 1], x_abm4[:, 2], label="ABM4", linewidth=2, linestyle=:dash)
xlabel!("x₁")
ylabel!("x₂")
title!("Фазовый портрет осциллятора Ван дер Поля")
```

## Достоинства и недостатки

### Достоинства методов Адамса

1. **Вычислительная эффективность**: Методы Адамса требуют только одно (для явных) или несколько (для неявных) вычислений функции правой части на каждом шаге, что эффективнее, чем методы Рунге-Кутта того же порядка.

2. **Высокая точность**: Методы высоких порядков обеспечивают очень высокую точность.

3. **Оценка локальной ошибки**: Разница между предиктором и корректором в методах предиктор-корректор даёт оценку локальной ошибки.

4. **Адаптивность**: Легко реализовать адаптивный выбор шага на основе оценки локальной ошибки.

### Недостатки методов Адамса

1. **Необходимость стартовых значений**: Требуют $k$ начальных значений для метода порядка $k$, которые обычно получают другими методами.

2. **Сложность изменения шага**: Изменение размера шага требует пересчета всех начальных значений.

3. **Меньшая область устойчивости**: Методы Адамса имеют меньшую область устойчивости по сравнению с методами Рунге-Кутта того же порядка.

4. **Сложность реализации**: Более сложны в реализации и отладке по сравнению с одношаговыми методами.

## Практические рекомендации

1. **Выбор порядка метода**:
   - Для нежестких задач: методы высоких порядков (4-5) обеспечивают хороший баланс между точностью и эффективностью.
   - Для жестких задач: рассмотрите методы низких порядков или специализированные жесткие методы.

2. **Выбор между явными и неявными методами**:
   - Явные методы Адамса-Башфорта: просты в реализации, но менее устойчивы.
   - Неявные методы Адамса-Мультона: более устойчивы, но требуют решения нелинейных уравнений на каждом шаге.
   - Методы предиктор-корректор: хороший компромисс между устойчивостью и эффективностью.

3. **Генерация начальных значений**:
   - Используйте методы Рунге-Кутта высокого порядка для получения начальных значений.
   - Для повышенной точности используйте шаг меньше основного при генерации начальных значений.

4. **Контроль ошибки и адаптивный шаг**:
   - В методах предиктор-корректор используйте разницу между предиктором и корректором для оценки локальной ошибки.
   - Для эффективного решения используйте адаптивный выбор шага на основе оценки ошибки.

## Заключение

Методы Адамса представляют собой мощный инструмент для численного решения обыкновенных дифференциальных уравнений, особенно когда требуется высокая вычислительная эффективность. Хотя они более сложны в реализации и использовании по сравнению с одношаговыми методами, их преимущества в скорости вычислений и точности делают их привлекательным выбором для многих практических задач.

Выбор между явными методами Адамса-Башфорта, неявными методами Адамса-Мультона или методами предиктор-корректор зависит от конкретной задачи, требований к точности, устойчивости и вычислительной эффективности. Комбинация методов Рунге-Кутта для получения начальных значений и методов Адамса для основного интегрирования часто дает оптимальный результат с точки зрения баланса между точностью и эффективностью. 