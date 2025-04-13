# OrdinaryDifferentialEquations

Модуль для численного решения обыкновенных дифференциальных уравнений в Julia.

## Возможности

- Метод Эйлера (явный и неявный)
- Методы Рунге-Кутты различных порядков
- Методы Адамса
- Решение систем ОДУ
- Решение ОДУ высших порядков
- Краевые задачи

## Установка

```julia
import Pkg
Pkg.develop(path="путь/к/LinearSystems/OrdinaryDifferentialEquations")
```

## Использование

```julia
using OrdinaryDifferentialEquations

# Определение ОДУ: y' = f(t, y)
f(t, y) = -2*y  # y' = -2y (решение: y = y0 * exp(-2t))

# Начальные условия
t0 = 0.0
y0 = 1.0

# Параметры интегрирования
t_end = 2.0
h = 0.1

# Решение методом Эйлера
t, y = euler_method(f, t0, y0, t_end, h)

# Решение методом Рунге-Кутты 4-го порядка
t, y = runge_kutta4(f, t0, y0, t_end, h)
```

## Структура модуля

- `OrdinaryDifferentialEquations.jl` - основной модуль
- `EulerMethod.jl` - реализация метода Эйлера
- `RungeKuttaMethods.jl` - методы Рунге-Кутты
- `AdamsMethods.jl` - методы Адамса
- `BoundaryValueProblems.jl` - решение краевых задач
- `ODESystems.jl` - решение систем ОДУ
- `HighOrderODE.jl` - решение ОДУ высших порядков

## Примеры

В директории `examples` содержатся примеры использования модуля.

## Документация

Подробная документация и математическое описание методов доступны в файлах:
- `doc_euler_method.md`
- `doc_runge_kutta_method.md`
- `doc_adams_method.md`
- `doc_systems_ode.md`
- `doc_high_order_ode.md`
- `doc_boundary_value_problems.md`
- `doc_jacobian_matrix.md`

## Лицензия

Этот проект лицензирован под MIT License. 