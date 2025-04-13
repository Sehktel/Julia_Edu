# Численное интегрирование в Julia

Эта библиотека предоставляет набор методов численного интегрирования, реализованных на языке Julia.

## Особенности

- Базовые методы: прямоугольников, трапеций, Симпсона
- Составные методы с заданным числом подынтервалов
- Адаптивные методы с контролем точности
- Методы Монте-Карло для сложных интегралов
- Многомерное интегрирование (двойные, тройные и n-мерные интегралы)
- Специализированные методы для функций с особенностями

## Установка

Скопируйте репозиторий и добавьте его в свой путь поиска Julia:

```julia
import Pkg
Pkg.develop(path="путь/к/NumericalIntegration")
```

## Использование

Основная функция библиотеки — `integrate`, которая принимает функцию, границы интегрирования и метод:

```julia
using NumericalIntegration

# Базовый метод трапеций
result = integrate(sin, 0, π, :trapezoid)

# Составной метод Симпсона с 100 подынтервалами
result = integrate(x -> x^2, 0, 1, :composite_simpson, n=100)

# Адаптивное интегрирование с точностью 10^-10
result = integrate(exp, 0, 1, :adaptive, tol=1e-10)

# Метод Монте-Карло для интеграла в многомерном пространстве
f(x) = sum(sin.(x))  # функция от вектора x
result, error = monte_carlo_integration(f, zeros(3), ones(3), 10000, dims=3)
```

## Доступные методы

### Базовые методы
- `:rectangle`, `:midpoint`, `:trapezoid`
- `:simpson`, `:simpson38`, `:boole`

### Составные методы
- `:composite_rectangle`, `:composite_midpoint`
- `:composite_trapezoid`, `:composite_simpson`
- `:romberg`

### Адаптивные методы
- `:adaptive`, `:adaptive_simpson`, `:adaptive_quadrature`

### Методы Монте-Карло
- `:monte_carlo`
- `:monte_carlo_importance`
- `:monte_carlo_stratified`
- `:monte_carlo_quasi_random`

### Многомерное интегрирование
- `double_integral`, `triple_integral`
- `multidimensional_integral`
- `iterated_integral`

### Методы для функций с особенностями
- `:tanh_sinh`, `:double_exponential`
- `:singularity_subtraction`, `:singularity_transformation`
- `:log_weighted`, `:cauchy_principal_value`

## Примеры

В каталоге `examples` содержатся примеры использования различных методов интегрирования.

```julia
# Запуск примеров
include("examples/integration_examples.jl")
```

## Лицензия

Эта библиотека распространяется под лицензией MIT. 