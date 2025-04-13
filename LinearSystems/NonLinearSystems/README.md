# NonLinearSystems

Модуль для численного решения нелинейных систем уравнений в Julia.

## Возможности

- Метод Ньютона
- Метод простой итерации
- Метод Зейделя
- Модифицированные методы
- Контроль сходимости
- Оценка скорости сходимости

## Установка

```julia
import Pkg
Pkg.develop(path="путь/к/LinearSystems/NonLinearSystems")
```

## Использование

```julia
using NonLinearSystems

# Определение системы нелинейных уравнений
function f!(F, x)
    F[1] = x[1]^2 + x[2]^2 - 4
    F[2] = exp(x[1]) + x[2] - 1
end

# Начальное приближение
x0 = [1.0, 1.0]

# Решение методом Ньютона
solution, iterations = newton_method(f!, x0, tol=1e-6, max_iter=50)

# Решение методом простой итерации
function g!(x_new, x)
    x_new[1] = sqrt(4 - x[2]^2)
    x_new[2] = 1 - exp(x[1])
end

solution, iterations = simple_iteration(g!, x0, tol=1e-6, max_iter=100)
```

## Структура модуля

- `NonLinearSystems.jl` - основной модуль
- `NewtonMethod.jl` - метод Ньютона
- `SimpleIteration.jl` - метод простой итерации
- `SeidelMethod.jl` - метод Зейделя
- `Convergence.jl` - анализ и контроль сходимости

## Примеры

В директории `examples` содержатся примеры использования модуля.

## Документация

Подробная документация и математическое описание методов доступны в файлах:
- `doc_newton_method.md`
- `doc_simple_iteration.md`
- `doc_seidel_method.md`

## Лицензия

Этот проект лицензирован под MIT License. 