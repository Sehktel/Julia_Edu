# LinearSystems.jl

Пакет `LinearSystems.jl` предоставляет реализацию методов решения систем линейных алгебраических уравнений (СЛАУ) на языке Julia. Проект разработан в образовательных целях для демонстрации различных алгоритмов решения СЛАУ и их особенностей.

## Содержание

Пакет включает следующие методы решения СЛАУ:

- **Прямые методы** - стандартный метод, LU-разложение, разложение Холецкого
- **Метод Гаусса** - классический метод с выбором главного элемента и без
- **Метод прогонки (алгоритм Томаса)** - специализированный метод для трехдиагональных систем
- **Метод Гаусса-Зейделя** - итерационный метод

## Установка

```julia
using Pkg
Pkg.add(url="путь_к_репозиторию")
```

## Быстрый старт

```julia
using LinearSystems

# Пример использования прямых методов
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
x = solve_direct(A, b)

# Пример использования метода Гаусса с выбором главного элемента
x = gauss_elimination_pivot(A, b)

# Пример решения трехдиагональной системы методом прогонки
n = 5
A_tridiag = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
b_tridiag = ones(n)
x = thomas_algorithm(A_tridiag, b_tridiag)

# Пример использования итерационного метода Гаусса-Зейделя
x = gauss_seidel(A, b, tol=1e-6, max_iter=100)
```

## Документация

Для более подробной информации о методах, пожалуйста, обратитесь к соответствующим разделам документации:

- [Руководство пользователя](manual.md)
- [Прямые методы](direct_methods.md)
- [Метод Гаусса](gaussian_elimination.md)
- [Метод прогонки](thomas_algorithm.md)
- [Метод Гаусса-Зейделя](gauss_seidel.md)
- [API-документация](api.md)

## Лицензия

Этот проект распространяется под лицензией MIT. 