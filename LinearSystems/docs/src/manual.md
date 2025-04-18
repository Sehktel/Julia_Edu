# Руководство пользователя

## Установка

Чтобы установить пакет `LinearSystems.jl`, выполните следующие команды в Julia REPL:

```julia
using Pkg
Pkg.add(url="путь_к_репозиторию")
```

Для использования пакета:

```julia
using LinearSystems
```

## Запуск примеров

Вы можете запустить примеры использования каждого метода, выполнив скрипт `examples.jl`:

```julia
include("examples.jl")
```

Для интерактивного исследования методов используйте скрипт `run_interactive.jl`:

```julia
include("run_interactive.jl")
```

## Структура проекта

- `src/` - исходный код пакета
  - `LinearSystems.jl` - основной модуль
  - `DirectMethods.jl` - прямые методы
  - `GaussianElimination.jl` - методы Гаусса
  - `ThomasAlgorithm.jl` - метод прогонки
  - `GaussSeidel.jl` - метод Гаусса-Зейделя
- `docs/` - документация
- `examples.jl` - примеры использования
- `run_interactive.jl` - интерактивное исследование

## Образовательные материалы

В директории проекта также доступны подробные документы, описывающие каждый метод решения:

- `doc_direct_methods.md` - прямые методы
- `doc_gaussian_elimination.md` - метод Гаусса
- `doc_thomas_algorithm.md` - метод прогонки
- `doc_gauss_seidel.md` - метод Гаусса-Зейделя

Эти документы содержат теоретическое обоснование методов, алгоритмы, особенности реализации на Julia и оценки вычислительной сложности.

## Практические задания

Для закрепления материала рекомендуется выполнить следующие задания:

1. Модифицируйте метод Гаусса для работы с разреженными матрицами
2. Реализуйте метод Гаусса-Зейделя с оптимальным параметром релаксации
3. Сравните производительность различных методов на больших системах
4. Исследуйте устойчивость методов для плохо обусловленных систем

## Дополнительные ресурсы

### Рекомендуемая литература

- Голуб Дж., Ван Лоун Ч. Матричные вычисления
- Самарский А.А., Николаев Е.С. Методы решения сеточных уравнений
- Деммель Дж. Вычислительная линейная алгебра

### Онлайн-ресурсы

- [Julia Documentation](https://docs.julialang.org)
- [Linear Algebra in Julia](https://julialang.org/learning/linear-algebra/) 