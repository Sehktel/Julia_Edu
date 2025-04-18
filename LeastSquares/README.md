# LeastSquares

Модуль для решения задач методом наименьших квадратов в Julia.

## Возможности

- Линейная регрессия с одной и несколькими переменными
- Полиномиальная регрессия
- Взвешенный метод наименьших квадратов
- Обобщенный метод наименьших квадратов
- Оценка коэффициентов и их статистическая значимость
- Оценка качества модели (R², скорректированный R², F-статистика)

## Установка

```julia
import Pkg
Pkg.develop(path="путь/к/LeastSquares")
```

## Использование

```julia
using LeastSquares

# Линейная регрессия
X = [1.0 2.5; 2.0 3.4; 3.0 4.5; 4.0 5.1; 5.0 6.0]  # Матрица признаков
y = [2.0, 3.5, 4.8, 6.1, 7.2]                      # Целевая переменная
coefficients = linear_regression(X, y)

# Полиномиальная регрессия
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.1, 4.0, 8.5, 17.0, 26.5]
degree = 2  # Степень полинома
coeffs = polynomial_regression(x_data, y_data, degree)

# Предсказание значений
new_x = [6.0, 7.0]
predictions = predict(coeffs, new_x)
```

## Структура модуля

- `LeastSquares.jl` - основной модуль
- `LinearRegression.jl` - линейная регрессия
- `PolynomialRegression.jl` - полиномиальная регрессия
- `WeightedLeastSquares.jl` - взвешенный метод наименьших квадратов
- `GeneralizedLeastSquares.jl` - обобщенный метод наименьших квадратов
- `Statistics.jl` - статистические метрики для оценки качества моделей

## Примеры

В директории `examples` содержатся примеры использования модуля.

## Лицензия

Этот проект лицензирован под MIT License. 