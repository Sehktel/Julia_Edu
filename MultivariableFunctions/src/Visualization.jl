"""
# Визуализация функций двух переменных

Данный модуль реализует функции для визуализации функций двух переменных,
включая построение графиков поверхностей, линий уровня и полей градиентов.
"""

using Plots

"""
    plot_function(f, x_range, y_range; title="График функции", kwargs...)

Строит трехмерный график поверхности функции двух переменных `f(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::AbstractRange`: диапазон значений по оси x
- `y_range::AbstractRange`: диапазон значений по оси y
- `title::String="График функции"`: заголовок графика
- `kwargs...`: дополнительные аргументы для функции plot

# Возвращаемое значение
- `Plot`: объект графика

# Примеры
```julia
f(x, y) = sin(x) + cos(y)
x_range = -2π:0.1:2π
y_range = -2π:0.1:2π
plot_function(f, x_range, y_range, title="f(x, y) = sin(x) + cos(y)")
```
"""
function plot_function(f::Function, x_range::AbstractRange, y_range::AbstractRange; 
                      title::String="График функции", kwargs...)
    # Вычисляем значения функции на сетке
    z = [f(x, y) for y in y_range, x in x_range]
    
    # Строим 3D график
    plt = surface(x_range, y_range, z, 
                  xlabel="x", ylabel="y", zlabel="f(x, y)",
                  title=title, 
                  camera=(30, 30),  # угол обзора
                  c=:viridis,       # цветовая схема
                  kwargs...)
    
    return plt
end

"""
    plot_contour(f, x_range, y_range; levels=20, title="Линии уровня", kwargs...)

Строит график линий уровня (контурный график) функции двух переменных `f(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::AbstractRange`: диапазон значений по оси x
- `y_range::AbstractRange`: диапазон значений по оси y
- `levels::Union{Int, Vector{<:Real}}=20`: количество линий уровня или их значения
- `title::String="Линии уровня"`: заголовок графика
- `kwargs...`: дополнительные аргументы для функции contour

# Возвращаемое значение
- `Plot`: объект графика

# Примеры
```julia
f(x, y) = x^2 + y^2
x_range = -2:0.1:2
y_range = -2:0.1:2
plot_contour(f, x_range, y_range, levels=10, title="Линии уровня x² + y²")
```
"""
function plot_contour(f::Function, x_range::AbstractRange, y_range::AbstractRange; 
                     levels::Union{Int, Vector{<:Real}}=20, 
                     title::String="Линии уровня", kwargs...)
    # Вычисляем значения функции на сетке
    z = [f(x, y) for y in y_range, x in x_range]
    
    # Строим контурный график
    plt = contour(x_range, y_range, z, 
                  levels=levels,
                  xlabel="x", ylabel="y",
                  title=title,
                  c=:viridis,       # цветовая схема
                  fill=true,        # заполнение цветом
                  colorbar=true,    # цветовая шкала
                  kwargs...)
    
    return plt
end

"""
    plot_gradient_field(f, x_range, y_range; scale=0.1, density=20, title="Поле градиента", kwargs...)

Строит поле градиентов функции двух переменных `f(x, y)`.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::AbstractRange`: диапазон значений по оси x
- `y_range::AbstractRange`: диапазон значений по оси y
- `scale::Real=0.1`: масштаб векторов градиента
- `density::Int=20`: плотность сетки векторов (количество по каждой оси)
- `title::String="Поле градиента"`: заголовок графика
- `kwargs...`: дополнительные аргументы для функции quiver

# Возвращаемое значение
- `Plot`: объект графика

# Примеры
```julia
f(x, y) = x^2 + y^2
x_range = -2:0.1:2
y_range = -2:0.1:2
plot_gradient_field(f, x_range, y_range, scale=0.05, density=15)
```
"""
function plot_gradient_field(f::Function, x_range::AbstractRange, y_range::AbstractRange; 
                            scale::Real=0.1, density::Int=20,
                            title::String="Поле градиента", kwargs...)
    # Создаем сетку точек для векторов
    x_points = range(minimum(x_range), maximum(x_range), length=density)
    y_points = range(minimum(y_range), maximum(y_range), length=density)
    
    # Вычисляем градиенты в каждой точке сетки
    grad_x = zeros(density, density)
    grad_y = zeros(density, density)
    
    for i in 1:density
        for j in 1:density
            x = x_points[j]
            y = y_points[i]
            grad = gradient(f, x, y)
            grad_x[i, j] = grad[1]
            grad_y[i, j] = grad[2]
        end
    end
    
    # Создаем основной график - контур функции
    plt = plot_contour(f, x_range, y_range, title=title)
    
    # Добавляем поле градиентов
    quiver!(plt, repeat(x_points, outer=(1, density)), 
                repeat(y_points', outer=(density, 1)), 
                quiver=(scale .* grad_x, scale .* grad_y),
                color=:black, alpha=0.7, kwargs...)
    
    return plt
end

"""
    plot_combined(f, x_range, y_range; title="Анализ функции", kwargs...)

Строит комбинированный график с изображением поверхности, линий уровня и поля градиентов.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::AbstractRange`: диапазон значений по оси x
- `y_range::AbstractRange`: диапазон значений по оси y
- `title::String="Анализ функции"`: общий заголовок
- `kwargs...`: дополнительные аргументы для функций построения графиков

# Возвращаемое значение
- `Plot`: объект графика

# Примеры
```julia
f(x, y) = sin(x) * cos(y)
x_range = -π:0.1:π
y_range = -π:0.1:π
plot_combined(f, x_range, y_range, title="Анализ функции sin(x)cos(y)")
```
"""
function plot_combined(f::Function, x_range::AbstractRange, y_range::AbstractRange; 
                      title::String="Анализ функции", kwargs...)
    # Создаем графики
    p1 = plot_function(f, x_range, y_range, title="Поверхность", kwargs...)
    p2 = plot_contour(f, x_range, y_range, title="Линии уровня", kwargs...)
    p3 = plot_gradient_field(f, x_range, y_range, title="Поле градиента", kwargs...)
    
    # Комбинируем графики
    plt = plot(p1, p2, p3, layout=(1, 3), 
               size=(1200, 400), 
               title=title)
    
    return plt
end

"""
    plot_critical_points(f, x_range, y_range, critical_points; title="Критические точки", kwargs...)

Строит график линий уровня функции двух переменных с отмеченными критическими точками.

# Аргументы
- `f::Function`: функция двух переменных
- `x_range::AbstractRange`: диапазон значений по оси x
- `y_range::AbstractRange`: диапазон значений по оси y
- `critical_points::Vector{Tuple{Vector{<:Real}, Symbol}}`: вектор критических точек и их типов
- `title::String="Критические точки"`: заголовок графика
- `kwargs...`: дополнительные аргументы для функции contour

# Возвращаемое значение
- `Plot`: объект графика

# Примеры
```julia
f(x, y) = x^2 + y^2
critical_points = [([0.0, 0.0], :minimum)]
plot_critical_points(f, -2:0.1:2, -2:0.1:2, critical_points)
```
"""
function plot_critical_points(f::Function, x_range::AbstractRange, y_range::AbstractRange, 
                             critical_points::Vector{Tuple{Vector{<:Real}, Symbol}}; 
                             title::String="Критические точки", kwargs...)
    # Создаем контурный график
    plt = plot_contour(f, x_range, y_range, title=title, kwargs...)
    
    # Определяем маркеры для разных типов критических точек
    markers = Dict(
        :minimum => :circle,
        :maximum => :star5,
        :saddle => :diamond,
        :undefined => :cross
    )
    
    # Добавляем критические точки на график
    for (point, type) in critical_points
        scatter!(plt, [point[1]], [point[2]], 
                marker=markers[type], 
                markersize=8, 
                color=:red, 
                label=string(type))
    end
    
    return plt
end 