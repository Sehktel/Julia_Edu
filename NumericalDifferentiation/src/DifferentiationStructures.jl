"""
    DifferentiationStructures

Модуль, содержащий основные структуры данных и перечисления для численного дифференцирования.
"""
module DifferentiationStructures

export DifferentiationMethod, forward, backward, central, extrapolation
export DifferentiationResult

"""
    DifferentiationMethod

Перечисление, представляющее различные методы численного дифференцирования.

# Значения
- `forward`: метод прямой (правой) разности
- `backward`: метод обратной (левой) разности
- `central`: метод центральной разности
- `extrapolation`: метод экстраполяции Ричардсона
"""
@enum DifferentiationMethod begin
    forward = 1     # Метод прямой разности
    backward = 2    # Метод обратной разности
    central = 3     # Метод центральной разности
    extrapolation = 4 # Метод экстраполяции Ричардсона
end

"""
    DifferentiationResult

Структура для хранения результатов численного дифференцирования.

# Поля
- `value::Float64`: значение вычисленной производной
- `error_estimate::Float64`: оценка погрешности
- `step_size::Float64`: использованный шаг дифференцирования
- `method::DifferentiationMethod`: использованный метод дифференцирования

# Пример
```julia
result = DifferentiationResult(2.0, 1e-8, 0.001, central)
println("Производная: \$(result.value), погрешность: \$(result.error_estimate)")
```
"""
struct DifferentiationResult
    value::Float64
    error_estimate::Float64
    step_size::Float64
    method::DifferentiationMethod
end

# Переопределение метода вывода для структуры DifferentiationResult
import Base: show
function show(io::IO, result::DifferentiationResult)
    method_names = Dict(
        forward => "прямая разность", 
        backward => "обратная разность", 
        central => "центральная разность",
        extrapolation => "экстраполяция Ричардсона"
    )
    
    println(io, "Результат численного дифференцирования:")
    println(io, "  Метод: $(method_names[result.method])")
    println(io, "  Значение: $(result.value)")
    println(io, "  Оценка погрешности: $(result.error_estimate)")
    println(io, "  Размер шага: $(result.step_size)")
end

end # module DifferentiationStructures 