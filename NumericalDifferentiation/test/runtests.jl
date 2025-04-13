using Test
using NumericalDifferentiation

# Запуск всех тестов
@testset "NumericalDifferentiation.jl" begin
    # Включаем файлы с тестами
    include("test_basic_methods.jl")
    # В будущем здесь можно включить дополнительные тесты
end 