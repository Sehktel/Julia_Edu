using Test
using NumericalDifferentiation

@testset "Базовые методы дифференцирования" begin
    # Тестовая функция и точка
    f(x) = sin(x)
    x0 = π/4
    
    # Точное значение производной
    exact_derivative = cos(x0)
    
    # Тестирование метода разности вперёд
    @testset "Метод разности вперёд" begin
        result = forward_difference(f, x0)
        @test isapprox(result, exact_derivative, atol=1e-5)
    end
    
    # Тестирование метода разности назад
    @testset "Метод разности назад" begin
        result = backward_difference(f, x0)
        @test isapprox(result, exact_derivative, atol=1e-5)
    end
    
    # Тестирование метода центральной разности
    @testset "Метод центральной разности" begin
        result = central_difference(f, x0)
        @test isapprox(result, exact_derivative, atol=1e-7)
    end
    
    # Тестирование производной второго порядка
    @testset "Производная второго порядка" begin
        # Точное значение второй производной sin(x) = -sin(x)
        exact_second_derivative = -sin(x0)
        result = second_derivative(f, x0)
        @test isapprox(result, exact_second_derivative, atol=1e-5)
    end
    
    # Тестирование высших производных
    @testset "Высшие производные" begin
        # Третья производная sin(x) = -cos(x)
        exact_third_derivative = -cos(x0)
        result = higher_derivative(f, x0, 3)
        @test isapprox(result, exact_third_derivative, atol=1e-5)
        
        # Четвертая производная sin(x) = sin(x)
        exact_fourth_derivative = sin(x0)
        result = higher_derivative(f, x0, 4)
        @test isapprox(result, exact_fourth_derivative, atol=1e-4)
    end
    
    # Тестирование унифицированного интерфейса
    @testset "Унифицированный интерфейс differentiate" begin
        # Проверка различных методов
        result_forward = differentiate(f, x0, :forward)
        result_backward = differentiate(f, x0, :backward)
        result_central = differentiate(f, x0, :central)
        
        @test isapprox(result_forward, exact_derivative, atol=1e-5)
        @test isapprox(result_backward, exact_derivative, atol=1e-5)
        @test isapprox(result_central, exact_derivative, atol=1e-7)
        
        # Проверка метода Ричардсона
        result_richardson = differentiate(f, x0, :richardson)
        @test isapprox(result_richardson, exact_derivative, atol=1e-9)
    end
end

@testset "Оптимальный шаг и адаптивное дифференцирование" begin
    f(x) = exp(x)
    x0 = 1.0
    
    # Точное значение производной
    exact_derivative = exp(x0)  # = e
    
    # Тестирование оптимального шага
    h_opt = optimal_step_size(f, x0)
    @test h_opt > 0
    
    # Тестирование адаптивного дифференцирования
    result = adaptive_differentiation(f, x0)
    @test isapprox(result, exact_derivative, atol=1e-10)
end

@testset "Экстраполяция Ричардсона" begin
    f(x) = log(x)
    x0 = 2.0
    
    # Точное значение производной
    exact_derivative = 1/x0
    
    # Тестирование базового метода Ричардсона
    result = richardson_extrapolation(f, x0)
    @test isapprox(result, exact_derivative, atol=1e-12)
    
    # Проверка улучшения точности с увеличением уровней
    result1 = richardson_extrapolation(f, x0, levels=1)
    result2 = richardson_extrapolation(f, x0, levels=2)
    result3 = richardson_extrapolation(f, x0, levels=3)
    
    # Погрешность должна уменьшаться
    @test abs(result1 - exact_derivative) > abs(result2 - exact_derivative)
    @test abs(result2 - exact_derivative) > abs(result3 - exact_derivative)
end 