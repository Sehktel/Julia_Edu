# Скрипт для генерации документации

# Установка пакета Documenter, если он отсутствует
using Pkg
Pkg.activate(@__DIR__)

# Проверяем и устанавливаем зависимости
println("Проверка зависимостей...")
Pkg.instantiate()

# Добавляем Documenter, если его нет
deps = Pkg.dependencies()
has_documenter = any(x -> x.name == "Documenter", values(deps))
if !has_documenter
    println("Устанавливаем Documenter...")
    Pkg.add("Documenter")
end

# Запуск make.jl для сборки документации
println("Генерация документации...")
include(joinpath(@__DIR__, "docs", "make.jl"))

# Вывод информации о расположении созданной документации
println("="^50)
println("Документация создана!")
println("="^50)
println("Откройте файл в браузере: file://$(joinpath(abspath(@__DIR__), "docs", "build", "index.html"))")
println("="^50) 