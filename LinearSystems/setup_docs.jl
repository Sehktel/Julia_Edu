#!/usr/bin/env julia

println("Настройка и генерация документации для проекта LinearSystems")
println("="^60)

# Активация проекта
using Pkg
println("Активация проекта LinearSystems...")
Pkg.activate(".")

# Установка необходимых пакетов
println("Установка необходимых пакетов...")
Pkg.add("Documenter")
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")

# Обновление пакетов
println("Обновление зависимостей...")
Pkg.update()

# Создание структуры директорий для документации
println("Проверка структуры директорий...")
isdir("docs") || mkdir("docs")
isdir("docs/src") || mkdir("docs/src")
isdir("docs/build") || mkdir("docs/build")
isdir("docs/src/assets") || mkdir("docs/src/assets")

# Генерация документации
println("Генерация документации...")
include("docs/make.jl")

# Вывод информации о созданной документации
println("\n", "="^60)
println("Документация успешно создана!")
println("="^60)
println("Для просмотра документации откройте в браузере файл:")
println("file://$(joinpath(abspath(@__DIR__), "docs", "build", "index.html"))")
println("="^60) 