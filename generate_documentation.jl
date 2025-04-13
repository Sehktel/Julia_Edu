#!/usr/bin/env julia

"""
Скрипт для генерации документации для всех модулей в проекте Julia_Edu
"""

using Pkg
using Documenter

# Список модулей в проекте
modules = [
    "NumericalIntegration",
    "NumericalDifferentiation",
    "MultivariableFunctions",
    "Interpolation",
    "Correlation",
    "LeastSquares",
    "LinearSystems"
]

# Словарь для хранения модулей, которые будут документированы
module_dict = Dict()

# Функция для генерации документации одного модуля
function generate_module_documentation(module_name)
    println("Generating documentation for $module_name...")
    
    # Добавляем модуль в текущий путь загрузки
    push!(LOAD_PATH, joinpath(@__DIR__, module_name))
    
    # Пытаемся загрузить модуль
    try
        module_dict[module_name] = Base.require(Main, Symbol(module_name))
        
        # Определяем пути для документации
        docs_src = joinpath(@__DIR__, module_name, "docs", "src")
        docs_build = joinpath(@__DIR__, module_name, "docs", "build")
        
        # Создаем директорию для документации, если её нет
        mkpath(docs_src)
        
        # Базовый файл index.md, если его нет
        index_file = joinpath(docs_src, "index.md")
        if !isfile(index_file)
            readme_file = joinpath(@__DIR__, module_name, "README.md")
            if isfile(readme_file)
                # Копируем содержимое README.md в index.md
                cp(readme_file, index_file, force=true)
            else
                # Создаем базовый index.md
                open(index_file, "w") do io
                    write(io, """
                    # $module_name

                    Документация модуля $module_name.
                    """)
                end
            end
        end
        
        # Генерируем документацию
        DocMeta.setdocmeta!(module_dict[module_name], :DocTestSetup, :(using $module_name); recursive=true)
        
        makedocs(
            modules = [module_dict[module_name]],
            format = Documenter.HTML(
                prettyurls = false,
                analytics = "UA-xxxxxxxxx-x",
                assets = ["assets/favicon.ico"]
            ),
            sitename = "$module_name Documentation",
            authors = "Julia Education Team",
            pages = [
                "Главная" => "index.md"
            ],
            doctest = false,
            repo = "",
            source = docs_src,
            build = docs_build
        )
        
        println("Documentation for $module_name generated successfully!")
    catch e
        println("Error generating documentation for $module_name: $e")
    end
end

# Генерация документации для всех модулей
for module_name in modules
    generate_module_documentation(module_name)
end

println("Documentation generation complete!") 