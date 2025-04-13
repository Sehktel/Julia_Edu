using Documenter
using LinearSystems

# Настройка документации
makedocs(
    sitename = "LinearSystems.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/custom.css"],
        lang = "ru"
    ),
    modules = [LinearSystems],
    authors = "Educational Project",
    pages = [
        "Главная" => "index.md",
        "Руководство" => "manual.md",
        "Методы" => [
            "Прямые методы" => "direct_methods.md",
            "Метод Гаусса" => "gaussian_elimination.md",
            "Метод прогонки" => "thomas_algorithm.md",
            "Метод Гаусса-Зейделя" => "gauss_seidel.md"
        ],
        "API-документация" => "api.md"
    ],
    remotes = nothing # Отключаем проверку удаленного репозитория
)

# Развертывание документации (по желанию)
# deploydocs(
#     repo = "github.com/username/LinearSystems.jl.git",
#     devbranch = "main"
# ) 