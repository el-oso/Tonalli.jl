using Tonalli
using Documenter
using DocumenterVitepress

makedocs(;
    sitename = "Tonalli.jl",
    authors = "el-oso",
    modules = [Tonalli],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/Tonalli.jl",
        devbranch = "master",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Inference" => "inference.md",
        "Fine-tuning" => "finetuning.md",
        "Diagnostics" => "diagnostics.md",
        "API reference" => "api.md",
        "Roadmap" => "roadmap.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/el-oso/Tonalli.jl",
    target = "build",
    devbranch = "master",
    branch = "gh-pages",
    push_preview = true,
)
