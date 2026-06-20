using Tonalli
using Documenter
using DocumenterVitepress

makedocs(;
    modules = [Tonalli],
    authors = "el-oso",
    repo = "https://github.com/el-oso/Tonalli.jl",
    sitename = "Tonalli.jl",
    # No `deploy_url`: for a standard github.io *project* site, DocumenterVitepress derives
    # the base path as "/<repo>/<devurl>/" from `repo`. Setting deploy_url (esp. without a
    # scheme) bakes the host into the asset base (/el-oso.github.io/Tonalli.jl/dev/) and the
    # theme assets 404 → unstyled page.
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/el-oso/Tonalli.jl",
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

# DocumenterVitepress.deploydocs (NOT Documenter.deploydocs): it post-processes the
# VitePress version-folder build into the final gh-pages layout (dev/index.html redirect,
# .nojekyll, correct base) — plain Documenter.deploydocs leaves content stranded in dev/1/.
DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/Tonalli.jl",
    devbranch = "master",
    push_preview = true,
)
