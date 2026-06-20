# Hugging Face download + GGUF inspection. Thin wrappers over HuggingFaceApi.jl and
# GGUFFiles.jl so the rest of Tonalli has one place to resolve model files and read GGUF
# metadata.

import HuggingFaceApi
import GGUFFiles

"""
    hf_download(repo_id, filename; revision="main", token=nothing) -> String

Download a single file from a Hugging Face repo and return its local cached path.
"""
function hf_download(repo_id::AbstractString, filename::AbstractString; revision::AbstractString = "main", token = HuggingFaceApi.get_token())
    return HuggingFaceApi.hf_hub_download(String(repo_id), String(filename); revision = String(revision), auth_token = token)
end

"""
    HFModel(repo_id; filename="", revision="main")

A Hugging Face model source. With `filename` set, `resolve` downloads that file.
"""
struct HFModel <: AbstractModelSource
    repo_id::String
    filename::String
    revision::String
end
HFModel(repo_id::AbstractString; filename::AbstractString = "", revision::AbstractString = "main") =
    HFModel(String(repo_id), String(filename), String(revision))

function resolve(m::HFModel)
    isempty(m.filename) && error("HFModel(\"$(m.repo_id)\") needs a `filename` to resolve a single file")
    return hf_download(m.repo_id, m.filename; revision = m.revision)
end

"""
    gguf_metadata(path) -> Dict{String,Any}

Read the key/value metadata block from a GGUF file (e.g. `general.architecture`,
`*.context_length`).
"""
function gguf_metadata(path::AbstractString)
    _, metadata_kv, _ = GGUFFiles.parse_gguf(String(path))
    return Dict{String, Any}(kv.key => kv.value for kv in metadata_kv)
end

function metadata(m::HFModel)
    p = resolve(m)
    endswith(p, ".gguf") && return gguf_metadata(p)
    return nothing
end
