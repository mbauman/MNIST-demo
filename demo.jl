### A Pluto.jl notebook ###
# v0.11.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 7cd4a358-ed6a-11ea-1ac6-6f9204a9697a
begin
    using Pkg
    using Flux, Plots, PlutoUI
    function ingredients(path::String)
        # https://github.com/fonsp/Pluto.jl/issues/115#issuecomment-661722426
        name = Symbol(basename(path))
        m = Module(name)
        Core.eval(m,
            Expr(:toplevel,
                 :(eval(x) = $(Expr(:core, :eval))($name, x)),
                 :(include(x) = $(Expr(:top, :include))($name, x)),
                 :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
                 :(include($path))))
        m
    end
    M = ingredients("model.jl")
    model = M.build_model(M.Args())
    md"### Define the model"
end

# ╔═╡ 5157ca6a-3686-11eb-144c-ab641f935d76
begin
    using JuliaHubClient
    ENV["JULIA_PKG_SERVER"] = "juliahub.com"
    auth, _ = JuliaHubClient.authenticate()
end

# ╔═╡ edd9dcc6-ed60-11ea-24b8-ddabed89f7ce
@bind img html"""
<canvas width="200" height="200" style="margin: auto; display: block;"></canvas>
<button style="font-size:3ex; margin: auto; display: block">Clear</button>
<script>
const canvas = this.querySelector("canvas")
const button = this.querySelector("button")
const ctx = canvas.getContext("2d")
function send_image(){
    // 🐸 We send the value back to Julia 🐸 //
    canvas.value = {
        width: 200,
        height: 200,
        data: ctx.getImageData(0,0,200,200).data,
    }
    canvas.dispatchEvent(new CustomEvent("input"))
}
var prev_pos = [0, 0]
function clear(){
    ctx.fillStyle = '#ffecec'
    ctx.fillRect(0, 0, 200, 200)
    send_image()
}
clear()
function onmove(e){
    const new_pos = [e.layerX, e.layerY]
    ctx.lineTo(...new_pos)
    ctx.stroke()
    prev_pos = new_pos
    send_image()
}
canvas.onmousedown = e => {
    prev_pos = [e.layerX, e.layerY]
    ctx.strokeStyle = '#003d6d'
    ctx.lineWidth = 12
    ctx.beginPath()
    ctx.moveTo(...prev_pos)
    canvas.onmousemove = onmove
}
button.onclick = clear
canvas.onmouseup = e => {
    canvas.onmousemove = null
}
</script>
"""

# ╔═╡ 0d398458-0d6a-11eb-0ab4-13f1165b3830
md"#### Connect to JuliaHub"

# ╔═╡ 9d2461a6-3681-11eb-115a-194d2faad27a
function html_results_table(name; auth)
    jobs = JuliaHubClient.get_jobs(auth = auth)
    hash = String(rand(['0':'9';'a':'z';'A':'Z'], 10))
    function make_table_row(job)
        time = job.timestamp
        url = JuliaHubClient.get_result_url(job, auth=auth)
        radio = job.status == "Completed" && !isnothing(url) ?
            """<input class="_radio" data-url="$(url)"
                type="radio" name="run"></input>""" : ""
        return """
            <tr>
                <td>$(radio)</td>
                <td>$(time)</td>
                <td>$(job.status)</td>
                <td>$(name)</td>
            </tr>"""
    end
    return HTML("""
    <table id="results-table-$hash">
        <thead>
            <td></td>
            <td>Time</td>
            <td>Status</td>
            <td>Name</td>
        </thead>
        <tbody>
            $(
            join((make_table_row(job) for job in jobs if get(job.inputs, "jobname", job.jobname) == name), "\n")
            )
        </tbody>
    </table>

    <script>
        const table = document.getElementById('results-table-$hash')
        const radios = table.getElementsByClassName('_radio')

        for (const r of radios) {
            r.oninput = ev => {
                table.value = r.dataset.url
                table.dispatchEvent(new CustomEvent("input"))
            }
        }
        table.value = null
        table.dispatchEvent(new CustomEvent("input"))
    </script>
    """)
end

# ╔═╡ 5c4e1366-0d69-11eb-026b-d7a1e71e89d0
@bind url html_results_table("MNIST-demo", auth=auth)

# ╔═╡ b44e03ac-3684-11eb-3652-cfbbd23efe8c
begin
    using Random
    using BSON
    if !isnothing(url)
        p = BSON.load(JuliaHubClient.get_result_file(url, tempname(), auth=auth))[:params]
        Flux.loadparams!(model, p)
    end
    md"### Load parameters"
end

# ╔═╡ 39bc90b0-ed67-11ea-12f6-5783fe7d3e7a
begin
    url
    bar(0:9, softmax(model(Float32.(transpose(reshape(reinterpret(UInt32, img["data"]), img["width"], img["height"])) .!= 0xffececff)[round.(Int, range(1, end, length=28)), round.(Int, range(1, end, length=28)), :, :])), ylim=[0,1], xticks=0:9, label="")
end

# ╔═╡ Cell order:
# ╟─5c4e1366-0d69-11eb-026b-d7a1e71e89d0
# ╟─b44e03ac-3684-11eb-3652-cfbbd23efe8c
# ╟─edd9dcc6-ed60-11ea-24b8-ddabed89f7ce
# ╟─39bc90b0-ed67-11ea-12f6-5783fe7d3e7a
# ╟─7cd4a358-ed6a-11ea-1ac6-6f9204a9697a
# ╟─0d398458-0d6a-11eb-0ab4-13f1165b3830
# ╟─5157ca6a-3686-11eb-144c-ab641f935d76
# ╟─9d2461a6-3681-11eb-115a-194d2faad27a
