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

# ╔═╡ 506e96d0-edfa-11ea-2165-17e39b78e479
begin
	using Random
	using BSON
	p = BSON.load("mnist_conv.bson")[:params]
	# rand!.(p)
	Flux.loadparams!(model, p)
	md"### Load parameters"
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


# ╔═╡ 39bc90b0-ed67-11ea-12f6-5783fe7d3e7a
bar(0:9, softmax(model(Float32.(transpose(reshape(reinterpret(UInt32, img["data"]), img["width"], img["height"])) .!= 0xffececff)[round.(Int, range(1, end, length=28)), round.(Int, range(1, end, length=28)), :, :])), ylim=[0,1], xticks=0:9, label="")


# ╔═╡ Cell order:
# ╟─edd9dcc6-ed60-11ea-24b8-ddabed89f7ce
# ╟─39bc90b0-ed67-11ea-12f6-5783fe7d3e7a
# ╟─7cd4a358-ed6a-11ea-1ac6-6f9204a9697a
# ╟─506e96d0-edfa-11ea-2165-17e39b78e479
