using Dash, DashHtmlComponents, DashCoreComponents, PlotlyBase
using Base64, BSON

include("model.jl")
model = build_model(Args())
Flux.loadparams!(model, BSON.load("mnist_conv.bson")[:params])

app = dash()

plotlayout = Layout(xaxis = Dict(:nticks=>10, :tick0=>0, :dtick=>1),
                    yaxis = Dict(:range=>[0, 1]))

app.layout = html_div() do
    dcc_interval(id="loader", interval=1),
    dcc_interval(id="updater", interval=50),
    html_h1("Hello Dash"),
    html_p("Dash.jl: Julia interface for Dash"),
    dcc_store(id="store", storage_type="memory"),
    html_canvas(id="canvas", width="200", height="200"),
    html_button(id="button", "Clear"),
    dcc_graph(id = "theplot", figure=Plot(0:9, ones(10)./10, plotlayout; kind=:bar))
end

callback!("""
    function(n) {
        const canvas = document.querySelector("canvas")
        const button = document.querySelector("button")
        if (!canvas || !button) {
            return false;
        }
        const ctx = canvas.getContext("2d")
        function clear(){
            ctx.fillStyle = '#ffecec'
            ctx.fillRect(0, 0, 200, 200)
        }
        clear()
        function onmove(e){
            const new_pos = [e.layerX, e.layerY]
            ctx.lineTo(...new_pos)
            ctx.stroke()
            prev_pos = new_pos
        }
        canvas.onmousedown = e => {
            prev_pos = [e.layerX, e.layerY]
            ctx.strokeStyle = '#003d6d'
            ctx.lineWidth = 12
            ctx.beginPath()
            ctx.moveTo(...prev_pos)
            canvas.onmousemove = onmove
        }
        canvas.onmouseup = e => {
            canvas.onmousemove = null
        }
        button.onclick = clear
        return true;
    }""", app, Output("loader", "disabled"), Input("loader", "n_intervals"))

callback!("""
    function(n){
        const canvas = document.querySelector("canvas")
        if (!canvas) {
            return {img: []};
        }
        const ctx = canvas.getContext("2d");
        var binary = '';
        var bytes = new Uint8Array( ctx.getImageData(0,0,200,200).data );
        var len = bytes.byteLength;
        for (var i = 0; i < len; i++) {
            binary += String.fromCharCode( bytes[ i ] );
        }
        return {img: btoa( binary )};
    }""", app, Output("store", "data"), Input("updater", "n_intervals"))

callback!(app, Output("theplot", "figure"), Input("store", "data")) do x
    (x === nothing || x.img isa Vector{Any}) && return Plot()
    probs = softmax(model(Float32.(transpose(reshape(reinterpret(UInt32, base64decode(x.img)), 200, 200)) .!= 0xffececff)[round.(Int, range(1, end, length=28)), round.(Int, range(1, end, length=28)), :, :]))
    return Plot(0:9, probs, plotlayout; kind=:bar)
end

run_server(app, "0.0.0.0", 8080)
