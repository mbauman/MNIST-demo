using Dash, DashCoreComponents, DashHtmlComponents

# Load the model definition and the trained model
using Base64, BSON
include("model.jl")
model = build_model(Args())
Flux.loadparams!(model, BSON.load("mnist_conv.bson")[:params])

using PlotlyBase
plotlayout = Layout(xaxis = Dict(:nticks=>10, :tick0=>0, :dtick=>1),
                    yaxis = Dict(:range=>[0, 1]))
placeholder = Plot(0:9, ones(10)./10, plotlayout; kind=:bar)


app = dash()

app.layout = html_div() do
    dcc_interval(id="loader", interval=1),
    dcc_interval(id="updater", interval=100),
    dcc_store(id="store", storage_type="memory"),
    html_h1("Hello Dash + Julia Webinar"),
    html_canvas(id="canvas", width=200, height=200, style=(margin="auto", display="block")),
    html_button(id="button", "Clear", style=(margin="auto", display="block")),
    dcc_graph(id="theplot", figure=placeholder)
end

# Support for the canvas element:
include("canvas.jl")
callback!(canvas_setup, app,
    Output("loader", "disabled"),
    Input("loader", "n_intervals"))
callback!(canvas_export, app,
    Output("store", "data"),
    Input("updater", "n_intervals"))

callback!(app, Output("theplot", "figure"), Input("store", "data")) do x
    (x === nothing || x.img isa Vector{Any}) && return placeholder
    probs = softmax(model(Float32.(transpose(reshape(reinterpret(UInt32, base64decode(x.img)), 200, 200)) .!= 0xffececff)[round.(Int, range(1, end, length=28)), round.(Int, range(1, end, length=28)), :, :]))
    return Plot(0:9, probs, plotlayout; kind=:bar)
end


run_server(app, "0.0.0.0", 8080)