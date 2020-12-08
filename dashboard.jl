using Dash, DashHtmlComponents, DashCoreComponents, PlotlyBase
using Base64, BSON
using Images: imresize

# Load the model definition and the trained model
include("model.jl")
model = build_model(Args())
Flux.loadparams!(model, BSON.load("mnist_conv.bson")[:params])

macro js_str(str) str end # For nice syntax highlighting in VS Code
js = js"""
var custom_components = {
  CanvasInput: function(props) {
    const {id, width, height, zoom} = props
    const canvasRef = React.useRef(null);
    const [coordinates, setCoordinates] = React.useState([]);
    const [isDrawing, setDrawing] = React.useState(false);

    React.useEffect(()=>{
      const canvasObj = canvasRef.current;
      const ctx = canvasObj.getContext('2d');
      var shoudUpdate = false;
      if (coordinates.length == 0) {
        ctx.fillStyle = '#ffecec';
        ctx.fillRect(0, 0, width, height);
        shouldUpdate = true;
      } else if (coordinates.length > 2 && isDrawing) {
        // draw all coordinates held in state
        ctx.strokeStyle = '#003d6d';
        ctx.lineWidth = (width/28)*1.25;
        ctx.beginPath();
        ctx.moveTo(coordinates[0].x, coordinates[0].y);
        ctx.lineTo(coordinates[1].x, coordinates[1].y);
        ctx.lineTo(coordinates[2].x, coordinates[2].y);
        ctx.stroke();
        shouldUpdate = true;
      }
      if (shouldUpdate && props.setProps) {
        // And update the state of the component... sending the array itself is SLLLOOOOWWWW
        var binary = '';
        var bytes = ctx.getImageData(0,0,width,height).data;
        var len = bytes.byteLength;
        for (var i = 0; i < len; i++) {
          binary += String.fromCharCode( bytes[ i ] );
        }
        props.setProps({buffer: btoa( binary )});
      }
    }, [canvasRef, coordinates, isDrawing]);

    const handleCanvasMove=(event)=>{
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const currentCoord = { x: (event.clientX - rect.left) * scaleX, y: (event.clientY - rect.top) * scaleY };

      // add the newest mouse location to an array in state 
      if (coordinates.length < 3) {
        setCoordinates([...coordinates, currentCoord]);
      } else {
        setCoordinates([...coordinates.slice(1), currentCoord]);
      }
    };

    const handleClearCanvas=(event)=>{
      setCoordinates([]);
    };

    return React.createElement('div', {id: id}, [
      React.createElement('canvas', {key: 'canvas', ref: canvasRef, width: width, height: height,
        style: {margin: "auto", display: "block", width:width*zoom, height:height*zoom},
        onMouseMove: handleCanvasMove, onMouseDown: e=>{setDrawing(true)}, onMouseUp: e=>{setDrawing(false)}, onMouseOut: e=>{setDrawing(false)}}),
      React.createElement('button', {key: 'button', onClick: handleClearCanvas, style: {margin: "auto", display: "block", fontSize: "3ex"}}, 'Clear')]);
  }
};
custom_components.CanvasInput.propTypes = {
    id: PropTypes.string,
    buffer: PropTypes.string,
    width: PropTypes.number,
    height: PropTypes.number,
    zoom: PropTypes.number,
    setProps: PropTypes.func
};
custom_components.CanvasInput.defaultProps = {
    buffer: '',
    width: 112,
    height: 112,
    zoom: 3
};
"""
custom_canvas_input(;id, kwargs...) = Dash.Component("CanvasInput", "CanvasInput", "custom_components", Symbol[:id, :buffer, :width, :height, :zoom], Symbol.(["data-","aria-"]), id=id; kwargs...)

app = dash()
app.title = "Digit tester"
push!(app.inline_scripts, js)

plotlayout = Layout(xaxis = Dict(:nticks=>10, :tick0=>0, :dtick=>1),
                    yaxis = Dict(:range=>[0, 1]), font_size=18)
placeholder = Plot(0:9, ones(10)./10, plotlayout; kind=:bar)

app.layout = html_div() do
    custom_canvas_input(id="canvas", width=112, height=112, zoom=2),
    dcc_graph(id="theplot", figure=placeholder)
end

callback!(app, Output("theplot", "figure"), Input.("canvas", ["buffer","width","height"])) do x, w, h
    (x === nothing || isempty(x)) && return placeholder
    img = Float32.(imresize(transpose(reshape(reinterpret(UInt32, base64decode(x)), w, h)) .!= 0xffececff, 28, 28))
    probs = softmax(model(reshape(img, size(img)..., 1, 1)))
    return Plot(0:9, probs, plotlayout; kind=:bar)
end

run_server(app, "0.0.0.0", something(tryparse(Int, get(ENV, "PORT", "")), 8080))
