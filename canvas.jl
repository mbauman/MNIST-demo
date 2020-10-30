canvas_setup = """
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
    }"""

canvas_export =  """
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
    }"""