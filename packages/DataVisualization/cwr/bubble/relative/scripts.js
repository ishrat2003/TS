function displayContext(container, data, width, height) {  
    var fontColor = '#000000';
    let dataCopy = Object.assign({}, data);

    var bySize = dataCopy['children'].slice(0);
    bySize.sort(function(a,b) {
        return b.size - a.size;
    });
    dataCopy['children'] = bySize;

    bySize.forEach(child => {
        child['children'] = child['children'].slice(0,15);
    });

    var layout = d3.pack().size([width, height]);
    var root = d3.hierarchy(dataCopy).sum(function (d) { return d.size ; });
    var nodes = root.descendants();
    layout(root);

    var svg = d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("class", "bubble")
        .append('g')
        .style("cursor", "pointer");
    
    var node = svg.selectAll(".node")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
    
    node.append("title")
        .text(function(d) {
            return d.data.name;
        });
    
    node.append("circle")
        .attr("r", (d) => !d.data.children ? d.r : (d.r + 2))
        .attr("fill", (d) => getCircleColor(d))
        .attr("pointer-events", d => !d.children ? "none" : null)
        .on("mouseover", function() { d3.select(this).attr("stroke", "#000"); })
        .on("mouseout", function() { d3.select(this).attr("stroke", null); });

    var label = node.append("text")
        .attr("dy", ".2em")
        .style("text-anchor", "middle")
        .text(function(d) {
            return d.data.name;
        })
        .attr("font-family", "sans-serif")
        .attr("font-size", function(d){
            return d.data.font_size;
        })
        .attr("fill", fontColor)
        .style("fill-opacity", d => d.children ? 0 : 1)
        .style("display", d => d.children  ? "none" : "inline");
    
    node.append("text")
        .attr("dy", "1.3em")
        .style("text-anchor", "middle")
        .text(function(d) {
            return d.data.value;
        })
        .attr("font-family",  "Gill Sans", "Gill Sans MT")
        .attr("font-size", function(d){
            return 10;
        })
        .attr("fill", fontColor);
}

// Create Event Handlers for mouse
function handleMouseOver() {
    // console.log('here');
    // d3.select(this).attr("fill", "red");
  }

function handleMouseOut() {
    // console.log('here out');
    // d3.select(this).attr("fill", fontColor);
  }

