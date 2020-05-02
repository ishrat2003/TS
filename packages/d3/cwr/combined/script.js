var height = 1000;
var width = 1000;

var format = d3.format(",d"),
    color = d3.scaleOrdinal(d3.schemeCategory20c);
var bubble = d3.pack()
    .size([width, height])
    .padding(1.5);

d3.json("http://localhost:8080/flare.json", function(error, data) {  
    if (error) throw error;
        // Declare d3 layout
        let view;
        
        var layout = d3.pack().size([width, height]);

        // Layout + Data
        var root = d3.hierarchy(data).sum(function (d) { return d.value; });
        var nodes = root.descendants();
        var focus = root;
        layout(root);
    
        var svg = d3.select("body")
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
            .attr("r", function(d) {
                return d.r;
            })
        .attr("fill", d => d.children ? color(d.depth) : "white")
        .attr("pointer-events", d => !d.children ? "none" : null)
        .on("mouseover", function() { d3.select(this).attr("stroke", "#000"); })
        .on("mouseout", function() { d3.select(this).attr("stroke", null); });
;
      
        var label = node.append("text")
            .attr("dy", ".2em")
            .style("text-anchor", "middle")
            .text(function(d) {
                return d.data.name;
            })
            .attr("font-family", "sans-serif")
            .attr("font-weight", function(d){
              return 5000;
             })
            .attr("font-size", function(d){
                return d.data.font_size;
            })
            .attr("fill", "white")
            .style("fill-opacity", d => d.parent === root ? 1 : 0)
          .style("display", d => d.parent === root ? "inline" : "none");
      
        node.append("text")
            .attr("dy", "1.3em")
            .style("text-anchor", "middle")
            .text(function(d) {
                return d.data.number_of_blocks;
            })
            .attr("font-family",  "Gill Sans", "Gill Sans MT")
            .attr("font-size", function(d){
                return 10;
            })
            .attr("fill", "white");

    

});


// Create Event Handlers for mouse
function handleMouseOver() {
    console.log('here');
    d3.select(this).attr("fill", "red");
  }

function handleMouseOut() {
    console.log('here out');
    d3.select(this).attr("fill", "blue");
  }


// function visualize(data) {
//     // Declare d3 layout
//     var layout = d3.pack().size([width, height]);

//     // Layout + Data
//     var root = d3.hierarchy(data).sum(function (d) { return d.value; });
//     var nodes = root.descendants();
//     layout(root);

//     var g = d3.select("body")
//     .append("svg")
//     .attr("width", width)
//     .attr("height", height)
//     .attr("class", "bubble")
//     .append('g')
//     .style("cursor", "pointer")
//     .on("click", () => zoom(root));
        
//     var node = g.selectAll(".node")
//         .data(nodes)
//         .enter()
//         .append("g")
//         .attr("class", "node")
//         .attr("transform", function(d) {
//             return "translate(" + d.x + "," + d.y + ")";
//         });
  
//     node.append("title")
//         .text(function(d) {
//             return d.data.name;
//         });
  
//     node.append("circle")
//         .attr("r", function(d) {
//             return d.r;
//         });
  
//     var label = node.append("text")
//         .attr("dy", ".2em")
//         .style("text-anchor", "middle")
//         .text(function(d) {
//             return d.data.name;
//         })
//         .attr("font-family", "sans-serif")
//         .attr("font-weight", function(d){
//           return 5000;
//          })
//         .attr("font-size", function(d){
//             return d.data.font_size;
//         })
//         .attr("fill", "white")
//         .style("fill-opacity", d => d.parent === root ? 1 : 0)
//       .style("display", d => d.parent === root ? "inline" : "none");
  
//     node.append("text")
//         .attr("dy", "1.3em")
//         .style("text-anchor", "middle")
//         .text(function(d) {
//             return d.data.number_of_blocks;
//         })
//         .attr("font-family",  "Gill Sans", "Gill Sans MT")
//         .attr("font-size", function(d){
//             return 10;
//         })
//         .attr("fill", "white");

//     // Draw on screen
//     // slices.attr('cx', function (d) { return d.x; })
//     //     .attr('cy', function (d) { return d.y; })
//     //     .attr('r', function (d) { return d.r; });
// }

  