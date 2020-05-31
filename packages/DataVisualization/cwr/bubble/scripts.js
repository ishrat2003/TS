var topicIndex = {
    'Topic 1': 0,
    'Topic 2': 2,
    'Topic 3': 3,
    'Topic 4': 4,
    'Topic 5': 5,
    'Topic 6': 6
}
var diameter = 1000;
d3.csv("http://localhost:8080/cwr/data/covid19/Topic1_cwr_gc.csv", function(error, root) {  
  var dataset = {
      children: root
  }
  
  var color = d3.scaleOrdinal(["#FFFFFF", "#C733FF","#339CFF","#9FFF33","#436C0B","#FFC133", "#FF4C33"]);

  var bubble = d3.pack(dataset)
      .size([diameter, diameter])
      .padding(1.5);

  var svg = d3.select("body")
      .append("svg")
      .attr("width", diameter)
      .attr("height", diameter)
      .attr("class", "bubble");

  var nodes = d3.hierarchy(dataset)
      .sum(function(d) { return d.number_of_blocks; });

  var node = svg.selectAll(".node")
      .data(bubble(nodes).descendants())
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", function(d) {
          return "translate(" + d.x + "," + d.y + ")";
      });

  node.append("title")
      .text(function(d) {
          return d.data.label + ": " + d.data.number_of_blocks;
      });

  node.append("circle")
      .attr("r", function(d) {
          return d.r;
      })
      .style("fill", function(d,i) {
          return color(topicIndex[d.data.topic]);
      });

  node.append("text")
      .attr("dy", ".2em")
      .style("text-anchor", "middle")
      .text(function(d) {
          console.log(d.data);
          return d.data.label;
      })
      .attr("font-family", "sans-serif")
      .attr("font-weight", function(d){
        return 5000;
       })
      .attr("font-size", function(d){
          return d.data.font_size;
      })
      .attr("fill", "white");

  node.append("text")
      .attr("dy", "1.3em")
      .style("text-anchor", "middle")
      .text(function(d) {
          return d.data.number_of_blocks;
      })
      .attr("font-family",  "Gill Sans", "Gill Sans MT")
      .attr("font-weight", function(d){
        return 5000;
       })
      .attr("font-size", function(d){
          return (d.data.number_of_blocks)/5;
      })
      .attr("fill", "white");

  d3.select(self.frameElement)
      .style("height", diameter + "px");
});

d3.select(self.frameElement).style("height", diameter + "px");

  