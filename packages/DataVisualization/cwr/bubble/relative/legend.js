var fontColor = '#000000';
var legends = {
    'negative_sentiment': {'name':'Negative sentiment', 'color': '#FCAC97', 'dy': 0},
    'positive_sentiment': {'name':'Positive sentiment', 'color': '#7FB924', 'dy': 15},
    'same_topic': {'name':'Same topic', 'color': '#AFC1EC', 'dy': 30},
    'core_context': {'name':'Core context', 'color': '#6DC6E7', 'dy': 45},
    'other_topic': {'name':'Other topics', 'color': '#B8A3F1', 'dy': 60}
};

function getCircleColor(d){
    var keys = Object.keys(legends);
    if (keys.includes(d.data.name)) {
        return legends[d.data.name]['color'];
    }
    
    var circleColor = d.data.children ? "#F3F2EA" : "white";
    return circleColor;
}

function addLegend(container, legendWidth, legendHeight){
    var legendData = Object.values(legends);
    var svg = d3.select(container)
        .append("svg")
        .attr("width", legendWidth)
        .attr("height", legendHeight);
    
    var groups = svg.selectAll(".groups")
        .data(legendData)
        .enter()
        .append("g")
        .attr("class", "legend")
        .attr("transform", function(d,i){
            return "translate(0," + d.dy + ")"; 
        });

    groups.append("rect")
        .attr("width", 8)
        .attr("height", 8)
        .style("fill", function(d){return d.color});
        
    groups.append("text")
        .attr("x", (d) => 15)
        .attr("dy", "1em")
        .text(function(d){return d.name;})
        .attr("font-family",  "Gill Sans", "Gill Sans MT")
        .attr("font-size", function(d){
            return 10;
        })
        .attr("fill", fontColor);
}



