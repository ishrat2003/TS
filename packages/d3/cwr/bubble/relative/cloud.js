function display(data, container, cloudWidth, cloudHeight, fontMultiplier){
    var allData = []
    data['children'].forEach(child => {
        var allChild = [];
        child['children'].forEach(grandChild => {
            grandChild['parent_name'] = child['name'];
            allChild = allChild.concat(grandChild);
        });
        allData = allData.concat(allChild);
    });

    if(!allData) return;
    var margin = {top: 0, right: 0, bottom: 0, left: 0},
    cloudWidth = cloudWidth - margin.left - margin.right,
    cloudHeight = cloudHeight - margin.top - margin.bottom;

    var svg = d3.select(container).append("svg")
            .attr("width", cloudWidth + margin.left + margin.right)
            .attr("height", cloudHeight + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var wordcloud = svg.append("g")
        .attr('class','wordcloud')
        .attr("transform", "translate(" + cloudWidth/2 + "," + cloudHeight/2 + ")");


    var layout = d3.layout.cloud()
        .timeInterval(10)
        .size([cloudWidth, cloudHeight])
        .words(allData)
        .rotate(function(d) { return 0; })
        .font('monospace')
        .fontSize(function(d,i) {return d.size * fontMultiplier; })
        .text(function(d) { return d.name; })
        .spiral("archimedean")
        .on("end", draw)
        .start();

    
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + cloudHeight + ")")
        .selectAll('text')
        .style('font-size','20px')
        .style('fill',function(d) { return legends[d.parent_name]['color']; })
        .style('font','sans-serif');

    function draw(words) {
        wordcloud.selectAll("text")
            .data(words)
            .enter().append("text")
            .attr('class','word')
            .style("font-size", function(d) { return d.size; })
            .style("font-family", function(d) { return d.font; })
            .style("fill", function(d) { 
                return legends[d.parent_name]['color'];
            })
            .attr("text-anchor", "middle")
            .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
            .text(function(d) { return d.name; });
    };
}