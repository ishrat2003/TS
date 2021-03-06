import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

class Plotter():    
    

    def __init__(self, wordInfo):
        self.wordInfo = wordInfo
        self.colors = ['crimson', 'fuchsia', 'pink', 'plum', 
            'violet', 'darkorchid', 'royalblue', 
            'dodgerblue', 'lightskyblue', 'aqua', 'aquamarine', 'green', 
            'yellowgreen', 'yellow', 'lightyellow', 'lightsalmon', 
            'coral', 'tomato', 'brown', 'maroon', 'gray']
        return


    def displayPlot(self, path):
        #rcParams['figure.figsize']=15,10
        mpl.rcParams.update({'font.size': 22})
        points = self.getPoints()
        if not points:
            print('No points to display')
            return

        plt.figure(figsize=(20, 20))  # in inches(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)[source]
        for point in points:
            plt.scatter(point['x'], point['y'], c = point['color'])
            plt.annotate(point['label'], 
                xy=(point['x'], point['y']), 
                xytext=(5, 2), 
                textcoords='offset points', 
                ha='right', 
                va='bottom')
        plt.savefig(path)
        plt.show()
        return


    def getPoints(self):
        if not len(self.wordInfo):
            return None
        
        topicColors = {}
        colorIndex = 0
        
        points = []
        for word in self.wordInfo:
            if word['topic'] in topicColors.keys():
                color = topicColors[word['topic']]
            else:
                color = self.colors[colorIndex]
                topicColors[word['topic']] = color
                colorIndex += 1
                
            point = {}
            point['x'] = word['x']
            point['y'] = word['y']
            point['color'] = color
            point['label'] = word['label']
            points.append(point)

        return points