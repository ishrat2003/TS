from packages.gc.vocab import Vocab
vocabProcessor = Vocab(dataProcessor)
if train:
    vocabProcessor.buildVocab()

from packages.gc.peripheral import Peripheral
peripheralProcessor = Peripheral(dataProcessor)


# Display top 50 points (all topics)
peripheralProcessor.setTotalAngle(360)
peripheralProcessor.setStartAngle(0)
wordInfo = peripheralProcessor.getPoints(50)
plotProcessor = Plotter(wordInfo)
plotProcessor.displayPlot()

peripheralProcessor.setStartAngle(0)
peripheralProcessor.setTotalAngle(360 / 10)

numberOfWords = 15
peripheralProcessor.setTopicFilter(1)
wordInfo = peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(2)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(3)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(4)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(5)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(6)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(7)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(8)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

peripheralProcessor.setTopicFilter(9)
wordInfo += peripheralProcessor.getPoints(numberOfWords)

plotProcessor = Plotter(wordInfo)
plotProcessor.displayPlot()
