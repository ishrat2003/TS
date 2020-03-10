import lc as LC

text = """
Brexit: Customs checks to be simplified in no-deal situation.
Lorries will be able to drive straight off ferries and Channel Tunnel trains without making customs declarations in the event of a no-deal Brexit, the government has announced.
New guidance for importers and hauliers says firms would file a simplified form online in advance and pay duty later.
Hauliers have warned that no-deal could result in long queues at Channel ports.
The industry said firms would still not be ready for a chaotic EU exit - even with these simplified procedures.
The UK is due to leave the EU at 23:00 GMT on Friday 29 March - with or without a deal.
Theresa May has said she is determined to deliver Brexit on time, but a number of cabinet ministers have indicated they would be willing to agree to a short extension to finalise legislation.
Media captionUp to 90 lorries assembled at Manston airfield as part of a no-deal Brexit exercise
These would allow an importer to file a very short customs form - a simplified frontier declaration" - only two hours before a lorry is due to cross the Channel by ferry, or one hour via the Channel Tunnel.
The truck would then be able to drive straight into the UK without any further paperwork being done at the border.
The importer would have to update the computer entry within 24 hours to tell HMRC the goods had arrived, and the duty would be payable as much as a month after the shipment had entered the UK.
The temporary system would be reviewed after three months, but is expected to last more than a year.
The latest guidance applies only to vehicles entering the UK, but additional customs checks may also be introduced for EU-bound lorries arriving at Calais, Coquelles and Dunkirk in the event of no-deal.
Charlie Elphicke, the Conservative MP for Dover - home to the UK's busiest Channel port - described the plans as a "common sense move".
He said he had long argued that checks can be done away from the border - so traffic can keep flowing smoothly.
However, Rod McKenzie, from the Road Haulage Association, said the guidance would not help trucking firms.
Business is simply not ready for a chaotic no-deal Brexit, he said.
The systems aren't in place, the staff are not trained, there isn't the time in the day for hauliers and businesses to do all the paperwork, he told the BBC
Last month, a convoy of 89 lorries took part in two runs from the disused Manston Airport, near Ramsgate in Kent, on a 20-mile route to the Port of Dover as part of an exercise to test plans for border disruption in the event of a no-deal Brexit.
It emerged on Tuesday that the government plans to pay a law firm £800,000 for advice in case Eurotunnel decides to sue over the effects of Brexit on its business.
"""

allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
#allowedTypes = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
#allowedTypes = ['JJ', 'JJR', 'JJS']
#allowedTypes = ['RB', 'RBR', 'RBS']
#allowedTypes = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
#allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

tsneProcessor = LC.TSNELC(text)
tsneProcessor.setAllowedPosTypes(allowedTypes)
tsneProcessor.setPerplexity(3)
tsneProcessor.setNumberOfComponents(2)
tsneProcessor.setNumberOfIterations(500)
tsneProcessor.setTopScorePercentage(0.8)
tsneProcessor.setFilterWords(0.2)
tsneProcessor.loadSentences(text)
tsneProcessor.loadFilteredWords()
tsneProcessor.train()
print(tsneProcessor.getWordInfo())

tsneProcessor.setMarkedWords(['brexit', 'uk', 'custom', 'channel', 'lorri'])
tsneProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/tsne.png')

peripheralProcessor = LC.Peripheral(text)
peripheralProcessor.setAllowedPosTypes(allowedTypes)
peripheralProcessor.setPositionContributingFactor(1)
peripheralProcessor.setOccuranceContributingFactor(1)
peripheralProcessor.setProperNounContributingFactor(1)
peripheralProcessor.setTopScorePercentage(0.6)
peripheralProcessor.setFilterWords(0.1)
peripheralProcessor.loadSentences(text)
peripheralProcessor.loadFilteredWords()
peripheralProcessor.train()
peripheralProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/peripheral.png')

linearProcessor = LC.Linear(text)
linearProcessor.setAllowedPosTypes(allowedTypes)
linearProcessor.setPositionContributingFactor(1)
linearProcessor.setOccuranceContributingFactor(1)
linearProcessor.setProperNounContributingFactor(1)
linearProcessor.setTopScorePercentage(0.6)
linearProcessor.setFilterWords(0.1)
linearProcessor.loadSentences(text)
linearProcessor.loadFilteredWords()
linearProcessor.train()
linearProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/linear.png')
print(linearProcessor.getProperNouns())

print(linearProcessor.getContrinutors())

