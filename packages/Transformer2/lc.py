import lc as LC

text1 = """
Over 60 years after the first excavations at Qumran, researchers from Hebrew University said Wednesday that they identified a twelfth cave near Qumran they believe contained Dead Sea Scrolls until it was plundered in the middle of the 20th century. 
  
 The latest excavation was conducted by Hebrew University and the Israel Antiquities Authority under the auspices of the IDF’s Civil Administration. 
  
 Get The Times of Israel's Daily Edition by email and never miss our top stories Free Sign Up 
  
 It yielded no new scrolls, but archaeologists found a small scrap of parchment in a jar and a collection of at least seven storage jugs identical to those found in the other Qumran caves. 
  
 Altogether there was “no doubt we have a new scroll cave,” Oren Gutfeld, head archaeologist from the dig, told The Times of Israel. 
  
 “Only the scrolls themselves are not there.” 
  
 The bit of parchment and other organic remains have been dated to the first century CE, when the community at Qumran was active during the twilight of the Second Temple period. 
  
 Pickaxes from the 1940s, a smoking gun from the Bedouin plunderers who dug in the cave, were found along with the ancient remains. 
  
 The dig in the cliffs west of Qumran, situated over the Green Line in the West Bank, was headed by Hebrew University’s Oren Gutfeld and Ahiad Ovadia with the collaboration of Randall Price and students from Virginia’s Liberty University. 
  
 “This exciting excavation is the closest we’ve come to discovering new Dead Sea Scrolls in 60 years,” Gutfeld said. “Until now, it was accepted that Dead Sea Scrolls were found only in 11 caves at Qumran, but now there is no doubt that this is the twelfth cave.” 
  
 At the same time, Gutfeld said, the cave’s association with the Dead Sea Scrolls means “we can no longer be certain that the original locations (Caves 1 through 11) attributed to the Dead Sea Scrolls that reached the market via the Bedouins are accurate.” 
  
 The first batch of ancient scrolls plundered from caves near the shores of the Dead Sea were purchased by Israeli scholars from the black market in 1947, and additional texts surfaced in the years following in excavations in the Jordanian-held West Bank and for sale on the black market. After Israel captured the West Bank in 1967, many of the scrolls stored in the Rockefeller Museum in East Jerusalem were transferred to the Israel Museum. 
  
 Altogether, the nearly 1,000 ancient Jewish texts dated to the Second Temple period comprise a vast corpus of historical and religious documents that include the earliest known copies of biblical texts. 
  
 Roughly a quarter of the manuscripts are made up of material belonging to the Hebrew Bible, while another quarter detail the Qumran community’s unique philosophy. 
  
 The various scrolls and scroll fragments are identified by the cave they were believed to be stored in over the centuries. The new cave’s discovery shakes things up. 
  
 “How can we know for sure that they only came from 11 caves? For sure there were 12 caves, and maybe more,” Gutfeld said. 
  
 Among the other finds discovered in the cavern, now designated Q12 to denote its inclusion in the Qumran cave complex, were a leather strap for binding scrolls and a cloth for wrapping them, the university said in a statement announcing the find. Other discoveries included flint blades, arrowheads, and a carnelian stamp seal, all of which point to the cave’s inhabitation as far back as the Chalcolithic and the Neolithic periods. 
  
 Experts at the Dead Sea Scroll Laboratories in Jerusalem found no writing on the scrap of parchment found in the jar, but they plan to carry out multispectral imaging of the artifact to reveal any ink invisible to the naked eye. 
  
 The Q12 study was carried out as part of the IAA’s efforts to systematically excavate Judean Desert caves that may hold ancient scroll caches in a bid to foil antiquities theft. The expedition to Qumran was the first of its kind in the northern Judean Desert. 
  
 The IAA announced in November that it was launching a massive project to find as yet undiscovered Dead Sea Scrolls in the desert. Last summer an IAA team excavated the Cave of the Skulls in Zeelim Valley after the antiquities watchdog caught thieves in the act. 
  
 Gutfeld said he and his team “absolutely” plan to survey more caves in the region of Qumran in the coming months to determine where else to dig. 
  
 ———————— 
  
 Follow Ilan Ben Zion on Twitter and Facebook. 
  
"""

text2 = """
Story highlights First discovery of a Dead Sea cave in over 60 years; "Operation Scroll" is a yearslong effort to survey Qumran cliffs 
  
 Archaeologists think the new cave was looted around the 1950s 
  
 (CNN) Excavations on the storied Judean cliffside revealed a new Dead Sea Scrolls cave, full of scroll storage jars and other antiquities, the first such discovery in over 60 years. 
  
 The discovery upends a decades-old theory in the archaeological community that Dead Sea Scrolls were only found in certain caves at the Qumran cliffs, which are managed by Israel in the West Bank. 
  
 Entrance of newly discovered Dead Sea Scrolls cave. 
  
 "Until now, it was accepted that Dead Sea Scrolls were found only in 11 caves at Qumran, but now there is no doubt that this is the 12th cave," said Dr. Oren Gutfeld, one of the project's lead archaeologists. 
  
 Pottery shards, broken scroll storage jars and their lids -- even neolithic flint tools and arrowheads -- littered the cave's entrance. Farther in, there appeared to be a cave-in. 
  
 Neolithic flint tools found inside the newly discovered cave. 
  
 After a bit of work with a small pickax, the team made a monumental find: an unbroken storage jar with a scroll. It was rushed to Hebrew University's conservation lab, where it was unfurled in a protected environment. 
  
 Read More

"""

summaryText = """
Israeli researchers have discovered what they believe is the first new Dead Sea Scrolls cave uncovered in more than 60 years—but looters got there long before them. The site at the Qumran cliffs, an Israeli-controlled site in the West Bank, has yielded artifacts including pieces of pottery, broken scroll storage jars, and even an unbroken jar containing a scroll, though researchers later found it was blank, CNN reports. Clues including old pickaxes have led the Hebrew University team to believe that the site was ransacked in the 1940s or 1950s by looters who made off with ancient scrolls. There is "no doubt we have a new scroll cave," Oren Gutfeld, chief archaeologist on the dig, tells the Times of Israel. "Only the scrolls themselves are not there." Researchers believe the scrolls looted from the cave were sold on the black market many years ago, possibly as long ago as 1947. Gutfeld says the discovery of the cave upends the theory that the scrolls were held in only 11 caves, because this was definitely a 12th. His team plans to survey more of the hundreds of caves in the area in the hope of finding more of the scrolls, which held ancient religious and historical writings. (Israeli authorities busted a gang that was trying to steal an ancient comb from a cave in the area.)
"""

filePrefix = 'text1-NN-NNP-NNs-NNPS'
text = text1
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
tsneProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/' + filePrefix + 'tsne.png')

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
peripheralProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/' + filePrefix + 'peripheral.png')

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
linearProcessor.displayPlot('/content/drive/My Drive/Colab Notebooks/data/images/lc/' + filePrefix + 'linear.png')
print(linearProcessor.getProperNouns())

print(linearProcessor.getContrinutors())



