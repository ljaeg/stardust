#duplicates in the yesCodes.txt?

yes_codes = open("/Users/loganjaeger/Desktop/stardust/yesCodes.txt", "r")
x = set()
i = 0
for line in yes_codes.read().splitlines():
	x.add(line)
	i+=1

print("length of yes codes: ", i)
print("number of unique elements: ", len(x))