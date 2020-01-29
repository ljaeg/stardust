#all totals
total = set()
f1 = open("verified_codes.txt", "r")
f2 = open("verified_codes_2.txt", "r")

i = 0
for name in f1.read().splitlines():
	i += 1
	total.add(name)

j = 0
for name in f2.read().splitlines():
	j += 1
	total.add(name)

f_total = open("all_codes_gen1", "w")

for code in total:
	f_total.write(code)
	f_total.write("\n")

print("f1: ", i)
print("f2: ", j)
print("total: ", len(total))