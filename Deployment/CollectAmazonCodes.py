#This is a program for going thru the SAH website and collecting amazon codes of images of craters 
#To store on the machine at Berkeley .

import mysql.connector
import os

mydb = mysql.connector.connect(
  host="flair.ssl.berkeley.edu",
  user="stardust",
  passwd="56y$Uq2CY",
  database="foils_20181207"
)

Dir = "/home/admin/Desktop/RawDataDeploy/"
fname = "disconf_is_0"
#note that I'm mirroring the data labeling system used in the SQL file for the variable fname
#double check that the fname and the database name match
already_seen = set()
seen_codes = open("yesCodes.txt", "r")
for c in seen_codes.read().splitlines():
	already_seen.add(c)
seen_codes.close()
# if os.path.isfile(Dir + fname):
# 	print("file already exists")
# else:
file = open(Dir + fname + ".txt", "w")
query = "SELECT amazon_key FROM `real_movie` WHERE tech = 0 AND disconf = 0"
cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()
i = 0
for key in result:
	i += 1
	if key in already_seen:
		continue
	file.write(key[0])
	file.write("\n")
print(i)
file.close()

