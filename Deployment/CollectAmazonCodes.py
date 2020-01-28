#This is a program for going thru the SAH website and collecting amazon codes of images of craters 
#To store on the machine at Berkeley (Big Dusty).

import mysql.connector
import os

mydb = mysql.connector.connect(
  host="flair.ssl.berkeley.edu",
  user="stardust",
  passwd="56y$Uq2CY",
  database="foils_1-30-2017"
)

Dir = "/home/admin/Desktop/RawDataDeploy/"
fname = "likely_2"
#note that I'm mirroring the data labeling system used in the SQL file for the variable fname
#double check that the fname and the database name match

# if os.path.isfile(Dir + fname):
# 	print("file already exists")
# else:
file = open(Dir + fname + ".txt", "w")
query = "SELECT amazon_key FROM `real_movie` WHERE tech = 0 AND conf / (conf + disconf) > .5 AND conf >= 2 AND bad_focus = 0 AND comment = ''"
cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()
for key in result:
	file.write(key[0])
	file.write("\n")
file.close()

