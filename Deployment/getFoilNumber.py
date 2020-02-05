
import mysql.connector
import os
import xlsxwriter

mydb = mysql.connector.connect(
  host="flair.ssl.berkeley.edu",
  user="stardust",
  passwd="56y$Uq2CY",
  database="foils_20181207"
)

Dir = "/Users/loganjaeger/Desktop/stardust/"
fname = "all_codes_batch1.txt"
#note that I'm mirroring the data labeling system used in the SQL file for the variable fname
#double check that the fname and the database name match

# if os.path.isfile(Dir + fname):
# 	print("file already exists")
# else:
file = open(Dir + fname, "r")
to_write = open("keys_to_foils.txt", "w")
workbook = xlsxwriter.Workbook('Key_to_foil.xlsx')
worksheet = workbook.add_worksheet()
row = 0

for key in file.read().splitlines():
	query = "SELECT file_comment FROM `real_movie` WHERE amazon_key = '{x}'".format(x = key)
	cursor = mydb.cursor()
	cursor.execute(query)
	result = cursor.fetchall()
	worksheet.write(row, 0, key)
	worksheet.write(row, 1, result[0][0][5:])
	row += 1

workbook.close()

