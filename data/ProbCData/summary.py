import csv 

label_info_file = "Problem C - CollegeScorecardDataDictionary-09-08-2015.csv"
data_file = "Problem C - Most Recent Cohorts Data (Scorecard Elements).csv"


with open(data_file) as csvfile:
	data = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	labels = data.fieldnames
print(labels)

with open(label_info_file) as csvfile:
	data = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	variable_dict = {}
	for row in data:
		#print(row["VARIABLE NAME"], row["VALUE"])
		variable_dict[row["VARIABLE NAME"]] = row["NAME OF DATA ELEMENT"]
	

for label in labels:
	print("{0}: {1}".format(label, variable_dict[label]))


#def main():
#	make_readable(bla)

#if __name__ == "__main__":
#	main()