import csv

# Open the CSV file in read mode and the text file in write mode
with open('similar_exercises.csv', 'r') as csv_file, open('similar.txt', 'w') as txt_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Join the row elements with a space separator
        row_txt = ' '.join(row)
        # Write the row to the text file
        txt_file.write(row_txt + '\n')