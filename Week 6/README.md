## Week 6: File Ingestion and Schema Validation

Take any csv/text file of 2+ GB of your choice. --- (You can do this assignment on Google colab)

Read the file ( Present approach of reading the file )

Try different methods of file reading eg: Dask, Modin, Ray, pandas and present your findings in term of computational efficiency

Perform basic validation on data columns : eg: remove special character , white spaces from the col name

As you already know the schema hence create a YAML file and write the column name in YAML file. --define separator of read and write file, column name in YAML

Validate number of columns and column name of ingested file with YAML.

Write the file in pipe separated text file (|) in gz format.

Create a summary of the file:

- Total number of rows,

- Total number of columns

- File size

