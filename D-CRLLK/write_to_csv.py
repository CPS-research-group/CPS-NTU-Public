import csv

def write_to_csv_runner(data_to_append,csv_file_path = 'runner_result.csv'):
    with open(csv_file_path, 'a+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data_to_append)

def write_to_csv_update_result(data_to_append,csv_file_path = 'update_result.csv'):
    with open(csv_file_path, 'a+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data_to_append)
