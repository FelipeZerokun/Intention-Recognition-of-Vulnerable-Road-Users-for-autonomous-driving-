import pandas as pd

class DataModifier:

    def __init__(self, data_file, column_to_change: str, initial_value, new_value):
        self.filename = data_file
        self.column_to_change = column_to_change
        self.data = pd.read_csv(self.filename)
        self.initial_value = initial_value
        self.new_value = new_value


    def change_column_value(self):
        self.data[self.column_to_change] = self.data[self.column_to_change].apply(
            lambda x: x.replace(self.initial_value, self.new_value)
        )

    def save_changes(self, output_file):
        self.data.to_csv(output_file, index=False)

if __name__ == "__main__":
    data_file = 'data_dir'
    output_file = 'output_dir'
    column_to_change = 'rgb_frame'
    target_data = 'actual_value'
    new_value = 'new_value'

    modifier = DataModifier(data_file, column_to_change, target_data, new_value)
    modifier.change_column_value()
    modifier.save_changes(output_file)