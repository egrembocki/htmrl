class Cell:
    def __init__(self, index, column_index):
        self.index = index

        self.column_index = column_index

        self.active = False

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False
