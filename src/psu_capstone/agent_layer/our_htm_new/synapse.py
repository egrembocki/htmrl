class Synapse:
    def __init__(self, source_input_index, permanence):

        self.source_input_index = source_input_index

        self.permanence = permanence

    def is_connected(self, connected_perm):
        return self.permanence >= connected_perm

    def increment_permanence(self, amount, connected_perm):
        was_connected = self.permanence >= connected_perm
        self.permanence = min(1.0, self.permanence + amount)
        now_connected = self.permanence >= connected_perm
        return 1 if (now_connected and not was_connected) else 0

    def decrement_permanence(self, amount, connected_perm):
        was_connected = self.permanence >= connected_perm
        self.permanence = max(0.0, self.permanence - amount)
        now_connected = self.permanence >= connected_perm
        return -1 if (was_connected and not now_connected) else 0
